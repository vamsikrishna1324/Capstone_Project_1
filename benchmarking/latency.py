"""
Inference Latency Benchmarking Module for Edge Deployment.

Measures high-resolution execution timing, throughput (samples/sec), and statistical
distribution percentiles (mean, median, std, min, max, p95, p99) across PyTorch,
ONNX Runtime, and TensorFlow Lite.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.model_size import resolve_artifact_path
from benchmarking.results import LatencyMetrics
from models.model_loader import load_model
from models.model_registry import (
    is_model_supported,
    list_supported_models,
)


def _synchronize_accelerator(device: str) -> None:
    """Synchronize asynchronous GPU/MPS accelerators before and after timing.

    Ensures that non-blocking kernel dispatches on Apple Silicon (MPS) or NVIDIA (CUDA)
    complete before the high-resolution timer is stopped.

    Args:
        device: Hardware device identifier (e.g., 'cpu', 'mps', 'cuda').
    """
    dev = device.strip().lower()
    if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_latency_statistics(
    durations_ms: List[float],
    batch_size: int = 1,
) -> LatencyMetrics:
    """Compute summary and percentile metrics from individual iteration durations.

    Calculates:
    - mean_ms, median_ms, std_ms, min_ms, max_ms
    - p95_ms (95th percentile), p99_ms (99th percentile)
    - throughput_fps: total images processed per second = (batch_size * N) / (total_sec)

    Args:
        durations_ms: List of measured inference pass times in milliseconds.
        batch_size: Number of images processed per forward pass.

    Returns:
        Populated LatencyMetrics dataclass.

    Raises:
        ValueError: If durations_ms is empty or batch_size < 1.
    """
    if not durations_ms:
        raise ValueError("Cannot compute latency statistics on empty durations list.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    durations = np.array(durations_ms, dtype=np.float64)

    mean_ms = float(np.mean(durations))
    median_ms = float(np.median(durations))
    std_ms = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
    min_ms = float(np.min(durations))
    max_ms = float(np.max(durations))
    p95_ms = float(np.percentile(durations, 95))
    p99_ms = float(np.percentile(durations, 99))

    total_time_sec = float(np.sum(durations)) / 1000.0
    if total_time_sec > 0:
        throughput_fps = (batch_size * len(durations)) / total_time_sec
    else:
        throughput_fps = 0.0

    return LatencyMetrics(
        mean_ms=round(mean_ms, 3),
        median_ms=round(median_ms, 3),
        std_ms=round(std_ms, 3),
        min_ms=round(min_ms, 3),
        max_ms=round(max_ms, 3),
        p95_ms=round(p95_ms, 3),
        p99_ms=round(p99_ms, 3),
        throughput_fps=round(throughput_fps, 2),
    )


def measure_pytorch_latency(
    model_id: str,
    config: BenchmarkConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LatencyMetrics:
    """Measure inference latency for a PyTorch model.

    Loads model into memory, sets evaluation mode, allocates deterministic dummy
    input tensor matching (batch_size, 3, H, W) on target device, executes warmup runs
    (excluded from metrics), and times measured runs using time.perf_counter() with
    accelerator synchronization.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        progress_callback: Optional intermediate progress callback.

    Returns:
        LatencyMetrics summarizing the inference profiling.
    """
    loaded = load_model(model_id, device=config.device)
    model = loaded.model
    model.eval()

    device = config.device.strip().lower()
    in_channels, height, width = config.input_size
    batch_size = config.batch_size
    total_steps = (config.warmup_runs + config.num_runs) * batch_size
    completed_steps = 0

    # Generate deterministic dummy input on the target device
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((batch_size, in_channels, height, width)).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_np).to(device)

    # 1. Warm-up iterations (excluded from timing)
    with torch.no_grad():
        for _ in range(config.warmup_runs):
            _ = model(dummy_tensor)
            _synchronize_accelerator(device)
            completed_steps += batch_size
            if progress_callback:
                progress_callback(completed_steps, total_steps)

    # 2. Measured iterations
    durations_ms: List[float] = []
    with torch.no_grad():
        for _ in range(config.num_runs):
            _synchronize_accelerator(device)
            t0 = time.perf_counter()
            _ = model(dummy_tensor)
            _synchronize_accelerator(device)
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)
            completed_steps += batch_size
            if progress_callback:
                progress_callback(completed_steps, total_steps)

    return compute_latency_statistics(durations_ms, batch_size=batch_size)


def measure_onnx_latency(
    model_id: str,
    config: BenchmarkConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LatencyMetrics:
    """Measure inference latency for an ONNX Runtime model.

    Loads the on-disk .onnx artifact, validates input dimensions (NCHW layout),
    generates deterministic dummy input array, executes warmup runs, and profiles
    measured forward passes using high-resolution timing.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        progress_callback: Optional intermediate progress callback.

    Returns:
        LatencyMetrics summarizing the inference profiling.

    Raises:
        FileNotFoundError: If the ONNX file does not exist on disk.
        ValueError: If batch size is incompatible with static input dimensions.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.ONNX, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"ONNX model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run ONNX conversion first."
        )

    providers = [config.backend_provider] if config.backend_provider else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(artifact_path), providers=providers)

    in_details = session.get_inputs()[0]
    input_name = in_details.name
    expected_shape = in_details.shape  # e.g., [1, 3, 224, 224]

    # Validate batch compatibility
    if isinstance(expected_shape[0], int) and expected_shape[0] != config.batch_size:
        raise ValueError(
            f"Batch size {config.batch_size} is incompatible with static ONNX input shape {expected_shape} "
            f"for model '{model_id}'."
        )

    in_channels, height, width = config.input_size
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((config.batch_size, in_channels, height, width)).astype(np.float32)
    feed = {input_name: dummy_np}
    total_steps = (config.warmup_runs + config.num_runs) * config.batch_size
    completed_steps = 0

    # 1. Warm-up iterations (excluded from timing)
    for _ in range(config.warmup_runs):
        _ = session.run(None, feed)
        completed_steps += config.batch_size
        if progress_callback:
            progress_callback(completed_steps, total_steps)

    # 2. Measured iterations
    durations_ms: List[float] = []
    for _ in range(config.num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, feed)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)
        completed_steps += config.batch_size
        if progress_callback:
            progress_callback(completed_steps, total_steps)

    return compute_latency_statistics(durations_ms, batch_size=config.batch_size)


def measure_tflite_latency(
    model_id: str,
    config: BenchmarkConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LatencyMetrics:
    """Measure inference latency for a TensorFlow Lite (.tflite) model.

    Loads the on-disk .tflite flatbuffer, configures input tensor with NHWC layout,
    resizes input tensor if batch_size > 1, executes warmup runs, and measures
    forward pass durations.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        progress_callback: Optional intermediate progress callback.

    Returns:
        LatencyMetrics summarizing the inference profiling.

    Raises:
        FileNotFoundError: If the TFLite file does not exist on disk.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.TFLITE, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"TFLite model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run TFLite conversion first."
        )

    in_channels, height, width = config.input_size
    batch_size = config.batch_size
    total_steps = (config.warmup_runs + config.num_runs) * batch_size
    completed_steps = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        interpreter = tf.lite.Interpreter(model_path=str(artifact_path))

        # Check and handle batch resizing if needed
        in_details = interpreter.get_input_details()[0]
        current_shape = list(in_details["shape"])
        target_shape = [batch_size, height, width, in_channels]

        if current_shape != target_shape:
            interpreter.resize_tensor_input(in_details["index"], target_shape)

        interpreter.allocate_tensors()

    in_idx = interpreter.get_input_details()[0]["index"]

    # Generate deterministic NHWC dummy input
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((batch_size, height, width, in_channels)).astype(np.float32)
    interpreter.set_tensor(in_idx, dummy_np)

    # 1. Warm-up iterations (excluded from timing)
    for _ in range(config.warmup_runs):
        interpreter.invoke()
        completed_steps += batch_size
        if progress_callback:
            progress_callback(completed_steps, total_steps)

    # 2. Measured iterations
    durations_ms: List[float] = []
    for _ in range(config.num_runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)
        completed_steps += batch_size
        if progress_callback:
            progress_callback(completed_steps, total_steps)

    return compute_latency_statistics(durations_ms, batch_size=batch_size)


def measure_latency(
    config: BenchmarkConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LatencyMetrics:
    """Orchestrate inference latency benchmarking across supported runtimes.

    Dispatches to runtime-specific runners (PyTorch, ONNX, TFLite) based on configuration.

    Args:
        config: Validated BenchmarkConfig specifying model, runtime, device, and iteration counts.
        progress_callback: Optional intermediate progress callback.

    Returns:
        LatencyMetrics containing latency percentiles and throughput.

    Raises:
        ValueError: If model_id or runtime is unsupported.
        FileNotFoundError: If a required on-disk artifact is missing.
    """
    if not is_model_supported(config.model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{config.model_id}'. Supported models: {supported}")

    norm_runtime = BenchmarkRuntime.from_string(config.runtime)

    if norm_runtime == BenchmarkRuntime.PYTORCH:
        return measure_pytorch_latency(config.model_id, config, progress_callback=progress_callback)
    elif norm_runtime == BenchmarkRuntime.ONNX:
        return measure_onnx_latency(config.model_id, config, progress_callback=progress_callback)
    elif norm_runtime == BenchmarkRuntime.TFLITE:
        return measure_tflite_latency(config.model_id, config, progress_callback=progress_callback)
    else:
        raise ValueError(f"Unhandled benchmark runtime '{norm_runtime}'")
