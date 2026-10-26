"""
Memory Footprint Benchmarking Module for Edge Deployment.

Measures process-level Resident Set Size (RSS) memory consumption in MiB using psutil
across model loading, warm-up, and measured inference passes for PyTorch,
ONNX Runtime, and TensorFlow Lite.
"""

import gc
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import onnxruntime as ort
import psutil
import tensorflow as tf
import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.latency import _synchronize_accelerator
from benchmarking.model_size import resolve_artifact_path
from benchmarking.results import MemoryMetrics
from models.model_loader import load_model
from models.model_registry import (
    is_model_supported,
    list_supported_models,
)


def get_process_rss_mb() -> float:
    """Sample the current process Resident Set Size (RSS) in MiB.

    Returns:
        Process RSS in MiB rounded to 2 decimal places.
    """
    process = psutil.Process(os.getpid())
    rss_bytes = process.memory_info().rss
    return round(rss_bytes / (1024.0 * 1024.0), 2)


class ProcessMemoryTracker:
    """Monitors process Resident Set Size (RSS) during execution using a daemon thread."""

    def __init__(self, interval_seconds: float = 0.002) -> None:
        """Initialize the process memory tracker.

        Args:
            interval_seconds: Polling period for sampling RSS during execution.
        """
        self.process = psutil.Process(os.getpid())
        self.interval = interval_seconds
        self.peak_bytes = self.process.memory_info().rss
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        """Continuously sample RSS while running."""
        while self._running:
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        """Start tracking process peak memory in the background."""
        self.peak_bytes = self.process.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def sample(self) -> float:
        """Take an explicit synchronous sample and update the peak."""
        rss = self.process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        return round(rss / (1024.0 * 1024.0), 2)

    def stop(self) -> float:
        """Stop tracking and return the maximum observed RSS in MiB.

        Returns:
            Peak RSS in MiB rounded to 2 decimal places.
        """
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        rss = self.process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        return round(self.peak_bytes / (1024.0 * 1024.0), 2)

    def __enter__(self) -> "ProcessMemoryTracker":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def measure_pytorch_memory(
    model_id: str,
    config: BenchmarkConfig,
) -> MemoryMetrics:
    """Measure memory footprint for a PyTorch model on CPU or Apple Silicon MPS.

    Tracks:
    - baseline_ram_mb: Process RSS prior to model loading (after garbage collection).
    - post_load_ram_mb: Process RSS after model instantiation and eval mode setup.
    - peak_ram_mb: Maximum process RSS recorded during the measured inference phase.
    - ram_delta_mb: Total process memory increase (peak_ram_mb - baseline_ram_mb).
    - inference_ram_delta_mb: Memory increase during inference (peak_ram_mb - post_load_ram_mb).
    - device_memory_mb: PyTorch tensor buffer allocation on MPS, or None on CPU.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.

    Returns:
        Populated MemoryMetrics dataclass.
    """
    gc.collect()
    baseline_ram_mb = get_process_rss_mb()

    device = config.device.strip().lower()
    loaded = load_model(model_id, device=config.device)
    model = loaded.model
    model.eval()

    post_load_ram_mb = get_process_rss_mb()

    in_channels, height, width = config.input_size
    batch_size = config.batch_size

    # Prepare dummy input tensor on target device
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((batch_size, in_channels, height, width)).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_np).to(device)

    # 1. Warm-up iterations (excluded from inference peak tracking)
    with torch.no_grad():
        for _ in range(config.warmup_runs):
            _ = model(dummy_tensor)
            _synchronize_accelerator(device)

    # 2. Measured inference phase: monitor peak memory
    tracker = ProcessMemoryTracker()
    tracker.start()
    with torch.no_grad():
        for _ in range(config.num_runs):
            _synchronize_accelerator(device)
            _ = model(dummy_tensor)
            _synchronize_accelerator(device)
            tracker.sample()

    peak_ram_mb = tracker.stop()
    peak_ram_mb = max(peak_ram_mb, post_load_ram_mb, baseline_ram_mb)

    # Handle accelerator-specific memory for Apple Silicon MPS
    device_memory_mb: Optional[float] = None
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        _synchronize_accelerator("mps")
        mps_bytes = torch.mps.current_allocated_memory()
        device_memory_mb = round(mps_bytes / (1024.0 * 1024.0), 2)
        notes = (
            "Process RSS measured via psutil. device_memory_mb reflects torch.mps.current_allocated_memory() "
            "in Apple Silicon unified memory; does not represent discrete GPU VRAM."
        )
    else:
        notes = (
            "Process RSS measured via psutil. Reflects process-level resident memory (RAM) on macOS; "
            "does not isolate model parameters from Python runtime overhead."
        )

    ram_delta_mb = round(max(0.0, peak_ram_mb - baseline_ram_mb), 2)
    inference_ram_delta_mb = round(max(0.0, peak_ram_mb - post_load_ram_mb), 2)

    return MemoryMetrics(
        peak_ram_mb=peak_ram_mb,
        ram_delta_mb=ram_delta_mb,
        baseline_ram_mb=baseline_ram_mb,
        post_load_ram_mb=post_load_ram_mb,
        inference_ram_delta_mb=inference_ram_delta_mb,
        device_memory_mb=device_memory_mb,
        notes=notes,
    )


def measure_onnx_memory(
    model_id: str,
    config: BenchmarkConfig,
) -> MemoryMetrics:
    """Measure memory footprint for an ONNX Runtime model.

    Tracks process RSS across baseline, session creation, warm-up, and inference.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.

    Returns:
        Populated MemoryMetrics dataclass.

    Raises:
        FileNotFoundError: If the ONNX artifact is missing from disk.
        ValueError: If batch size is incompatible with static input dimensions.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.ONNX, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"ONNX model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run ONNX conversion first."
        )

    gc.collect()
    baseline_ram_mb = get_process_rss_mb()

    providers = [config.backend_provider] if config.backend_provider else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(artifact_path), providers=providers)

    in_details = session.get_inputs()[0]
    input_name = in_details.name
    expected_shape = in_details.shape

    if isinstance(expected_shape[0], int) and expected_shape[0] != config.batch_size:
        raise ValueError(
            f"Batch size {config.batch_size} is incompatible with static ONNX input shape {expected_shape} "
            f"for model '{model_id}'."
        )

    post_load_ram_mb = get_process_rss_mb()

    in_channels, height, width = config.input_size
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((config.batch_size, in_channels, height, width)).astype(np.float32)
    feed = {input_name: dummy_np}

    # 1. Warm-up iterations (excluded from inference peak tracking)
    for _ in range(config.warmup_runs):
        _ = session.run(None, feed)

    # 2. Measured inference phase
    tracker = ProcessMemoryTracker()
    tracker.start()
    for _ in range(config.num_runs):
        _ = session.run(None, feed)
        tracker.sample()

    peak_ram_mb = tracker.stop()
    peak_ram_mb = max(peak_ram_mb, post_load_ram_mb, baseline_ram_mb)

    ram_delta_mb = round(max(0.0, peak_ram_mb - baseline_ram_mb), 2)
    inference_ram_delta_mb = round(max(0.0, peak_ram_mb - post_load_ram_mb), 2)

    notes = (
        "Process RSS measured via psutil for ONNX Runtime session and inference; "
        "does not represent on-disk artifact size."
    )

    return MemoryMetrics(
        peak_ram_mb=peak_ram_mb,
        ram_delta_mb=ram_delta_mb,
        baseline_ram_mb=baseline_ram_mb,
        post_load_ram_mb=post_load_ram_mb,
        inference_ram_delta_mb=inference_ram_delta_mb,
        device_memory_mb=None,
        notes=notes,
    )


def measure_tflite_memory(
    model_id: str,
    config: BenchmarkConfig,
) -> MemoryMetrics:
    """Measure memory footprint for a TensorFlow Lite (.tflite) flatbuffer model.

    Tracks process RSS across baseline, flatbuffer interpreter loading,
    tensor allocation, warm-up, and inference.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.

    Returns:
        Populated MemoryMetrics dataclass.

    Raises:
        FileNotFoundError: If the TFLite artifact is missing from disk.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.TFLITE, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"TFLite model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run TFLite conversion first."
        )

    gc.collect()
    baseline_ram_mb = get_process_rss_mb()

    in_channels, height, width = config.input_size
    batch_size = config.batch_size

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        interpreter = tf.lite.Interpreter(model_path=str(artifact_path))

        in_details = interpreter.get_input_details()[0]
        current_shape = list(in_details["shape"])
        target_shape = [batch_size, height, width, in_channels]

        if current_shape != target_shape:
            interpreter.resize_tensor_input(in_details["index"], target_shape)

        interpreter.allocate_tensors()

    post_load_ram_mb = get_process_rss_mb()

    in_idx = interpreter.get_input_details()[0]["index"]
    rng = np.random.default_rng(42)
    dummy_np = rng.standard_normal((batch_size, height, width, in_channels)).astype(np.float32)
    interpreter.set_tensor(in_idx, dummy_np)

    # 1. Warm-up iterations (excluded from inference peak tracking)
    for _ in range(config.warmup_runs):
        interpreter.invoke()

    # 2. Measured inference phase
    tracker = ProcessMemoryTracker()
    tracker.start()
    for _ in range(config.num_runs):
        interpreter.invoke()
        tracker.sample()

    peak_ram_mb = tracker.stop()
    peak_ram_mb = max(peak_ram_mb, post_load_ram_mb, baseline_ram_mb)

    ram_delta_mb = round(max(0.0, peak_ram_mb - baseline_ram_mb), 2)
    inference_ram_delta_mb = round(max(0.0, peak_ram_mb - post_load_ram_mb), 2)

    notes = (
        "Process RSS measured via psutil for TensorFlow Lite flatbuffer interpreter and execution."
    )

    return MemoryMetrics(
        peak_ram_mb=peak_ram_mb,
        ram_delta_mb=ram_delta_mb,
        baseline_ram_mb=baseline_ram_mb,
        post_load_ram_mb=post_load_ram_mb,
        inference_ram_delta_mb=inference_ram_delta_mb,
        device_memory_mb=None,
        notes=notes,
    )


def measure_memory(config: BenchmarkConfig) -> MemoryMetrics:
    """Orchestrate memory footprint benchmarking across supported runtimes.

    Dispatches to runtime-specific runners (PyTorch, ONNX, TFLite) based on configuration.

    Args:
        config: Validated BenchmarkConfig specifying model, runtime, device, and iteration counts.

    Returns:
        Populated MemoryMetrics dataclass.

    Raises:
        ValueError: If model_id or runtime is unsupported.
        FileNotFoundError: If a required on-disk artifact is missing.
    """
    if not is_model_supported(config.model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{config.model_id}'. Supported models: {supported}")

    norm_runtime = BenchmarkRuntime.from_string(config.runtime)

    if norm_runtime == BenchmarkRuntime.PYTORCH:
        return measure_pytorch_memory(config.model_id, config)
    elif norm_runtime == BenchmarkRuntime.ONNX:
        return measure_onnx_memory(config.model_id, config)
    elif norm_runtime == BenchmarkRuntime.TFLITE:
        return measure_tflite_memory(config.model_id, config)
    else:
        raise ValueError(f"Unhandled benchmark runtime '{norm_runtime}'")
