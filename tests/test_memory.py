"""
Unit and Integration Tests for Step 4B-3: Memory Footprint Benchmarking.

Verifies:
1. Process RSS measurement via psutil (get_process_rss_mb).
2. ProcessMemoryTracker background polling and peak tracking.
3. MemoryMetrics dataclass attributes, defaults, and dictionary serialization.
4. PyTorch memory measurement on CPU and Apple Silicon MPS.
5. ONNX Runtime memory measurement, static batch validation, and missing artifact handling.
6. TensorFlow Lite memory measurement, dynamic batch resizing, and missing artifact handling.
7. Dispatcher (measure_memory) across runtimes and invalid input validation.
8. Coverage across all 4 registered models (MobileNetV3, ResNet50, EfficientNet-B0, DenseNet121).
9. BenchmarkEngine and BenchmarkResult integration, error handling, and serialization.
10. Sample benchmarks CLI execution with memory metrics enabled.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine
from benchmarking.memory import (
    ProcessMemoryTracker,
    get_process_rss_mb,
    measure_memory,
    measure_onnx_memory,
    measure_pytorch_memory,
    measure_tflite_memory,
)
from benchmarking.results import BenchmarkResult, MemoryMetrics
from benchmarking.sample_benchmarks import run_sample_benchmarks
from models.model_registry import list_supported_models


# ============================================================================
# 1. Process Memory Sampling & Tracker Tests
# ============================================================================
def test_get_process_rss_mb() -> None:
    """Verify get_process_rss_mb returns a positive float representing process RSS in MiB."""
    rss = get_process_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_process_memory_tracker_lifecycle() -> None:
    """Verify ProcessMemoryTracker accurately monitors and records peak RSS."""
    tracker = ProcessMemoryTracker(interval_seconds=0.001)
    tracker.start()

    # Allocate a temporary buffer to ensure memory increases
    buf = bytearray(10 * 1024 * 1024)  # 10 MB
    sample_val = tracker.sample()
    assert sample_val > 0.0

    peak = tracker.stop()
    assert peak >= sample_val
    del buf


def test_process_memory_tracker_context_manager() -> None:
    """Verify ProcessMemoryTracker works cleanly as a context manager."""
    with ProcessMemoryTracker(interval_seconds=0.001) as tracker:
        sample_val = tracker.sample()
        assert sample_val > 0.0
    # After exiting context, tracker is stopped
    assert tracker._running is False


# ============================================================================
# 2. MemoryMetrics Structure & Serialization Tests
# ============================================================================
def test_memory_metrics_defaults() -> None:
    """Verify MemoryMetrics default attributes are None."""
    metrics = MemoryMetrics()
    assert metrics.peak_ram_mb is None
    assert metrics.ram_delta_mb is None
    assert metrics.baseline_ram_mb is None
    assert metrics.post_load_ram_mb is None
    assert metrics.inference_ram_delta_mb is None
    assert metrics.device_memory_mb is None
    assert metrics.notes is None

    d = metrics.to_dict()
    assert isinstance(d, dict)
    assert d["peak_ram_mb"] is None


def test_memory_metrics_populated_serialization() -> None:
    """Verify MemoryMetrics serialization with populated metrics."""
    metrics = MemoryMetrics(
        peak_ram_mb=320.5,
        ram_delta_mb=45.2,
        baseline_ram_mb=275.3,
        post_load_ram_mb=310.1,
        inference_ram_delta_mb=10.4,
        device_memory_mb=12.5,
        notes="Test notes for memory profiling",
    )
    assert metrics.peak_ram_mb == 320.5
    assert metrics.ram_delta_mb == 45.2
    assert metrics.baseline_ram_mb == 275.3
    assert metrics.post_load_ram_mb == 310.1
    assert metrics.inference_ram_delta_mb == 10.4
    assert metrics.device_memory_mb == 12.5
    assert metrics.notes == "Test notes for memory profiling"

    d = metrics.to_dict()
    assert d["peak_ram_mb"] == 320.5
    assert d["ram_delta_mb"] == 45.2
    assert d["device_memory_mb"] == 12.5


# ============================================================================
# 3. PyTorch Memory Benchmarking Tests
# ============================================================================
def test_measure_pytorch_memory_cpu() -> None:
    """Verify PyTorch memory measurement on CPU."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        warmup_runs=1,
        num_runs=2,
        device="cpu",
    )
    metrics = measure_pytorch_memory("mobilenet_v3_small", config)

    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0
    assert metrics.baseline_ram_mb is not None and metrics.baseline_ram_mb > 0.0
    assert metrics.post_load_ram_mb is not None and metrics.post_load_ram_mb > 0.0
    assert metrics.ram_delta_mb is not None and metrics.ram_delta_mb >= 0.0
    assert metrics.inference_ram_delta_mb is not None and metrics.inference_ram_delta_mb >= 0.0
    assert metrics.device_memory_mb is None
    assert metrics.notes is not None
    assert "Process RSS" in metrics.notes


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Apple Silicon MPS is not available on this host.",
)
def test_measure_pytorch_memory_mps() -> None:
    """Verify PyTorch memory measurement on Apple Silicon MPS."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        warmup_runs=1,
        num_runs=2,
        device="mps",
    )
    metrics = measure_pytorch_memory("mobilenet_v3_small", config)

    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0
    assert metrics.baseline_ram_mb is not None and metrics.baseline_ram_mb > 0.0
    assert metrics.post_load_ram_mb is not None and metrics.post_load_ram_mb > 0.0
    assert metrics.device_memory_mb is not None and metrics.device_memory_mb > 0.0
    assert metrics.notes is not None
    assert "torch.mps.current_allocated_memory()" in metrics.notes


# ============================================================================
# 4. ONNX Runtime Memory Benchmarking Tests
# ============================================================================
def test_measure_onnx_memory_mobilenet_v3_small() -> None:
    """Verify ONNX Runtime memory measurement on existing MobileNetV3 artifact."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_onnx_memory("mobilenet_v3_small", config)

    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0
    assert metrics.baseline_ram_mb is not None and metrics.baseline_ram_mb > 0.0
    assert metrics.post_load_ram_mb is not None and metrics.post_load_ram_mb > 0.0
    assert metrics.ram_delta_mb is not None and metrics.ram_delta_mb >= 0.0
    assert metrics.device_memory_mb is None
    assert metrics.notes is not None
    assert "ONNX Runtime" in metrics.notes


def test_measure_onnx_memory_incompatible_batch_size_error() -> None:
    """Verify ONNX memory measurement raises ValueError for static batch mismatch."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        batch_size=4,
        warmup_runs=1,
        num_runs=2,
    )
    with pytest.raises(ValueError, match="is incompatible with static ONNX input shape"):
        measure_onnx_memory("mobilenet_v3_small", config)


def test_measure_onnx_memory_missing_artifact_error(tmp_path: Path) -> None:
    """Verify ONNX memory measurement raises FileNotFoundError for missing file."""
    missing_file = tmp_path / "nonexistent.onnx"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        artifact_path=missing_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        measure_onnx_memory("mobilenet_v3_small", config)


# ============================================================================
# 5. TensorFlow Lite Memory Benchmarking Tests
# ============================================================================
def test_measure_tflite_memory_mobilenet_v3_small() -> None:
    """Verify TFLite memory measurement on existing MobileNetV3 artifact."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_tflite_memory("mobilenet_v3_small", config)

    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0
    assert metrics.baseline_ram_mb is not None and metrics.baseline_ram_mb > 0.0
    assert metrics.post_load_ram_mb is not None and metrics.post_load_ram_mb > 0.0
    assert metrics.ram_delta_mb is not None and metrics.ram_delta_mb >= 0.0
    assert metrics.device_memory_mb is None
    assert metrics.notes is not None
    assert "TensorFlow Lite" in metrics.notes


def test_measure_tflite_memory_dynamic_batch_resizing() -> None:
    """Verify TFLite memory measurement handles dynamic batch resizing."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        batch_size=2,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_tflite_memory("mobilenet_v3_small", config)

    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0


def test_measure_tflite_memory_missing_artifact_error(tmp_path: Path) -> None:
    """Verify TFLite memory measurement raises FileNotFoundError for missing file."""
    missing_file = tmp_path / "nonexistent.tflite"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        artifact_path=missing_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        measure_tflite_memory("mobilenet_v3_small", config)


# ============================================================================
# 6. Dispatcher (measure_memory) Validation Tests
# ============================================================================
def test_measure_memory_dispatcher_pytorch() -> None:
    """Verify measure_memory dispatches to PyTorch."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="pytorch",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_memory(config)
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0


def test_measure_memory_dispatcher_onnx() -> None:
    """Verify measure_memory dispatches to ONNX."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="onnx",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_memory(config)
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0


def test_measure_memory_dispatcher_tflite() -> None:
    """Verify measure_memory dispatches to TFLite."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="tflite",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_memory(config)
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0


def test_measure_memory_invalid_model() -> None:
    """Verify measure_memory raises ValueError for unsupported model ID."""
    config = BenchmarkConfig.__new__(BenchmarkConfig)
    config.model_id = "invalid_model_abc"
    config.runtime = BenchmarkRuntime.PYTORCH

    with pytest.raises(ValueError, match="Unsupported model ID"):
        measure_memory(config)


# ============================================================================
# 7. Coverage Across All 4 Supported Models
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_measure_memory_all_four_models_onnx(model_id: str) -> None:
    """Verify memory measurement succeeds across all 4 models via ONNX."""
    config = BenchmarkConfig(
        model_id=model_id,
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_memory(config)
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0
    assert metrics.ram_delta_mb is not None and metrics.ram_delta_mb >= 0.0


# ============================================================================
# 8. BenchmarkEngine and BenchmarkResult Integration Tests
# ============================================================================
def test_benchmark_engine_memory_method() -> None:
    """Verify BenchmarkEngine.benchmark_memory executes and returns MemoryMetrics."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    metrics = engine.benchmark_memory()
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.peak_ram_mb is not None and metrics.peak_ram_mb > 0.0


def test_benchmark_engine_run_benchmark_all_three_metrics() -> None:
    """Verify run_benchmark() populates model_size, latency, and memory metrics."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark(
        measure_model_size=True,
        measure_latency=True,
        measure_memory=True,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.error_message is None
    assert result.model_size is not None
    assert result.latency is not None
    assert result.memory_footprint is not None
    assert result.memory_footprint.peak_ram_mb is not None
    assert result.memory_footprint.peak_ram_mb > 0.0
    assert result.accuracy is None


def test_benchmark_engine_run_benchmark_without_memory() -> None:
    """Verify run_benchmark with measure_memory=False leaves memory_footprint as None."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark(
        measure_model_size=True,
        measure_latency=True,
        measure_memory=False,
    )
    assert result.success is True
    assert result.memory_footprint is None
    assert result.model_size is not None
    assert result.latency is not None


def test_benchmark_engine_run_benchmark_captures_memory_error(tmp_path: Path) -> None:
    """Verify run_benchmark captures memory measurement failure gracefully."""
    missing_file = tmp_path / "missing_model.onnx"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        artifact_path=missing_file,
    )
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark(
        measure_model_size=False,
        measure_latency=False,
        measure_memory=True,
    )
    assert result.success is False
    assert result.memory_footprint is None
    assert result.error_message is not None
    assert "Memory benchmarking failed" in result.error_message


def test_end_to_end_memory_benchmarking_serialization() -> None:
    """Verify complete end-to-end benchmark run with memory serialization."""
    config = BenchmarkConfig(
        model_id="efficientnet_b0",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark()

    assert result.success is True
    assert result.memory_footprint is not None

    d = result.to_dict()
    assert d["memory_footprint"] is not None
    assert "peak_ram_mb" in d["memory_footprint"]
    assert d["memory_footprint"]["peak_ram_mb"] == result.memory_footprint.peak_ram_mb

    summary = result.summary()
    assert "Memory Footprint:" in summary


def test_run_sample_benchmarks_with_memory(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify run_sample_benchmarks outputs memory columns when enabled."""
    run_sample_benchmarks(
        models=["mobilenet_v3_small"],
        warmup_runs=1,
        num_runs=1,
        measure_memory=True,
    )
    captured = capsys.readouterr()
    assert "Edge Model Benchmark" in captured.out
    assert "Peak RAM" in captured.out
    assert "RAM Delta" in captured.out
