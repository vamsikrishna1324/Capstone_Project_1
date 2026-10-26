"""
Unit and Integration Tests for Step 4B-2: Inference Latency Benchmarking.

Verifies:
1. compute_latency_statistics calculation (mean, median, std, min, max, p95, p99, throughput).
2. Warm-up exclusion from timing and metric calculations.
3. Accelerator synchronization across CPU, MPS, and CUDA.
4. PyTorch latency benchmarking (CPU and MPS).
5. ONNX Runtime latency benchmarking, batch shape validation, and missing artifact handling.
6. TFLite latency benchmarking, dynamic batch resizing, and missing artifact handling.
7. Dispatcher (measure_latency) across runtimes and invalid input validation.
8. Coverage across all 4 registered models (MobileNetV3, ResNet50, EfficientNet-B0, DenseNet121).
9. BenchmarkEngine and BenchmarkResult integration, error handling, and serialization.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine
from benchmarking.latency import (
    _synchronize_accelerator,
    compute_latency_statistics,
    measure_latency,
    measure_onnx_latency,
    measure_pytorch_latency,
    measure_tflite_latency,
)
from benchmarking.results import BenchmarkResult, LatencyMetrics
from models.model_registry import list_supported_models


# ============================================================================
# 1. Latency Statistical Calculation Tests
# ============================================================================
def test_compute_latency_statistics_known_values() -> None:
    """Verify statistical metrics calculation against known reference values."""
    durations = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics = compute_latency_statistics(durations, batch_size=1)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms == 30.0
    assert metrics.median_ms == 30.0
    assert metrics.min_ms == 10.0
    assert metrics.max_ms == 50.0
    # Sample standard deviation ddof=1 for [10, 20, 30, 40, 50] is sqrt(250) ~= 15.811
    assert metrics.std_ms == 15.811
    assert metrics.p95_ms == round(float(np.percentile(durations, 95)), 3)
    assert metrics.p99_ms == round(float(np.percentile(durations, 99)), 3)

    # Throughput: (1 batch * 5 runs) / (150 ms / 1000) = 5 / 0.15 = 33.33 fps
    assert metrics.throughput_fps == 33.33


def test_compute_latency_statistics_single_element() -> None:
    """Verify statistics when only a single measurement iteration is provided."""
    durations = [25.0]
    metrics = compute_latency_statistics(durations, batch_size=1)

    assert metrics.mean_ms == 25.0
    assert metrics.median_ms == 25.0
    assert metrics.std_ms == 0.0
    assert metrics.min_ms == 25.0
    assert metrics.max_ms == 25.0
    assert metrics.p95_ms == 25.0
    assert metrics.p99_ms == 25.0
    # (1 * 1) / (0.025 sec) = 40.0 fps
    assert metrics.throughput_fps == 40.0


def test_compute_latency_statistics_batch_scaling() -> None:
    """Verify throughput scales proportionally with batch size."""
    durations = [10.0, 10.0]
    # batch_size=4: total images = 4 * 2 = 8; total time = 20ms = 0.02s -> 400 fps
    metrics = compute_latency_statistics(durations, batch_size=4)

    assert metrics.mean_ms == 10.0
    assert metrics.throughput_fps == 400.0


def test_compute_latency_statistics_empty_durations_error() -> None:
    """Verify ValueError is raised if durations list is empty."""
    with pytest.raises(ValueError, match="Cannot compute latency statistics on empty durations list"):
        compute_latency_statistics([], batch_size=1)


def test_compute_latency_statistics_invalid_batch_size() -> None:
    """Verify ValueError is raised if batch_size < 1."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        compute_latency_statistics([10.0, 15.0], batch_size=0)


# ============================================================================
# 2. Warm-up Exclusion Tests
# ============================================================================
def test_warmup_exclusion_pytorch() -> None:
    """Verify warm-up iterations are invoked but excluded from recorded durations."""
    call_records: list[str] = []

    class DummyPyTorchModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            call_records.append("call")
            return x

    dummy_loaded = MagicMock()
    dummy_loaded.model = DummyPyTorchModel()

    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        warmup_runs=3,
        num_runs=5,
        device="cpu",
    )

    with patch("benchmarking.latency.load_model", return_value=dummy_loaded):
        metrics = measure_pytorch_latency("mobilenet_v3_small", config)

    # 3 warmup + 5 measured = 8 forward calls total
    assert len(call_records) == 8
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms is not None
    assert metrics.throughput_fps is not None


def test_warmup_exclusion_onnx() -> None:
    """Verify ONNX runtime excludes warmup runs from timing."""
    run_records: list[str] = []

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 224, 224]
    mock_session.get_inputs.return_value = [mock_input]

    def mock_run(output_names, input_feed):
        run_records.append("onnx_run")
        return [np.zeros((1, 1000), dtype=np.float32)]

    mock_session.run.side_effect = mock_run

    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=4,
        num_runs=6,
        artifact_path="/dummy/path/mobilenet_v3_small.onnx",
    )

    with patch("benchmarking.latency.resolve_artifact_path", return_value=Path("/dummy/path/mobilenet_v3_small.onnx")):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                with patch("onnxruntime.InferenceSession", return_value=mock_session):
                    metrics = measure_onnx_latency("mobilenet_v3_small", config)

    # 4 warmup + 6 measured = 10 runs
    assert len(run_records) == 10
    assert isinstance(metrics, LatencyMetrics)


# ============================================================================
# 3. Accelerator Synchronization Tests
# ============================================================================
def test_synchronize_accelerator_cpu() -> None:
    """Verify _synchronize_accelerator executes cleanly for CPU."""
    _synchronize_accelerator("cpu")


def test_synchronize_accelerator_mps() -> None:
    """Verify _synchronize_accelerator executes cleanly for MPS device."""
    _synchronize_accelerator("mps")


def test_synchronize_accelerator_cuda_fallback() -> None:
    """Verify _synchronize_accelerator does not crash on CUDA when unavailable."""
    _synchronize_accelerator("cuda")


# ============================================================================
# 4. PyTorch Latency Benchmarking Tests
# ============================================================================
def test_measure_pytorch_latency_cpu() -> None:
    """Verify PyTorch latency measurement on CPU for mobilenet_v3_small."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        warmup_runs=1,
        num_runs=2,
        device="cpu",
    )
    metrics = measure_pytorch_latency("mobilenet_v3_small", config)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.median_ms > 0
    assert metrics.min_ms > 0
    assert metrics.max_ms >= metrics.min_ms
    assert metrics.p95_ms >= metrics.min_ms
    assert metrics.p99_ms >= metrics.min_ms
    assert metrics.throughput_fps > 0


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Apple Silicon MPS is not available on this host.",
)
def test_measure_pytorch_latency_mps() -> None:
    """Verify PyTorch latency measurement on Apple Silicon MPS with synchronization."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        warmup_runs=1,
        num_runs=2,
        device="mps",
    )
    metrics = measure_pytorch_latency("mobilenet_v3_small", config)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.throughput_fps > 0


# ============================================================================
# 5. ONNX Runtime Latency Benchmarking Tests
# ============================================================================
def test_measure_onnx_latency_mobilenet_v3_small() -> None:
    """Verify ONNX Runtime latency measurement on existing MobileNetV3 artifact."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=2,
        num_runs=5,
    )
    metrics = measure_onnx_latency("mobilenet_v3_small", config)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.median_ms > 0
    assert metrics.min_ms > 0
    assert metrics.max_ms >= metrics.min_ms
    assert metrics.throughput_fps > 0


def test_measure_onnx_latency_incompatible_batch_size_error() -> None:
    """Verify ONNX raises ValueError when batch_size does not match static dimension."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        batch_size=4,  # ONNX artifact was exported with static batch=1
        warmup_runs=1,
        num_runs=2,
    )
    with pytest.raises(ValueError, match="is incompatible with static ONNX input shape"):
        measure_onnx_latency("mobilenet_v3_small", config)


def test_measure_onnx_latency_missing_artifact_error(tmp_path: Path) -> None:
    """Verify ONNX raises FileNotFoundError when the artifact does not exist."""
    missing_file = tmp_path / "nonexistent.onnx"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        artifact_path=missing_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        measure_onnx_latency("mobilenet_v3_small", config)


# ============================================================================
# 6. TFLite Latency Benchmarking Tests
# ============================================================================
def test_measure_tflite_latency_mobilenet_v3_small() -> None:
    """Verify TFLite latency measurement on existing MobileNetV3 artifact."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        warmup_runs=2,
        num_runs=5,
    )
    metrics = measure_tflite_latency("mobilenet_v3_small", config)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.median_ms > 0
    assert metrics.min_ms > 0
    assert metrics.max_ms >= metrics.min_ms
    assert metrics.throughput_fps > 0


def test_measure_tflite_latency_dynamic_batch_resizing() -> None:
    """Verify TFLite supports dynamic batch resizing (batch_size=2)."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        batch_size=2,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_tflite_latency("mobilenet_v3_small", config)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.throughput_fps > 0


def test_measure_tflite_latency_missing_artifact_error(tmp_path: Path) -> None:
    """Verify TFLite raises FileNotFoundError when the artifact does not exist."""
    missing_file = tmp_path / "nonexistent.tflite"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        artifact_path=missing_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        measure_tflite_latency("mobilenet_v3_small", config)


# ============================================================================
# 7. Dispatcher (measure_latency) Validation Tests
# ============================================================================
def test_measure_latency_dispatcher_pytorch() -> None:
    """Verify measure_latency dispatches properly to PyTorch."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="pytorch",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_latency(config)
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0


def test_measure_latency_dispatcher_onnx() -> None:
    """Verify measure_latency dispatches properly to ONNX."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="onnx",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_latency(config)
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0


def test_measure_latency_dispatcher_tflite() -> None:
    """Verify measure_latency dispatches properly to TFLite."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime="tflite",
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_latency(config)
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0


def test_measure_latency_invalid_model() -> None:
    """Verify measure_latency raises ValueError for unsupported model ID."""
    config = BenchmarkConfig.__new__(BenchmarkConfig)
    config.model_id = "invalid_model_xyz"
    config.runtime = BenchmarkRuntime.PYTORCH

    with pytest.raises(ValueError, match="Unsupported model ID"):
        measure_latency(config)


# ============================================================================
# 8. Registered Models Coverage Tests (All 4 Models)
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_measure_latency_all_four_models_onnx(model_id: str) -> None:
    """Verify inference latency measurement works across all 4 supported models via ONNX."""
    config = BenchmarkConfig(
        model_id=model_id,
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    metrics = measure_latency(config)
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0
    assert metrics.throughput_fps > 0


# ============================================================================
# 9. BenchmarkEngine and BenchmarkResult Integration Tests
# ============================================================================
def test_benchmark_engine_latency_method() -> None:
    """Verify BenchmarkEngine.benchmark_latency executes and returns LatencyMetrics."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    metrics = engine.benchmark_latency()
    assert isinstance(metrics, LatencyMetrics)
    assert metrics.mean_ms > 0


def test_benchmark_engine_run_benchmark_both_metrics() -> None:
    """Verify run_benchmark() populates both model_size and latency metrics."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=3,
    )
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark(measure_model_size=True, measure_latency=True)

    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.error_message is None
    assert result.model_size is not None
    assert result.model_size.file_size_mb > 0
    assert result.latency is not None
    assert result.latency.mean_ms > 0
    assert result.latency.throughput_fps > 0


def test_benchmark_engine_run_benchmark_captures_latency_error(tmp_path: Path) -> None:
    """Verify run_benchmark captures latency failure without crashing and flags success=False."""
    missing_file = tmp_path / "missing_model.onnx"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        artifact_path=missing_file,
    )
    engine = BenchmarkEngine(config)
    # Turn off model_size measurement so failure occurs at latency stage
    result = engine.run_benchmark(measure_model_size=False, measure_latency=True)

    assert result.success is False
    assert result.latency is None
    assert result.error_message is not None
    assert "Latency benchmarking failed" in result.error_message


# ============================================================================
# 10. End-to-End Latency Benchmark Tests (ONNX and TFLite)
# ============================================================================
def test_end_to_end_latency_benchmark_onnx_and_tflite() -> None:
    """Verify complete end-to-end benchmark run for ONNX and TFLite models."""
    # 1. End-to-end ONNX (ResNet50)
    onnx_cfg = BenchmarkConfig(
        model_id="resnet50",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=2,
        num_runs=3,
    )
    onnx_engine = BenchmarkEngine(onnx_cfg)
    onnx_res = onnx_engine.run_benchmark()

    assert onnx_res.success is True
    assert onnx_res.latency is not None
    assert onnx_res.latency.mean_ms > 0
    assert onnx_res.latency.throughput_fps > 0
    assert onnx_res.model_size is not None

    onnx_dict = onnx_res.to_dict()
    assert onnx_dict["latency"]["mean_ms"] == onnx_res.latency.mean_ms
    assert onnx_dict["latency"]["throughput_fps"] == onnx_res.latency.throughput_fps
    assert "Latency:" in onnx_res.summary()

    # 2. End-to-end TFLite (EfficientNet-B0)
    tflite_cfg = BenchmarkConfig(
        model_id="efficientnet_b0",
        runtime=BenchmarkRuntime.TFLITE,
        warmup_runs=2,
        num_runs=3,
    )
    tflite_engine = BenchmarkEngine(tflite_cfg)
    tflite_res = tflite_engine.run_benchmark()

    assert tflite_res.success is True
    assert tflite_res.latency is not None
    assert tflite_res.latency.mean_ms > 0
    assert tflite_res.latency.throughput_fps > 0
    assert tflite_res.model_size is not None

    tflite_dict = tflite_res.to_dict()
    assert tflite_dict["latency"]["mean_ms"] == tflite_res.latency.mean_ms
    assert tflite_dict["latency"]["throughput_fps"] == tflite_res.latency.throughput_fps
    assert "Latency:" in tflite_res.summary()


def test_run_sample_benchmarks_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify run_sample_benchmarks executes cleanly and outputs formatted table."""
    from benchmarking.sample_benchmarks import run_sample_benchmarks

    run_sample_benchmarks(
        models=["mobilenet_v3_small"],
        warmup_runs=1,
        num_runs=1,
    )
    captured = capsys.readouterr()
    assert "Edge Model Benchmark" in captured.out
    assert "mobilenet_v3_small" in captured.out
    assert "FPS" in captured.out
