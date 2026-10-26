"""
Test Suite for Step 4B-1: Model Size Benchmarking.

Verifies:
- Accurate parameter count calculation from official PyTorch architectures.
- Actual on-disk file size measurement for all 4 supported models across ONNX and TFLite.
- Byte-to-MB conversion consistency.
- Correct handling of PyTorch in-memory model representation (file size None when no serialized artifact exists).
- Correct handling of custom serialized PyTorch checkpoints.
- Controlled FileNotFoundError exceptions when artifact files are missing.
- Controlled ValueError exceptions on invalid model IDs and runtimes.
- Clean integration with BenchmarkEngine, BenchmarkConfig, ModelSizeMetrics, and BenchmarkResult.
- End-to-end model-size benchmarking runs for ONNX and TFLite models.
"""

from pathlib import Path
from typing import Dict
import pytest

from benchmarking import (
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkRuntime,
    ModelSizeMetrics,
    bytes_to_mb,
    get_model_parameter_count,
    measure_model_size,
    resolve_artifact_path,
)
from models.model_registry import list_supported_models

# Expected parameter counts from official torchvision architectures
EXPECTED_PARAM_COUNTS: Dict[str, int] = {
    "mobilenet_v3_small": 2542856,
    "resnet50": 25557032,
    "efficientnet_b0": 5288548,
    "densenet121": 7978856,
}


# ============================================================================
# 1. Byte to MB Conversion & Parameter Count Tests
# ============================================================================
@pytest.mark.parametrize(
    "byte_count,expected_mb",
    [
        (0, 0.0),
        (1048576, 1.0),
        (102154114, 97.42),
        (51112080, 48.74),
        (5131628, 4.89),
        (21167527, 20.19),
        (32339559, 30.84),
    ],
)
def test_bytes_to_mb_conversion(byte_count: int, expected_mb: float) -> None:
    """Verify byte-to-MB conversion produces expected rounded megabyte figures."""
    assert bytes_to_mb(byte_count) == expected_mb


@pytest.mark.parametrize("model_id,expected_count", EXPECTED_PARAM_COUNTS.items())
def test_parameter_count_all_models(model_id: str, expected_count: int) -> None:
    """Verify official PyTorch parameter counts for all 4 supported models."""
    count = get_model_parameter_count(model_id)
    assert count == expected_count, f"Parameter count mismatch for {model_id}: {count} != {expected_count}"


# ============================================================================
# 2. ONNX Model Size Measurement Tests
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_onnx_model_size_measurement_all_models(model_id: str) -> None:
    """Verify ONNX model size measurement on actual generated artifacts on disk."""
    metrics = measure_model_size(model_id, runtime=BenchmarkRuntime.ONNX)

    assert isinstance(metrics, ModelSizeMetrics)
    assert metrics.file_size_bytes is not None and metrics.file_size_bytes > 0
    assert metrics.file_size_mb is not None and metrics.file_size_mb > 0
    assert metrics.file_size_mb == bytes_to_mb(metrics.file_size_bytes)
    assert metrics.param_count == EXPECTED_PARAM_COUNTS[model_id]
    assert metrics.artifact_path is not None
    assert Path(metrics.artifact_path).exists()
    assert Path(metrics.artifact_path).is_file()


# ============================================================================
# 3. TFLite Model Size Measurement Tests
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_tflite_model_size_measurement_all_models(model_id: str) -> None:
    """Verify TFLite model size measurement on actual generated artifacts on disk."""
    metrics = measure_model_size(model_id, runtime=BenchmarkRuntime.TFLITE)

    assert isinstance(metrics, ModelSizeMetrics)
    assert metrics.file_size_bytes is not None and metrics.file_size_bytes > 0
    assert metrics.file_size_mb is not None and metrics.file_size_mb > 0
    assert metrics.file_size_mb == bytes_to_mb(metrics.file_size_bytes)
    assert metrics.param_count == EXPECTED_PARAM_COUNTS[model_id]
    assert metrics.artifact_path is not None
    assert Path(metrics.artifact_path).exists()
    assert Path(metrics.artifact_path).is_file()


# ============================================================================
# 4. PyTorch Model Size Handling Tests
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_pytorch_model_size_default_in_memory(model_id: str) -> None:
    """Verify that PyTorch in-memory models report parameter count but None file size on disk."""
    metrics = measure_model_size(model_id, runtime=BenchmarkRuntime.PYTORCH)

    assert isinstance(metrics, ModelSizeMetrics)
    assert metrics.file_size_bytes is None
    assert metrics.file_size_mb is None
    assert metrics.param_count == EXPECTED_PARAM_COUNTS[model_id]
    assert metrics.artifact_path is None


def test_pytorch_model_size_with_custom_artifact(tmp_path: Path) -> None:
    """Verify that if a serialized PyTorch checkpoint is provided, its file size is measured."""
    dummy_ckpt = tmp_path / "custom_checkpoint.pt"
    dummy_ckpt.write_bytes(b"\x00" * 8192)

    metrics = measure_model_size("mobilenet_v3_small", runtime="pytorch", artifact_path=dummy_ckpt)

    assert metrics.file_size_bytes == 8192
    assert metrics.file_size_mb == bytes_to_mb(8192)
    assert metrics.param_count == EXPECTED_PARAM_COUNTS["mobilenet_v3_small"]
    assert metrics.artifact_path == str(dummy_ckpt)


# ============================================================================
# 5. Error Handling Tests
# ============================================================================
@pytest.mark.parametrize("runtime", [BenchmarkRuntime.ONNX, BenchmarkRuntime.TFLITE])
def test_nonexistent_artifact_handling_onnx_tflite(tmp_path: Path, runtime: BenchmarkRuntime) -> None:
    """Verify FileNotFoundError is raised when an artifact does not exist on disk."""
    missing_path = tmp_path / f"missing_model.{runtime.value}"
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        measure_model_size("resnet50", runtime=runtime, artifact_path=missing_path)


def test_nonexistent_artifact_handling_pytorch_explicit(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when an explicit PyTorch artifact path is missing."""
    missing_path = tmp_path / "missing_pytorch_weights.pt"
    with pytest.raises(FileNotFoundError, match="Specified PyTorch model artifact for 'resnet50' does not exist"):
        measure_model_size("resnet50", runtime="pytorch", artifact_path=missing_path)


@pytest.mark.parametrize("invalid_id", ["unknown_model", "vgg16", "", "   "])
def test_invalid_model_id_handling(invalid_id: str) -> None:
    """Verify ValueError is raised for unregistered model IDs."""
    with pytest.raises(ValueError):
        measure_model_size(invalid_id, runtime="onnx")


@pytest.mark.parametrize("invalid_runtime", ["tensorrt", "openvino", "bad_runtime"])
def test_invalid_runtime_handling(invalid_runtime: str) -> None:
    """Verify ValueError is raised for unsupported runtime strings."""
    with pytest.raises(ValueError, match="Unsupported benchmark runtime"):
        measure_model_size("resnet50", runtime=invalid_runtime)


# ============================================================================
# 6. BenchmarkEngine Integration Tests
# ============================================================================
@pytest.mark.parametrize("runtime", [BenchmarkRuntime.ONNX, BenchmarkRuntime.TFLITE])
def test_benchmark_engine_model_size_integration(runtime: BenchmarkRuntime) -> None:
    """Verify BenchmarkEngine.benchmark_model_size returns accurate disk metrics."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime=runtime)
    engine = BenchmarkEngine(config)

    metrics = engine.benchmark_model_size()
    assert isinstance(metrics, ModelSizeMetrics)
    assert metrics.file_size_bytes is not None and metrics.file_size_bytes > 0
    assert metrics.file_size_mb is not None and metrics.file_size_mb > 0
    assert metrics.param_count == EXPECTED_PARAM_COUNTS["mobilenet_v3_small"]


def test_benchmark_engine_run_benchmark_integration() -> None:
    """Verify BenchmarkEngine.run_benchmark populates model_size in the BenchmarkResult."""
    config = BenchmarkConfig(model_id="resnet50", runtime="onnx")
    engine = BenchmarkEngine(config)

    result = engine.run_benchmark()
    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.error_message is None
    assert result.model_id == "resnet50"
    assert result.runtime == BenchmarkRuntime.ONNX
    assert result.model_size is not None
    assert result.model_size.file_size_bytes > 50_000_000
    assert result.model_size.file_size_mb == bytes_to_mb(result.model_size.file_size_bytes)
    assert result.model_size.param_count == 25557032

    # Latency is populated in Step 4B-2
    assert result.latency is not None
    assert result.latency.mean_ms > 0
    # Memory is populated in Step 4B-3
    assert result.memory_footprint is not None
    assert result.memory_footprint.peak_ram_mb > 0
    # Remaining deferred metrics remain None
    assert result.accuracy is None


def test_benchmark_engine_run_benchmark_missing_artifact_captured(tmp_path: Path) -> None:
    """Verify that run_benchmark gracefully captures missing artifact errors in BenchmarkResult."""
    missing_file = tmp_path / "missing.onnx"
    config = BenchmarkConfig(model_id="resnet50", runtime="onnx", artifact_path=missing_file)
    engine = BenchmarkEngine(config)

    result = engine.run_benchmark()
    assert result.success is False
    assert result.model_size is None
    assert result.error_message is not None
    assert "Model size measurement failed" in result.error_message


# ============================================================================
# 7. End-to-End Tests for Valid ONNX and TFLite Models
# ============================================================================
def test_end_to_end_model_size_benchmarking_onnx_and_tflite() -> None:
    """End-to-end benchmark run for ONNX and TFLite models verifying results, to_dict, and summary."""
    # 1. ONNX Model Run (EfficientNet-B0)
    onnx_cfg = BenchmarkConfig(model_id="efficientnet_b0", runtime=BenchmarkRuntime.ONNX)
    onnx_engine = BenchmarkEngine(onnx_cfg)
    onnx_res = onnx_engine.run_benchmark()

    assert onnx_res.success is True
    assert onnx_res.model_size is not None
    assert onnx_res.model_size.file_size_bytes > 10_000_000
    assert onnx_res.model_size.file_size_mb == bytes_to_mb(onnx_res.model_size.file_size_bytes)
    assert onnx_res.model_size.param_count == 5288548

    onnx_dict = onnx_res.to_dict()
    assert onnx_dict["model_size"]["file_size_mb"] == onnx_res.model_size.file_size_mb
    assert onnx_dict["model_size"]["param_count"] == 5288548
    assert "Model Size" in onnx_res.summary()

    # 2. TFLite Model Run (DenseNet121)
    tflite_cfg = BenchmarkConfig(model_id="densenet121", runtime=BenchmarkRuntime.TFLITE)
    tflite_engine = BenchmarkEngine(tflite_cfg)
    tflite_res = tflite_engine.run_benchmark()

    assert tflite_res.success is True
    assert tflite_res.model_size is not None
    assert tflite_res.model_size.file_size_bytes > 10_000_000
    assert tflite_res.model_size.file_size_mb == bytes_to_mb(tflite_res.model_size.file_size_bytes)
    assert tflite_res.model_size.param_count == 7978856

    tflite_dict = tflite_res.to_dict()
    assert tflite_dict["model_size"]["file_size_mb"] == tflite_res.model_size.file_size_mb
    assert tflite_dict["model_size"]["param_count"] == 7978856
    assert "Model Size" in tflite_res.summary()
