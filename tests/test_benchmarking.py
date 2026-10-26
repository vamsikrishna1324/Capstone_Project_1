"""
Test Suite for Step 4A: Benchmarking Engine Foundation.

Verifies:
- BenchmarkRuntime enum conversion and validation.
- BenchmarkConfig defaults, custom parameters, and validation rules.
- Controlled ValueError exceptions on invalid configuration parameters.
- BenchmarkResult container and sub-metric dataclass creation and metric handling.
- Serialization via to_dict() and text summary generation.
- BenchmarkEngine initialization, environment capture, and model metadata binding.
- Foundation method contracts (NotImplementedError on deferred metrics).
- End-to-end execution of the foundation run_benchmark() lifecycle.
"""

from typing import Any, Dict
import pytest

from benchmarking import (
    AccuracyMetrics,
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkRuntime,
    LatencyMetrics,
    MemoryMetrics,
    ModelSizeMetrics,
    SUPPORTED_RUNTIMES,
    bytes_to_mb,
    get_environment_info,
)
from models.model_registry import list_supported_models


# ============================================================================
# 1. BenchmarkRuntime Tests
# ============================================================================
def test_supported_runtimes_list() -> None:
    """Verify that supported runtimes include PyTorch, ONNX, and TFLite."""
    assert "pytorch" in SUPPORTED_RUNTIMES
    assert "onnx" in SUPPORTED_RUNTIMES
    assert "tflite" in SUPPORTED_RUNTIMES
    assert len(SUPPORTED_RUNTIMES) == 3


@pytest.mark.parametrize(
    "input_val,expected_enum",
    [
        ("pytorch", BenchmarkRuntime.PYTORCH),
        ("PyTorch", BenchmarkRuntime.PYTORCH),
        ("PYTORCH", BenchmarkRuntime.PYTORCH),
        ("onnx", BenchmarkRuntime.ONNX),
        ("ONNX", BenchmarkRuntime.ONNX),
        ("tflite", BenchmarkRuntime.TFLITE),
        ("TFLite", BenchmarkRuntime.TFLITE),
        (BenchmarkRuntime.PYTORCH, BenchmarkRuntime.PYTORCH),
        (BenchmarkRuntime.ONNX, BenchmarkRuntime.ONNX),
        (BenchmarkRuntime.TFLITE, BenchmarkRuntime.TFLITE),
    ],
)
def test_benchmark_runtime_from_string_valid(
    input_val: Any, expected_enum: BenchmarkRuntime
) -> None:
    """Verify case-insensitive string resolution to BenchmarkRuntime enum."""
    assert BenchmarkRuntime.from_string(input_val) == expected_enum


@pytest.mark.parametrize("invalid_val", ["tensorrt", "openvino", "keras", "", "   "])
def test_benchmark_runtime_from_string_invalid(invalid_val: str) -> None:
    """Verify ValueError is raised for unrecognized runtime names."""
    with pytest.raises(ValueError, match="Unsupported benchmark runtime"):
        BenchmarkRuntime.from_string(invalid_val)


# ============================================================================
# 2. BenchmarkConfig Defaults & Customization Tests
# ============================================================================
def test_benchmark_config_defaults() -> None:
    """Verify default parameters of BenchmarkConfig."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small")
    assert config.model_id == "mobilenet_v3_small"
    assert config.runtime == BenchmarkRuntime.PYTORCH
    assert config.warmup_runs == 10
    assert config.num_runs == 100
    assert config.input_size == (3, 224, 224)
    assert config.batch_size == 1
    assert config.device == "cpu"
    assert config.backend_provider is None
    assert config.metadata == {}


@pytest.mark.parametrize("model_id", list_supported_models())
@pytest.mark.parametrize("runtime", [BenchmarkRuntime.PYTORCH, BenchmarkRuntime.ONNX, BenchmarkRuntime.TFLITE])
def test_benchmark_config_custom_valid(model_id: str, runtime: BenchmarkRuntime) -> None:
    """Verify valid custom configuration instantiation across all models and runtimes."""
    config = BenchmarkConfig(
        model_id=model_id,
        runtime=runtime,
        warmup_runs=5,
        num_runs=50,
        input_size=(3, 224, 224),
        batch_size=4,
        device="mps",
        backend_provider="CoreMLExecutionProvider",
        metadata={"user": "tester", "env": "ci"},
    )
    assert config.model_id == model_id
    assert config.runtime == runtime
    assert config.warmup_runs == 5
    assert config.num_runs == 50
    assert config.batch_size == 4
    assert config.device == "mps"
    assert config.backend_provider == "CoreMLExecutionProvider"
    assert config.metadata["user"] == "tester"


def test_benchmark_config_to_dict() -> None:
    """Verify dictionary serialization of BenchmarkConfig."""
    config = BenchmarkConfig(model_id="resnet50", runtime="onnx", batch_size=2)
    data = config.to_dict()
    assert isinstance(data, dict)
    assert data["model_id"] == "resnet50"
    assert data["runtime"] == "onnx"
    assert data["batch_size"] == 2
    assert data["warmup_runs"] == 10
    assert data["num_runs"] == 100


# ============================================================================
# 3. BenchmarkConfig Validation & Error Handling Tests
# ============================================================================
@pytest.mark.parametrize("invalid_model", ["unknown_model", "vgg16", "", "   "])
def test_benchmark_config_invalid_model(invalid_model: str) -> None:
    """Verify ValueError is raised when an unsupported or empty model_id is given."""
    with pytest.raises(ValueError):
        BenchmarkConfig(model_id=invalid_model)


def test_benchmark_config_invalid_runtime() -> None:
    """Verify ValueError is raised when an unsupported runtime string is given."""
    with pytest.raises(ValueError, match="Unsupported benchmark runtime"):
        BenchmarkConfig(model_id="mobilenet_v3_small", runtime="invalid_runtime")


@pytest.mark.parametrize("bad_warmup", [-1, -50, True, "10"])
def test_benchmark_config_invalid_warmup_runs(bad_warmup: Any) -> None:
    """Verify ValueError is raised for non-integer or negative warmup_runs."""
    with pytest.raises(ValueError, match="warmup_runs must be a non-negative integer"):
        BenchmarkConfig(model_id="mobilenet_v3_small", warmup_runs=bad_warmup)


@pytest.mark.parametrize("bad_num_runs", [0, -1, -100, False, "100"])
def test_benchmark_config_invalid_num_runs(bad_num_runs: Any) -> None:
    """Verify ValueError is raised for non-integer or non-positive num_runs."""
    with pytest.raises(ValueError, match="num_runs must be a positive integer"):
        BenchmarkConfig(model_id="mobilenet_v3_small", num_runs=bad_num_runs)


@pytest.mark.parametrize("bad_batch_size", [0, -1, False, "1"])
def test_benchmark_config_invalid_batch_size(bad_batch_size: Any) -> None:
    """Verify ValueError is raised for non-integer or batch_size < 1."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        BenchmarkConfig(model_id="mobilenet_v3_small", batch_size=bad_batch_size)


@pytest.mark.parametrize(
    "bad_input_size",
    [
        (224, 224),
        (3, 224, 224, 1),
        (3, -224, 224),
        (0, 224, 224),
        (3, 224, "224"),
        [3, 224, 224],
    ],
)
def test_benchmark_config_invalid_input_size(bad_input_size: Any) -> None:
    """Verify ValueError is raised for invalid input_size shapes and types."""
    with pytest.raises(ValueError, match="input_size must be a 3-tuple"):
        BenchmarkConfig(model_id="mobilenet_v3_small", input_size=bad_input_size)


@pytest.mark.parametrize("bad_device", ["", "   "])
def test_benchmark_config_invalid_device(bad_device: str) -> None:
    """Verify ValueError is raised for empty device string."""
    with pytest.raises(ValueError, match="device must be a non-empty string"):
        BenchmarkConfig(model_id="mobilenet_v3_small", device=bad_device)


# ============================================================================
# 4. Result Structure & Metric Handling Tests
# ============================================================================
def test_benchmark_result_defaults() -> None:
    """Verify default attributes and None metric states in BenchmarkResult."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small")
    result = BenchmarkResult(
        model_id=config.model_id,
        runtime=config.runtime,
        config=config,
    )
    assert result.model_id == "mobilenet_v3_small"
    assert result.runtime == BenchmarkRuntime.PYTORCH
    assert result.accuracy is None
    assert result.latency is None
    assert result.memory_footprint is None
    assert result.model_size is None
    assert result.success is True
    assert result.error_message is None
    assert result.timestamp is not None
    assert len(result.timestamp) > 0


def test_benchmark_result_populated_metrics() -> None:
    """Verify populated metric dataclasses inside BenchmarkResult."""
    config = BenchmarkConfig(model_id="resnet50", runtime="onnx")
    acc = AccuracyMetrics(top1=0.802, top5=0.951, num_samples=50000)
    lat = LatencyMetrics(
        mean_ms=12.4,
        median_ms=12.2,
        std_ms=0.5,
        min_ms=11.8,
        max_ms=15.1,
        p95_ms=13.0,
        p99_ms=14.2,
        throughput_fps=80.6,
    )
    mem = MemoryMetrics(peak_ram_mb=245.8, ram_delta_mb=42.1)
    size = ModelSizeMetrics(file_size_bytes=102154114, file_size_mb=97.42, param_count=25557032)

    result = BenchmarkResult(
        model_id=config.model_id,
        runtime=config.runtime,
        config=config,
        accuracy=acc,
        latency=lat,
        memory_footprint=mem,
        model_size=size,
    )

    assert result.accuracy.top1 == 0.802
    assert result.latency.mean_ms == 12.4
    assert result.memory_footprint.peak_ram_mb == 245.8
    assert result.model_size.file_size_mb == 97.42

    # Check to_dict() nested conversion
    d = result.to_dict()
    assert d["model_id"] == "resnet50"
    assert d["runtime"] == "onnx"
    assert d["accuracy"]["top1"] == 0.802
    assert d["latency"]["throughput_fps"] == 80.6
    assert d["memory_footprint"]["peak_ram_mb"] == 245.8
    assert d["model_size"]["param_count"] == 25557032

    # Check summary() text contains expected sections
    summary_text = result.summary()
    assert "resnet50" in summary_text
    assert "onnx" in summary_text
    assert "PASSED" in summary_text
    assert "Accuracy" in summary_text
    assert "Latency" in summary_text
    assert "Memory Footprint" in summary_text
    assert "Model Size" in summary_text


def test_benchmark_result_failed_state() -> None:
    """Verify BenchmarkResult captures failed executions correctly."""
    config = BenchmarkConfig(model_id="efficientnet_b0")
    result = BenchmarkResult(
        model_id=config.model_id,
        runtime=config.runtime,
        config=config,
        success=False,
        error_message="Simulated runtime error during inference execution",
    )
    assert result.success is False
    assert "Simulated runtime error" in result.error_message
    summary_text = result.summary()
    assert "FAILED" in summary_text


# ============================================================================
# 5. BenchmarkEngine Initialization & Interface Tests
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_benchmark_engine_init_valid(model_id: str) -> None:
    """Verify BenchmarkEngine initializes with a valid BenchmarkConfig."""
    config = BenchmarkConfig(model_id=model_id, runtime="pytorch")
    engine = BenchmarkEngine(config)
    assert engine.config.model_id == model_id
    assert engine.model_metadata.model_id == model_id
    assert isinstance(engine.environment, dict)
    assert "python_version" in engine.environment
    assert "torch_version" in engine.environment


def test_benchmark_engine_init_with_dict() -> None:
    """Verify BenchmarkEngine accepts a raw configuration dictionary."""
    config_dict: Dict[str, Any] = {
        "model_id": "mobilenet_v3_small",
        "runtime": "tflite",
        "num_runs": 20,
    }
    engine = BenchmarkEngine(config_dict)
    assert engine.config.model_id == "mobilenet_v3_small"
    assert engine.config.runtime == BenchmarkRuntime.TFLITE
    assert engine.config.num_runs == 20


def test_benchmark_engine_init_invalid_type() -> None:
    """Verify BenchmarkEngine raises ValueError when given an invalid configuration type."""
    with pytest.raises(ValueError, match="config must be a BenchmarkConfig instance or dict"):
        BenchmarkEngine(config=123)  # type: ignore


def test_benchmark_engine_init_invalid_model() -> None:
    """Verify BenchmarkEngine rejects unregistered model IDs."""
    with pytest.raises(ValueError, match="Unsupported model ID"):
        BenchmarkEngine({"model_id": "nonexistent_model"})


def test_benchmark_engine_metric_methods() -> None:
    """Verify all benchmark metric methods (size, latency, memory, accuracy) execute and return metrics."""
    config = BenchmarkConfig(model_id="densenet121", runtime="pytorch", warmup_runs=1, num_runs=2)
    engine = BenchmarkEngine(config)

    # Accuracy is now implemented in Step 4C
    with pytest.raises(ValueError, match="No evaluation dataset provided"):
        engine.benchmark_accuracy()

    # Model size was implemented in Step 4B-1
    size_metrics = engine.benchmark_model_size()
    assert isinstance(size_metrics, ModelSizeMetrics)
    assert size_metrics.param_count == 7978856

    # Latency was implemented in Step 4B-2
    lat_metrics = engine.benchmark_latency()
    assert isinstance(lat_metrics, LatencyMetrics)
    assert lat_metrics.mean_ms > 0
    assert lat_metrics.throughput_fps > 0

    # Memory was implemented in Step 4B-3
    mem_metrics = engine.benchmark_memory()
    assert isinstance(mem_metrics, MemoryMetrics)
    assert mem_metrics.peak_ram_mb is not None
    assert mem_metrics.peak_ram_mb > 0


def test_benchmark_engine_run_benchmark_foundation() -> None:
    """Verify run_benchmark() method returns a valid BenchmarkResult with size, latency, and memory populated."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="onnx", warmup_runs=2, num_runs=5)
    engine = BenchmarkEngine(config)
    result = engine.run_benchmark()

    assert isinstance(result, BenchmarkResult)
    assert result.model_id == "mobilenet_v3_small"
    assert result.runtime == BenchmarkRuntime.ONNX
    assert result.config.num_runs == 5
    assert result.success is True
    assert result.error_message is None
    assert result.accuracy is None
    assert result.latency is not None
    assert result.latency.mean_ms > 0
    assert result.latency.throughput_fps > 0
    assert result.memory_footprint is not None
    assert result.memory_footprint.peak_ram_mb is not None
    assert result.memory_footprint.peak_ram_mb > 0
    assert result.model_size is not None
    assert result.model_size.file_size_bytes > 1_000_000
    assert result.model_size.file_size_mb == bytes_to_mb(result.model_size.file_size_bytes)
    assert result.model_size.param_count == 2542856
    assert isinstance(result.environment, dict)
    assert "device_accelerator" in result.environment

    # If measure_latency=False, latency remains None
    result_no_lat = engine.run_benchmark(measure_latency=False)
    assert result_no_lat.latency is None
    assert result_no_lat.model_size is not None

    # If measure_memory=False, memory_footprint remains None
    result_no_mem = engine.run_benchmark(measure_memory=False)
    assert result_no_mem.memory_footprint is None


def test_get_environment_info_structure() -> None:
    """Verify get_environment_info returns complete hardware and library metadata."""
    env = get_environment_info()
    assert isinstance(env, dict)
    assert "python_version" in env
    assert "platform" in env
    assert "system" in env
    assert "torch_version" in env
    assert "torchvision_version" in env
    assert "numpy_version" in env
    assert "onnx_version" in env
    assert "onnxruntime_version" in env
    assert "tensorflow_version" in env
    assert "device_accelerator" in env
    assert "mps_available" in env
    assert "cuda_available" in env
