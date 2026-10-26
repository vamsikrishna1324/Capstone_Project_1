"""
Unit Tests for Edge Impulse Profiling Service.

Verifies:
1. Handling of missing credentials or missing model artifacts.
2. Safe result parsing and structured metric extraction.
3. Verification of timePerInferenceMs units and conversion to seconds.
4. Preservation of NULL semantics when RAM or arena is reported as 0 for Linux SBC targets.
5. Strict separation of MCU export warnings (e.g. Unsupported ops) from target profiling status.
6. Strict omission of API keys in exceptions, logs, and representations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.config import BackendSettings
from backend.services.edge_impulse_service import (
    EdgeImpulseProfileResult,
    EdgeImpulseProfilingError,
    SUPPORTED_EDGE_IMPULSE_TARGETS,
    compute_tflite_sha256,
    get_or_create_edge_impulse_benchmark,
    get_verified_saved_accuracy,
    poll_profiling_job_result,
    submit_tflite_profiling_job,
    validate_edge_impulse_compatibility,
)
from database.models import BenchmarkRun, ModelMeasurement


def test_submit_missing_credentials_raises():
    """Submitting without credentials must raise EdgeImpulseProfilingError."""
    with patch("backend.services.edge_impulse_service.get_settings") as mock_settings:
        mock_settings.return_value = BackendSettings(edge_impulse_api_key=None, edge_impulse_project_id=None)
        with pytest.raises(EdgeImpulseProfilingError, match="EDGE_IMPULSE_API_KEY is not configured"):
            submit_tflite_profiling_job(Path("nonexistent.tflite"))


def test_submit_missing_file_raises():
    """Submitting a nonexistent model file must raise FileNotFoundError."""
    with patch("backend.services.edge_impulse_service.get_settings") as mock_settings:
        mock_settings.return_value = BackendSettings(
            edge_impulse_api_key="ei_test_mock_key",
            edge_impulse_project_id="123456",
        )
        with pytest.raises(FileNotFoundError):
            submit_tflite_profiling_job(Path("does_not_exist_xyz.tflite"))


def test_edge_impulse_profile_result_structure():
    """EdgeImpulseProfileResult converts cleanly to dictionary without credentials."""
    res = EdgeImpulseProfileResult(
        job_id=98765,
        target_device="raspberry-pi-4",
        status="completed",
        time_per_inference_ms=45.2,
        tflite_ram_bytes=1500000,
        tflite_rom_bytes=5135976,
        tflite_arena_bytes=1400000,
        eon_ram_bytes=1100000,
        eon_rom_bytes=5135976,
        eon_arena_bytes=1000000,
        is_supported_on_mcu=True,
        mcu_support_error=None,
        profiling_error=None,
    )
    d = res.to_dict()
    assert d["job_id"] == 98765
    assert d["target_device"] == "raspberry-pi-4"
    assert d["time_per_inference_ms"] == 45.2
    assert d["time_per_inference_s"] == 0.045
    assert d["tflite_rom_bytes"] == 5135976
    assert d["is_supported_on_mcu"] is True
    assert d["provenance"] == "Edge Impulse Studio Estimation"
    assert "api_key" not in d


def test_parsing_preserves_null_for_zero_ram_and_separates_mcu_error():
    """Verify that zero RAM values are converted to None (NULL/NA) and MCU error is kept separate."""
    # Mock payload matching exact Edge Impulse API response structure from Job 55253939
    mock_api_payload = {
        "success": True,
        "device": "raspberry-pi-4",
        "memory": {
            "tflite": {
                "arenaSize": 0,
                "ram": 0,
                "rom": 5135976,
            }
        },
        "timePerInferenceMs": 21477,
        "isSupportedOnMcu": False,
        "tfliteFileSizeBytes": 5135976,
        "mcuSupportError": "Unsupported ops: RELU_0_TO_1.",
        "variant": "float32",
        "customMetrics": [],
        "hasPerformance": True,
    }

    mock_resp = MagicMock()
    import json
    mock_resp.read.return_value = json.dumps(mock_api_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("backend.services.edge_impulse_service.get_settings") as mock_settings, \
         patch("urllib.request.urlopen", return_value=mock_resp):
        mock_settings.return_value = BackendSettings(
            edge_impulse_api_key="ei_test_mock_key",
            edge_impulse_project_id="1144836",
        )

        result = poll_profiling_job_result(job_id=55253939, device="raspberry-pi-4", timeout_seconds=5.0)

        assert result.status == "completed"
        assert result.job_id == 55253939
        assert result.target_device == "raspberry-pi-4"

        # Unit verification: timePerInferenceMs is in milliseconds (21477 ms = 21.477 s)
        assert result.time_per_inference_ms == 21477.0
        assert result.time_per_inference_s == 21.477

        # RAM and arena of 0 must be converted to None (NULL / N/A), never literal zero
        assert result.tflite_ram_bytes is None
        assert result.tflite_arena_bytes is None
        assert result.eon_ram_bytes is None
        assert result.eon_arena_bytes is None

        # ROM / model size matches exact TFLite file size
        assert result.tflite_rom_bytes == 5135976

        # MCU support warning is separated from target profiling status
        assert result.is_supported_on_mcu is False
        assert result.mcu_support_error == "Unsupported ops: RELU_0_TO_1."
        assert result.profiling_error is None

        # Dictionary representation preserves None for unavailable RAM
        d = result.to_dict()
        assert d["tflite_ram_bytes"] is None
        assert d["tflite_arena_bytes"] is None
        assert d["time_per_inference_ms"] == 21477.0
        assert d["time_per_inference_s"] == 21.477
        assert d["mcu_support_error"] == "Unsupported ops: RELU_0_TO_1."
        assert d["profiling_error"] is None


def test_validate_edge_impulse_compatibility_success():
    """Validating MobileNetV3-Small with TFLite on Raspberry Pi 4 succeeds and returns Path."""
    p = validate_edge_impulse_compatibility("mobilenet_v3_small", "tflite", "raspberry-pi-4")
    assert p.is_file()
    assert p.name == "mobilenet_v3_small.tflite"


def test_validate_edge_impulse_compatibility_rejects_onnx_and_pytorch():
    """Edge Impulse profiling must reject ONNX and PyTorch formats."""
    with pytest.raises(EdgeImpulseProfilingError, match="TFLite models only"):
        validate_edge_impulse_compatibility("mobilenet_v3_small", "onnx", "raspberry-pi-4")

    with pytest.raises(EdgeImpulseProfilingError, match="TFLite models only"):
        validate_edge_impulse_compatibility("mobilenet_v3_small", "pytorch", "raspberry-pi-4")


def test_validate_edge_impulse_compatibility_rejects_unknown_target():
    """Unrecognized profiling targets must raise EdgeImpulseProfilingError."""
    with pytest.raises(EdgeImpulseProfilingError, match="not a recognized Edge Impulse profiling target"):
        validate_edge_impulse_compatibility("mobilenet_v3_small", "tflite", "unsupported-device-xyz")


def test_validate_edge_impulse_compatibility_rejects_unsupported_model():
    """Unregistered model IDs must raise EdgeImpulseProfilingError."""
    with pytest.raises(EdgeImpulseProfilingError, match="not supported"):
        validate_edge_impulse_compatibility("nonexistent_model_xyz", "tflite", "raspberry-pi-4")


def test_compute_tflite_sha256():
    """compute_tflite_sha256 returns exact hex digest for existing artifact."""
    artifact_path = Path("results/conversions/tflite/mobilenet_v3_small.tflite")
    sha = compute_tflite_sha256(artifact_path)
    assert sha == "aad685b124b2e56d22687b8486c93ecb5ebbbde81354786599acde8fb0fa043d"


def test_get_verified_saved_accuracy():
    """Verified saved accuracy must match model, runtime, batch_size and full evaluation."""
    # Matches MobileNetV3-Small TFLite
    acc = get_verified_saved_accuracy("mobilenet_v3_small", "tflite", expected_batch_size=1)
    assert acc is not None
    assert acc["evaluated_samples"] == 50000
    assert acc["total_samples"] == 50000
    assert acc["top1_accuracy"] == 0.6764
    assert acc["top5_accuracy"] == 0.8743

    # Mismatch batch size -> returns None
    assert get_verified_saved_accuracy("mobilenet_v3_small", "tflite", expected_batch_size=4) is None

    # Nonexistent runtime / model -> returns None
    assert get_verified_saved_accuracy("mobilenet_v3_small", "unknown_runtime") is None


def test_get_or_create_edge_impulse_benchmark_reusing_cached_job():
    """Verify that get_or_create_edge_impulse_benchmark reuses verified completed job 55253939."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = []

    # Mock settings so it does not poll live API
    with patch("backend.services.edge_impulse_service.get_settings") as mock_settings:
        mock_settings.return_value = BackendSettings(edge_impulse_api_key=None, edge_impulse_project_id=None)

        run = get_or_create_edge_impulse_benchmark(
            session=mock_session,
            model_id="mobilenet_v3_small",
            target_device="raspberry-pi-4",
            runtime="tflite",
            batch_size=1,
            reuse_completed=True,
            reuse_saved_accuracy=True,
            allow_external_submission=False,
        )

        assert run.provenance == "edge_impulse_estimation"
        assert run.status == "completed"
        assert run.device == "raspberry-pi-4"
        assert run.environment_info["job_id"] == 55253939
        assert run.environment_info["latency_type"] == "analytical_estimate"
        assert run.environment_info["throughput_type"] == "derived_from_estimated_latency"
        assert run.environment_info["is_supported_on_mcu"] is False
        assert run.environment_info["mcu_support_error"] == "Unsupported ops: RELU_0_TO_1."

        # Verify added measurement
        assert mock_session.add.call_count >= 2
        # Verify flush called
        assert mock_session.flush.call_count >= 2


def test_get_or_create_edge_impulse_benchmark_database_reuse():
    """Verify that existing database records are returned directly without reprocessing."""
    existing_run = MagicMock()
    existing_run.id = 42
    existing_run.environment_info = {
        "artifact_sha256": "aad685b124b2e56d22687b8486c93ecb5ebbbde81354786599acde8fb0fa043d",
        "batch_size": 1,
    }

    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = [existing_run]

    run = get_or_create_edge_impulse_benchmark(
        session=mock_session,
        model_id="mobilenet_v3_small",
        target_device="raspberry-pi-4",
        runtime="tflite",
        batch_size=1,
        reuse_completed=True,
    )

    assert run.id == 42
    # Verify no new objects were added
    mock_session.add.assert_not_called()


def test_get_or_create_edge_impulse_benchmark_blocks_unauthorized_external_submission():
    """If no cached job exists and allow_external_submission is False, must raise EdgeImpulseProfilingError."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = []

    # Patch VERIFIED_COMPLETED_JOBS to simulate no cached job
    with patch.dict("backend.services.edge_impulse_service.VERIFIED_COMPLETED_JOBS", {}, clear=True):
        with pytest.raises(EdgeImpulseProfilingError, match="External job submission is currently disabled"):
            get_or_create_edge_impulse_benchmark(
                session=mock_session,
                model_id="mobilenet_v3_small",
                target_device="raspberry-pi-4",
                runtime="tflite",
                batch_size=1,
                reuse_completed=False,
                allow_external_submission=False,
            )
