"""
Focused Test Suite for Accuracy Dataset Wiring and Missing Precomputed Accuracy Handling.

Verifies:
1. Dataset manifest path resolution from project root for fresh live evaluation.
2. Verified precomputed accuracy artifact loading across supported configurations.
3. Missing DenseNet121 TFLite precomputed accuracy handling (None/N/A values + non-fatal warning).
4. Missing optional accuracy preservation of valid latency, throughput, memory, and model size.
5. BenchmarkEngine wiring receives dataset path without executing real inference.
6. API contracts: GET /api/models and POST /api/benchmarks warning compatibility.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# Enforce test database isolation strictly before application import
os.environ["MYSQL_DATABASE"] = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test")
os.environ["SEED_BASELINE_ON_STARTUP"] = "false"

from backend.dependencies import get_engine, reset_engine
from backend.main import app
from backend.services.accuracy_service import (
    find_saved_accuracy_file,
    get_missing_accuracy_warning,
    is_saved_accuracy_available,
    load_saved_accuracy,
)
from backend.services.benchmark_service import (
    DEFAULT_MANIFEST_PATH,
    PROJECT_ROOT,
    execute_comparison_benchmark,
    resolve_dataset_manifest_path,
)
from backend.services.progress_tracker import progress_tracker
from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.results import (
    AccuracyMetrics,
    BenchmarkResult,
    LatencyMetrics,
    MemoryMetrics,
    ModelSizeMetrics,
)
from database.connection import session_scope
from database.models import Base, BenchmarkRun, ModelMeasurement


@pytest.fixture(scope="module")
def db_session_factory():
    """Isolated session factory bound to edge_model_benchmark_test."""
    os.environ["MYSQL_DATABASE"] = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test")
    os.environ["SEED_BASELINE_ON_STARTUP"] = "false"
    reset_engine()

    engine = get_engine()
    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        conn.execute(text("TRUNCATE TABLE model_measurements;"))
        conn.execute(text("TRUNCATE TABLE benchmark_runs;"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        conn.commit()

    from backend.dependencies import get_session_factory
    factory = get_session_factory()
    yield factory
    reset_engine()


@pytest.fixture(scope="module")
def client(db_session_factory):
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. Dataset Manifest Path Resolution
# ==============================================================================

def test_manifest_path_resolution_from_project_root():
    """Verify DEFAULT_MANIFEST_PATH resolves to dataset/val_manifest.csv at project root."""
    assert DEFAULT_MANIFEST_PATH.is_absolute()
    assert DEFAULT_MANIFEST_PATH.name == "val_manifest.csv"
    assert DEFAULT_MANIFEST_PATH.parent.name == "dataset"
    assert DEFAULT_MANIFEST_PATH.exists(), f"Expected manifest file at {DEFAULT_MANIFEST_PATH}"
    assert DEFAULT_MANIFEST_PATH == PROJECT_ROOT / "dataset" / "val_manifest.csv"


# ==============================================================================
# 2. Existing Precomputed Accuracy Loading
# ==============================================================================

def test_existing_precomputed_accuracy_artifacts_load_correctly():
    """Verify existing precomputed artifacts load genuine top-1/top-5 fractions."""
    # DenseNet121 ONNX
    dense_onnx = load_saved_accuracy("densenet121", "onnx")
    assert dense_onnx is not None
    assert dense_onnx["top1_accuracy"] == 0.7444
    assert dense_onnx["top5_accuracy"] == 0.9197
    assert dense_onnx["evaluated_samples"] == 50000

    # MobileNetV3-Small TFLite
    mb_tflite = load_saved_accuracy("mobilenet_v3_small", "tflite")
    assert mb_tflite is not None
    assert mb_tflite["top1_accuracy"] == 0.6764
    assert mb_tflite["top5_accuracy"] == 0.8743

    # ResNet50 TFLite
    res_tflite = load_saved_accuracy("resnet50", "tflite")
    assert res_tflite is not None
    assert res_tflite["top1_accuracy"] == 0.8086
    assert res_tflite["top5_accuracy"] == 0.9544


# ==============================================================================
# 3. Missing DenseNet121 TFLite Precomputed Accuracy Handling
# ==============================================================================

def test_missing_densenet121_tflite_precomputed_accuracy():
    """DenseNet121 TFLite has no precomputed artifact; should return None and clear warning."""
    # File lookup returns None
    assert find_saved_accuracy_file("densenet121", "tflite") is None
    assert is_saved_accuracy_available("densenet121", "tflite") is False

    # Loading returns None without throwing exceptions or fabricating values
    result = load_saved_accuracy("densenet121", "tflite")
    assert result is None

    # Warning text is non-sensitive and informative
    warning = get_missing_accuracy_warning("densenet121", "tflite")
    assert "DenseNet121" in warning
    assert "TFLITE" in warning
    assert "unavailable" in warning.lower()
    assert "fresh" in warning.lower()


# ==============================================================================
# 4. Mocked Fresh Accuracy Dataset Wiring
# ==============================================================================

def test_fresh_accuracy_supplies_dataset_to_engine(db_session_factory):
    """When fresh accuracy is requested, BenchmarkConfig and engine receive the manifest path."""
    mock_engine_run = MagicMock()
    mock_engine_run.return_value = BenchmarkResult(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        config=BenchmarkConfig(model_id="mobilenet_v3_small", runtime=BenchmarkRuntime.PYTORCH),
        accuracy=AccuracyMetrics(top1=0.68, top5=0.88, num_samples=10),
        latency=LatencyMetrics(mean_ms=15.0, p95_ms=18.0, throughput_fps=66.6),
        memory_footprint=MemoryMetrics(peak_ram_mb=300.0),
        model_size=ModelSizeMetrics(file_size_mb=10.0, param_count=2500000),
        success=True,
    )

    with patch("backend.services.benchmark_service.BenchmarkEngine") as mock_engine_cls:
        instance = mock_engine_cls.return_value
        instance.run_benchmark = mock_engine_run

        with session_scope(db_session_factory) as session:
            run = execute_comparison_benchmark(
                session=session,
                model_ids=["mobilenet_v3_small"],
                runtimes=["pytorch"],
                metrics=["latency", "accuracy"],
                run_accuracy_eval=True,
                reuse_saved_accuracy=False,
                max_accuracy_samples=10,
            )

            assert run.status == "completed"
            # Verify BenchmarkConfig was constructed with dataset_path
            init_call_config = mock_engine_cls.call_args[0][0]
            assert init_call_config.dataset_path == str(DEFAULT_MANIFEST_PATH)
            assert init_call_config.max_samples == 10

            # Verify engine.run_benchmark was called with dataset parameter
            mock_engine_run.assert_called_once()
            call_kwargs = mock_engine_run.call_args[1]
            assert call_kwargs["dataset"] == str(DEFAULT_MANIFEST_PATH)
            assert call_kwargs["measure_accuracy"] is True
            assert call_kwargs["max_accuracy_samples"] == 10


# ==============================================================================
# 5. Missing Accuracy Preserves Valid Performance Measurements
# ==============================================================================

def test_missing_accuracy_preserves_performance_measurements(db_session_factory):
    """Missing accuracy for DenseNet121 TFLite stores NULL accuracy but keeps latency/memory/status completed."""
    mock_engine_run = MagicMock()
    mock_engine_run.return_value = BenchmarkResult(
        model_id="densenet121",
        runtime=BenchmarkRuntime.TFLITE,
        config=BenchmarkConfig(model_id="densenet121", runtime=BenchmarkRuntime.TFLITE),
        accuracy=None,
        latency=LatencyMetrics(mean_ms=101.55, p95_ms=122.7, throughput_fps=9.85),
        memory_footprint=MemoryMetrics(peak_ram_mb=472.78),
        model_size=ModelSizeMetrics(file_size_mb=15.35, param_count=7978856),
        success=True,
    )

    with patch("backend.services.benchmark_service.BenchmarkEngine") as mock_engine_cls:
        instance = mock_engine_cls.return_value
        instance.run_benchmark = mock_engine_run

        with session_scope(db_session_factory) as session:
            run = execute_comparison_benchmark(
                session=session,
                model_ids=["densenet121"],
                runtimes=["tflite"],
                metrics=["latency", "memory", "model_size", "accuracy"],
                reuse_saved_accuracy=True,
                run_accuracy_eval=False,
            )

            # Benchmark run and measurements succeed
            assert run.status == "completed"
            assert len(run.measurements) == 1
            m = run.measurements[0]
            assert m.status == "completed"

            # Performance metrics are intact
            assert m.latency_mean_ms == pytest.approx(101.55, rel=1e-2)
            assert m.peak_ram_mb == pytest.approx(472.78, rel=1e-2)
            assert m.file_size_mb == pytest.approx(15.35, rel=1e-2)

            # Accuracy is cleanly NULL (unavailable, not fabricated)
            assert m.top1_accuracy is None
            assert m.top5_accuracy is None
            assert m.evaluated_samples is None

            # Warning is captured in environment_info and to_dict
            run_dict = run.to_dict()
            assert "warnings" in run_dict
            assert len(run_dict["warnings"]) > 0
            assert "DenseNet121" in run_dict["warnings"][0]
            assert "TFLITE" in run_dict["warnings"][0]


# ==============================================================================
# 6. API Contracts Compatibility
# ==============================================================================

def test_models_endpoint_reports_has_precomputed_accuracy(client: TestClient):
    """GET /api/models exposes has_precomputed_accuracy in artifact details."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    models = {m["model_id"]: m for m in data["models"]}
    assert "densenet121" in models
    dense_artifacts = models["densenet121"]["artifacts"]

    assert dense_artifacts["pytorch"]["has_precomputed_accuracy"] is True
    assert dense_artifacts["onnx"]["has_precomputed_accuracy"] is True
    assert dense_artifacts["tflite"]["has_precomputed_accuracy"] is False

    res_artifacts = models["resnet50"]["artifacts"]
    assert res_artifacts["tflite"]["has_precomputed_accuracy"] is True


def test_benchmarks_endpoint_response_schema_compatibility(client: TestClient, db_session_factory):
    """POST /api/benchmarks and GET /api/benchmarks/{id} return warnings list."""
    mock_engine_run = MagicMock()
    mock_engine_run.return_value = BenchmarkResult(
        model_id="densenet121",
        runtime=BenchmarkRuntime.TFLITE,
        config=BenchmarkConfig(model_id="densenet121", runtime=BenchmarkRuntime.TFLITE),
        accuracy=None,
        latency=LatencyMetrics(mean_ms=99.0, throughput_fps=10.1),
        memory_footprint=MemoryMetrics(peak_ram_mb=450.0),
        model_size=ModelSizeMetrics(file_size_mb=15.35, param_count=7978856),
        success=True,
    )

    with patch("backend.services.benchmark_service.BenchmarkEngine") as mock_engine_cls:
        instance = mock_engine_cls.return_value
        instance.run_benchmark = mock_engine_run

        payload = {
            "model_ids": ["densenet121"],
            "runtimes": ["tflite"],
            "metrics": ["latency", "memory", "model_size", "accuracy"],
            "reuse_saved_accuracy": True,
            "run_accuracy_eval": False,
            "async_run": False,
        }

        post_res = client.post("/api/benchmarks", json=payload)
        assert post_res.status_code == 201
        run_data = post_res.json()

        assert run_data["status"] == "completed"
        assert "warnings" in run_data
        assert any("DenseNet121" in w for w in run_data["warnings"])

        # Retrieve by ID
        get_res = client.get(f"/api/benchmarks/{run_data['id']}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert "warnings" in get_data
        assert any("DenseNet121" in w for w in get_data["warnings"])


# ==============================================================================
# 7. Robust Dataset Path Resolution & Validation
# ==============================================================================

def test_resolve_dataset_manifest_path_relative_and_whitespace():
    """Relative dataset paths resolve against PROJECT_ROOT, whitespace/empty paths default."""
    # Relative path
    rel_resolved = resolve_dataset_manifest_path("dataset/val_manifest.csv")
    assert rel_resolved == DEFAULT_MANIFEST_PATH

    # Whitespace and empty
    assert resolve_dataset_manifest_path("   \t\n  ") == DEFAULT_MANIFEST_PATH
    assert resolve_dataset_manifest_path("") == DEFAULT_MANIFEST_PATH
    assert resolve_dataset_manifest_path(None) == DEFAULT_MANIFEST_PATH

    # Absolute path
    custom_abs = Path("/tmp/custom_manifest.csv")
    assert resolve_dataset_manifest_path(custom_abs) == custom_abs.resolve()


def test_live_accuracy_missing_manifest_raises_filenotfound(db_session_factory):
    """Initiating live accuracy evaluation with a non-existent manifest raises FileNotFoundError immediately."""
    with session_scope(db_session_factory) as session:
        with pytest.raises(FileNotFoundError) as exc_info:
            execute_comparison_benchmark(
                session=session,
                model_ids=["mobilenet_v3_small"],
                runtimes=["pytorch"],
                metrics=["accuracy"],
                run_accuracy_eval=True,
                dataset_path="nonexistent_manifest_file_xyz.csv",
            )
        assert "Validation dataset manifest file not found" in str(exc_info.value)


# ==============================================================================
# 8. Transaction Commit Ordering & Progress Tracker Warnings
# ==============================================================================

def test_transaction_commit_precedes_progress_completion(db_session_factory):
    """Database transaction must commit before progress tracker reports the job completed."""
    call_log = []

    mock_engine_run = MagicMock()
    mock_engine_run.return_value = BenchmarkResult(
        model_id="densenet121",
        runtime=BenchmarkRuntime.TFLITE,
        config=BenchmarkConfig(model_id="densenet121", runtime=BenchmarkRuntime.TFLITE),
        accuracy=None,
        latency=LatencyMetrics(mean_ms=105.0, throughput_fps=9.5),
        memory_footprint=MemoryMetrics(peak_ram_mb=460.0),
        model_size=ModelSizeMetrics(file_size_mb=15.35, param_count=7978856),
        success=True,
    )

    with patch("backend.services.benchmark_service.BenchmarkEngine") as mock_engine_cls, \
         patch.object(progress_tracker, "complete_job", side_effect=lambda *args, **kwargs: (call_log.append("progress_complete"), kwargs)[1]):
        instance = mock_engine_cls.return_value
        instance.run_benchmark = mock_engine_run

        with session_scope(db_session_factory) as session:
            orig_commit = session.commit

            def _tracking_commit():
                call_log.append("session_commit")
                return orig_commit()

            session.commit = _tracking_commit

            run = execute_comparison_benchmark(
                session=session,
                model_ids=["densenet121"],
                runtimes=["tflite"],
                metrics=["latency", "accuracy"],
                reuse_saved_accuracy=True,
                run_accuracy_eval=False,
            )

            assert "session_commit" in call_log
            assert "progress_complete" in call_log
            commit_idx = call_log.index("session_commit")
            progress_idx = call_log.index("progress_complete")
            assert commit_idx < progress_idx, "session.commit() must occur BEFORE progress_tracker.complete_job()"


def test_progress_endpoint_exposes_warnings(client: TestClient):
    """GET /api/benchmarks/{run_id}/progress exposes non-fatal warnings directly."""
    test_run_id = 998877
    progress_tracker.init_job(test_run_id, total_images=100)
    progress_tracker.complete_job(
        test_run_id,
        status="completed",
        warnings=["Notice: Precomputed accuracy is unavailable for DenseNet121 on TFLITE."],
    )

    res = client.get(f"/api/benchmarks/{test_run_id}/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "warnings" in data
    assert len(data["warnings"]) == 1
    assert "DenseNet121" in data["warnings"][0]
