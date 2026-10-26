"""
Comprehensive Test Suite for FastAPI Backend API.

Verifies:
1. Health check endpoint (GET /api/health) and credential masking.
2. Models catalog endpoint (GET /api/models) and accurate artifact inspections.
3. Benchmark results endpoint (GET /api/results) with filtering and pagination.
4. Benchmark comparison runner (POST /api/benchmarks) and retrieval (GET /api/benchmarks/{id}).
5. Transactional integrity and rollback on persistence failure.
6. Multi-criteria recommendation engine (POST /api/recommendations), constraint filtering,
   trade-off analysis, and transparent missing metric handling.
"""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# Enforce test database isolation strictly before application import
os.environ["MYSQL_DATABASE"] = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test")
os.environ["SEED_BASELINE_ON_STARTUP"] = "false"

from backend.dependencies import get_engine, get_session_factory, reset_engine
from backend.main import app
from backend.services.benchmark_service import import_baseline_benchmarks_if_needed
from database.connection import session_scope
from database.models import Base, BenchmarkRun, ModelMeasurement


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture strictly bound to edge_model_benchmark_test."""
    os.environ["MYSQL_DATABASE"] = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test")
    os.environ["SEED_BASELINE_ON_STARTUP"] = "false"
    reset_engine()

    engine = get_engine()
    Base.metadata.create_all(engine)

    # Clean only the test database tables
    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        conn.execute(text("TRUNCATE TABLE model_measurements;"))
        conn.execute(text("TRUNCATE TABLE benchmark_runs;"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        conn.commit()

    with TestClient(app) as test_client:
        yield test_client

    reset_engine()


# ==============================================================================
# 1. Health Check Endpoint Tests
# ==============================================================================

def test_health_endpoint_success(client: TestClient):
    """GET /api/health should report online API, live MySQL, and safe Edge Impulse status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("ok", "degraded")
    assert data["api"] == "online"
    assert "version" in data
    assert "timestamp" in data

    db_info = data["database"]
    assert "connected" in db_info
    assert db_info["dialect"] == "mysql+pymysql"
    assert "host" in db_info
    assert "port" in db_info
    assert db_info["database"] == "edge_model_benchmark_test"

    # Verify no database credentials or secret keys are exposed
    assert "password" not in db_info
    assert "user" not in db_info
    assert "MYSQL_PASSWORD" not in response.text

    # Verify Edge Impulse safe status reporting
    ei_info = data["edge_impulse"]
    assert "configured" in ei_info
    assert "status" in ei_info
    assert "has_api_key" in ei_info
    assert "has_project_id" in ei_info
    assert "api_key" not in ei_info
    assert "EDGE_IMPULSE_API_KEY" not in response.text


def test_health_endpoint_degraded_when_unreachable(client: TestClient):
    """GET /api/health should report status 'degraded' when database is unreachable."""
    mocked_info = {
        "connected": False,
        "dialect": "mysql+pymysql",
        "host": "127.0.0.1",
        "port": 3306,
        "database": "edge_model_benchmark",
    }
    with patch("backend.routers.health.check_db_health", return_value=mocked_info):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"]["connected"] is False
        assert "edge_impulse" in data


def test_edge_impulse_status_endpoint_safe_reporting(client: TestClient):
    """GET /api/edge-impulse/status should report unconfigured status safely without secrets."""
    response = client.get("/api/edge-impulse/status")
    assert response.status_code == 200
    data = response.json()

    assert "configured" in data
    assert "status" in data
    assert "has_api_key" in data
    assert "has_project_id" in data
    assert "api_key" not in data
    assert "EDGE_IMPULSE_API_KEY" not in response.text



# ==============================================================================
# 2. Supported Models and Runtimes Endpoint Tests
# ==============================================================================

def test_models_endpoint_returns_registered_models(client: TestClient):
    """GET /api/models should return all four registered models and their runtimes."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    assert data["total_models"] == 4
    model_ids = [m["model_id"] for m in data["models"]]
    expected_models = ["mobilenet_v3_small", "resnet50", "efficientnet_b0", "densenet121"]
    for expected in expected_models:
        assert expected in model_ids

    for m in data["models"]:
        assert set(m["supported_runtimes"]) == {"pytorch", "onnx", "tflite"}
        assert m["num_classes"] == 1000
        assert m["expected_input_size"] == [3, 224, 224]
        assert "artifacts" in m
        assert "onnx" in m["artifacts"]
        assert "tflite" in m["artifacts"]
        assert "pytorch" in m["artifacts"]


def test_models_endpoint_reports_artifacts_accurately(client: TestClient):
    """GET /api/models should accurately report artifact file existence and sizes."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    mobilenet = next(m for m in data["models"] if m["model_id"] == "mobilenet_v3_small")
    onnx_art = mobilenet["artifacts"]["onnx"]
    assert onnx_art["exists"] is True
    assert onnx_art["path"] == "results/conversions/onnx/mobilenet_v3_small.onnx"
    assert onnx_art["file_size_bytes"] is not None
    assert onnx_art["file_size_bytes"] > 0
    assert onnx_art["file_size_mb"] is not None

    tflite_art = mobilenet["artifacts"]["tflite"]
    assert tflite_art["exists"] is True
    assert tflite_art["path"] == "results/conversions/tflite/mobilenet_v3_small.tflite"
    assert tflite_art["file_size_bytes"] > 0


# ==============================================================================
# 3. Available Benchmark Results Endpoint Tests
# ==============================================================================

def test_results_endpoint_pagination_and_structure(client: TestClient):
    """GET /api/results should return paginated measurements."""
    response = client.get("/api/results?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()

    assert "total" in data
    assert data["limit"] == 5
    assert data["offset"] == 0
    assert isinstance(data["items"], list)
    assert len(data["items"]) <= 5

    if data["items"]:
        item = data["items"][0]
        assert "id" in item
        assert "model_id" in item
        assert "runtime" in item
        assert "device" in item
        assert "status" in item


def test_results_endpoint_filtering(client: TestClient):
    """GET /api/results should filter by model_ids and runtimes."""
    # Filter by model_ids
    r_model = client.get("/api/results?model_ids=resnet50")
    assert r_model.status_code == 200
    data_m = r_model.json()
    for item in data_m["items"]:
        assert item["model_id"] == "resnet50"

    # Filter by runtimes
    r_rt = client.get("/api/results?runtimes=onnx")
    assert r_rt.status_code == 200
    data_rt = r_rt.json()
    for item in data_rt["items"]:
        assert item["runtime"] == "onnx"


# ==============================================================================
# 4. Benchmark Comparison Execution and Retrieval Tests
# ==============================================================================

def test_benchmarks_input_validation(client: TestClient):
    """POST /api/benchmarks should reject unsupported models or runtimes."""
    # Unsupported model
    r_bad_model = client.post(
        "/api/benchmarks",
        json={"model_ids": ["invalid_vision_model"], "runtimes": ["onnx"]},
    )
    assert r_bad_model.status_code == 422

    # Unsupported runtime
    r_bad_rt = client.post(
        "/api/benchmarks",
        json={"model_ids": ["mobilenet_v3_small"], "runtimes": ["tensorrt"]},
    )
    assert r_bad_rt.status_code == 422

    # Empty model list
    r_empty = client.post(
        "/api/benchmarks",
        json={"model_ids": [], "runtimes": ["onnx"]},
    )
    assert r_empty.status_code == 422


def test_benchmarks_execute_and_retrieve_comparison(client: TestClient):
    """POST /api/benchmarks executes comparison and GET /api/benchmarks/{id} retrieves it."""
    payload = {
        "model_ids": ["mobilenet_v3_small"],
        "runtimes": ["onnx"],
        "metrics": ["latency", "model_size", "accuracy"],
        "num_runs": 5,
        "warmup_runs": 1,
        "reuse_saved_accuracy": True,
        "run_name": "API Test Comparison Run",
    }
    r_post = client.post("/api/benchmarks", json=payload)
    assert r_post.status_code == 201
    run_data = r_post.json()

    run_id = run_data["id"]
    assert run_id > 0
    assert run_data["status"] == "completed"
    assert run_data["provenance"] == "live_execution"
    assert "methodology_disclaimer" in run_data
    assert len(run_data["measurements"]) == 1

    m = run_data["measurements"][0]
    assert m["model_id"] == "mobilenet_v3_small"
    assert m["runtime"] == "onnx"
    assert m["latency"]["mean_ms"] is not None
    assert m["latency"]["mean_ms"] > 0
    assert m["model_size"]["file_size_mb"] is not None
    assert m["accuracy"]["top1_accuracy"] == 0.6767

    # Retrieve comparison run by ID
    r_get = client.get(f"/api/benchmarks/{run_id}")
    assert r_get.status_code == 200
    retrieved = r_get.json()
    assert retrieved["id"] == run_id
    assert retrieved["status"] == "completed"
    assert len(retrieved["measurements"]) == 1


def test_benchmarks_get_nonexistent_returns_404(client: TestClient):
    """GET /api/benchmarks/{run_id} returns 404 if ID does not exist."""
    response = client.get("/api/benchmarks/99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_benchmarks_async_execution(client: TestClient):
    """POST /api/benchmarks with async_run=True returns immediately with status 'running'."""
    payload = {
        "model_ids": ["mobilenet_v3_small"],
        "runtimes": ["onnx"],
        "metrics": ["model_size"],
        "num_runs": 1,
        "warmup_runs": 0,
        "async_run": True,
        "run_name": "Async Benchmark Test",
    }
    response = client.post("/api/benchmarks", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "running"
    assert data["id"] > 0


def test_benchmarks_transaction_rollback_on_persistence_failure():
    """Ensure database transactions roll back if persistence fails."""
    factory = get_session_factory()
    with pytest.raises(RuntimeError):
        with session_scope(factory) as session:
            run = BenchmarkRun(
                run_name="Rollback Test Run",
                device="cpu",
                status="running",
            )
            session.add(run)
            session.flush()
            # Force an unhandled error to trigger rollback
            raise RuntimeError("Forced simulation of persistence failure")

    # Verify run was rolled back and does not exist in DB
    with session_scope(factory) as session:
        found = session.query(BenchmarkRun).filter_by(run_name="Rollback Test Run").first()
        assert found is None


# ==============================================================================
# 5. Recommendation Engine Endpoint Tests
# ==============================================================================

def test_recommendations_empty_database(client: TestClient):
    """POST /api/recommendations against an empty database returns a valid empty response with explanation."""
    # Temporarily clean test tables in test database to guarantee 0 records
    factory = get_session_factory()
    with session_scope(factory) as session:
        session.query(ModelMeasurement).delete()
        session.query(BenchmarkRun).delete()

    response = client.post("/api/recommendations", json={"priority_preset": "balanced"})
    assert response.status_code == 200
    data = response.json()

    assert data["recommended_model"] is None
    assert data["recommended_runtime"] is None
    assert data["composite_score"] is None
    assert len(data["rankings"]) == 0
    assert "No benchmark measurements are available" in data["explanation"]
    assert "Run benchmarks from the dashboard" in data["trade_off_analysis"]


def test_recommendations_preset_balanced(client: TestClient):
    """POST /api/recommendations with balanced preset evaluates models across all criteria."""
    # Seed test database with baseline measurements strictly in test database
    factory = get_session_factory()
    with session_scope(factory) as session:
        import_baseline_benchmarks_if_needed(session)

    response = client.post("/api/recommendations", json={"priority_preset": "balanced"})
    assert response.status_code == 200
    data = response.json()

    assert data["recommended_model"] is not None
    assert data["recommended_runtime"] is not None
    assert data["composite_score"] is not None
    assert isinstance(data["explanation"], str)
    assert len(data["explanation"]) > 20
    assert isinstance(data["trade_off_analysis"], str)
    assert "methodology_disclaimer" in data
    assert len(data["rankings"]) > 0

    # Ensure rankings are in descending order of composite score for eligible models
    eligible_scores = [c["composite_score"] for c in data["rankings"] if c["eligible"]]
    assert eligible_scores == sorted(eligible_scores, reverse=True)


def test_recommendations_custom_weights(client: TestClient):
    """POST /api/recommendations should accept and normalize custom weights."""
    payload = {
        "weights": {
            "accuracy": 0.50,
            "latency": 0.30,
            "memory": 0.10,
            "model_size": 0.10,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()

    applied = data["criteria_weights_applied"]
    assert pytest.approx(sum(applied.values()), 0.001) == 1.0
    assert applied["accuracy"] == 0.50
    assert applied["latency"] == 0.30


def test_recommendations_hard_constraints_filtering(client: TestClient):
    """POST /api/recommendations should disqualify candidates violating hard constraints."""
    # Disqualify all models with latency > 5.0ms
    payload = {
        "constraints": {
            "max_latency_ms": 5.0,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()

    for candidate in data["rankings"]:
        lat = candidate["raw_metrics"]["latency"]
        if lat is not None and lat > 5.0:
            assert candidate["eligible"] is False
            assert len(candidate["disqualification_reasons"]) > 0
            assert "exceeds constraint" in candidate["disqualification_reasons"][0]


def test_recommendations_missing_metrics_handled_transparently(client: TestClient):
    """Candidates with missing metrics must be handled transparently without invented zeros."""
    response = client.post("/api/recommendations", json={"priority_preset": "balanced"})
    assert response.status_code == 200
    data = response.json()

    # Find candidates with missing metrics (e.g. PyTorch missing file_size_mb, or unevaluated accuracy)
    has_candidate_with_missing_metric = False
    for c in data["rankings"]:
        if c["missing_metrics"]:
            has_candidate_with_missing_metric = True
            assert c["data_complete"] is False
            for m in c["missing_metrics"]:
                assert c["raw_metrics"][m] is None
                assert c["subscores"][m] is None
            # Notes must explicitly report missing metrics
            assert any("Metrics missing" in note for note in c["notes"])

    assert has_candidate_with_missing_metric is True


def test_recommendations_all_disqualified_handled_gracefully(client: TestClient):
    """When all candidates violate constraints, return None winner with clear explanation."""
    payload = {
        "constraints": {
            "max_latency_ms": 0.01,  # Impossible threshold
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["recommended_model"] is None
    assert data["recommended_runtime"] is None
    assert data["composite_score"] is None
    assert "No model satisfied the specified edge constraints" in data["explanation"]


def test_recommendations_ignores_edge_impulse_analytical_estimates(client: TestClient):
    """POST /api/recommendations must ignore Edge Impulse analytical estimates and evaluate live Host CPU only."""
    factory = get_session_factory()
    with session_scope(factory) as session:
        import_baseline_benchmarks_if_needed(session)

        # Insert an Edge Impulse estimation run for Raspberry Pi 4
        edge_run = BenchmarkRun(
            run_name="Edge Impulse Estimation Test Run",
            device="raspberry-pi-4",
            provenance="edge_impulse_estimation",
            status="completed",
        )
        session.add(edge_run)
        session.flush()

        edge_meas = ModelMeasurement(
            run_id=edge_run.id,
            model_id="mobilenet_v3_small",
            architecture="MobileNetV3-Small",
            runtime="tflite",
            device="raspberry-pi-4",
            batch_size=1,
            top1_accuracy=0.6767,
            top5_accuracy=0.8741,
            latency_mean_ms=21477.0,
            peak_ram_mb=None,
            status="completed",
        )
        session.add(edge_meas)
        session.commit()
        edge_run_id = edge_run.id

    # 1. General recommendations must ignore the Edge Impulse estimation run
    response = client.post("/api/recommendations", json={"priority_preset": "balanced"})
    assert response.status_code == 200
    data = response.json()

    for candidate in data["rankings"]:
        # Verify no raspberry-pi-4 candidates are evaluated in MCDA
        assert candidate["device"] == "cpu"
        # Latency should not be 21477.0
        assert candidate["raw_metrics"]["latency"] < 1000.0

    # 2. Specifying the Edge Impulse run_id explicitly must be rejected with 400
    resp_run = client.post("/api/recommendations", json={"run_id": edge_run_id})
    assert resp_run.status_code == 400
    assert "Edge Impulse analytical estimation runs cannot be ranked" in resp_run.json()["detail"]


def test_recommendations_invalid_preset_rejected(client: TestClient):
    """POST /api/recommendations should reject invalid preset with 422."""
    response = client.post(
        "/api/recommendations",
        json={"priority_preset": "unsupported_preset_mode"},
    )
    assert response.status_code == 422


def test_recommendations_all_zero_custom_weights_rejected(client: TestClient):
    """POST /api/recommendations should reject all-zero custom weights with 422."""
    payload = {
        "weights": {
            "accuracy": 0.0,
            "latency": 0.0,
            "memory": 0.0,
            "model_size": 0.0,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 422
    assert "Set at least one criterion weight above zero." in response.text


def test_recommendations_individual_zero_custom_weights_accepted(client: TestClient):
    """POST /api/recommendations should allow individual zero weights and score accordingly."""
    payload = {
        "weights": {
            "accuracy": 0.8,
            "latency": 0.0,
            "memory": 0.2,
            "model_size": 0.0,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    applied = data["criteria_weights_applied"]
    assert applied["latency"] == 0.0
    assert applied["model_size"] == 0.0
    assert applied["accuracy"] == 0.8
    assert applied["memory"] == 0.2


def test_recommendations_top_3_returned_and_explained(client: TestClient):
    """POST /api/recommendations returns up to three ranked eligible candidates with explanations and satisfied constraints."""
    response = client.post("/api/recommendations", json={"priority_preset": "balanced"})
    assert response.status_code == 200
    data = response.json()

    assert "top_recommendations" in data
    top_recs = data["top_recommendations"]
    assert 1 <= len(top_recs) <= 3

    for idx, cand in enumerate(top_recs, start=1):
        assert cand["rank"] == idx
        assert cand["eligible"] is True
        assert cand["composite_score"] is not None
        assert cand["rationale"] is not None
        assert len(cand["rationale"]) > 0
        assert cand["trade_offs"] is not None
        assert isinstance(cand["satisfied_constraints"], list)


def test_recommendations_fewer_than_3_eligible(client: TestClient):
    """When strict constraints leave only 1 candidate, top_recommendations contains only that candidate."""
    # Constrain latency tightly so only the fastest model qualifies
    payload = {
        "constraints": {
            "max_latency_ms": 2.5,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()

    top_recs = data["top_recommendations"]
    assert len(top_recs) < 3
    for cand in top_recs:
        assert cand["eligible"] is True
        assert cand["raw_metrics"]["latency"] <= 2.5
        assert any("Latency" in sc for sc in cand["satisfied_constraints"])


def test_recommendations_missing_accuracy_fails_min_accuracy(client: TestClient):
    """A candidate with missing accuracy must not qualify for min_accuracy constraint."""
    from recommendation.engine import RecommendationEngine

    mock_candidates = [
        {
            "model_id": "mobilenet_v3_small",
            "runtime": "onnx",
            "device": "cpu",
            "latency_mean_ms": 1.5,
            "peak_ram_mb": 600.0,
            "file_size_mb": 10.0,
            "top1_accuracy": None,  # Accuracy unmeasured
        },
        {
            "model_id": "resnet50",
            "runtime": "onnx",
            "device": "cpu",
            "latency_mean_ms": 12.0,
            "peak_ram_mb": 800.0,
            "file_size_mb": 95.0,
            "top1_accuracy": 0.80,  # Measured accuracy
        },
    ]

    engine = RecommendationEngine(constraints={"min_accuracy": 0.50})
    result = engine.evaluate_candidates(mock_candidates)

    top_recs = result["top_recommendations"]
    assert len(top_recs) == 1
    assert top_recs[0]["model_id"] == "resnet50"

    disqualified = [c for c in result["rankings"] if not c["eligible"]]
    assert len(disqualified) == 1
    assert disqualified[0]["model_id"] == "mobilenet_v3_small"
    assert any("Accuracy is unmeasured" in r for r in disqualified[0]["disqualification_reasons"])


def test_recommendations_invalid_negative_constraints_rejected(client: TestClient):
    """POST /api/recommendations should reject negative constraints with 422."""
    payload = {
        "constraints": {
            "max_latency_ms": -5.0,
        }
    }
    response = client.post("/api/recommendations", json=payload)
    assert response.status_code == 422


# ==============================================================================
# 7. Edge Impulse Profiling Endpoint Tests
# ==============================================================================

def test_edge_impulse_targets_endpoint(client: TestClient):
    """GET /api/edge-impulse/targets should return verified hardware targets."""
    response = client.get("/api/edge-impulse/targets")
    assert response.status_code == 200
    data = response.json()

    assert data["total_targets"] >= 1
    target_ids = [t["target_id"] for t in data["targets"]]
    assert "raspberry-pi-4" in target_ids

    rpi4 = next(t for t in data["targets"] if t["target_id"] == "raspberry-pi-4")
    assert rpi4["display_name"] == "Raspberry Pi 4"
    assert rpi4["supported_formats"] == ["tflite"]
    assert rpi4["is_supported"] is True


def test_edge_impulse_profile_endpoint_rejects_onnx(client: TestClient):
    """POST /api/edge-impulse/profile must reject ONNX models with 400."""
    response = client.post(
        "/api/edge-impulse/profile",
        json={
            "model_id": "mobilenet_v3_small",
            "runtime": "onnx",
            "target_device": "raspberry-pi-4",
        },
    )
    assert response.status_code == 400
    assert "TFLite models only" in response.json()["detail"]


def test_edge_impulse_profile_endpoint_rejects_unknown_target(client: TestClient):
    """POST /api/edge-impulse/profile must reject unsupported target devices with 400."""
    response = client.post(
        "/api/edge-impulse/profile",
        json={
            "model_id": "mobilenet_v3_small",
            "runtime": "tflite",
            "target_device": "unknown_microcontroller",
        },
    )
    assert response.status_code == 400
    assert "not a recognized Edge Impulse profiling target" in response.json()["detail"]


def test_edge_impulse_profile_endpoint_rejects_unregistered_model(client: TestClient):
    """POST /api/edge-impulse/profile must reject unregistered model IDs with 400."""
    response = client.post(
        "/api/edge-impulse/profile",
        json={
            "model_id": "fantasy_net_v9",
            "runtime": "tflite",
            "target_device": "raspberry-pi-4",
        },
    )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_edge_impulse_profile_endpoint_success_and_reuse(client: TestClient):
    """POST /api/edge-impulse/profile persists estimation with NULL RAM and reuses existing record."""
    payload = {
        "model_id": "mobilenet_v3_small",
        "target_device": "raspberry-pi-4",
        "runtime": "tflite",
        "batch_size": 1,
        "reuse_completed": True,
        "reuse_saved_accuracy": True,
    }

    # Initial profile request: reuses verified completed job 55253939
    response = client.post("/api/edge-impulse/profile", json=payload)
    assert response.status_code == 201
    data = response.json()

    first_run_id = data["id"]
    assert data["provenance"] == "edge_impulse_estimation"
    assert data["status"] == "completed"
    assert data["device"] == "raspberry-pi-4"

    # Environment info contains provenance, MCU warnings, and estimation disclaimers
    env = data["environment_info"]
    assert env["provider"] == "edge_impulse"
    assert env["job_id"] == 55253939
    assert env["latency_type"] == "analytical_estimate"
    assert env["throughput_type"] == "derived_from_estimated_latency"
    assert env["is_supported_on_mcu"] is False
    assert env["mcu_support_error"] == "Unsupported ops: RELU_0_TO_1."

    # Measurement checks
    assert len(data["measurements"]) == 1
    m = data["measurements"][0]
    assert m["model_id"] == "mobilenet_v3_small"
    assert m["runtime"] == "tflite"
    assert m["device"] == "raspberry-pi-4"
    assert m["status"] == "completed"

    # Latency & throughput
    assert m["latency"]["mean_ms"] == 21477.0
    assert m["latency"]["p95_ms"] is None
    assert m["latency"]["throughput_fps"] is not None

    # Memory: unmeasured RAM is represented as None (NULL/NA), never zero
    assert m["memory"] is None

    # Model size & parameters
    assert m["model_size"]["file_size_mb"] == 4.9
    assert m["model_size"]["parameter_count"] == 2542856

    # Reused verified accuracy
    assert m["accuracy"]["top1_accuracy"] == 0.6764
    assert m["accuracy"]["evaluated_samples"] == 50000

    # Second profile request: must reuse the exact same run from database without creating a new run
    response_reuse = client.post("/api/edge-impulse/profile", json=payload)
    assert response_reuse.status_code == 201
    data_reuse = response_reuse.json()
    assert data_reuse["id"] == first_run_id


def test_edge_impulse_profile_endpoint_blocks_unauthorized_external_submission(client: TestClient):
    """POST /api/edge-impulse/profile must reject submission when no cached result is available."""
    with patch.dict("backend.services.edge_impulse_service.VERIFIED_COMPLETED_JOBS", {}, clear=True):
        response = client.post(
            "/api/edge-impulse/profile",
            json={
                "model_id": "mobilenet_v3_small",
                "target_device": "raspberry-pi-4",
                "runtime": "tflite",
                "reuse_completed": False,
            },
        )
        assert response.status_code == 400
        assert "External job submission is currently disabled" in response.json()["detail"]
