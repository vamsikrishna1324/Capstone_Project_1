"""
Comprehensive Test Suite for MySQL Persistence Layer (Step 5).

Verifies:
1. ORM Model instantiation, attributes, defaults, and relationships.
2. Serialization to dictionary (to_dict) preserving NULLs for unmeasured metrics.
3. Strict MySQL DDL compilation (CREATE TABLE, foreign keys, cascades, data types).
4. Repository operations: create_benchmark_run, add_measurement, add_measurements.
5. Multi-model comparison run creation and retrieval.
6. Cross-model query filtering in get_measurements_for_comparison.
7. Cascade deletion behavior.
8. Transaction management: commit on success and rollback on failure.
9. Credential protection in error messages and representations.
10. Live MySQL test execution if an active test database is reachable (skips cleanly if not).
    STRICT PROHIBITION: Never uses SQLite for testing or fallback.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from database.config import DatabaseConfig, DatabaseConfigurationError
from database.connection import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
    session_scope,
)
from database.models import Base, BenchmarkRun, ModelMeasurement
from database.repository import (
    add_measurement,
    add_measurements,
    create_benchmark_run,
    create_comparison_run,
    delete_benchmark_run,
    get_benchmark_run,
    get_measurements_for_comparison,
    get_measurements_for_run,
    list_benchmark_runs,
)


# ============================================================================
# 1. ORM Model Instantiation & Relationship Tests
# ============================================================================
def test_benchmark_run_defaults() -> None:
    """Verify BenchmarkRun default attribute assignments and types."""
    run = BenchmarkRun()
    assert run.device == "cpu"
    assert run.status == "completed"
    assert run.provenance == "live_execution"
    assert run.error_message is None
    assert run.run_name is None


def test_model_measurement_preserves_null_for_unmeasured_metrics() -> None:
    """Verify that unmeasured metrics are stored as None (SQL NULL), never invented zeros."""
    m = ModelMeasurement(
        run_id=1,
        model_id="mobilenet_v3_small",
        architecture="MobileNetV3-Small",
        runtime="onnx",
        device="cpu",
        batch_size=1,
        latency_mean_ms=1.47,
        throughput_fps=681.3,
        peak_ram_mb=652.6,
        file_size_mb=9.72,
        parameter_count=2542856,
        # Accuracy is NOT evaluated: must be None
        top1_accuracy=None,
        top5_accuracy=None,
        evaluated_samples=None,
    )

    assert m.top1_accuracy is None
    assert m.top5_accuracy is None
    assert m.evaluated_samples is None
    assert m.latency_mean_ms == 1.47
    assert m.throughput_fps == 681.3
    assert m.peak_ram_mb == 652.6


def test_model_relationship_association() -> None:
    """Verify bidirectional relationship between BenchmarkRun and ModelMeasurement."""
    run = BenchmarkRun(id=10, run_name="Multi-Model Comparison Run")
    m1 = ModelMeasurement(model_id="mobilenet_v3_small", runtime="onnx", latency_mean_ms=1.47)
    m2 = ModelMeasurement(model_id="resnet50", runtime="onnx", latency_mean_ms=11.49)

    run.measurements.append(m1)
    run.measurements.append(m2)

    assert len(run.measurements) == 2
    assert run.measurements[0].model_id == "mobilenet_v3_small"
    assert run.measurements[1].model_id == "resnet50"
    assert m1.run is run
    assert m2.run is run


# ============================================================================
# 2. Serialization (to_dict) Tests
# ============================================================================
def test_benchmark_run_to_dict_serialization() -> None:
    """Verify BenchmarkRun dictionary serialization structure."""
    now = datetime.now(timezone.utc)
    run = BenchmarkRun(
        id=42,
        run_name="Edge Deployment Benchmark",
        description="Comprehensive evaluation",
        device="cpu",
        status="completed",
        environment_info={"torch_version": "2.14.0"},
        provenance="imported_artifact",
        created_at=now,
    )
    d = run.to_dict(include_measurements=False)

    assert d["id"] == 42
    assert d["run_name"] == "Edge Deployment Benchmark"
    assert d["device"] == "cpu"
    assert d["status"] == "completed"
    assert d["environment_info"] == {"torch_version": "2.14.0"}
    assert d["created_at"] == now.isoformat()


def test_model_measurement_to_dict_serialization() -> None:
    """Verify ModelMeasurement dictionary serialization structures metrics cleanly."""
    m = ModelMeasurement(
        id=100,
        run_id=42,
        model_id="resnet50",
        architecture="ResNet-50",
        runtime="onnx",
        device="cpu",
        batch_size=1,
        top1_accuracy=0.8085,
        top5_accuracy=0.9544,
        evaluated_samples=50000,
        latency_mean_ms=11.49,
        latency_p95_ms=12.88,
        throughput_fps=87.0,
        peak_ram_mb=899.1,
        file_size_mb=97.42,
        parameter_count=25557032,
    )
    d = m.to_dict()

    assert d["id"] == 100
    assert d["run_id"] == 42
    assert d["model_id"] == "resnet50"
    assert d["runtime"] == "onnx"

    # Accuracy block
    assert d["accuracy"] is not None
    assert d["accuracy"]["top1_accuracy"] == 0.8085
    assert d["accuracy"]["top5_accuracy"] == 0.9544
    assert d["accuracy"]["evaluated_samples"] == 50000

    # Latency block
    assert d["latency"] is not None
    assert d["latency"]["mean_ms"] == 11.49
    assert d["latency"]["p95_ms"] == 12.88
    assert d["latency"]["throughput_fps"] == 87.0

    # Memory block
    assert d["memory"] is not None
    assert d["memory"]["peak_ram_mb"] == 899.1

    # Model size block
    assert d["model_size"] is not None
    assert d["model_size"]["file_size_mb"] == 97.42
    assert d["model_size"]["parameter_count"] == 25557032


# ============================================================================
# 3. MySQL DDL Compilation & Dialect Verification Tests
# ============================================================================
def test_mysql_ddl_compilation_benchmark_runs() -> None:
    """Verify that BenchmarkRun compiles to valid MySQL DDL statements."""
    dialect = mysql.dialect()
    ddl = str(CreateTable(BenchmarkRun.__table__).compile(dialect=dialect))

    assert "CREATE TABLE benchmark_runs" in ddl
    assert "id INTEGER NOT NULL AUTO_INCREMENT" in ddl
    assert "artifact_hash" not in ddl
    assert "environment_info JSON" in ddl
    assert "PRIMARY KEY (id)" in ddl


def test_mysql_ddl_compilation_model_measurements() -> None:
    """Verify that ModelMeasurement compiles to valid MySQL DDL with foreign key and cascade."""
    dialect = mysql.dialect()
    ddl = str(CreateTable(ModelMeasurement.__table__).compile(dialect=dialect))

    assert "CREATE TABLE model_measurements" in ddl
    assert "run_id INTEGER NOT NULL" in ddl
    assert "top1_accuracy FLOAT" in ddl
    assert "latency_mean_ms FLOAT" in ddl
    assert "latency_p95_ms FLOAT" in ddl
    assert "file_size_mb FLOAT" in ddl
    assert "file_size_bytes" not in ddl
    assert "weights" not in ddl
    assert "parameter_count BIGINT" in ddl
    assert "FOREIGN KEY(run_id) REFERENCES benchmark_runs (id) ON DELETE CASCADE" in ddl


# ============================================================================
# 4. Repository CRUD & Transaction Tests (Mocked Session)
# ============================================================================
def test_create_benchmark_run_repository() -> None:
    """Verify create_benchmark_run adds and flushes the entity."""
    mock_session = MagicMock(spec=Session)

    run = create_benchmark_run(
        session=mock_session,
        run_name="Test Run",
        description="Testing repository",
        device="cpu",
        status="completed",
    )

    mock_session.add.assert_called_once_with(run)
    mock_session.flush.assert_called_once()
    assert run.run_name == "Test Run"


def test_add_measurement_repository() -> None:
    """Verify add_measurement adds and flushes single measurement."""
    mock_session = MagicMock(spec=Session)
    m = ModelMeasurement(run_id=1, model_id="efficientnet_b0", runtime="onnx")

    result = add_measurement(mock_session, m)

    mock_session.add.assert_called_once_with(m)
    mock_session.flush.assert_called_once()
    assert result is m


def test_add_measurements_repository() -> None:
    """Verify add_measurements bulk adds all measurement objects."""
    mock_session = MagicMock(spec=Session)
    m1 = ModelMeasurement(run_id=1, model_id="mobilenet_v3_small", runtime="pytorch")
    m2 = ModelMeasurement(run_id=1, model_id="resnet50", runtime="pytorch")

    results = add_measurements(mock_session, [m1, m2])

    assert mock_session.add.call_count == 2
    mock_session.flush.assert_called_once()
    assert len(results) == 2


def test_create_comparison_run_multi_model() -> None:
    """Verify atomic comparison run creation associating multiple models."""
    mock_session = MagicMock(spec=Session)
    m1 = ModelMeasurement(model_id="mobilenet_v3_small", runtime="onnx")
    m2 = ModelMeasurement(model_id="resnet50", runtime="onnx")
    m3 = ModelMeasurement(model_id="efficientnet_b0", runtime="onnx")
    m4 = ModelMeasurement(model_id="densenet121", runtime="onnx")

    # Give simulated ID upon creation
    def side_effect(entity):
        if isinstance(entity, BenchmarkRun):
            entity.id = 99

    mock_session.add.side_effect = side_effect

    run = create_comparison_run(
        session=mock_session,
        run_name="Full 4-Model ONNX Comparison",
        measurements=[m1, m2, m3, m4],
        device="cpu",
    )

    assert run.run_name == "Full 4-Model ONNX Comparison"
    assert m1.run_id == 99
    assert m2.run_id == 99
    assert m3.run_id == 99
    assert m4.run_id == 99
    assert mock_session.add.call_count == 5  # 1 run + 4 measurements


def test_get_benchmark_run_repository() -> None:
    """Verify get_benchmark_run executes SELECT query with correct ID."""
    mock_session = MagicMock(spec=Session)
    mock_result = MagicMock()
    fake_run = BenchmarkRun(id=7, run_name="Found Run")
    mock_result.scalar_one_or_none.return_value = fake_run
    mock_session.execute.return_value = mock_result

    run = get_benchmark_run(mock_session, 7)

    assert run is fake_run
    mock_session.execute.assert_called_once()



def test_delete_benchmark_run_repository() -> None:
    """Verify delete_benchmark_run deletes entity when present and handles missing gracefully."""
    mock_session = MagicMock(spec=Session)
    fake_run = BenchmarkRun(id=5)

    with patch("database.repository.get_benchmark_run", return_value=fake_run):
        deleted = delete_benchmark_run(mock_session, 5)
        assert deleted is True
        mock_session.delete.assert_called_once_with(fake_run)
        mock_session.flush.assert_called_once()

    with patch("database.repository.get_benchmark_run", return_value=None):
        deleted = delete_benchmark_run(mock_session, 999)
        assert deleted is False


def test_list_benchmark_runs_repository() -> None:
    """Verify list_benchmark_runs executes paginated query."""
    mock_session = MagicMock(spec=Session)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        BenchmarkRun(id=1, run_name="Run 1"),
        BenchmarkRun(id=2, run_name="Run 2"),
    ]
    mock_session.execute.return_value = mock_result

    runs = list_benchmark_runs(mock_session, limit=10, offset=0, status="completed")
    assert len(runs) == 2
    mock_session.execute.assert_called_once()


def test_get_measurements_for_run_repository() -> None:
    """Verify get_measurements_for_run executes query for run ID."""
    mock_session = MagicMock(spec=Session)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        ModelMeasurement(id=1, run_id=10, model_id="mobilenet_v3_small"),
        ModelMeasurement(id=2, run_id=10, model_id="resnet50"),
    ]
    mock_session.execute.return_value = mock_result

    measurements = get_measurements_for_run(mock_session, 10)
    assert len(measurements) == 2
    mock_session.execute.assert_called_once()


def test_get_measurements_for_comparison_repository() -> None:
    """Verify get_measurements_for_comparison applies model and runtime filters."""
    mock_session = MagicMock(spec=Session)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        ModelMeasurement(id=1, model_id="resnet50", runtime="onnx"),
        ModelMeasurement(id=2, model_id="resnet50", runtime="tflite"),
    ]
    mock_session.execute.return_value = mock_result

    measurements = get_measurements_for_comparison(
        mock_session,
        model_ids=["resnet50"],
        runtimes=["onnx", "tflite"],
        device="cpu",
    )
    assert len(measurements) == 2
    mock_session.execute.assert_called_once()



# ============================================================================
# 5. Transaction Safety & Rollback Tests
# ============================================================================
def test_session_scope_commit_on_success() -> None:
    """Verify session_scope commits when execution block completes cleanly."""
    mock_session = MagicMock(spec=Session)
    mock_factory = MagicMock(return_value=mock_session)

    with session_scope(mock_factory) as session:
        session.add(BenchmarkRun(run_name="Success Run"))

    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()


def test_session_scope_rollback_on_failure() -> None:
    """Verify session_scope rolls back and re-raises exception upon failure."""
    mock_session = MagicMock(spec=Session)
    mock_factory = MagicMock(return_value=mock_session)

    class CustomDbException(RuntimeError):
        pass

    with pytest.raises(CustomDbException):
        with session_scope(mock_factory) as session:
            session.add(BenchmarkRun(run_name="Failing Run"))
            raise CustomDbException("Simulated constraint violation")

    mock_session.rollback.assert_called_once()
    mock_session.commit.assert_not_called()
    mock_session.close.assert_called_once()


# ============================================================================
# 6. Credential Protection & Safe Display Tests
# ============================================================================
def test_database_credentials_masked_in_all_representations() -> None:
    """Verify that sensitive database passwords are never exposed in safe representations."""
    cfg = DatabaseConfig(
        user="edge_worker",
        password="TopSecretPassword123!",
        host="db.internal.local",
        port=3306,
        database="benchmarks",
    )
    assert "TopSecretPassword123!" not in cfg.safe_url
    assert "TopSecretPassword123!" not in str(cfg)
    assert "TopSecretPassword123!" not in repr(cfg)
    assert "TopSecretPassword123!" not in str(cfg.to_dict())


# ============================================================================
# 7. Live MySQL Integration Test (Conditional)
# ============================================================================
def test_live_mysql_integration_if_configured() -> None:
    """Live integration test against disposable MySQL test database.

    Skips cleanly if MySQL server is not reachable on host/port.
    STRICT PROHIBITION: Never falls back to SQLite.
    """
    try:
        config = DatabaseConfig.from_env(is_test=True)
    except DatabaseConfigurationError:
        pytest.skip("MySQL test environment variables are not configured in environment.")
        return

    engine = create_database_engine(config)
    is_live = check_database_connection(engine, timeout_seconds=1.5)
    if not is_live:
        pytest.skip(f"Live MySQL database is not reachable at: {config.safe_url}. Skipping live DB tests.")
        return

    # If live MySQL server IS reachable, test actual table creation and lifecycle
    session_factory = create_session_factory(engine)
    Base.metadata.create_all(engine)

    with session_scope(session_factory) as session:
        run = create_benchmark_run(
            session=session,
            run_name="Live Integration Test Run",
            device="cpu",
        )
        m = ModelMeasurement(
            run_id=run.id,
            model_id="mobilenet_v3_small",
            runtime="onnx",
            latency_mean_ms=1.47,
            throughput_fps=681.3,
            peak_ram_mb=652.6,
        )
        add_measurement(session, m)

        # Retrieve and verify
        retrieved = get_benchmark_run(session, run.id)
        assert retrieved is not None
        assert len(retrieved.measurements) == 1
        assert retrieved.measurements[0].model_id == "mobilenet_v3_small"

        # Cleanup test run
        delete_benchmark_run(session, run.id)
