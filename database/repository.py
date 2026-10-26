"""
Repository Operations for Benchmark Persistence and Comparison.

Provides clean CRUD functions for benchmark comparison runs and their
associated model measurements, supporting multi-model comparisons,
query filtering, and transactional safety.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session, joinedload

from database.models import BenchmarkRun, ModelMeasurement


def create_benchmark_run(
    session: Session,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    device: str = "cpu",
    environment_info: Optional[Dict[str, Any]] = None,
    provenance: str = "live_execution",
    status: str = "completed",
    error_message: Optional[str] = None,
) -> BenchmarkRun:
    """Create and persist a new benchmark run header.

    Args:
        session: Active SQLAlchemy database session.
        run_name: Optional user-friendly name for the run.
        description: Optional textual notes describing the benchmark run.
        device: Target execution device (e.g. 'cpu', 'mps').
        environment_info: Hardware, OS, and library metadata.
        provenance: Origin of data ('live_execution' or 'imported_artifact').
        status: Run completion status ('completed', 'failed', 'running').
        error_message: Optional error message if the run failed.

    Returns:
        Persisted BenchmarkRun instance with populated ID.
    """
    run = BenchmarkRun(
        run_name=run_name,
        description=description,
        device=device,
        environment_info=environment_info,
        provenance=provenance,
        status=status,
        error_message=error_message,
        completed_at=datetime.now(timezone.utc) if status in ("completed", "failed") else None,
    )
    session.add(run)
    session.flush()
    return run


def add_measurement(
    session: Session,
    measurement: ModelMeasurement,
) -> ModelMeasurement:
    """Add a single model measurement record to a benchmark run.

    Args:
        session: Active SQLAlchemy database session.
        measurement: ModelMeasurement instance with run_id set.

    Returns:
        Persisted ModelMeasurement with populated ID.
    """
    session.add(measurement)
    session.flush()
    return measurement


def add_measurements(
    session: Session,
    measurements: Sequence[ModelMeasurement],
) -> List[ModelMeasurement]:
    """Bulk-add multiple model measurements to their respective benchmark runs.

    Args:
        session: Active SQLAlchemy database session.
        measurements: Sequence of ModelMeasurement instances.

    Returns:
        List of persisted ModelMeasurement instances.
    """
    for m in measurements:
        session.add(m)
    session.flush()
    return list(measurements)


def create_comparison_run(
    session: Session,
    run_name: str,
    measurements: Sequence[ModelMeasurement],
    description: Optional[str] = None,
    device: str = "cpu",
    environment_info: Optional[Dict[str, Any]] = None,
    provenance: str = "live_execution",
) -> BenchmarkRun:
    """Atomically create a comparison run associating multiple model measurements.

    Args:
        session: Active SQLAlchemy database session.
        run_name: Name of the comparison session.
        measurements: Sequence of ModelMeasurement records to associate.
        description: Optional notes or tags.
        device: Device identifier.
        environment_info: Optional environment dictionary.
        provenance: Data origin.

    Returns:
        Persisted BenchmarkRun containing all associated measurements.
    """
    run = create_benchmark_run(
        session=session,
        run_name=run_name,
        description=description,
        device=device,
        environment_info=environment_info,
        provenance=provenance,
    )
    for m in measurements:
        m.run_id = run.id
        session.add(m)
    session.flush()
    return run


def get_benchmark_run(
    session: Session,
    run_id: int,
) -> Optional[BenchmarkRun]:
    """Retrieve a benchmark run by its primary key ID, including measurements.

    Args:
        session: Active SQLAlchemy database session.
        run_id: Primary key ID of the run.

    Returns:
        BenchmarkRun instance or None if not found.
    """
    stmt = select(BenchmarkRun).where(BenchmarkRun.id == run_id)
    return session.execute(stmt).scalar_one_or_none()


def list_benchmark_runs(
    session: Session,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
) -> List[BenchmarkRun]:
    """List benchmark runs ordered by creation timestamp descending.

    Args:
        session: Active SQLAlchemy database session.
        limit: Maximum number of runs to return.
        offset: Offset for pagination.
        status: Optional filter by status ('completed', 'failed').

    Returns:
        List of BenchmarkRun instances.
    """
    stmt = select(BenchmarkRun).order_by(desc(BenchmarkRun.created_at))
    if status is not None:
        stmt = stmt.where(BenchmarkRun.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.execute(stmt).scalars().all())


def get_measurements_for_run(
    session: Session,
    run_id: int,
) -> List[ModelMeasurement]:
    """Retrieve all model measurements associated with a specific benchmark run.

    Args:
        session: Active SQLAlchemy database session.
        run_id: Target benchmark run ID.

    Returns:
        List of ModelMeasurement instances ordered by measurement ID.
    """
    stmt = (
        select(ModelMeasurement)
        .where(ModelMeasurement.run_id == run_id)
        .order_by(ModelMeasurement.id)
    )
    return list(session.execute(stmt).scalars().all())


def get_measurements_for_comparison(
    session: Session,
    model_ids: Optional[Sequence[str]] = None,
    runtimes: Optional[Sequence[str]] = None,
    run_ids: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
) -> List[ModelMeasurement]:
    """Query measurements across runs and models for cross-model comparative visualization.

    Args:
        session: Active SQLAlchemy database session.
        model_ids: Optional list of model IDs to filter by (e.g. ['mobilenet_v3_small', 'resnet50']).
        runtimes: Optional list of runtimes to filter by (e.g. ['pytorch', 'onnx', 'tflite']).
        run_ids: Optional list of run IDs to restrict search to.
        device: Optional target device filter (e.g. 'cpu').

    Returns:
        List of matching ModelMeasurement records.
    """
    stmt = (
        select(ModelMeasurement)
        .options(joinedload(ModelMeasurement.run))
        .order_by(
            ModelMeasurement.model_id,
            ModelMeasurement.runtime,
            desc(ModelMeasurement.created_at),
        )
    )

    if model_ids:
        stmt = stmt.where(ModelMeasurement.model_id.in_(model_ids))
    if runtimes:
        stmt = stmt.where(ModelMeasurement.runtime.in_(runtimes))
    if run_ids:
        stmt = stmt.where(ModelMeasurement.run_id.in_(run_ids))
    if device:
        stmt = stmt.where(ModelMeasurement.device == device)

    return list(session.execute(stmt).scalars().all())


def delete_benchmark_run(
    session: Session,
    run_id: int,
) -> bool:
    """Delete a benchmark run and all associated measurements (via CASCADE).

    Args:
        session: Active SQLAlchemy database session.
        run_id: ID of the run to delete.

    Returns:
        True if a run was deleted, False if no run matched the ID.
    """
    run = get_benchmark_run(session, run_id)
    if run is None:
        return False
    session.delete(run)
    session.flush()
    return True
