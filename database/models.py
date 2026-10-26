"""
SQLAlchemy 2.0 ORM Models for Automated Image Model Benchmarking.

Defines MySQL persistence schema for benchmark comparison runs and their
associated model/runtime performance metrics.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all benchmarking database models."""
    pass


class BenchmarkRun(Base):
    """Represents a benchmark execution or comparison session.

    Can group multiple model and runtime measurements together for cross-model
    and cross-runtime comparative evaluation.
    """

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device: Mapped[str] = mapped_column(String(64), default="cpu", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    environment_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    provenance: Mapped[str] = mapped_column(
        String(32),
        default="live_execution",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    measurements: Mapped[List["ModelMeasurement"]] = relationship(
        "ModelMeasurement",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ModelMeasurement.id",
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("status", "completed")
        kwargs.setdefault("provenance", "live_execution")
        kwargs.setdefault("created_at", datetime.now(timezone.utc))
        super().__init__(**kwargs)

    def to_dict(self, include_measurements: bool = True) -> Dict[str, Any]:
        """Convert benchmark run to dictionary representation."""
        failed_count = (
            sum(1 for m in self.measurements if m.status == "failed")
            if self.measurements
            else 0
        )
        total_m = len(self.measurements) if self.measurements else 0

        if total_m > 0 and failed_count == total_m:
            eval_status = "failed"
        elif total_m > 0 and failed_count > 0:
            eval_status = "partial"
        else:
            eval_status = self.status

        warnings = (
            self.environment_info.get("warnings", [])
            if isinstance(self.environment_info, dict)
            else []
        )

        data: Dict[str, Any] = {
            "id": self.id,
            "run_name": self.run_name,
            "description": self.description,
            "device": self.device,
            "status": self.status,
            "evaluation_status": eval_status,
            "failed_measurements_count": failed_count,
            "error_message": self.error_message,
            "warnings": warnings,
            "environment_info": self.environment_info,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        if include_measurements and self.measurements:
            data["measurements"] = [m.to_dict() for m in self.measurements]
        elif include_measurements:
            data["measurements"] = []
        return data

    def __repr__(self) -> str:
        return f"<BenchmarkRun(id={self.id}, name='{self.run_name}', status='{self.status}')>"


class ModelMeasurement(Base):
    """Represents a discrete benchmark measurement for a model under a specific runtime.

    Captures accuracy, latency, memory footprint, model size, and execution settings.
    Unavailable metrics are explicitly stored as NULL (never invented zeros).
    """

    __tablename__ = "model_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("benchmark_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Model Identification & Architecture
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    architecture: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Runtime & Execution Configuration
    runtime: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    device: Mapped[str] = mapped_column(String(32), default="cpu", nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    warmup_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Dataset & Classification Accuracy (NULL if not evaluated)
    evaluated_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    top1_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top5_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Inference Latency & Throughput (NULL if not evaluated)
    latency_mean_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_p95_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    throughput_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Memory Footprint Metrics (NULL if not evaluated)
    peak_ram_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model Storage & Parameter Metrics (NULL if not evaluated)
    file_size_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    parameter_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", back_populates="measurements")

    __table_args__ = (
        Index("ix_measurement_lookup", "model_id", "runtime", "device"),
        Index("ix_measurement_run_model", "run_id", "model_id"),
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("status", "completed")
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("created_at", datetime.now(timezone.utc))
        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurement record to serializable dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "runtime": self.runtime,
            "device": self.device,
            "batch_size": self.batch_size,
            "warmup_runs": self.warmup_runs,
            "num_runs": self.num_runs,
            "status": self.status,
            "error_message": self.error_message,
            "provenance": self.run.provenance if self.run else getattr(self, "provenance", None),
            "accuracy": {
                "evaluated_samples": self.evaluated_samples,
                "top1_accuracy": self.top1_accuracy,
                "top5_accuracy": self.top5_accuracy,
            } if self.top1_accuracy is not None or self.evaluated_samples is not None else None,
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "p95_ms": self.latency_p95_ms,
                "throughput_fps": self.throughput_fps,
            } if self.latency_mean_ms is not None else None,
            "memory": {
                "peak_ram_mb": self.peak_ram_mb,
            } if self.peak_ram_mb is not None else None,
            "model_size": {
                "file_size_mb": self.file_size_mb,
                "parameter_count": self.parameter_count,
            } if self.file_size_mb is not None or self.parameter_count is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<ModelMeasurement(id={self.id}, run_id={self.run_id}, "
            f"model_id='{self.model_id}', runtime='{self.runtime}')>"
        )
