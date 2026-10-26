"""
Benchmark Results Module for Edge Deployment.

Defines structured result classes for capturing and analyzing benchmark outputs:
accuracy, latency, memory footprint, model size, and environment metadata.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime


@dataclass
class AccuracyMetrics:
    """Classification accuracy evaluation metrics."""

    top1: Optional[float] = None
    top5: Optional[float] = None
    num_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation."""
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Latency and throughput performance profiling metrics."""

    mean_ms: Optional[float] = None
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    throughput_fps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation."""
        return asdict(self)


@dataclass
class MemoryMetrics:
    """Memory consumption and footprint profiling metrics.

    Attributes:
        peak_ram_mb: Maximum process resident memory (RSS) observed during benchmarking in MiB.
        ram_delta_mb: Net process memory increase from baseline to peak (peak_ram_mb - baseline_ram_mb) in MiB.
        baseline_ram_mb: Process resident memory (RSS) prior to model loading and inference in MiB.
        post_load_ram_mb: Process resident memory (RSS) after model loading / initialization in MiB.
        inference_ram_delta_mb: Memory increase during the inference phase (peak_ram_mb - post_load_ram_mb) in MiB.
        device_memory_mb: Accelerator-specific memory allocated (e.g. PyTorch MPS current allocated memory in MiB), or None for CPU.
        notes: Methodological notes clarifying scope, unit, and hardware-specific limitations.
    """

    peak_ram_mb: Optional[float] = None
    ram_delta_mb: Optional[float] = None
    baseline_ram_mb: Optional[float] = None
    post_load_ram_mb: Optional[float] = None
    inference_ram_delta_mb: Optional[float] = None
    device_memory_mb: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation."""
        return asdict(self)


@dataclass
class ModelSizeMetrics:
    """Storage and parameter volume metrics for a model artifact."""

    file_size_bytes: Optional[int] = None
    file_size_mb: Optional[float] = None
    param_count: Optional[int] = None
    artifact_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Compute file_size_mb automatically if file_size_bytes is supplied."""
        if self.file_size_bytes is not None and self.file_size_mb is None:
            self.file_size_mb = round(self.file_size_bytes / (1024.0 * 1024.0), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result container for a single model and runtime configuration.

    Attributes:
        model_id: Logical identifier of the benchmarked model.
        runtime: Execution runtime used (PyTorch, ONNX, TFLite).
        config: Full benchmark configuration specification.
        accuracy: Classification accuracy metrics (None if not measured).
        latency: Latency and throughput metrics (None if not measured).
        memory_footprint: Memory consumption metrics (None if not measured).
        model_size: Storage and parameter metrics (None if not measured).
        environment: Hardware and library environment metadata.
        timestamp: ISO 8601 timestamp string of the benchmark run.
        success: Boolean flag indicating if benchmark executed without unhandled errors.
        error_message: Optional error message if the benchmark failed.
    """

    model_id: str
    runtime: BenchmarkRuntime
    config: BenchmarkConfig
    accuracy: Optional[Union[AccuracyMetrics, float]] = None
    latency: Optional[Union[LatencyMetrics, float]] = None
    memory_footprint: Optional[Union[MemoryMetrics, float]] = None
    model_size: Optional[Union[ModelSizeMetrics, float]] = None
    environment: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not explicitly provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(self.runtime, str):
            self.runtime = BenchmarkRuntime.from_string(self.runtime)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to a serializable dictionary."""
        def _metric_to_dict(metric: Any) -> Any:
            if hasattr(metric, "to_dict"):
                return metric.to_dict()
            return metric

        return {
            "model_id": self.model_id,
            "runtime": self.runtime.value if isinstance(self.runtime, BenchmarkRuntime) else str(self.runtime),
            "config": self.config.to_dict(),
            "accuracy": _metric_to_dict(self.accuracy),
            "latency": _metric_to_dict(self.latency),
            "memory_footprint": _metric_to_dict(self.memory_footprint),
            "model_size": _metric_to_dict(self.model_size),
            "environment": self.environment,
            "timestamp": self.timestamp,
            "success": self.success,
            "error_message": self.error_message,
        }

    def summary(self) -> str:
        """Produce a human-readable text summary of the benchmark result."""
        status = "PASSED" if self.success else f"FAILED ({self.error_message})"
        lines = [
            f"Benchmark Result: {self.model_id} [{self.runtime.value}] - Status: {status}",
            f"  Timestamp: {self.timestamp}",
            f"  Batch Size: {self.config.batch_size}, Num Runs: {self.config.num_runs}, Device: {self.config.device}",
        ]
        if self.accuracy is not None:
            lines.append(f"  Accuracy: {self.accuracy}")
        if self.latency is not None:
            lines.append(f"  Latency: {self.latency}")
        if self.memory_footprint is not None:
            lines.append(f"  Memory Footprint: {self.memory_footprint}")
        if self.model_size is not None:
            lines.append(f"  Model Size: {self.model_size}")
        return "\n".join(lines)
