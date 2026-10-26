"""
Pydantic v2 Schemas for Automated Image Model Benchmarking API.

Defines validation rules, request payloads, and serialized response structures
for health checks, model catalogs, benchmark executions, measurement queries,
and edge deployment recommendations.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from benchmarking.config import BenchmarkRuntime, SUPPORTED_RUNTIMES
from models.model_registry import is_model_supported, list_supported_models


# ==============================================================================
# Health Check Schemas
# ==============================================================================

class DatabaseHealthInfo(BaseModel):
    """Database connectivity status without exposing credentials."""

    connected: bool
    dialect: str
    host: str
    port: int
    database: str


class EdgeImpulseStatusResponse(BaseModel):
    """Edge Impulse configuration status without exposing credentials."""

    configured: bool
    status: str
    has_api_key: bool
    has_project_id: bool
    project_id: Optional[str] = None


class EdgeImpulseTargetDetail(BaseModel):
    """Supported Edge Impulse profiling target hardware device."""

    target_id: str
    display_name: str
    supported_formats: List[str]
    is_supported: bool
    notes: Optional[str] = None


class EdgeImpulseTargetsResponse(BaseModel):
    """Catalog of verified Edge Impulse profiling hardware targets."""

    targets: List[EdgeImpulseTargetDetail]
    total_targets: int


class EdgeImpulseProfileRequest(BaseModel):
    """Payload to initiate or reuse an Edge Impulse profiling evaluation."""

    model_id: str = Field(
        ...,
        description="Registered model identifier to profile (e.g. mobilenet_v3_small).",
    )
    target_device: str = Field(
        default="raspberry-pi-4",
        description="Target edge device identifier (e.g. raspberry-pi-4).",
    )
    runtime: str = Field(
        default="tflite",
        description="Model format/runtime. Edge Impulse profiling is currently TFLite only.",
    )
    batch_size: int = Field(default=1, ge=1, le=64)
    reuse_completed: bool = Field(
        default=True,
        description="Whether to reuse completed matching results without duplicate API calls.",
    )
    reuse_saved_accuracy: bool = Field(
        default=True,
        description="Whether to verify and attach matching pre-computed classification accuracy.",
    )


class HealthResponse(BaseModel):
    """API health status, database liveness, and integration status response."""

    status: str
    api: str
    version: str
    database: DatabaseHealthInfo
    edge_impulse: EdgeImpulseStatusResponse
    timestamp: str


# ==============================================================================
# Model Catalog & Artifact Schemas
# ==============================================================================

class ArtifactDetail(BaseModel):
    """Metadata regarding a serialized model artifact file on disk."""

    format: str
    exists: bool
    path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_size_mb: Optional[float] = None
    has_precomputed_accuracy: Optional[bool] = None


class ModelInfo(BaseModel):
    """Metadata describing a registered computer vision model."""

    model_id: str
    display_name: str
    architecture_name: str
    torchvision_model_name: str
    weights_identifier: str
    num_classes: int
    expected_input_size: List[int]
    supported_runtimes: List[str]
    artifacts: Dict[str, ArtifactDetail]


class ModelsResponse(BaseModel):
    """Catalog of registered models and on-disk artifact statuses."""

    models: List[ModelInfo]
    total_models: int


# ==============================================================================
# Benchmark Measurement & Run Schemas
# ==============================================================================

class AccuracyMetricsResponse(BaseModel):
    """Classification accuracy evaluation metrics."""

    dataset_path: Optional[str] = None
    total_samples: Optional[int] = None
    evaluated_samples: Optional[int] = None
    skipped_images: Optional[int] = None
    top1_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    evaluation_time_seconds: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    accuracy_source: Optional[str] = None


class LatencyMetricsResponse(BaseModel):
    """Latency distribution and throughput profiling metrics."""

    mean_ms: Optional[float] = None
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    throughput_fps: Optional[float] = None


class MemoryMetricsResponse(BaseModel):
    """Process memory (RSS) footprint profiling metrics."""

    peak_ram_mb: Optional[float] = None
    ram_delta_mb: Optional[float] = None
    baseline_ram_mb: Optional[float] = None
    post_load_ram_mb: Optional[float] = None
    inference_ram_delta_mb: Optional[float] = None
    device_memory_mb: Optional[float] = None
    notes: Optional[str] = None


class ModelSizeMetricsResponse(BaseModel):
    """Storage footprint and parameter count metrics."""

    file_size_bytes: Optional[int] = None
    file_size_mb: Optional[float] = None
    parameter_count: Optional[int] = None
    artifact_path: Optional[str] = None
    artifact_version: Optional[str] = None


class MeasurementResponse(BaseModel):
    """A single discrete model benchmark measurement record."""

    id: int
    run_id: int
    model_id: str
    architecture: Optional[str] = None
    weights: Optional[str] = None
    runtime: str
    device: str
    backend_provider: Optional[str] = None
    batch_size: int
    warmup_runs: Optional[int] = None
    num_runs: Optional[int] = None
    input_shape: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    accuracy: Optional[AccuracyMetricsResponse] = None
    latency: Optional[LatencyMetricsResponse] = None
    memory: Optional[MemoryMetricsResponse] = None
    model_size: Optional[ModelSizeMetricsResponse] = None
    provenance: Optional[str] = None
    created_at: Optional[str] = None


class BenchmarkRunResponse(BaseModel):
    """A benchmark run session container and its associated measurements."""

    id: int
    run_name: Optional[str] = None
    description: Optional[str] = None
    device: str
    status: str
    provenance: str
    evaluation_status: Optional[str] = None
    failed_measurements_count: Optional[int] = None
    error_message: Optional[str] = None
    artifact_hash: Optional[str] = None
    environment_info: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal notices or warnings regarding this run session.",
    )
    methodology_disclaimer: str = (
        "Benchmarking was executed on host CPU environment. CPU measurements serve as "
        "comparative baselines and do not represent physical edge device (NPU/DSP/embedded) execution."
    )
    measurements: List[MeasurementResponse] = []


class ResultsPaginatedResponse(BaseModel):
    """Paginated collection of benchmark measurements."""

    total: int
    limit: int
    offset: int
    items: List[MeasurementResponse]


class BenchmarkProgressResponse(BaseModel):
    """Real-time intermediate benchmark evaluation progress."""

    run_id: int
    status: str
    images_completed: int
    total_images: int
    images_remaining: int
    percentage: float
    elapsed_seconds: Optional[float] = None
    estimated_remaining_seconds: Optional[float] = None
    processing_speed_fps: Optional[float] = None
    error_message: Optional[str] = None
    stage: Optional[str] = None
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal notices or warnings regarding this run.",
    )


# ==============================================================================
# Benchmark Comparison Execution Request
# ==============================================================================

class BenchmarkRunRequest(BaseModel):
    """Payload to initiate a multi-model benchmark comparison session."""

    model_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of registered model IDs to benchmark.",
    )
    runtimes: List[str] = Field(
        ...,
        min_length=1,
        description="List of runtimes to benchmark (pytorch, onnx, tflite).",
    )
    metrics: List[str] = Field(
        default=["latency", "memory", "model_size", "accuracy"],
        description="Metrics to evaluate: 'latency', 'memory', 'model_size', 'accuracy'.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="User-friendly title for this comparison run.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional notes, tags, or hardware description.",
    )
    batch_size: int = Field(default=1, ge=1, le=64)
    num_runs: int = Field(default=100, ge=1, le=1000)
    warmup_runs: int = Field(default=10, ge=0, le=100)
    device: str = Field(default="cpu")
    reuse_saved_accuracy: bool = Field(
        default=True,
        description="Whether to reuse pre-computed 50,000-image accuracy results.",
    )
    run_accuracy_eval: bool = Field(
        default=False,
        description="Whether to execute live sample accuracy benchmarking.",
    )
    max_accuracy_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description="Sample limit for live accuracy evaluation.",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Optional custom path to validation manifest CSV/JSON file.",
    )
    async_run: bool = Field(
        default=False,
        description="Whether to run the benchmark asynchronously in the background.",
    )

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, v: List[str]) -> List[str]:
        supported = list_supported_models()
        for mid in v:
            if not is_model_supported(mid):
                raise ValueError(
                    f"Unsupported model ID '{mid}'. Supported models: {', '.join(supported)}"
                )
        return list(dict.fromkeys(v))

    @field_validator("runtimes")
    @classmethod
    def validate_runtimes(cls, v: List[str]) -> List[str]:
        valid_runtimes = set(SUPPORTED_RUNTIMES)
        normalized = []
        for r in v:
            r_lower = r.strip().lower()
            if r_lower not in valid_runtimes:
                raise ValueError(
                    f"Unsupported runtime '{r}'. Supported runtimes: {', '.join(SUPPORTED_RUNTIMES)}"
                )
            normalized.append(r_lower)
        return list(dict.fromkeys(normalized))

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        valid_metrics = {"latency", "memory", "model_size", "accuracy"}
        normalized = []
        for m in v:
            m_lower = m.strip().lower()
            if m_lower not in valid_metrics:
                raise ValueError(
                    f"Unsupported metric '{m}'. Valid metrics: {', '.join(sorted(valid_metrics))}"
                )
            normalized.append(m_lower)
        return list(dict.fromkeys(normalized))


# ==============================================================================
# Recommendation Engine Schemas
# ==============================================================================

class CriteriaWeights(BaseModel):
    """User-defined importance weights for multi-criteria deployment decision."""

    accuracy: float = Field(default=0.35, ge=0.0, le=1.0)
    latency: float = Field(default=0.30, ge=0.0, le=1.0)
    memory: float = Field(default=0.20, ge=0.0, le=1.0)
    model_size: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_nonzero_sum(self) -> "CriteriaWeights":
        if self.accuracy == 0.0 and self.latency == 0.0 and self.memory == 0.0 and self.model_size == 0.0:
            raise ValueError("Set at least one criterion weight above zero.")
        return self


class ConstraintFilter(BaseModel):
    """Optional hard edge constraints to filter out ineligible models."""

    max_latency_ms: Optional[float] = Field(default=None, gt=0)
    max_memory_mb: Optional[float] = Field(default=None, gt=0)
    max_model_size_mb: Optional[float] = Field(default=None, gt=0)
    min_accuracy: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class CandidateScoreDetail(BaseModel):
    """Detailed MCDA score evaluation for a specific model/runtime candidate."""

    model_id: str
    display_name: str
    runtime: str
    device: str
    eligible: bool
    disqualification_reasons: List[str] = []
    satisfied_constraints: List[str] = []
    composite_score: float
    rank: int
    subscores: Dict[str, Optional[float]]
    raw_metrics: Dict[str, Optional[float]]
    missing_metrics: List[str] = []
    data_complete: bool
    notes: List[str] = []
    rationale: Optional[str] = None
    trade_offs: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Payload to request model deployment recommendations."""

    model_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of models to compare. Defaults to all 4 registered models.",
    )
    runtimes: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of runtimes to evaluate. Defaults to all supported.",
    )
    run_id: Optional[int] = Field(
        default=None,
        description="Optional benchmark run ID to pull measurements from.",
    )
    priority_preset: Optional[str] = Field(
        default=None,
        description="Weight preset: 'balanced', 'speed_optimized', 'edge_constrained', 'accuracy_first'.",
    )
    weights: Optional[CriteriaWeights] = None
    constraints: Optional[ConstraintFilter] = None

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        supported = list_supported_models()
        for mid in v:
            if not is_model_supported(mid):
                raise ValueError(
                    f"Unsupported model ID '{mid}'. Supported models: {', '.join(supported)}"
                )
        return list(dict.fromkeys(v))

    @field_validator("runtimes")
    @classmethod
    def validate_runtimes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        valid_runtimes = set(SUPPORTED_RUNTIMES)
        normalized = []
        for r in v:
            r_lower = r.strip().lower()
            if r_lower not in valid_runtimes:
                raise ValueError(
                    f"Unsupported runtime '{r}'. Supported runtimes: {', '.join(SUPPORTED_RUNTIMES)}"
                )
            normalized.append(r_lower)
        return list(dict.fromkeys(normalized))

    @field_validator("priority_preset")
    @classmethod
    def validate_priority_preset(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid_presets = {"balanced", "speed_optimized", "edge_constrained", "accuracy_first"}
        normalized = v.strip().lower()
        if normalized not in valid_presets:
            raise ValueError(
                f"Invalid preset '{v}'. Supported presets: {', '.join(sorted(valid_presets))}"
            )
        return normalized


class RecommendationResponse(BaseModel):
    """Ranked recommendations with score breakdown and trade-off rationale."""

    recommended_model: Optional[str] = None
    recommended_display_name: Optional[str] = None
    recommended_runtime: Optional[str] = None
    composite_score: Optional[float] = None
    explanation: str
    trade_off_analysis: str
    criteria_weights_applied: Dict[str, float]
    constraints_applied: Dict[str, Any]
    rankings: List[CandidateScoreDetail]
    top_recommendations: List[CandidateScoreDetail] = Field(default_factory=list)
    methodology_disclaimer: str = (
        "Benchmarking was executed on host CPU environment. CPU measurements serve as "
        "comparative baselines and do not represent physical edge device (NPU/DSP/embedded) execution."
    )
