"""
Edge Impulse Profiling Service Module.

Provides single-job model profiling submission and asynchronous result retrieval
against Edge Impulse Studio API for target hardware estimates (latency, RAM, ROM/Flash)
using pure Python standard library HTTP facilities.

Strict Security:
Never logs, prints, serializes, or exposes the API key or secret tokens.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.services.accuracy_service import find_saved_accuracy_file
from benchmarking.model_size import get_model_parameter_count
from database.models import BenchmarkRun, ModelMeasurement
from models.model_registry import (
    get_model_metadata,
    is_model_supported,
    list_supported_models,
)

logger = logging.getLogger("backend.services.edge_impulse")

EDGE_IMPULSE_API_BASE = "https://studio.edgeimpulse.com/v1/api"

# Supported target device catalog
SUPPORTED_EDGE_IMPULSE_TARGETS: Dict[str, Dict[str, Any]] = {
    "raspberry-pi-4": {
        "target_id": "raspberry-pi-4",
        "display_name": "Raspberry Pi 4",
        "supported_formats": ["tflite"],
        "is_supported": True,
        "notes": "Verified analytical profiling for TFLite models. Evaluates Float32/Int8 performance estimates.",
    }
}

# Verified completed profiling jobs catalog: (model_id, runtime, target_device, artifact_sha256) -> job_id
VERIFIED_COMPLETED_JOBS: Dict[Tuple[str, str, str, str], int] = {
    (
        "mobilenet_v3_small",
        "tflite",
        "raspberry-pi-4",
        "aad685b124b2e56d22687b8486c93ecb5ebbbde81354786599acde8fb0fa043d",
    ): 55253939,
}

# Verified offline result fallback if Edge Impulse API is unreachable
VERIFIED_JOB_RESULTS: Dict[int, Dict[str, Any]] = {
    55253939: {
        "timePerInferenceMs": 21477,
        "memory": {
            "tflite": {
                "arenaSize": 0,
                "ram": 0,
                "rom": 5135976,
            }
        },
        "tfliteFileSizeBytes": 5135976,
        "isSupportedOnMcu": False,
        "mcuSupportError": "Unsupported ops: RELU_0_TO_1.",
        "hasPerformance": True,
        "device": "raspberry-pi-4",
        "status": "completed",
    }
}


class EdgeImpulseProfilingError(RuntimeError):
    """Raised when an Edge Impulse profiling request fails or returns an error."""
    pass


@dataclass
class EdgeImpulseProfileResult:
    """Structured representation of Edge Impulse profiling estimation metrics."""

    job_id: int
    target_device: str
    status: str  # "completed", "failed", "timeout"
    time_per_inference_ms: Optional[float] = None
    tflite_ram_bytes: Optional[int] = None
    tflite_rom_bytes: Optional[int] = None
    tflite_arena_bytes: Optional[int] = None
    eon_ram_bytes: Optional[int] = None
    eon_rom_bytes: Optional[int] = None
    eon_arena_bytes: Optional[int] = None
    is_supported_on_mcu: Optional[bool] = None
    mcu_support_error: Optional[str] = None
    profiling_error: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None
    provenance: str = "Edge Impulse Studio Estimation"

    @property
    def time_per_inference_s(self) -> Optional[float]:
        """Convert inference latency from milliseconds to seconds."""
        if self.time_per_inference_ms is not None:
            return round(self.time_per_inference_ms / 1000.0, 3)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation preserving NULL semantics."""
        return {
            "job_id": self.job_id,
            "target_device": self.target_device,
            "status": self.status,
            "time_per_inference_ms": self.time_per_inference_ms,
            "time_per_inference_s": self.time_per_inference_s,
            "tflite_ram_bytes": self.tflite_ram_bytes,
            "tflite_rom_bytes": self.tflite_rom_bytes,
            "tflite_arena_bytes": self.tflite_arena_bytes,
            "eon_ram_bytes": self.eon_ram_bytes,
            "eon_rom_bytes": self.eon_rom_bytes,
            "eon_arena_bytes": self.eon_arena_bytes,
            "is_supported_on_mcu": self.is_supported_on_mcu,
            "mcu_support_error": self.mcu_support_error,
            "profiling_error": self.profiling_error,
            "provenance": self.provenance,
        }



def submit_tflite_profiling_job(
    tflite_path: Path,
    device: str = "raspberry-pi-4",
    submission_timeout: float = 180.0,
) -> int:
    """Submit exactly one TFLite model profiling job to Edge Impulse.

    Does NOT retry job creation automatically on failure or timeout.
    """
    settings = get_settings()
    api_key = settings.edge_impulse_api_key
    project_id = settings.edge_impulse_project_id

    if not api_key:
        raise EdgeImpulseProfilingError("EDGE_IMPULSE_API_KEY is not configured.")
    if not project_id:
        raise EdgeImpulseProfilingError("EDGE_IMPULSE_PROJECT_ID is not configured.")

    if not tflite_path.exists() or not tflite_path.is_file():
        raise FileNotFoundError(f"TFLite model file not found at: {tflite_path}")

    model_bytes = tflite_path.read_bytes()
    b64_str = base64.b64encode(model_bytes).decode("ascii")

    url = f"{EDGE_IMPULSE_API_BASE}/{project_id}/jobs/profile-tflite"
    payload = json.dumps({"tfliteFileBase64": b64_str, "device": device}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=submission_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            err_msg = err_body.get("error", str(exc))
        except Exception:
            err_msg = f"HTTP {exc.code} {exc.reason}"
        raise EdgeImpulseProfilingError(f"Profiling job submission failed: {err_msg}") from exc
    except urllib.error.URLError as exc:
        raise EdgeImpulseProfilingError(f"Network error submitting profiling job: {exc.reason}") from exc
    except Exception as exc:
        raise EdgeImpulseProfilingError(f"Network error submitting profiling job: {type(exc).__name__}") from exc


    if not data.get("success"):
        raise EdgeImpulseProfilingError(f"API rejected profiling job: {data.get('error', 'Unknown error')}")

    job_id = data.get("id")
    if not job_id:
        raise EdgeImpulseProfilingError("API reported success but returned no job ID.")

    logger.info("Successfully created Edge Impulse profiling job ID %s for target %s", job_id, device)
    return int(job_id)


def poll_profiling_job_result(
    job_id: int,
    device: str = "raspberry-pi-4",
    timeout_seconds: float = 180.0,
    poll_interval: float = 3.0,
) -> EdgeImpulseProfileResult:
    """Poll the result endpoint for an existing profiling job until completed or timed out.

    Does NOT create any new jobs under any circumstance.
    """
    settings = get_settings()
    api_key = settings.edge_impulse_api_key
    project_id = settings.edge_impulse_project_id

    result_url = f"{EDGE_IMPULSE_API_BASE}/{project_id}/jobs/profile-tflite/{job_id}/result"
    status_url = f"{EDGE_IMPULSE_API_BASE}/{project_id}/jobs/{job_id}/status"

    start_time = time.time()

    while (time.time() - start_time) < timeout_seconds:
        # Check result endpoint
        req = urllib.request.Request(
            result_url,
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("success") and ("result" in data or "timePerInferenceMs" in data or "memory" in data):
                    res_payload = data["result"] if "result" in data else data
                    mem = res_payload.get("memory", {})
                    tflite_mem = mem.get("tflite", {})
                    eon_mem = mem.get("eonRamOptimized", {}) or mem.get("eon", {})

                    # Preserve NULL semantics for unavailable metrics.
                    # For Linux/SBC targets, 0 indicates memory is dynamically managed by the OS
                    # and not measured as a static MCU tensor arena. Do not substitute zero.
                    raw_tflite_ram = tflite_mem.get("ram")
                    raw_tflite_arena = tflite_mem.get("arenaSize")
                    raw_tflite_rom = tflite_mem.get("rom") or res_payload.get("tfliteFileSizeBytes")

                    tflite_ram = int(raw_tflite_ram) if (raw_tflite_ram is not None and raw_tflite_ram > 0) else None
                    tflite_arena = int(raw_tflite_arena) if (raw_tflite_arena is not None and raw_tflite_arena > 0) else None
                    tflite_rom = int(raw_tflite_rom) if (raw_tflite_rom is not None and raw_tflite_rom > 0) else None

                    raw_eon_ram = eon_mem.get("ram")
                    raw_eon_arena = eon_mem.get("arenaSize")
                    raw_eon_rom = eon_mem.get("rom")

                    eon_ram = int(raw_eon_ram) if (raw_eon_ram is not None and raw_eon_ram > 0) else None
                    eon_arena = int(raw_eon_arena) if (raw_eon_arena is not None and raw_eon_arena > 0) else None
                    eon_rom = int(raw_eon_rom) if (raw_eon_rom is not None and raw_eon_rom > 0) else None

                    logger.info("Profiling job ID %s completed successfully.", job_id)
                    return EdgeImpulseProfileResult(
                        job_id=job_id,
                        target_device=device,
                        status="completed",
                        time_per_inference_ms=float(res_payload["timePerInferenceMs"]) if res_payload.get("timePerInferenceMs") is not None else None,
                        tflite_ram_bytes=tflite_ram,
                        tflite_rom_bytes=tflite_rom,
                        tflite_arena_bytes=tflite_arena,
                        eon_ram_bytes=eon_ram,
                        eon_rom_bytes=eon_rom,
                        eon_arena_bytes=eon_arena,
                        is_supported_on_mcu=res_payload.get("isSupportedOnMcu"),
                        mcu_support_error=res_payload.get("mcuSupportError"),
                        profiling_error=res_payload.get("profilingError"),
                        raw_result=res_payload,
                    )

        except urllib.error.HTTPError as exc:
            # HTTP 404/400 while job is in progress is expected before results are compiled
            if exc.code not in (404, 400):
                logger.debug("HTTP %d received while polling job %d result.", exc.code, job_id)
        except Exception as exc:
            logger.debug("Transient error while polling job %d result: %s", job_id, type(exc).__name__)

        # Check job lifecycle status
        try:
            status_req = urllib.request.Request(
                status_url,
                headers={"x-api-key": api_key, "Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(status_req, timeout=10) as s_resp:
                s_data = json.loads(s_resp.read().decode("utf-8"))
                job_info = s_data.get("job", {})
                if job_info.get("finished"):
                    if not job_info.get("finishedSuccessful", True):
                        err = job_info.get("error", "Job marked finished without success.")
                        logger.warning("Job %d finished with failure: %s", job_id, err)
                        return EdgeImpulseProfileResult(
                            job_id=job_id,
                            target_device=device,
                            status="failed",
                            profiling_error=err,
                        )
        except Exception:
            pass

        time.sleep(poll_interval)

    logger.warning("Profiling job %d timed out after %.1f seconds.", job_id, timeout_seconds)
    return EdgeImpulseProfileResult(
        job_id=job_id,
        target_device=device,
        status="timeout",
        profiling_error=f"Timed out after {timeout_seconds}s waiting for Edge Impulse profiling job {job_id}.",
    )


def run_single_mobilenet_v3_small_rpi4_validation(
    timeout_seconds: float = 180.0,
) -> EdgeImpulseProfileResult:
    """Execute exactly one real profiling job for MobileNetV3-Small TFLite on Raspberry Pi 4.

    Controlled single validation. Does not retry job creation.
    """
    artifact_path = Path("results/conversions/tflite/mobilenet_v3_small.tflite")
    job_id = submit_tflite_profiling_job(artifact_path, device="raspberry-pi-4")
    return poll_profiling_job_result(job_id=job_id, device="raspberry-pi-4", timeout_seconds=timeout_seconds)


def compute_tflite_sha256(artifact_path: Path) -> str:
    """Compute hex-encoded SHA-256 digest for a local TFLite file."""
    if not artifact_path.is_file():
        raise FileNotFoundError(f"TFLite artifact not found at: {artifact_path}")
    hasher = hashlib.sha256()
    with open(artifact_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_edge_impulse_compatibility(
    model_id: str,
    runtime: str,
    target_device: str,
) -> Path:
    """Verify format, model, and device compatibility before profiling.

    Raises:
        EdgeImpulseProfilingError: If runtime is not TFLite, model is unsupported,
                                   or target device is not in supported catalog.
        FileNotFoundError: If model's TFLite artifact is missing on disk.
    """
    runtime_clean = runtime.strip().lower()
    if runtime_clean != "tflite":
        raise EdgeImpulseProfilingError(
            f"Edge Impulse profiling is supported for TFLite models only. "
            f"Runtime '{runtime}' cannot be profiled via Edge Impulse."
        )

    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise EdgeImpulseProfilingError(
            f"Model '{model_id}' is not supported. Supported models: {supported}"
        )

    if target_device not in SUPPORTED_EDGE_IMPULSE_TARGETS:
        supported_targets = ", ".join(SUPPORTED_EDGE_IMPULSE_TARGETS.keys())
        raise EdgeImpulseProfilingError(
            f"Target device '{target_device}' is not a recognized Edge Impulse profiling target. "
            f"Supported targets: {supported_targets}"
        )

    artifact_path = Path(f"results/conversions/tflite/{model_id}.tflite")
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"TFLite model artifact for '{model_id}' not found at '{artifact_path}'."
        )

    return artifact_path


def get_verified_saved_accuracy(
    model_id: str,
    runtime: str = "tflite",
    expected_batch_size: int = 1,
) -> Optional[Dict[str, Any]]:
    """Load and strictly verify that a saved accuracy artifact matches the evaluation configuration."""
    path = find_saved_accuracy_file(model_id, runtime)
    if not path or not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_id") != model_id:
            return None
        if data.get("runtime") != runtime:
            return None
        if data.get("batch_size") != expected_batch_size:
            return None
        if data.get("total_samples") != data.get("evaluated_samples"):
            return None
        if data.get("top1_accuracy") is None:
            return None

        return {
            "top1_accuracy": data.get("top1_accuracy"),
            "top5_accuracy": data.get("top5_accuracy"),
            "evaluated_samples": data.get("evaluated_samples"),
            "total_samples": data.get("total_samples"),
            "accuracy_source": "saved_result",
            "artifact_file": path.name,
        }
    except Exception:
        return None


def get_or_create_edge_impulse_benchmark(
    session: Session,
    model_id: str,
    target_device: str = "raspberry-pi-4",
    runtime: str = "tflite",
    batch_size: int = 1,
    reuse_completed: bool = True,
    reuse_saved_accuracy: bool = True,
    allow_external_submission: bool = False,
) -> BenchmarkRun:
    """Retrieve existing completed Edge Impulse benchmark or persist verified completed job.

    Strict Constraints:
    - Never submits an external profiling job when allow_external_submission is False.
    - Preserves NULL/N/A for unmeasured RAM and arena values (never zero).
    - Labels latency explicitly as analytical estimate and throughput as derived.
    - Isolates MCU compiler warnings in environment_info without failing the SBC run.
    """
    artifact_path = validate_edge_impulse_compatibility(model_id, runtime, target_device)
    artifact_sha = compute_tflite_sha256(artifact_path)
    metadata = get_model_metadata(model_id)

    # 1. Tier 1: Check active MySQL session for an existing matching completed benchmark
    if reuse_completed:
        stmt = (
            select(BenchmarkRun)
            .join(ModelMeasurement)
            .where(
                BenchmarkRun.provenance == "edge_impulse_estimation",
                BenchmarkRun.device == target_device,
                BenchmarkRun.status == "completed",
                ModelMeasurement.model_id == model_id,
                ModelMeasurement.runtime == runtime,
                ModelMeasurement.device == target_device,
                ModelMeasurement.status == "completed",
            )
        )
        existing_runs = session.execute(stmt).scalars().all()
        for r in existing_runs:
            env = r.environment_info or {}
            if env.get("artifact_sha256") == artifact_sha and env.get("batch_size", 1) == batch_size:
                logger.info(
                    "Reusing existing completed Edge Impulse benchmark run ID %s from MySQL for %s on %s.",
                    r.id,
                    model_id,
                    target_device,
                )
                return r

    # 2. Tier 2: Check verified completed jobs cache (e.g. Job 55253939)
    job_key = (model_id, runtime, target_device, artifact_sha)
    job_id = VERIFIED_COMPLETED_JOBS.get(job_key)

    result: Optional[EdgeImpulseProfileResult] = None
    if job_id is not None:
        # Attempt to retrieve verified result from API if credentials configured, otherwise fallback to cache
        settings = get_settings()
        if settings.edge_impulse_api_key and settings.edge_impulse_project_id:
            try:
                result = poll_profiling_job_result(
                    job_id=job_id,
                    device=target_device,
                    timeout_seconds=5.0,
                    poll_interval=1.0,
                )
            except Exception as exc:
                logger.debug("Could not poll API for job %s, using verified cached result: %s", job_id, exc)

        if result is None or result.status != "completed":
            cached_raw = VERIFIED_JOB_RESULTS.get(job_id)
            if cached_raw:
                mem = cached_raw.get("memory", {}).get("tflite", {})
                raw_ram = mem.get("ram")
                raw_arena = mem.get("arenaSize")
                raw_rom = mem.get("rom") or cached_raw.get("tfliteFileSizeBytes")
                result = EdgeImpulseProfileResult(
                    job_id=job_id,
                    target_device=target_device,
                    status="completed",
                    time_per_inference_ms=float(cached_raw["timePerInferenceMs"]),
                    tflite_ram_bytes=int(raw_ram) if raw_ram and raw_ram > 0 else None,
                    tflite_rom_bytes=int(raw_rom) if raw_rom and raw_rom > 0 else None,
                    tflite_arena_bytes=int(raw_arena) if raw_arena and raw_arena > 0 else None,
                    is_supported_on_mcu=cached_raw.get("isSupportedOnMcu"),
                    mcu_support_error=cached_raw.get("mcuSupportError"),
                    raw_result=cached_raw,
                )

    # 3. Tier 3: External submission guard
    if result is None or result.status != "completed":
        if not allow_external_submission:
            raise EdgeImpulseProfilingError(
                f"No verified completed Edge Impulse profiling result available for "
                f"model '{model_id}' on '{target_device}' with SHA-256 '{artifact_sha[:12]}...'. "
                f"External job submission is currently disabled to prevent duplicate jobs."
            )
        # External submission is strictly guarded
        job_id = submit_tflite_profiling_job(artifact_path, device=target_device)
        result = poll_profiling_job_result(job_id=job_id, device=target_device)

    if result.status != "completed" or result.time_per_inference_ms is None:
        raise EdgeImpulseProfilingError(
            f"Edge Impulse profiling result for job {result.job_id} failed or returned no latency metric."
        )

    # 4. Accuracy extraction (verified matching artifact only)
    accuracy_data: Optional[Dict[str, Any]] = None
    if reuse_saved_accuracy:
        accuracy_data = get_verified_saved_accuracy(
            model_id=model_id,
            runtime=runtime,
            expected_batch_size=batch_size,
        )

    # 5. Model size and parameters
    file_size_mb = (
        round(result.tflite_rom_bytes / (1024.0 * 1024.0), 2)
        if result.tflite_rom_bytes
        else round(artifact_path.stat().st_size / (1024.0 * 1024.0), 2)
    )
    param_count = get_model_parameter_count(model_id)

    # Derived throughput
    throughput_fps = (
        round(1000.0 / result.time_per_inference_ms, 4)
        if result.time_per_inference_ms and result.time_per_inference_ms > 0
        else None
    )

    # Environment Info & Provenance Metadata
    env_info = {
        "provider": "edge_impulse",
        "target_device": target_device,
        "job_id": result.job_id,
        "artifact_sha256": artifact_sha,
        "batch_size": batch_size,
        "provenance": "Edge Impulse Studio Estimation",
        "estimation_type": "analytical_simulation",
        "latency_type": "analytical_estimate",
        "throughput_type": "derived_from_estimated_latency",
        "is_supported_on_mcu": result.is_supported_on_mcu,
        "mcu_support_error": result.mcu_support_error,
        "notes": (
            "Performance estimates computed via Edge Impulse Studio analytical profiler. "
            "Latency represents unaccelerated single-core analytical estimation. "
            "RAM is dynamically allocated by Linux OS and is not measured (N/A). "
            "MCU compiler warnings do not affect Linux SBC deployment."
        ),
    }

    # Persist BenchmarkRun
    run = BenchmarkRun(
        run_name=f"Edge Impulse Estimation ({metadata.display_name}) [{target_device}]",
        description=(
            f"Edge Impulse Studio profiling estimate for {metadata.display_name} TFLite on {target_device}."
        ),
        device=target_device,
        environment_info=env_info,
        provenance="edge_impulse_estimation",
        status="completed",
        completed_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.flush()

    # Persist ModelMeasurement
    # Unavailable RAM is explicitly None (stored as SQL NULL)
    measurement = ModelMeasurement(
        run_id=run.id,
        model_id=model_id,
        architecture=metadata.display_name,
        runtime=runtime,
        device=target_device,
        batch_size=batch_size,
        status="completed",
        # Latency & Throughput
        latency_mean_ms=result.time_per_inference_ms,
        latency_p95_ms=None,
        throughput_fps=throughput_fps,
        # Memory: Peak RAM is NULL / N/A
        peak_ram_mb=None,
        # Model Size
        file_size_mb=file_size_mb,
        parameter_count=param_count,
        # Accuracy
        evaluated_samples=accuracy_data.get("evaluated_samples") if accuracy_data else None,
        top1_accuracy=accuracy_data.get("top1_accuracy") if accuracy_data else None,
        top5_accuracy=accuracy_data.get("top5_accuracy") if accuracy_data else None,
    )
    session.add(measurement)
    session.flush()

    logger.info(
        "Persisted Edge Impulse profiling run ID %s (measurement ID %s) for %s on %s.",
        run.id,
        measurement.id,
        model_id,
        target_device,
    )
    return run
