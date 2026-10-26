"""
Benchmark Execution and Persistence Service.

Coordinates multi-model comparison benchmarking using BenchmarkEngine,
reusing verified 50k-sample accuracy artifacts when requested,
managing live sample evaluation, and persisting results transactionally to MySQL.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import select
from sqlalchemy.orm import Session

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine, get_environment_info
from backend.services.accuracy_service import (
    get_missing_accuracy_warning,
    load_saved_accuracy,
)
from backend.services.progress_tracker import progress_tracker
from database.models import BenchmarkRun, ModelMeasurement
from database.repository import (
    create_benchmark_run,
    create_comparison_run,
    get_benchmark_run,
)
from models.model_registry import get_model_metadata, is_model_supported

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "dataset" / "val_manifest.csv"


def resolve_dataset_manifest_path(dataset_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve validation dataset manifest path against the project root.

    Treats empty or whitespace-only paths as unspecified, falling back to
    the project-root default `dataset/val_manifest.csv`.

    Args:
        dataset_path: Optional custom manifest path (absolute or relative to PROJECT_ROOT).

    Returns:
        Resolved absolute Path to manifest.
    """
    if dataset_path is not None:
        raw_str = str(dataset_path).strip()
        if raw_str:
            p = Path(raw_str)
            return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    return DEFAULT_MANIFEST_PATH


def execute_comparison_benchmark(
    session: Session,
    model_ids: List[str],
    runtimes: List[str],
    metrics: List[str],
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    batch_size: int = 1,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
    reuse_saved_accuracy: bool = True,
    run_accuracy_eval: bool = False,
    max_accuracy_samples: Optional[int] = None,
    existing_run_id: Optional[int] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> BenchmarkRun:
    """Execute benchmarking across multiple models and runtimes, persisting to MySQL.

    Args:
        session: Active database session.
        model_ids: List of registered model identifiers.
        runtimes: List of execution runtimes ('pytorch', 'onnx', 'tflite').
        metrics: Selected metrics ('latency', 'memory', 'model_size', 'accuracy').
        run_name: Title of this comparison session.
        description: User notes or description.
        batch_size: Batch size for inference.
        num_runs: Number of timed runs.
        warmup_runs: Warmup iteration count.
        device: Target execution device.
        reuse_saved_accuracy: If True, attach matching 50k-image ImageNet accuracy.
        run_accuracy_eval: If True, perform live evaluation on dataset.
        max_accuracy_samples: Sample limit for live accuracy evaluation.
        existing_run_id: Optional ID of an existing BenchmarkRun to update instead of creating a new one.
        dataset_path: Optional path to validation manifest CSV/JSON. Defaults to dataset/val_manifest.csv.

    Returns:
        Persisted BenchmarkRun instance with populated measurements.
    """
    measure_latency = "latency" in metrics
    measure_memory = "memory" in metrics
    measure_model_size = "model_size" in metrics
    measure_accuracy = "accuracy" in metrics and run_accuracy_eval

    env_info = get_environment_info()
    manifest_path = resolve_dataset_manifest_path(dataset_path)

    if measure_accuracy and not manifest_path.is_file():
        raise FileNotFoundError(
            f"Validation dataset manifest file not found at: {manifest_path}. "
            "Please verify the manifest path before initiating live accuracy evaluation."
        )

    if existing_run_id is not None:
        run = get_benchmark_run(session, existing_run_id)
        if run is None:
            raise ValueError(f"Existing benchmark run with ID {existing_run_id} not found.")
        run.status = "running"
        run.environment_info = env_info
    else:
        run = create_benchmark_run(
            session=session,
            run_name=run_name or f"Comparison ({', '.join(model_ids)}) [{', '.join(runtimes)}]",
            description=description,
            device=device,
            environment_info=env_info,
            provenance="live_execution",
            status="running",
        )

    # Calculate total workload across models and runtimes for real-time progress tracking
    total_configs = len(model_ids) * len(runtimes)
    if measure_accuracy:
        samples_per_config = max_accuracy_samples or 50
        total_workload = total_configs * samples_per_config
    else:
        total_workload = total_configs * (warmup_runs + num_runs) * batch_size

    job_stage = "accuracy_evaluation" if measure_accuracy else "performance_benchmarking"
    progress_tracker.init_job(run.id, total_images=total_workload, status="running", stage=job_stage)

    measurements: List[ModelMeasurement] = []
    completed_accumulator = 0
    run_warnings: List[str] = []

    try:
        for mid in model_ids:
            metadata = get_model_metadata(mid) if is_model_supported(mid) else None
            arch_name = metadata.display_name if metadata else mid

            for rt in runtimes:
                rt_enum = BenchmarkRuntime.from_string(rt)

                config = BenchmarkConfig(
                    model_id=mid,
                    runtime=rt_enum,
                    batch_size=batch_size,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                    device=device,
                    dataset_path=str(manifest_path),
                    max_samples=max_accuracy_samples if measure_accuracy else None,
                )

                engine = BenchmarkEngine(config)

                def _progress_cb(completed_subtask: int, total_subtask: int) -> None:
                    current_completed = completed_accumulator + completed_subtask
                    progress_tracker.update_progress(
                        run.id,
                        completed_images=current_completed,
                        total_images=total_workload,
                        stage=job_stage,
                    )

                # Execute requested benchmarks
                bench_result = engine.run_benchmark(
                    dataset=str(manifest_path) if measure_accuracy else None,
                    measure_model_size=measure_model_size,
                    measure_latency=measure_latency,
                    measure_memory=measure_memory,
                    measure_accuracy=measure_accuracy,
                    max_accuracy_samples=max_accuracy_samples if measure_accuracy else None,
                    progress_callback=_progress_cb,
                )

                subtask_size = (
                    (max_accuracy_samples or 50)
                    if measure_accuracy
                    else (warmup_runs + num_runs) * batch_size
                )
                completed_accumulator += subtask_size
                progress_tracker.update_progress(
                    run.id,
                    completed_images=completed_accumulator,
                    total_images=total_workload,
                    stage=job_stage,
                )

                # Accuracy metric extraction
                accuracy_data: Optional[Dict[str, Any]] = None
                if measure_accuracy and bench_result.accuracy:
                    acc_obj = bench_result.accuracy
                    accuracy_data = {
                        "dataset_path": str(manifest_path),
                        "top1_accuracy": acc_obj.top1,
                        "top5_accuracy": acc_obj.top5,
                        "evaluated_samples": acc_obj.num_samples,
                        "total_samples": acc_obj.num_samples,
                        "skipped_images": 0,
                        "accuracy_source": "live_evaluation",
                    }
                elif reuse_saved_accuracy and "accuracy" in metrics:
                    accuracy_data = load_saved_accuracy(mid, rt)
                    if accuracy_data is None:
                        warn = get_missing_accuracy_warning(mid, rt)
                        if warn not in run_warnings:
                            run_warnings.append(warn)

                lat = bench_result.latency
                mem = bench_result.memory_footprint
                msize = bench_result.model_size

                measurement = ModelMeasurement(
                    run_id=run.id,
                    model_id=mid,
                    architecture=arch_name,
                    runtime=rt_enum.value,
                    device=device,
                    batch_size=batch_size,
                    warmup_runs=warmup_runs,
                    num_runs=num_runs,
                    status="completed" if bench_result.success else "failed",
                    error_message=bench_result.error_message,
                    # Accuracy
                    evaluated_samples=accuracy_data.get("evaluated_samples") if accuracy_data else None,
                    top1_accuracy=accuracy_data.get("top1_accuracy") if accuracy_data else None,
                    top5_accuracy=accuracy_data.get("top5_accuracy") if accuracy_data else None,
                    # Latency & Throughput
                    latency_mean_ms=lat.mean_ms if lat else None,
                    latency_p95_ms=lat.p95_ms if lat else None,
                    throughput_fps=lat.throughput_fps if lat else None,
                    # Memory
                    peak_ram_mb=mem.peak_ram_mb if mem else None,
                    # Model Size & Parameters (PyTorch retains NULL for file_size_mb)
                    file_size_mb=msize.file_size_mb if msize else None,
                    parameter_count=msize.param_count if msize else None,
                )
                session.add(measurement)
                measurements.append(measurement)

        if run_warnings:
            env = dict(run.environment_info or {})
            env["warnings"] = run_warnings
            run.environment_info = env

        failed_count = sum(1 for m in measurements if m.status == "failed")
        if failed_count == len(measurements) and len(measurements) > 0:
            run.status = "failed"
            first_err = next(
                (m.error_message for m in measurements if m.error_message),
                "All benchmark measurements failed.",
            )
            run.error_message = first_err
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
            progress_tracker.fail_job(run.id, error_message=run.error_message, warnings=run_warnings)
            return run
        elif failed_count > 0:
            run.status = "partial"
            run.error_message = f"{failed_count} of {len(measurements)} measurements failed."
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
            progress_tracker.complete_job(run.id, status="partial", error_message=run.error_message, warnings=run_warnings)
            return run

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()
        progress_tracker.complete_job(run.id, status="completed", warnings=run_warnings)
        return run
    except Exception as exc:
        progress_tracker.fail_job(run.id, error_message=str(exc))
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = datetime.now(timezone.utc)
        session.flush()
        raise



def import_baseline_benchmarks_if_needed(
    session: Session,
    summary_path: Optional[Union[str, Path]] = None,
) -> Optional[BenchmarkRun]:
    """Idempotently ingest pre-calculated edge benchmark matrix into MySQL.

    Reads `results/edge_benchmarks_summary.json` and attaches matching 50k-image
    accuracy evaluations, populating the database with historical baseline results.

    Args:
        session: Active database session.
        summary_path: Path to edge_benchmarks_summary.json.

    Returns:
        BenchmarkRun instance or None if file does not exist.
    """
    resolved_path = (
        Path(summary_path).resolve()
        if summary_path
        else (PROJECT_ROOT / "results" / "edge_benchmarks_summary.json")
    )
    if not resolved_path.is_file():
        return None

    stmt = select(BenchmarkRun).where(BenchmarkRun.provenance == "imported_artifact")
    existing_run = session.execute(stmt).scalars().first()
    if existing_run:
        return existing_run

    raw_bytes = resolved_path.read_bytes()
    summary_data = json.loads(raw_bytes.decode("utf-8"))
    environment = summary_data.get("environment", {})
    summary_items = summary_data.get("summary", [])

    measurements: List[ModelMeasurement] = []
    for item in summary_items:
        mid = item.get("model_id")
        rt = item.get("runtime")
        if not mid or not rt:
            continue

        metadata = get_model_metadata(mid) if is_model_supported(mid) else None
        accuracy_data = load_saved_accuracy(mid, rt)

        measurement = ModelMeasurement(
            model_id=mid,
            architecture=item.get("display_name") or (metadata.display_name if metadata else mid),
            runtime=rt,
            device=item.get("device", "cpu"),
            batch_size=item.get("batch_size", 1),
            warmup_runs=summary_data.get("warmup_runs", 10),
            num_runs=summary_data.get("num_runs", 100),
            status="completed" if item.get("success", True) else "failed",
            error_message=item.get("error_message"),
            # Accuracy
            evaluated_samples=accuracy_data.get("evaluated_samples") if accuracy_data else None,
            top1_accuracy=accuracy_data.get("top1_accuracy") if accuracy_data else None,
            top5_accuracy=accuracy_data.get("top5_accuracy") if accuracy_data else None,
            # Latency & Throughput
            latency_mean_ms=item.get("latency_mean_ms"),
            latency_p95_ms=item.get("latency_p95_ms"),
            throughput_fps=item.get("throughput_fps"),
            # Memory
            peak_ram_mb=item.get("peak_ram_mb"),
            # Model size & Parameters (PyTorch retains NULL for file_size_mb)
            file_size_mb=item.get("file_size_mb"),
            parameter_count=item.get("parameter_count"),
        )
        measurements.append(measurement)

    run = create_comparison_run(
        session=session,
        run_name="ImageNet Edge Benchmark Baseline (Imported)",
        measurements=measurements,
        description="Comprehensive 12-configuration benchmark matrix across PyTorch, ONNX, and TFLite.",
        device=summary_data.get("device", "cpu"),
        environment_info=environment,
        provenance="imported_artifact",
    )
    return run
