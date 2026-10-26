"""
Benchmark Comparison Execution and Retrieval Router.

Provides endpoints to trigger multi-model benchmark runs across PyTorch, ONNX,
and TFLite, and retrieve persisted comparison runs by ID from MySQL.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.dependencies import get_db, get_session_factory
from backend.schemas import (
    BenchmarkProgressResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)
from backend.services.benchmark_service import execute_comparison_benchmark
from backend.services.progress_tracker import progress_tracker
from database.connection import session_scope
from database.models import BenchmarkRun
from database.repository import create_benchmark_run, get_benchmark_run

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Benchmarks"])


def _run_comparison_worker(
    run_id: int,
    request: BenchmarkRunRequest,
) -> None:
    """Background worker for asynchronous benchmark execution."""
    factory = get_session_factory()
    try:
        with session_scope(factory) as session:
            run = get_benchmark_run(session, run_id)
            if not run:
                return

            execute_comparison_benchmark(
                session=session,
                model_ids=request.model_ids,
                runtimes=request.runtimes,
                metrics=request.metrics,
                run_name=request.run_name,
                description=request.description,
                batch_size=request.batch_size,
                num_runs=request.num_runs,
                warmup_runs=request.warmup_runs,
                device=request.device,
                reuse_saved_accuracy=request.reuse_saved_accuracy,
                run_accuracy_eval=request.run_accuracy_eval,
                max_accuracy_samples=request.max_accuracy_samples,
                existing_run_id=run_id,
                dataset_path=request.dataset_path,
            )
    except Exception as exc:
        logger.exception("Background benchmark execution failed: %s", exc)
        with session_scope(factory) as session:
            failed_run = get_benchmark_run(session, run_id)
            if failed_run:
                failed_run.status = "failed"
                failed_run.error_message = str(exc)
                failed_run.completed_at = datetime.now(timezone.utc)


@router.post(
    "/benchmarks",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute Benchmark Comparison",
    description="Run latency, memory, size, and accuracy benchmarks across multiple models/runtimes and persist to MySQL.",
)
async def create_benchmark_comparison(
    request: BenchmarkRunRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Execute a multi-model benchmark comparison and persist measurements."""
    # Asynchronous background execution branch
    if request.async_run:
        initial_run = create_benchmark_run(
            session=session,
            run_name=request.run_name or f"Comparison ({', '.join(request.model_ids)})",
            description=request.description,
            device=request.device,
            status="running",
            provenance="live_execution",
        )
        session.commit()

        background_tasks.add_task(_run_comparison_worker, initial_run.id, request)
        return BenchmarkRunResponse(**initial_run.to_dict())

    # Synchronous threadpool execution (non-blocking event loop)
    factory = get_session_factory()

    def _sync_worker() -> dict:
        with session_scope(factory) as thread_session:
            completed_run = execute_comparison_benchmark(
                session=thread_session,
                model_ids=request.model_ids,
                runtimes=request.runtimes,
                metrics=request.metrics,
                run_name=request.run_name,
                description=request.description,
                batch_size=request.batch_size,
                num_runs=request.num_runs,
                warmup_runs=request.warmup_runs,
                device=request.device,
                reuse_saved_accuracy=request.reuse_saved_accuracy,
                run_accuracy_eval=request.run_accuracy_eval,
                max_accuracy_samples=request.max_accuracy_samples,
                dataset_path=request.dataset_path,
            )
            return completed_run.to_dict()

    try:
        run_data = await asyncio.to_thread(_sync_worker)
        return BenchmarkRunResponse(**run_data)
    except Exception as exc:
        logger.exception("Benchmark comparison execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {exc}",
        ) from exc


@router.get(
    "/benchmarks/{run_id}",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Benchmark Comparison Run",
    description="Retrieve a specific benchmark run session and its associated model measurements by run ID.",
)
def get_benchmark_comparison(
    run_id: int,
    session: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Retrieve persisted comparison run by ID."""
    run = get_benchmark_run(session, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark comparison run with ID {run_id} not found.",
        )

    return BenchmarkRunResponse(**run.to_dict())


@router.get(
    "/benchmarks/{run_id}/progress",
    response_model=BenchmarkProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Real-Time Benchmark Progress",
    description="Retrieve live execution progress and elapsed/remaining timing for an active or completed benchmark run.",
)
def get_benchmark_progress(
    run_id: int,
    session: Session = Depends(get_db),
) -> BenchmarkProgressResponse:
    """Retrieve live intermediate progress for a specific benchmark run."""
    progress = progress_tracker.get_progress(run_id, session=session)
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run with ID {run_id} not found.",
        )

    return BenchmarkProgressResponse(**progress)
