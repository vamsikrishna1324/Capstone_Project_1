"""
Deployment Recommendation Router.

Provides Multi-Criteria Decision Analysis (MCDA) recommendations across
user-selected models, priority presets, custom importance weights,
and hardware edge constraints.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.schemas import RecommendationRequest, RecommendationResponse
from database.repository import (
    get_benchmark_run,
    get_measurements_for_comparison,
    get_measurements_for_run,
)
from recommendation.engine import RecommendationEngine

router = APIRouter(tags=["Recommendations"])


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtain Model Deployment Recommendation",
    description="Multi-criteria deployment recommendation based on accuracy, latency, memory, and model size trade-offs.",
)
def get_recommendation(
    request: RecommendationRequest,
    session: Session = Depends(get_db),
) -> RecommendationResponse:
    """Recommend the optimal model and runtime configuration for edge deployment."""
    # 1. Fetch relevant measurement candidates
    if request.run_id is not None:
        run = get_benchmark_run(session, request.run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark run with ID {request.run_id} not found.",
            )
        if run.provenance == "edge_impulse_estimation" or run.device != "cpu":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deployment recommendations evaluate standardized Host CPU live execution benchmarks only. Edge Impulse analytical estimation runs cannot be ranked via this recommendation workflow.",
            )
        raw_measurements = get_measurements_for_run(session, request.run_id)
    else:
        raw_measurements = get_measurements_for_comparison(
            session=session,
            model_ids=request.model_ids,
            runtimes=request.runtimes,
            device="cpu",
        )

    # 2. Filter and deduplicate candidates by (model_id, runtime)
    filtered_measurements = []
    seen_pairs = set()

    for m in raw_measurements:
        if request.model_ids and m.model_id not in request.model_ids:
            continue
        if request.runtimes and m.runtime not in request.runtimes:
            continue
        if m.status != "completed":
            continue
        if m.device != "cpu":
            continue

        pair = (m.model_id, m.runtime)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            filtered_measurements.append({
                "model_id": m.model_id,
                "runtime": m.runtime,
                "device": m.device,
                "top1_accuracy": m.top1_accuracy,
                "top5_accuracy": m.top5_accuracy,
                "latency_mean_ms": m.latency_mean_ms,
                "peak_ram_mb": m.peak_ram_mb,
                "file_size_mb": m.file_size_mb,
            })

    # 3. Instantiate MCDA recommendation engine
    custom_weights = request.weights.model_dump() if request.weights else None
    constraints = request.constraints.model_dump() if request.constraints else None

    try:
        engine = RecommendationEngine(
            preset=request.priority_preset,
            custom_weights=custom_weights,
            constraints=constraints,
        )
        result = engine.evaluate_candidates(filtered_measurements)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return RecommendationResponse(**result)
