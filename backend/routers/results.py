"""
Benchmark Results Query and Retrieval Router.

Retrieves historical measurements from MySQL with multi-dimensional filtering
(model IDs, runtimes, device, run ID) and pagination.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.schemas import MeasurementResponse, ResultsPaginatedResponse
from database.repository import get_measurements_for_comparison

router = APIRouter(tags=["Results"])


def _parse_filter_param(values: Optional[List[str]]) -> Optional[List[str]]:
    """Flatten comma-separated strings or list query parameters."""
    if not values:
        return None
    flattened = []
    for item in values:
        for piece in item.split(","):
            piece_clean = piece.strip()
            if piece_clean:
                flattened.append(piece_clean)
    return flattened if flattened else None


@router.get(
    "/results",
    response_model=ResultsPaginatedResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Benchmark Results",
    description="Retrieve paginated benchmark measurements from MySQL with optional filters.",
)
def get_benchmark_results(
    model_ids: Optional[List[str]] = Query(None, description="Filter by model ID(s), e.g. 'mobilenet_v3_small'"),
    runtimes: Optional[List[str]] = Query(None, description="Filter by runtime(s), e.g. 'onnx', 'tflite'"),
    device: Optional[str] = Query(None, description="Filter by device, e.g. 'cpu'"),
    run_id: Optional[int] = Query(None, description="Filter by specific benchmark run ID"),
    limit: int = Query(50, ge=1, le=500, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: Session = Depends(get_db),
) -> ResultsPaginatedResponse:
    """Retrieve saved benchmark results matching filter criteria."""
    parsed_models = _parse_filter_param(model_ids)
    parsed_runtimes = _parse_filter_param(runtimes)
    run_ids = [run_id] if run_id is not None else None

    # Retrieve matching records from MySQL repository
    measurements = get_measurements_for_comparison(
        session=session,
        model_ids=parsed_models,
        runtimes=parsed_runtimes,
        run_ids=run_ids,
        device=device,
    )

    total = len(measurements)
    paginated_slice = measurements[offset : offset + limit]

    items = [MeasurementResponse(**m.to_dict()) for m in paginated_slice]

    return ResultsPaginatedResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )
