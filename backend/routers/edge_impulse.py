from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.dependencies import get_db
from backend.schemas import (
    BenchmarkRunResponse,
    EdgeImpulseProfileRequest,
    EdgeImpulseStatusResponse,
    EdgeImpulseTargetDetail,
    EdgeImpulseTargetsResponse,
)
from backend.services.edge_impulse_service import (
    EdgeImpulseProfilingError,
    SUPPORTED_EDGE_IMPULSE_TARGETS,
    get_or_create_edge_impulse_benchmark,
)

router = APIRouter(prefix="/edge-impulse", tags=["Edge Impulse"])


@router.get(
    "/status",
    response_model=EdgeImpulseStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Edge Impulse Configuration Status",
    description="Inspect whether Edge Impulse credentials are configured without exposing secrets.",
)
def get_edge_impulse_status() -> EdgeImpulseStatusResponse:
    """Return safe Edge Impulse configuration status.

    Never returns the actual API key string. Returns boolean indicator flags
    and non-sensitive metadata only.
    """
    return EdgeImpulseStatusResponse(**get_settings().get_edge_impulse_status())


@router.get(
    "/targets",
    response_model=EdgeImpulseTargetsResponse,
    status_code=status.HTTP_200_OK,
    summary="List Supported Edge Impulse Profiling Targets",
    description="Retrieve verified Edge Impulse hardware targets and supported model formats.",
)
def get_supported_targets() -> EdgeImpulseTargetsResponse:
    """Return catalog of supported Edge Impulse profiling hardware targets."""
    targets = [
        EdgeImpulseTargetDetail(
            target_id=info["target_id"],
            display_name=info["display_name"],
            supported_formats=info["supported_formats"],
            is_supported=info["is_supported"],
            notes=info.get("notes"),
        )
        for info in SUPPORTED_EDGE_IMPULSE_TARGETS.values()
    ]
    return EdgeImpulseTargetsResponse(targets=targets, total_targets=len(targets))


@router.post(
    "/profile",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Profile Model via Edge Impulse",
    description="Profile or reuse verified Edge Impulse estimation for a supported TFLite model on target hardware.",
)
def profile_model_edge_impulse(
    request: EdgeImpulseProfileRequest,
    session: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Execute or reuse Edge Impulse profiling for a supported TFLite model."""
    try:
        run = get_or_create_edge_impulse_benchmark(
            session=session,
            model_id=request.model_id,
            target_device=request.target_device,
            runtime=request.runtime,
            batch_size=request.batch_size,
            reuse_completed=request.reuse_completed,
            reuse_saved_accuracy=request.reuse_saved_accuracy,
            allow_external_submission=False,
        )
        session.commit()
        return BenchmarkRunResponse(**run.to_dict())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except EdgeImpulseProfilingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

