"""
Health Check Router.

Provides API liveness and database connectivity verification
without exposing credentials or sensitive environment details.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, status

from backend.config import get_settings
from backend.dependencies import check_db_health
from backend.schemas import DatabaseHealthInfo, EdgeImpulseStatusResponse, HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Inspect API operational status, database connectivity, and Edge Impulse integration status without exposing credentials.",
)
def get_health() -> HealthResponse:
    """Return API health status, MySQL connectivity, and safe Edge Impulse status."""
    db_info = check_db_health(timeout_seconds=2.0)
    api_status = "ok" if db_info["connected"] else "degraded"
    current_settings = get_settings()

    return HealthResponse(
        status=api_status,
        api="online",
        version=current_settings.version,
        database=DatabaseHealthInfo(**db_info),
        edge_impulse=EdgeImpulseStatusResponse(**current_settings.get_edge_impulse_status()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


