"""
FastAPI Application Entrypoint for Automated Image Model Benchmarking.

Configures CORS middleware, registers modular API routers under /api,
and handles database baseline synchronization upon startup.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.dependencies import check_db_health, get_session_factory
from backend.routers.benchmarks import router as benchmarks_router
from backend.routers.edge_impulse import router as edge_impulse_router
from backend.routers.health import router as health_router
from backend.routers.models import router as models_router
from backend.routers.recommendations import router as recommendations_router
from backend.routers.results import router as results_router
from backend.services.benchmark_service import import_baseline_benchmarks_if_needed
from database.connection import session_scope

logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan handler for FastAPI application startup and shutdown events."""
    logger.info("Initializing Automated Image Model Benchmarking API...")

    if settings.seed_baseline_on_startup:
        try:
            health = check_db_health(timeout_seconds=2.0)
            if health.get("connected"):
                factory = get_session_factory()
                with session_scope(factory) as session:
                    imported_run = import_baseline_benchmarks_if_needed(session)
                    if imported_run:
                        logger.info("Baseline benchmark matrix verified in MySQL (run_id=%s).", imported_run.id)
            else:
                logger.warning("Database unreachable during startup; baseline import skipped.")
        except Exception as exc:
            logger.warning("Baseline benchmark import skipped during startup: %s", exc)

    yield
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers under /api prefix
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(edge_impulse_router, prefix=settings.api_prefix)
    app.include_router(models_router, prefix=settings.api_prefix)
    app.include_router(results_router, prefix=settings.api_prefix)
    app.include_router(benchmarks_router, prefix=settings.api_prefix)
    app.include_router(recommendations_router, prefix=settings.api_prefix)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "title": settings.title,
            "version": settings.version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
            "edge_impulse_status": f"{settings.api_prefix}/edge-impulse/status",
        }

    return app



app = create_app()
