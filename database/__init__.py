"""
Database Persistence Layer Package.

Provides MySQL database configuration, connection management, and session handling
for the Automated Image Model Benchmarking framework.
"""

from database.config import DatabaseConfig, DatabaseConfigurationError
from database.connection import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
    session_scope,
)
from database.models import Base, BenchmarkRun, ModelMeasurement
from database.repository import (
    add_measurement,
    add_measurements,
    create_benchmark_run,
    create_comparison_run,
    delete_benchmark_run,
    get_benchmark_run,
    get_measurements_for_comparison,
    get_measurements_for_run,
    list_benchmark_runs,
)

__all__ = [
    "DatabaseConfig",
    "DatabaseConfigurationError",
    "create_database_engine",
    "create_session_factory",
    "check_database_connection",
    "session_scope",
    "Base",
    "BenchmarkRun",
    "ModelMeasurement",
    "create_benchmark_run",
    "add_measurement",
    "add_measurements",
    "create_comparison_run",
    "get_benchmark_run",
    "list_benchmark_runs",
    "get_measurements_for_run",
    "get_measurements_for_comparison",
    "delete_benchmark_run",
]

