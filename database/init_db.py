"""
Database Initialization and Schema Migration Utility.

Provides programmatic and CLI operations to:
1. Verify MySQL connectivity.
2. Initialize schema tables (CREATE TABLE IF NOT EXISTS) using SQLAlchemy ORM metadata.
3. Print pure MySQL DDL statements for manual migration without live database access.

Usage:
    # Print MySQL DDL
    python -m database.init_db --sql

    # Check connection
    python -m database.init_db --check

    # Initialize tables
    python -m database.init_db --init
"""

import argparse
import sys
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.schema import CreateTable

from database.config import DatabaseConfig, DatabaseConfigurationError
from database.connection import check_database_connection, create_database_engine
from database.models import Base, BenchmarkRun, ModelMeasurement


def get_schema_ddl() -> str:
    """Generate pure MySQL DDL statements directly from SQLAlchemy 2.0 ORM metadata.

    Does NOT require an active database connection.
    """
    from sqlalchemy.dialects import mysql

    ddl_statements = []
    dialect = mysql.dialect()

    for table in Base.metadata.sorted_tables:
        create_stmt = str(CreateTable(table).compile(dialect=dialect)).strip()
        ddl_statements.append(f"{create_stmt};")

    return "\n\n".join(ddl_statements)


def init_database(
    config: Optional[DatabaseConfig] = None,
    echo: bool = False,
) -> bool:
    """Initialize database tables using the configured DatabaseConfig.

    Args:
        config: Validated DatabaseConfig. If None, loaded from environment via .env.
        echo: Whether to echo SQL execution statements.

    Returns:
        True if tables were created successfully, False otherwise.
    """
    if config is None:
        try:
            config = DatabaseConfig.from_env()
        except DatabaseConfigurationError as exc:
            print(f"[ERROR] Database configuration error: {exc}", file=sys.stderr)
            return False

    engine = create_database_engine(config, echo=echo)

    print(f"Connecting to database at {config.safe_url}...")
    if not check_database_connection(engine, timeout_seconds=3.0):
        print(f"[ERROR] Cannot connect to MySQL at {config.safe_url}. Please verify MySQL is running.", file=sys.stderr)
        return False

    print("Connection successful. Creating database tables if not exist...")
    Base.metadata.create_all(engine)
    print("Database schema successfully initialized.")
    return True


def main() -> int:
    """CLI entrypoint for database management operations."""
    parser = argparse.ArgumentParser(
        description="Initialize MySQL persistence schema for Automated Image Model Benchmarking."
    )
    parser.add_argument(
        "--sql",
        action="store_true",
        help="Print pure MySQL DDL schema statements without connecting to database.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connection to the configured MySQL database without altering schema.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Connect to MySQL and create database tables if they do not exist.",
    )
    args = parser.parse_args()

    if args.sql:
        print(get_schema_ddl())
        return 0

    if args.check:
        try:
            config = DatabaseConfig.from_env()
        except DatabaseConfigurationError as exc:
            print(f"[ERROR] Configuration error: {exc}", file=sys.stderr)
            return 1

        engine = create_database_engine(config)
        is_live = check_database_connection(engine, timeout_seconds=3.0)
        if is_live:
            print(f"[SUCCESS] MySQL database is reachable at: {config.safe_url}")
            return 0
        else:
            print(f"[FAILURE] MySQL database is NOT reachable at: {config.safe_url}")
            return 1

    if args.init:
        success = init_database()
        return 0 if success else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
