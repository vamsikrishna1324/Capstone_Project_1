"""
FastAPI Dependency Injection Module for Database and Engine Resources.

Provides thread-safe SQLAlchemy engine and session management using existing
database connection utilities. SQLite and other fallback dialects are strictly prohibited.
"""

from typing import Generator, Optional
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.config import DatabaseConfig
from database.connection import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
    session_scope,
)


_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine() -> Engine:
    """Retrieve or initialize the singleton SQLAlchemy Engine."""
    global _engine
    if _engine is None:
        config = DatabaseConfig.from_env()
        _engine = create_database_engine(config)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Retrieve or initialize the thread-safe SQLAlchemy sessionmaker factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory(get_engine())
    return _session_factory


def reset_engine() -> None:
    """Reset singleton engine and session factory to allow clean environment switching."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def get_db() -> Generator[Session, None, None]:
    """FastAPI request-scoped database session dependency.

    Automatically commits successful transactions, rolls back upon any exception,
    and unconditionally closes the session.
    """
    factory = get_session_factory()
    with session_scope(factory) as session:
        yield session


def check_db_health(timeout_seconds: float = 2.0) -> dict:
    """Inspect database connectivity safely without exposing credentials."""
    try:
        config = DatabaseConfig.from_env()
        engine = get_engine()
        is_live = check_database_connection(engine, timeout_seconds=timeout_seconds)
        return {
            "connected": is_live,
            "dialect": config.SUPPORTED_DIALECT,
            "host": config.host,
            "port": config.port,
            "database": config.database,
        }
    except Exception:
        return {
            "connected": False,
            "dialect": "mysql+pymysql",
            "host": "unknown",
            "port": 3306,
            "database": "unknown",
        }


def reset_engine_for_testing() -> None:
    """Helper for testing to reset cached engine and factory."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
