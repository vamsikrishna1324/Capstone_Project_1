"""
Database Connection and Session Management Module.

Provides SQLAlchemy 2.x engine and session factory creation for MySQL (PyMySQL dialect).
Does NOT establish or attempt connections during module import.
Connection checks occur ONLY when explicitly requested via `check_database_connection`.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from database.config import DatabaseConfig, DatabaseConfigurationError


def create_database_engine(
    config: DatabaseConfig,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy 2.x Engine configured for MySQL via PyMySQL.

    Does not open a physical connection upon creation. Connections are pooled lazily.
    Strictly verifies that the dialect is mysql+pymysql. Prohibits SQLite.

    Args:
        config: Validated DatabaseConfig instance.
        echo: Whether to enable SQLAlchemy SQL query logging.

    Returns:
        SQLAlchemy Engine instance.

    Raises:
        DatabaseConfigurationError: If the config dialect is not mysql+pymysql.
    """
    if not isinstance(config, DatabaseConfig):
        raise TypeError(f"Expected DatabaseConfig instance, got {type(config).__name__}")

    url = config.get_connection_url()
    if not url.startswith("mysql+pymysql://"):
        raise DatabaseConfigurationError(
            f"Invalid connection URL protocol. Expected 'mysql+pymysql://', got '{url[:15]}...'. "
            "SQLite and other dialects are strictly prohibited."
        )

    engine = create_engine(
        url,
        pool_size=config.pool_size,
        pool_recycle=config.pool_recycle,
        pool_pre_ping=config.pool_pre_ping,
        echo=echo,
        future=True,
    )
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a thread-safe SQLAlchemy sessionmaker factory bound to an engine.

    Args:
        engine: Active SQLAlchemy Engine.

    Returns:
        Configured sessionmaker instance.
    """
    if not isinstance(engine, Engine):
        raise TypeError(f"Expected SQLAlchemy Engine instance, got {type(engine).__name__}")

    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def check_database_connection(
    engine: Engine,
    timeout_seconds: float = 3.0,
) -> bool:
    """Explicitly test database liveness by executing a lightweight ping query (`SELECT 1`).

    This is the ONLY function that attempts an active network connection.
    It does not silently fall back to any other database engine.

    Args:
        engine: SQLAlchemy Engine instance to test.
        timeout_seconds: Timeout threshold for connection attempt.

    Returns:
        True if connection succeeded and query returned 1, False otherwise.
    """
    if not isinstance(engine, Engine):
        raise TypeError(f"Expected SQLAlchemy Engine instance, got {type(engine).__name__}")

    try:
        # Connect explicitly with execution options
        with engine.connect().execution_options(timeout=timeout_seconds) as connection:
            result = connection.execute(text("SELECT 1"))
            val = result.scalar()
            return val == 1
    except Exception:
        return False


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Context manager providing transactional session management.

    Commits on successful block completion, rolls back on exception,
    and unconditionally closes the session upon exit.
    """
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
