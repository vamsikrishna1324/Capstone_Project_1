"""
Test Suite for Database Configuration and Connection Foundation (Step 5).

Verifies:
- Explicit credential enforcement (no empty user or password).
- Password masking in connection URLs, repr, str, and dictionary exports.
- Strict MySQL+PyMySQL dialect enforcement (prohibition of SQLite and other dialects).
- Port and connection pool parameter validation.
- Environment variable parsing with separate production and test databases.
- SQLAlchemy 2.x Engine construction without opening active socket connections.
- Session factory configuration.
- Graceful health-check failure when target MySQL server is unreachable.
"""

from pathlib import Path
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from database.config import DatabaseConfig, DatabaseConfigurationError
from database.connection import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
)


# ============================================================================
# 1. DatabaseConfig Parameter & Credential Validation Tests
# ============================================================================
def test_database_config_valid_construction() -> None:
    """Verify valid DatabaseConfig initialization and attribute assignment."""
    config = DatabaseConfig(
        user="benchmark_user",
        password="super_secure_password",
        host="127.0.0.1",
        port=3306,
        database="edge_model_benchmark",
        pool_size=10,
        pool_recycle=1800,
        pool_pre_ping=True,
        is_test=False,
    )
    assert config.user == "benchmark_user"
    assert config.password == "super_secure_password"
    assert config.host == "127.0.0.1"
    assert config.port == 3306
    assert config.database == "edge_model_benchmark"
    assert config.pool_size == 10
    assert config.pool_recycle == 1800
    assert config.pool_pre_ping is True
    assert config.is_test is False

    expected_url = "mysql+pymysql://benchmark_user:super_secure_password@127.0.0.1:3306/edge_model_benchmark?charset=utf8mb4"
    assert config.get_connection_url() == expected_url


def test_database_config_safe_url_masks_password() -> None:
    """Verify that safe_url, __str__, __repr__, and to_dict() never expose plain-text passwords."""
    raw_password = "very_secret_cleartext_password_123"
    config = DatabaseConfig(
        user="test_user",
        password=raw_password,
        host="localhost",
        port=3306,
        database="test_db",
    )

    safe_str = config.safe_url
    assert raw_password not in safe_str
    assert "***" in safe_str or "secret" not in safe_str.lower()
    assert "test_user" in safe_str
    assert "test_db" in safe_str

    # Verify __str__ and __repr__
    assert raw_password not in str(config)
    assert raw_password not in repr(config)

    # Verify to_dict()
    info = config.to_dict()
    assert "password" not in info
    assert raw_password not in str(info)
    assert info["safe_url"] == safe_str


@pytest.mark.parametrize(
    "invalid_user,invalid_password",
    [
        ("", "valid_password"),
        ("   ", "valid_password"),
        (None, "valid_password"),
        ("valid_user", ""),
        ("valid_user", "   "),
        ("valid_user", None),
    ],
)
def test_database_config_requires_explicit_credentials(invalid_user, invalid_password) -> None:
    """Verify that missing, empty, or whitespace-only credentials raise DatabaseConfigurationError."""
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig(
            user=invalid_user,
            password=invalid_password,
            host="127.0.0.1",
            port=3306,
            database="test_db",
        )


@pytest.mark.parametrize("invalid_port", [-1, 0, 65536, 70000])
def test_database_config_invalid_port(invalid_port: int) -> None:
    """Verify that out-of-range port numbers raise DatabaseConfigurationError."""
    with pytest.raises(DatabaseConfigurationError, match="port"):
        DatabaseConfig(
            user="user",
            password="password",
            port=invalid_port,
        )


@pytest.mark.parametrize("invalid_pool_size", [0, -1, -10])
def test_database_config_invalid_pool_size(invalid_pool_size: int) -> None:
    """Verify that non-positive pool sizes raise DatabaseConfigurationError."""
    with pytest.raises(DatabaseConfigurationError, match="Pool size"):
        DatabaseConfig(
            user="user",
            password="password",
            pool_size=invalid_pool_size,
        )


# ============================================================================
# 2. Strict MySQL Dialect Enforcement (Zero SQLite Policy)
# ============================================================================
@pytest.mark.parametrize(
    "prohibited_url,expected_error_snippet",
    [
        ("sqlite:///test.db", "SQLite and other database engines are strictly prohibited"),
        ("sqlite:///:memory:", "SQLite and other database engines are strictly prohibited"),
        ("postgresql://user:pass@localhost:5432/db", "Unsupported database dialect 'postgresql'"),
        ("mysql://user:pass@localhost:3306/db", "Unsupported database dialect 'mysql'"),
        ("mysql+mysqlconnector://user:pass@localhost/db", "Unsupported database dialect"),
    ],
)
def test_database_config_strictly_enforces_mysql_pymysql_dialect(
    prohibited_url: str,
    expected_error_snippet: str,
) -> None:
    """Verify that passing SQLite or non-PyMySQL dialects raises DatabaseConfigurationError."""
    with pytest.raises(DatabaseConfigurationError, match=expected_error_snippet):
        DatabaseConfig(
            user="fallback_user",
            password="fallback_password",
            database_url=prohibited_url,
        )


def test_database_config_accepts_valid_mysql_pymysql_url() -> None:
    """Verify that a properly formatted mysql+pymysql URL is accepted."""
    valid_url = "mysql+pymysql://custom_user:custom_pass@dbhost.internal:3307/custom_db?charset=utf8mb4"
    config = DatabaseConfig(
        user="ignored_user",
        password="ignored_password",
        database_url=valid_url,
    )
    assert config.user == "custom_user"
    assert config.password == "custom_pass"
    assert config.host == "dbhost.internal"
    assert config.port == 3307
    assert config.database == "custom_db"
    assert config.get_connection_url() == valid_url
    assert "custom_pass" not in config.safe_url


# ============================================================================
# 3. Environment Variable Resolution & Test Separation Tests
# ============================================================================
def test_database_config_from_env_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loading configuration from environment variables for production database."""
    monkeypatch.setenv("MYSQL_USER", "prod_user")
    monkeypatch.setenv("MYSQL_PASSWORD", "prod_secret")
    monkeypatch.setenv("MYSQL_HOST", "192.168.1.50")
    monkeypatch.setenv("MYSQL_PORT", "3308")
    monkeypatch.setenv("MYSQL_DATABASE", "prod_benchmark_db")
    monkeypatch.setenv("MYSQL_TEST_DATABASE", "test_benchmark_db")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = DatabaseConfig.from_env(is_test=False)
    assert config.user == "prod_user"
    assert config.password == "prod_secret"
    assert config.host == "192.168.1.50"
    assert config.port == 3308
    assert config.database == "prod_benchmark_db"
    assert config.is_test is False


def test_database_config_from_env_test_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that is_test=True cleanly routes to MYSQL_TEST_DATABASE."""
    monkeypatch.setenv("MYSQL_USER", "test_runner")
    monkeypatch.setenv("MYSQL_PASSWORD", "test_secret")
    monkeypatch.setenv("MYSQL_DATABASE", "prod_db")
    monkeypatch.setenv("MYSQL_TEST_DATABASE", "isolated_test_db")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = DatabaseConfig.from_env(is_test=True)
    assert config.database == "isolated_test_db"
    assert config.is_test is True
    assert "isolated_test_db" in config.get_connection_url()


def test_database_config_from_env_missing_user_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that missing MYSQL_USER raises DatabaseConfigurationError."""
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.setenv("MYSQL_PASSWORD", "some_password")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError, match="MYSQL_USER"):
        DatabaseConfig.from_env()


def test_database_config_from_env_missing_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that missing MYSQL_PASSWORD raises DatabaseConfigurationError."""
    monkeypatch.setenv("MYSQL_USER", "some_user")
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError, match="MYSQL_PASSWORD"):
        DatabaseConfig.from_env()


# ============================================================================
# 4. Engine & Connection Factory Tests (Without Active Server Connections)
# ============================================================================
def test_create_database_engine_without_connecting() -> None:
    """Verify SQLAlchemy 2.x engine construction without opening physical network sockets."""
    config = DatabaseConfig(
        user="mock_user",
        password="mock_password",
        host="127.0.0.1",
        port=3306,
        database="mock_db",
        pool_size=7,
        pool_recycle=2400,
        pool_pre_ping=True,
    )

    engine = create_database_engine(config)
    assert isinstance(engine, Engine)
    assert engine.dialect.name == "mysql"
    assert engine.dialect.driver == "pymysql"
    assert engine.pool.size() == 7

    # Verify session factory binds to the engine
    session_factory = create_session_factory(engine)
    assert isinstance(session_factory, sessionmaker)


def test_create_database_engine_type_validation() -> None:
    """Verify create_database_engine rejects non-DatabaseConfig inputs."""
    with pytest.raises(TypeError, match="DatabaseConfig"):
        create_database_engine("mysql+pymysql://user:pass@localhost/db")  # type: ignore[arg-type]


def test_check_database_connection_fails_gracefully_when_server_unreachable() -> None:
    """Verify that check_database_connection returns False when MySQL server is unreachable.

    Ensures it handles connection failures gracefully and does NOT crash or substitute SQLite.
    """
    # Point to an unreachable loopback port where no MySQL daemon is running
    config = DatabaseConfig(
        user="test_user",
        password="test_password",
        host="127.0.0.1",
        port=59999,
        database="nonexistent_db",
    )
    engine = create_database_engine(config)

    # Must return False without unhandled exception
    is_alive = check_database_connection(engine, timeout_seconds=0.5)
    assert is_alive is False
