"""
Database Configuration Module for Automated Image Model Benchmarking.

Manages environment variable resolution, credential validation, URL construction,
and secure credential masking for MySQL database persistence.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

_DOTENV_LOADED = False



class DatabaseConfigurationError(ValueError):
    """Raised when database configuration parameters are missing, malformed, or invalid."""



class DatabaseConfig:
    """Encapsulates MySQL database configuration and connection parameters.

    Strictly requires MySQL with the pure-Python PyMySQL dialect (`mysql+pymysql`).
    Prohibits SQLite or any fallback dialect. Requires explicit credentials and
    masks passwords in all string representations.
    """

    SUPPORTED_DIALECT = "mysql+pymysql"

    def __init__(
        self,
        user: str,
        password: str,
        host: str = "127.0.0.1",
        port: int = 3306,
        database: str = "edge_model_benchmark",
        pool_size: int = 5,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        database_url: Optional[str] = None,
        is_test: bool = False,
    ) -> None:
        if not user or not user.strip():
            raise DatabaseConfigurationError("Database user (MYSQL_USER) must be explicitly provided and non-empty.")
        if not password or not password.strip():
            raise DatabaseConfigurationError(
                "Database password (MYSQL_PASSWORD) must be explicitly provided and non-empty. "
                "Connecting with an empty password is not permitted."
            )
        if not (1 <= port <= 65535):
            raise DatabaseConfigurationError(f"Invalid database port: {port}. Must be between 1 and 65535.")
        if pool_size <= 0:
            raise DatabaseConfigurationError(f"Pool size must be a positive integer, got {pool_size}.")
        if pool_recycle <= 0:
            raise DatabaseConfigurationError(f"Pool recycle must be a positive integer, got {pool_recycle}.")

        self.user = user.strip()
        self.password = password
        self.host = host.strip()
        self.port = port
        self.database = database.strip()
        self.pool_size = pool_size
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.is_test = is_test

        if database_url is not None:
            self._validate_and_set_url(database_url)
        else:
            self.database_url = None

    def _validate_and_set_url(self, raw_url: str) -> None:
        """Validate and set a pre-configured database URL.

        Ensures that only mysql+pymysql is accepted. Rejects SQLite and other dialects.
        """
        try:
            url_obj = make_url(raw_url)
        except Exception as exc:
            raise DatabaseConfigurationError(f"Malformed DATABASE_URL: {exc}") from exc

        drivername = url_obj.drivername.lower()
        if drivername != self.SUPPORTED_DIALECT:
            raise DatabaseConfigurationError(
                f"Unsupported database dialect '{drivername}'. "
                f"Only '{self.SUPPORTED_DIALECT}' is supported. "
                "SQLite and other database engines are strictly prohibited."
            )

        if not url_obj.username:
            raise DatabaseConfigurationError("DATABASE_URL must include an explicit username.")
        if not url_obj.password:
            raise DatabaseConfigurationError("DATABASE_URL must include an explicit non-empty password.")

        self.database_url = raw_url
        self.user = url_obj.username
        self.password = url_obj.password
        self.host = url_obj.host or self.host
        self.port = url_obj.port or self.port
        self.database = url_obj.database or self.database

    @classmethod
    def from_env(
        cls,
        env_file: Optional[Union[str, Path]] = None,
        is_test: bool = False,
    ) -> "DatabaseConfig":
        """Load database configuration from environment variables.

        Loads .env via python-dotenv with override=False to preserve active shell
        variables. Requires explicit credentials.
        """
        global _DOTENV_LOADED
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        elif not _DOTENV_LOADED:
            default_env = Path(".env")
            if default_env.exists():
                load_dotenv(dotenv_path=default_env, override=False)
            _DOTENV_LOADED = True


        raw_url = os.environ.get("DATABASE_URL")
        if raw_url and raw_url.strip():
            # Validate and instantiate from URL
            try:
                url_obj = make_url(raw_url.strip())
            except Exception as exc:
                raise DatabaseConfigurationError(f"Malformed DATABASE_URL: {exc}") from exc

            return cls(
                user=url_obj.username or "",
                password=url_obj.password or "",
                host=url_obj.host or "127.0.0.1",
                port=url_obj.port or 3306,
                database=url_obj.database or ("edge_model_benchmark_test" if is_test else "edge_model_benchmark"),
                pool_size=int(os.environ.get("DB_POOL_SIZE", 5)),
                pool_recycle=int(os.environ.get("DB_POOL_RECYCLE", 3600)),
                pool_pre_ping=os.environ.get("DB_POOL_PRE_PING", "true").lower() in ("true", "1", "yes"),
                database_url=raw_url.strip(),
                is_test=is_test,
            )

        # Resolve individual variables
        user = os.environ.get("MYSQL_USER", "").strip()
        password = os.environ.get("MYSQL_PASSWORD", "")

        if not user:
            raise DatabaseConfigurationError(
                "Missing required environment variable 'MYSQL_USER'. "
                "Explicit credentials are required. Connecting as root or without credentials is not permitted."
            )
        if not password:
            raise DatabaseConfigurationError(
                "Missing required environment variable 'MYSQL_PASSWORD'. "
                "Explicit credentials are required. Connecting with an empty password is not permitted."
            )

        host = os.environ.get("MYSQL_HOST", "127.0.0.1").strip()
        raw_port = os.environ.get("MYSQL_PORT", "3306")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise DatabaseConfigurationError(f"MYSQL_PORT must be an integer, got '{raw_port}'") from exc

        if is_test:
            database = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test").strip()
        else:
            database = os.environ.get("MYSQL_DATABASE", "edge_model_benchmark").strip()

        try:
            pool_size = int(os.environ.get("DB_POOL_SIZE", "5"))
        except ValueError as exc:
            raise DatabaseConfigurationError(f"DB_POOL_SIZE must be an integer, got '{os.environ.get('DB_POOL_SIZE')}'") from exc

        try:
            pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))
        except ValueError as exc:
            raise DatabaseConfigurationError(f"DB_POOL_RECYCLE must be an integer, got '{os.environ.get('DB_POOL_RECYCLE')}'") from exc

        pool_pre_ping = os.environ.get("DB_POOL_PRE_PING", "true").lower() in ("true", "1", "yes")

        return cls(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            is_test=is_test,
        )

    def get_connection_url(self) -> str:
        """Construct the complete MySQL connection URL with driver specification."""
        if self.database_url:
            return self.database_url

        escaped_user = quote_plus(self.user)
        escaped_password = quote_plus(self.password)
        return (
            f"{self.SUPPORTED_DIALECT}://{escaped_user}:{escaped_password}@"
            f"{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )

    @property
    def safe_url(self) -> str:
        """Return the connection URL with password masked for safe logging and display."""
        url = self.get_connection_url()
        try:
            url_obj = make_url(url)
            return url_obj.render_as_string(hide_password=True)
        except Exception:
            return f"{self.SUPPORTED_DIALECT}://{self.user}:***@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration metadata dictionary with credentials redacted."""
        return {
            "dialect": self.SUPPORTED_DIALECT,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "is_test": self.is_test,
            "pool_size": self.pool_size,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "safe_url": self.safe_url,
        }

    def __repr__(self) -> str:
        return f"DatabaseConfig(safe_url='{self.safe_url}', is_test={self.is_test})"

    def __str__(self) -> str:
        return self.safe_url
