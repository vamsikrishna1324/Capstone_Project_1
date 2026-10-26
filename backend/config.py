"""
Backend Configuration Module for Edge Model Benchmarking API.

Defines API routing settings, CORS configuration, environment properties,
and safe, optional Edge Impulse integration settings without exposing secrets.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# Recognized placeholder tokens that must NOT be treated as valid credentials
_PLACEHOLDER_CREDENTIAL_TOKENS = {
    "",
    "none",
    "null",
    "your_edge_impulse_api_key_here",
    "your_api_key_here",
    "your_edge_impulse_project_id_here",
    "your_project_id_here",
    "placeholder",
}


@dataclass
class BackendSettings:
    """FastAPI application configuration settings."""

    api_prefix: str = "/api"
    title: str = "Automated Image Model Benchmarking API"
    description: str = (
        "REST API for automated computer vision model benchmarking, "
        "MySQL performance persistence, and edge deployment recommendations."
    )
    version: str = "1.0.0"
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    seed_baseline_on_startup: bool = False
    edge_impulse_api_key: Optional[str] = field(default=None, repr=False)
    edge_impulse_project_id: Optional[str] = None

    @property
    def is_edge_impulse_configured(self) -> bool:
        """Check whether valid Edge Impulse API key credentials are configured."""
        if not self.edge_impulse_api_key:
            return False
        clean_key = self.edge_impulse_api_key.strip()
        return bool(clean_key and clean_key.lower() not in _PLACEHOLDER_CREDENTIAL_TOKENS)

    def get_edge_impulse_status(self) -> Dict[str, Any]:
        """Return safe Edge Impulse configuration status without exposing credentials.

        Never returns the actual API key string. Returns boolean indicator flags
        and non-sensitive metadata only.
        """
        is_cfg = self.is_edge_impulse_configured
        has_proj = bool(
            self.edge_impulse_project_id
            and self.edge_impulse_project_id.strip()
            and self.edge_impulse_project_id.strip().lower() not in _PLACEHOLDER_CREDENTIAL_TOKENS
        )
        return {
            "configured": is_cfg,
            "status": "configured" if is_cfg else "not_configured",
            "has_api_key": is_cfg,
            "has_project_id": has_proj,
            "project_id": self.edge_impulse_project_id.strip() if has_proj else None,
        }

    def __repr__(self) -> str:
        ei_status = "configured" if self.is_edge_impulse_configured else "not_configured"
        return (
            f"BackendSettings(api_prefix='{self.api_prefix}', version='{self.version}', "
            f"seed_baseline_on_startup={self.seed_baseline_on_startup}, "
            f"edge_impulse_status='{ei_status}')"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_env(cls, env_file: Optional[Union[str, Path]] = None) -> "BackendSettings":
        """Instantiate settings with optional environment overrides."""
        default_env = Path(".env")
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        elif default_env.exists():
            load_dotenv(dotenv_path=default_env, override=False)

        origins_str = os.environ.get("CORS_ORIGINS", "")
        if origins_str.strip():
            origins = [o.strip() for o in origins_str.split(",") if o.strip()]
        else:
            origins = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        seed = os.environ.get("SEED_BASELINE_ON_STARTUP", "false").lower() in ("true", "1", "yes")

        raw_ei_key = os.environ.get("EDGE_IMPULSE_API_KEY", "").strip()
        ei_key = raw_ei_key if (raw_ei_key and raw_ei_key.lower() not in _PLACEHOLDER_CREDENTIAL_TOKENS) else None

        raw_ei_proj = os.environ.get("EDGE_IMPULSE_PROJECT_ID", "").strip()
        ei_proj = raw_ei_proj if (raw_ei_proj and raw_ei_proj.lower() not in _PLACEHOLDER_CREDENTIAL_TOKENS) else None

        return cls(
            cors_origins=origins,
            seed_baseline_on_startup=seed,
            edge_impulse_api_key=ei_key,
            edge_impulse_project_id=ei_proj,
        )


def get_settings() -> BackendSettings:
    """Retrieve the active BackendSettings singleton instance."""
    global settings
    return settings


def reload_settings(env_file: Optional[Union[str, Path]] = None) -> BackendSettings:
    """Reload settings from active environment or specified .env file in-place."""
    global settings
    new_settings = BackendSettings.from_env(env_file=env_file)
    settings.__dict__.clear()
    settings.__dict__.update(new_settings.__dict__)
    return settings


settings = BackendSettings.from_env()


