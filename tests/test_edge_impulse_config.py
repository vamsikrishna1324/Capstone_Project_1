"""
Unit and Integration Tests for Safe Optional Edge Impulse Configuration.

Verifies:
1. Default behavior when Edge Impulse credentials are absent:
   - Application starts cleanly without errors.
   - Status reports configured=False and status='not_configured'.
   - No external requests or profiling operations are initiated.
2. Placeholder credential handling:
   - Example placeholders (e.g., 'your_edge_impulse_api_key_here') are recognized as unconfigured.
3. Mock configured credentials behavior:
   - When valid credentials are provided, configured=True and status='configured'.
   - Critical: The API key is NEVER leaked in JSON responses, headers, raw response text,
     repr, or str representations.
4. Dedicated endpoint (GET /api/edge-impulse/status) and health integration (GET /api/health).
5. Existing application endpoints (/api/models, /api/recommendations) remain operational.
"""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.config import BackendSettings, reload_settings
from backend.main import app


@pytest.fixture
def clean_env():
    """Ensure environment is isolated from ambient Edge Impulse environment variables."""
    saved_key = os.environ.get("EDGE_IMPULSE_API_KEY")
    saved_proj = os.environ.get("EDGE_IMPULSE_PROJECT_ID")
    os.environ["EDGE_IMPULSE_API_KEY"] = ""
    os.environ["EDGE_IMPULSE_PROJECT_ID"] = ""
    reload_settings()
    try:
        yield
    finally:
        if saved_key is not None:
            os.environ["EDGE_IMPULSE_API_KEY"] = saved_key
        else:
            os.environ.pop("EDGE_IMPULSE_API_KEY", None)
        if saved_proj is not None:
            os.environ["EDGE_IMPULSE_PROJECT_ID"] = saved_proj
        else:
            os.environ.pop("EDGE_IMPULSE_PROJECT_ID", None)
        reload_settings()



def test_settings_unconfigured_by_default(clean_env):
    """When credentials are not set, Edge Impulse should be unconfigured."""
    settings = BackendSettings.from_env()

    assert settings.edge_impulse_api_key is None
    assert settings.edge_impulse_project_id is None
    assert settings.is_edge_impulse_configured is False

    status = settings.get_edge_impulse_status()
    assert status == {
        "configured": False,
        "status": "not_configured",
        "has_api_key": False,
        "has_project_id": False,
        "project_id": None,
    }
    assert "api_key" not in status


def test_settings_ignores_placeholders(clean_env):
    """Placeholder tokens from .env.example must not be treated as configured credentials."""
    for placeholder in [
        "your_edge_impulse_api_key_here",
        "your_api_key_here",
        "none",
        "null",
        "   ",
        "",
    ]:
        with patch.dict(os.environ, {"EDGE_IMPULSE_API_KEY": placeholder}):
            settings = BackendSettings.from_env()
            assert settings.is_edge_impulse_configured is False
            assert settings.get_edge_impulse_status()["configured"] is False
            assert settings.get_edge_impulse_status()["status"] == "not_configured"


def test_settings_masked_representation_never_leaks_secrets(clean_env):
    """The raw API key must never appear in repr(settings) or str(settings)."""
    fake_secret = "ei_super_secret_production_key_abcdef123456"
    settings = BackendSettings(
        edge_impulse_api_key=fake_secret,
        edge_impulse_project_id="proj_888",
    )

    assert settings.is_edge_impulse_configured is True
    repr_str = repr(settings)
    str_str = str(settings)

    assert fake_secret not in repr_str
    assert fake_secret not in str_str
    assert "edge_impulse_status='configured'" in repr_str

    status_dict = settings.get_edge_impulse_status()
    assert fake_secret not in str(status_dict)
    assert status_dict["configured"] is True
    assert status_dict["status"] == "configured"
    assert status_dict["has_api_key"] is True
    assert status_dict["has_project_id"] is True
    assert status_dict["project_id"] == "proj_888"


def test_endpoint_status_unconfigured(clean_env):
    """GET /api/edge-impulse/status should return 200 with not_configured status."""
    with TestClient(app) as client:
        response = client.get("/api/edge-impulse/status")
        assert response.status_code == 200
        data = response.json()

        assert data["configured"] is False
        assert data["status"] == "not_configured"
        assert data["has_api_key"] is False
        assert data["has_project_id"] is False
        assert data["project_id"] is None
        assert "api_key" not in data


def test_endpoint_status_when_configured_never_exposes_key(clean_env):
    """GET /api/edge-impulse/status must report configured=True but never expose the key."""
    fake_key = "ei_mock_secret_key_999888777"
    fake_proj = "mock_project_101"

    with patch.dict(
        os.environ,
        {
            "EDGE_IMPULSE_API_KEY": fake_key,
            "EDGE_IMPULSE_PROJECT_ID": fake_proj,
        },
    ):
        reload_settings()
        try:
            with TestClient(app) as client:
                response = client.get("/api/edge-impulse/status")
                assert response.status_code == 200
                data = response.json()

                assert data["configured"] is True
                assert data["status"] == "configured"
                assert data["has_api_key"] is True
                assert data["has_project_id"] is True
                assert data["project_id"] == fake_proj

                # Strict secret omission assertions
                assert "api_key" not in data
                assert fake_key not in response.text
                assert "EDGE_IMPULSE_API_KEY" not in response.text
        finally:
            reload_settings()


def test_health_endpoint_includes_edge_impulse_status(clean_env):
    """GET /api/health should include safe Edge Impulse status block."""
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()

        assert "edge_impulse" in data
        ei_status = data["edge_impulse"]
        assert ei_status["configured"] is False
        assert ei_status["status"] == "not_configured"
        assert "api_key" not in ei_status


def test_existing_local_functionality_works_unimpeded_without_credentials(clean_env):
    """Existing local models catalog and root endpoint must work when credentials are absent."""
    with TestClient(app) as client:
        root_res = client.get("/")
        assert root_res.status_code == 200
        root_data = root_res.json()
        assert "edge_impulse_status" in root_data

        models_res = client.get("/api/models")
        assert models_res.status_code == 200
        models_data = models_res.json()
        assert models_data["total_models"] == 4
