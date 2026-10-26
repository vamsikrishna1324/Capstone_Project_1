"""
Tests for Benchmark Evaluation Real-Time Progress Tracker and Progress API.

Validates intermediate count calculations, elapsed/remaining time estimations,
job isolation, duplicate prevention, and FastAPI endpoint contracts.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.progress_tracker import BenchmarkProgressTracker
from database.models import BenchmarkRun


@pytest.fixture
def tracker():
    """Create a fresh isolated tracker instance."""
    return BenchmarkProgressTracker()


def test_progress_tracker_initialization(tracker):
    """Tracker initializes job with correct initial counts, zero percentage, and stage."""
    job = tracker.init_job(run_id=101, total_images=50, status="preparing", stage="performance_benchmarking")
    assert job["run_id"] == 101
    assert job["status"] == "preparing"
    assert job["stage"] == "performance_benchmarking"
    assert job["images_completed"] == 0
    assert job["total_images"] == 50
    assert job["images_remaining"] == 50
    assert job["percentage"] == 0.0
    assert job["processing_speed_fps"] is None
    assert job["estimated_remaining_seconds"] is None
    assert job["error_message"] is None


def test_progress_tracker_stage_tracking(tracker):
    """Tracker preserves and updates stage between performance and accuracy evaluation."""
    job = tracker.init_job(run_id=106, total_images=50, stage="accuracy_evaluation")
    assert job["stage"] == "accuracy_evaluation"

    updated = tracker.update_progress(run_id=106, completed_images=10, stage="accuracy_evaluation")
    assert updated["stage"] == "accuracy_evaluation"


def test_progress_tracker_intermediate_updates(tracker):
    """Tracker accurately derives remaining images and percentage from completed count."""
    tracker.init_job(run_id=102, total_images=100, status="running")

    updated = tracker.update_progress(run_id=102, completed_images=25)
    assert updated["images_completed"] == 25
    assert updated["images_remaining"] == 75
    assert updated["percentage"] == 25.0

    # Halfway update
    updated = tracker.update_progress(run_id=102, completed_images=50)
    assert updated["images_completed"] == 50
    assert updated["images_remaining"] == 50
    assert updated["percentage"] == 50.0


def test_progress_tracker_speed_and_eta_calculation(tracker):
    """Tracker computes processing speed and estimated remaining time after observation window."""
    tracker.init_job(run_id=103, total_images=100)

    # Artificially set start_time to 2.0 seconds ago
    with tracker._lock:
        tracker._jobs[103]["start_time"] = time.time() - 2.0

    updated = tracker.update_progress(run_id=103, completed_images=20)
    assert updated["elapsed_seconds"] >= 2.0
    assert updated["processing_speed_fps"] is not None
    assert updated["processing_speed_fps"] > 0
    assert updated["estimated_remaining_seconds"] is not None
    # 80 remaining / ~10 fps ~= 8 seconds
    assert 6.0 <= updated["estimated_remaining_seconds"] <= 10.0


def test_progress_tracker_completion(tracker):
    """Completed job sets 100% percentage, zero remaining, and terminal status."""
    tracker.init_job(run_id=104, total_images=50)
    completed = tracker.complete_job(run_id=104, status="completed")
    assert completed["status"] == "completed"
    assert completed["images_completed"] == 50
    assert completed["images_remaining"] == 0
    assert completed["percentage"] == 100.0
    assert completed["estimated_remaining_seconds"] == 0.0


def test_progress_tracker_failure(tracker):
    """Failed job sets failed status and preserves error message."""
    tracker.init_job(run_id=105, total_images=50)
    failed = tracker.fail_job(run_id=105, error_message="Inference kernel crashed")
    assert failed["status"] == "failed"
    assert failed["error_message"] == "Inference kernel crashed"
    assert failed["estimated_remaining_seconds"] is None


def test_progress_tracker_fallback_to_database(tracker):
    """Tracker reads terminal status from database when job is not in active memory."""
    mock_session = MagicMock()
    mock_run = MagicMock(spec=BenchmarkRun)
    mock_run.id = 999
    mock_run.status = "completed"
    mock_run.error_message = None

    with patch("backend.services.progress_tracker.get_benchmark_run", return_value=mock_run):
        progress = tracker.get_progress(run_id=999, session=mock_session)
        assert progress is not None
        assert progress["run_id"] == 999
        assert progress["status"] == "completed"
        assert progress["percentage"] == 100.0
        assert progress["images_remaining"] == 0


def test_api_benchmark_progress_not_found():
    """GET /api/benchmarks/{run_id}/progress returns 404 for unknown run."""
    client = TestClient(app)
    with patch("backend.routers.benchmarks.progress_tracker.get_progress", return_value=None):
        resp = client.get("/api/benchmarks/88888/progress")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


def test_api_benchmark_progress_active():
    """GET /api/benchmarks/{run_id}/progress returns real progress payload for in-flight run."""
    client = TestClient(app)
    mock_progress = {
        "run_id": 42,
        "status": "running",
        "stage": "performance_benchmarking",
        "images_completed": 30,
        "total_images": 60,
        "images_remaining": 30,
        "percentage": 50.0,
        "elapsed_seconds": 3.2,
        "estimated_remaining_seconds": 3.2,
        "processing_speed_fps": 9.38,
        "error_message": None,
    }
    with patch("backend.routers.benchmarks.progress_tracker.get_progress", return_value=mock_progress):
        resp = client.get("/api/benchmarks/42/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == 42
        assert data["status"] == "running"
        assert data["stage"] == "performance_benchmarking"
        assert data["images_completed"] == 30
        assert data["total_images"] == 60
        assert data["images_remaining"] == 30
        assert data["percentage"] == 50.0
        assert data["processing_speed_fps"] == 9.38
        assert data["estimated_remaining_seconds"] == 3.2
