"""
Benchmark Evaluation Real-Time Progress Tracker.

Maintains thread-safe in-memory intermediate progress state for active benchmark runs,
exposing real evaluation metrics: images completed, total images, images remaining,
percentage, elapsed time, processing speed, and estimated time remaining.
"""

from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.repository import get_benchmark_run


class BenchmarkProgressTracker:
    """Thread-safe tracker for active benchmark run progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[int, Dict[str, Any]] = {}

    def init_job(
        self,
        run_id: int,
        total_images: int,
        status: str = "preparing",
        stage: Optional[str] = "performance_benchmarking",
    ) -> Dict[str, Any]:
        """Initialize progress tracking for a new or existing benchmark run."""
        with self._lock:
            now = time.time()
            total = max(1, total_images)
            record: Dict[str, Any] = {
                "run_id": run_id,
                "status": status,
                "stage": stage,
                "images_completed": 0,
                "total_images": total,
                "images_remaining": total,
                "percentage": 0.0,
                "start_time": now,
                "elapsed_seconds": 0.0,
                "estimated_remaining_seconds": None,
                "processing_speed_fps": None,
                "error_message": None,
                "warnings": [],
            }
            self._jobs[run_id] = record
            return dict(record)

    def update_progress(
        self,
        run_id: int,
        completed_images: int,
        total_images: Optional[int] = None,
        status: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update intermediate progress counts and compute derived metrics."""
        with self._lock:
            record = self._jobs.get(run_id)
            if not record:
                return None

            if stage:
                record["stage"] = stage

            if total_images is not None and total_images > 0:
                record["total_images"] = total_images

            record["images_completed"] = min(completed_images, record["total_images"])
            record["images_remaining"] = max(0, record["total_images"] - record["images_completed"])
            record["percentage"] = round(
                (record["images_completed"] / record["total_images"]) * 100.0,
                1,
            )

            if status:
                record["status"] = status
            elif record["status"] == "preparing" and record["images_completed"] > 0:
                record["status"] = "running"

            now = time.time()
            elapsed = max(0.01, now - record["start_time"])
            record["elapsed_seconds"] = round(elapsed, 2)

            # Compute processing speed and remaining time after an observation interval
            if elapsed >= 0.5 and record["images_completed"] > 0:
                speed = round(record["images_completed"] / elapsed, 2)
                record["processing_speed_fps"] = speed
                if speed > 0:
                    record["estimated_remaining_seconds"] = round(
                        record["images_remaining"] / speed,
                        1,
                    )
                else:
                    record["estimated_remaining_seconds"] = None
            else:
                record["processing_speed_fps"] = None
                record["estimated_remaining_seconds"] = None

            return dict(record)

    def complete_job(
        self,
        run_id: int,
        status: str = "completed",
        error_message: Optional[str] = None,
        warnings: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a job as finished, setting final counts, warnings, and terminal status."""
        with self._lock:
            record = self._jobs.get(run_id)
            now = time.time()
            if not record:
                record = {
                    "run_id": run_id,
                    "status": status,
                    "images_completed": 1,
                    "total_images": 1,
                    "images_remaining": 0,
                    "percentage": 100.0 if status == "completed" else 0.0,
                    "start_time": now,
                    "elapsed_seconds": 0.0,
                    "estimated_remaining_seconds": 0.0 if status == "completed" else None,
                    "processing_speed_fps": None,
                    "error_message": error_message,
                    "stage": "performance_benchmarking",
                    "warnings": list(warnings) if warnings else [],
                }
                self._jobs[run_id] = record
                return dict(record)

            record["status"] = status
            record["error_message"] = error_message
            if warnings is not None:
                record["warnings"] = list(warnings)
            elif "warnings" not in record:
                record["warnings"] = []

            elapsed = max(0.01, now - record["start_time"])
            record["elapsed_seconds"] = round(elapsed, 2)

            if status == "completed":
                record["images_completed"] = record["total_images"]
                record["images_remaining"] = 0
                record["percentage"] = 100.0
                record["estimated_remaining_seconds"] = 0.0
                if elapsed > 0:
                    record["processing_speed_fps"] = round(record["total_images"] / elapsed, 2)
            elif status == "failed":
                record["estimated_remaining_seconds"] = None

            return dict(record)

    def fail_job(
        self,
        run_id: int,
        error_message: str,
        warnings: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a job as failed with an error message and optional warnings."""
        return self.complete_job(run_id, status="failed", error_message=error_message, warnings=warnings)

    def get_progress(
        self,
        run_id: int,
        session: Optional[Session] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the current progress for a given run ID."""
        with self._lock:
            if run_id in self._jobs:
                rec = dict(self._jobs[run_id])
                rec.setdefault("warnings", [])
                if rec["status"] == "running":
                    elapsed = max(0.01, time.time() - rec["start_time"])
                    rec["elapsed_seconds"] = round(elapsed, 2)
                    if elapsed >= 0.5 and rec["images_completed"] > 0:
                        speed = round(rec["images_completed"] / elapsed, 2)
                        rec["processing_speed_fps"] = speed
                        if speed > 0:
                            rec["estimated_remaining_seconds"] = round(
                                rec["images_remaining"] / speed,
                                1,
                            )
                return rec

        # Fallback to database lookup if session is provided
        if session is not None:
            db_run = get_benchmark_run(session, run_id)
            if db_run:
                is_completed = db_run.status == "completed"
                warnings = (
                    db_run.environment_info.get("warnings", [])
                    if isinstance(db_run.environment_info, dict)
                    else []
                )
                return {
                    "run_id": db_run.id,
                    "status": db_run.status,
                    "images_completed": 1 if is_completed else 0,
                    "total_images": 1,
                    "images_remaining": 0 if is_completed else 1,
                    "percentage": 100.0 if is_completed else 0.0,
                    "start_time": None,
                    "elapsed_seconds": None,
                    "estimated_remaining_seconds": 0.0 if is_completed else None,
                    "processing_speed_fps": None,
                    "error_message": db_run.error_message,
                    "warnings": warnings,
                }

        return None


# Global singleton instance for in-process background job monitoring
progress_tracker = BenchmarkProgressTracker()
