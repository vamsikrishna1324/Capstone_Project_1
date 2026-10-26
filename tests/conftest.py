"""
Pytest Global Configuration and Test Environment Isolation.

Ensures that all test executions strictly target the dedicated test database
('edge_model_benchmark_test') with baseline auto-seeding disabled, protecting
the main database ('edge_model_benchmark') from test mutations.
"""

import os
from pathlib import Path
import sys
import pytest

# Ensure project root is on sys.path for test discovery and module imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Enforce test database isolation before any test modules or application code are imported
os.environ["MYSQL_DATABASE"] = os.environ.get("MYSQL_TEST_DATABASE", "edge_model_benchmark_test")
os.environ["SEED_BASELINE_ON_STARTUP"] = "false"

