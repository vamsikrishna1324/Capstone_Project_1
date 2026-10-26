"""
Saved Accuracy Results Retrieval and Mapping Service.

Facilitates reusing verified 50,000-image ImageNet accuracy evaluation results
across PyTorch, ONNX, and TFLite without incurring redundant multi-hour runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def find_saved_accuracy_file(model_id: str, runtime: str) -> Optional[Path]:
    """Locate existing saved accuracy JSON file for a given model and runtime.

    Checks both the project-root resolved results directory and relative cwd path.

    Args:
        model_id: Logical model identifier (e.g. 'mobilenet_v3_small').
        runtime: Execution runtime ('pytorch', 'onnx', 'tflite').

    Returns:
        Path to existing JSON file, or None if not found.
    """
    runtime_clean = runtime.strip().lower()

    filenames = []
    if runtime_clean == "pytorch":
        filenames = [
            f"accuracy_{model_id}.json",
            f"accuracy_{model_id}_pytorch.json",
        ]
    elif runtime_clean in ("onnx", "onnxruntime"):
        filenames = [
            f"accuracy_{model_id}_onnx.json",
        ]
    elif runtime_clean in ("tflite", "tensorflow_lite", "tensorflow"):
        filenames = [
            f"accuracy_{model_id}_tflite.json",
        ]

    for fname in filenames:
        candidates = [
            RESULTS_DIR / fname,
            Path("results") / fname,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None


def is_saved_accuracy_available(model_id: str, runtime: str) -> bool:
    """Check whether a precomputed accuracy artifact exists on disk."""
    return find_saved_accuracy_file(model_id, runtime) is not None


def get_missing_accuracy_warning(model_id: str, runtime: str) -> str:
    """Return a clear, non-sensitive warning when precomputed accuracy is missing."""
    arch_name = model_id
    try:
        from models.model_registry import get_model_metadata, is_model_supported
        if is_model_supported(model_id):
            meta = get_model_metadata(model_id)
            arch_name = meta.display_name
    except Exception:
        pass
    rt_label = runtime.upper() if runtime else ""
    return (
        f"Verified precomputed accuracy is unavailable for {arch_name} on {rt_label}. "
        f"Top-1 and Top-5 accuracy remain N/A unless fresh live evaluation is performed."
    )


def load_saved_accuracy(model_id: str, runtime: str) -> Optional[Dict[str, Any]]:
    """Load and parse pre-computed classification accuracy metrics.

    Args:
        model_id: Model identifier.
        runtime: Execution runtime.

    Returns:
        Structured accuracy dictionary or None if no matching artifact exists.
    """
    path = find_saved_accuracy_file(model_id, runtime)
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "dataset_path": data.get("manifest_path") or data.get("dataset_path"),
            "total_samples": data.get("total_samples"),
            "evaluated_samples": data.get("evaluated_samples"),
            "skipped_images": data.get("skipped_images", 0),
            "top1_accuracy": data.get("top1_accuracy"),
            "top5_accuracy": data.get("top5_accuracy"),
            "evaluation_time_seconds": data.get("evaluation_time_seconds"),
            "throughput_samples_per_sec": data.get("throughput_samples_per_sec"),
            "accuracy_source": "saved_result",
            "artifact_file": path.name,
        }
    except Exception:
        return None

