"""
Model Size Benchmarking Module for Edge Deployment.

Measures actual disk storage footprints for serialized model artifacts (ONNX, TFLite,
PyTorch) and calculates theoretical parameter volumes directly from official PyTorch architectures.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from benchmarking.config import BenchmarkRuntime
from benchmarking.results import ModelSizeMetrics
from models.model_loader import load_model
from models.model_registry import (
    is_model_supported,
    list_supported_models,
)

DEFAULT_ONNX_DIR = Path("results/conversions/onnx")
DEFAULT_TFLITE_DIR = Path("results/conversions/tflite")
DEFAULT_PYTORCH_DIR = Path("results/conversions/pytorch")

# In-memory cache for computed parameter counts to avoid repeated model loading
_PARAM_COUNT_CACHE: Dict[str, int] = {}


def bytes_to_mb(file_size_bytes: int) -> float:
    """Convert a byte count to binary megabytes (MiB), rounded to two decimal places.

    Args:
        file_size_bytes: Size in integer bytes.

    Returns:
        Size in megabytes as a float rounded to 2 decimal places.
    """
    return round(file_size_bytes / (1024.0 * 1024.0), 2)


def get_model_parameter_count(model_id: str) -> int:
    """Compute the total parameter count from the PyTorch model definition.

    Args:
        model_id: Logical identifier of the registered model.

    Returns:
        Total number of parameters across all layers.

    Raises:
        ValueError: If model_id is not registered.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    if model_id not in _PARAM_COUNT_CACHE:
        loaded = load_model(model_id, device="cpu")
        _PARAM_COUNT_CACHE[model_id] = sum(p.numel() for p in loaded.model.parameters())

    return _PARAM_COUNT_CACHE[model_id]


def resolve_artifact_path(
    model_id: str,
    runtime: Union[BenchmarkRuntime, str],
    artifact_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Determine the expected disk storage path for a model artifact.

    Args:
        model_id: Registered model identifier.
        runtime: Target runtime / format.
        artifact_path: Optional user-specified path overriding default location.

    Returns:
        Resolved Path to the artifact file.

    Raises:
        ValueError: If model_id or runtime is invalid.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    norm_runtime = BenchmarkRuntime.from_string(runtime)

    if artifact_path is not None:
        return Path(artifact_path)

    if norm_runtime == BenchmarkRuntime.ONNX:
        return DEFAULT_ONNX_DIR / f"{model_id}.onnx"
    elif norm_runtime == BenchmarkRuntime.TFLITE:
        return DEFAULT_TFLITE_DIR / f"{model_id}.tflite"
    elif norm_runtime == BenchmarkRuntime.PYTORCH:
        return DEFAULT_PYTORCH_DIR / f"{model_id}.pt"
    else:
        raise ValueError(f"Unsupported runtime '{norm_runtime}'")


def measure_model_size(
    model_id: str,
    runtime: Union[BenchmarkRuntime, str],
    artifact_path: Optional[Union[str, Path]] = None,
) -> ModelSizeMetrics:
    """Measure actual on-disk model artifact size and compute parameter count.

    - For ONNX and TFLite: inspects the existing serialized files under results/conversions.
    - For PyTorch: calculates parameter count from official torchvision weights. If an on-disk
      serialized checkpoint does not exist, file size is explicitly set to None rather than
      silently inventing a size or creating duplicate files.
    - Parameter count is kept conceptually separate from serialized file size.

    Args:
        model_id: Registered model identifier.
        runtime: Target inference runtime (PyTorch, ONNX, TFLite).
        artifact_path: Optional custom path to the model file on disk.

    Returns:
        ModelSizeMetrics instance containing file_size_bytes, file_size_mb,
        param_count, and artifact_path.

    Raises:
        ValueError: If model_id or runtime is unsupported.
        FileNotFoundError: If a required ONNX/TFLite artifact does not exist on disk,
                           or if a custom artifact_path is not found.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    norm_runtime = BenchmarkRuntime.from_string(runtime)
    param_count = get_model_parameter_count(model_id)
    target_path = resolve_artifact_path(model_id, norm_runtime, artifact_path)

    # 1. ONNX and TFLite: Must exist on disk
    if norm_runtime in (BenchmarkRuntime.ONNX, BenchmarkRuntime.TFLITE):
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(
                f"{norm_runtime.value.upper()} model artifact for '{model_id}' does not exist on disk at: {target_path}. "
                f"Please run {norm_runtime.value.upper()} conversion first."
            )
        size_bytes = target_path.stat().st_size
        size_mb = bytes_to_mb(size_bytes)
        return ModelSizeMetrics(
            file_size_bytes=size_bytes,
            file_size_mb=size_mb,
            param_count=param_count,
            artifact_path=str(target_path),
        )

    # 2. PyTorch: In-memory representation by default unless serialized artifact exists
    if norm_runtime == BenchmarkRuntime.PYTORCH:
        if artifact_path is not None:
            # Custom path was explicitly requested: must exist
            if not target_path.exists() or not target_path.is_file():
                raise FileNotFoundError(
                    f"Specified PyTorch model artifact for '{model_id}' does not exist at: {target_path}"
                )
            size_bytes = target_path.stat().st_size
            size_mb = bytes_to_mb(size_bytes)
            return ModelSizeMetrics(
                file_size_bytes=size_bytes,
                file_size_mb=size_mb,
                param_count=param_count,
                artifact_path=str(target_path),
            )

        # Default PyTorch: check if an on-disk serialized checkpoint exists
        if target_path.exists() and target_path.is_file():
            size_bytes = target_path.stat().st_size
            size_mb = bytes_to_mb(size_bytes)
            return ModelSizeMetrics(
                file_size_bytes=size_bytes,
                file_size_mb=size_mb,
                param_count=param_count,
                artifact_path=str(target_path),
            )

        # PyTorch model exists in-memory from torchvision weights; no on-disk file exists
        return ModelSizeMetrics(
            file_size_bytes=None,
            file_size_mb=None,
            param_count=param_count,
            artifact_path=None,
        )

    raise ValueError(f"Unhandled runtime '{norm_runtime}'")
