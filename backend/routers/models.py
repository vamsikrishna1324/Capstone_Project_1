"""
Supported Models and Runtimes Catalog Router.

Returns the four registered computer vision models, their supported runtimes,
and checks on-disk serialized artifacts (ONNX and TFLite) with file sizes.
"""

from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, status

from backend.schemas import ArtifactDetail, ModelInfo, ModelsResponse
from backend.services.accuracy_service import is_saved_accuracy_available
from benchmarking.config import SUPPORTED_RUNTIMES
from models.model_registry import get_model_metadata, list_supported_models

router = APIRouter(tags=["Models"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _inspect_artifact(path_str: str, fmt: str, model_id: Optional[str] = None) -> ArtifactDetail:
    """Inspect file existence and size for a serialized model artifact."""
    p = Path(path_str)
    target_path = p if p.is_file() else (PROJECT_ROOT / path_str)

    has_acc = is_saved_accuracy_available(model_id, fmt) if model_id else None
    if target_path.is_file():
        size_bytes = target_path.stat().st_size
        size_mb = round(size_bytes / (1024.0 * 1024.0), 2)
        return ArtifactDetail(
            format=fmt,
            exists=True,
            path=path_str,
            file_size_bytes=size_bytes,
            file_size_mb=size_mb,
            has_precomputed_accuracy=has_acc,
        )
    return ArtifactDetail(
        format=fmt,
        exists=False,
        path=path_str,
        file_size_bytes=None,
        file_size_mb=None,
        has_precomputed_accuracy=has_acc,
    )



@router.get(
    "/models",
    response_model=ModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="List Supported Models and Runtimes",
    description="Retrieve registered models, input requirements, supported runtimes, and disk artifact statuses.",
)
def get_supported_models() -> ModelsResponse:
    """Return catalog of all supported models, runtimes, and verified artifacts."""
    model_ids = list_supported_models()
    models_info = []

    for mid in model_ids:
        meta = get_model_metadata(mid)

        artifacts: Dict[str, ArtifactDetail] = {
            "pytorch": ArtifactDetail(
                format="pytorch",
                exists=True,
                path=None,
                file_size_bytes=None,
                file_size_mb=None,
                has_precomputed_accuracy=is_saved_accuracy_available(mid, "pytorch"),
            ),
            "onnx": _inspect_artifact(f"results/conversions/onnx/{mid}.onnx", "onnx", mid),
            "tflite": _inspect_artifact(f"results/conversions/tflite/{mid}.tflite", "tflite", mid),
        }


        models_info.append(
            ModelInfo(
                model_id=meta.model_id,
                display_name=meta.display_name,
                architecture_name=meta.architecture_name,
                torchvision_model_name=meta.torchvision_model_name,
                weights_identifier=meta.weights_identifier,
                num_classes=meta.num_classes,
                expected_input_size=list(meta.expected_input_size),
                supported_runtimes=list(SUPPORTED_RUNTIMES),
                artifacts=artifacts,
            )
        )

    return ModelsResponse(models=models_info, total_models=len(models_info))
