"""
Model Registry for Automated Image Model Benchmarking.

Provides a centralized, single source of truth for all supported pretrained
computer vision models, their architectures, official torchvision identifiers,
and weights specifications.

Supported Models:
- mobilenet_v3_small: MobileNetV3-Small (MobileNet_V3_Small_Weights.DEFAULT)
- resnet50: ResNet50 (ResNet50_Weights.DEFAULT)
- efficientnet_b0: EfficientNet-B0 (EfficientNet_B0_Weights.DEFAULT)
- densenet121: DenseNet121 (DenseNet121_Weights.DEFAULT)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import torchvision.models as tv_models


@dataclass(frozen=True)
class ModelMetadata:
    """Structured metadata describing a registered model."""

    model_id: str
    display_name: str
    architecture_name: str
    torchvision_model_name: str
    weights_identifier: str
    num_classes: int
    expected_input_size: tuple[int, int, int]
    preprocessing_info: Dict[str, Any]


@dataclass(frozen=True)
class _ModelEntry:
    """Internal registry entry binding metadata, builder, and weights."""

    metadata: ModelMetadata
    builder: Callable[..., Any]
    weights: Any


# Centralized registry definitions
_MODEL_REGISTRY: Dict[str, _ModelEntry] = {
    "mobilenet_v3_small": _ModelEntry(
        metadata=ModelMetadata(
            model_id="mobilenet_v3_small",
            display_name="MobileNetV3-Small",
            architecture_name="MobileNetV3",
            torchvision_model_name="mobilenet_v3_small",
            weights_identifier="MobileNet_V3_Small_Weights.DEFAULT",
            num_classes=1000,
            expected_input_size=(3, 224, 224),
            preprocessing_info={
                "crop_size": 224,
                "resize_size": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bilinear",
            },
        ),
        builder=tv_models.mobilenet_v3_small,
        weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT,
    ),
    "resnet50": _ModelEntry(
        metadata=ModelMetadata(
            model_id="resnet50",
            display_name="ResNet50",
            architecture_name="ResNet",
            torchvision_model_name="resnet50",
            weights_identifier="ResNet50_Weights.DEFAULT",
            num_classes=1000,
            expected_input_size=(3, 224, 224),
            preprocessing_info={
                "crop_size": 224,
                "resize_size": 232,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bilinear",
            },
        ),
        builder=tv_models.resnet50,
        weights=tv_models.ResNet50_Weights.DEFAULT,
    ),
    "efficientnet_b0": _ModelEntry(
        metadata=ModelMetadata(
            model_id="efficientnet_b0",
            display_name="EfficientNet-B0",
            architecture_name="EfficientNet",
            torchvision_model_name="efficientnet_b0",
            weights_identifier="EfficientNet_B0_Weights.DEFAULT",
            num_classes=1000,
            expected_input_size=(3, 224, 224),
            preprocessing_info={
                "crop_size": 224,
                "resize_size": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bicubic",
            },
        ),
        builder=tv_models.efficientnet_b0,
        weights=tv_models.EfficientNet_B0_Weights.DEFAULT,
    ),
    "densenet121": _ModelEntry(
        metadata=ModelMetadata(
            model_id="densenet121",
            display_name="DenseNet121",
            architecture_name="DenseNet",
            torchvision_model_name="densenet121",
            weights_identifier="DenseNet121_Weights.DEFAULT",
            num_classes=1000,
            expected_input_size=(3, 224, 224),
            preprocessing_info={
                "crop_size": 224,
                "resize_size": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bilinear",
            },
        ),
        builder=tv_models.densenet121,
        weights=tv_models.DenseNet121_Weights.DEFAULT,
    ),
}


def list_supported_models() -> List[str]:
    """Return a list of all registered model identifiers."""
    return list(_MODEL_REGISTRY.keys())


def is_model_supported(model_id: str) -> bool:
    """Check whether a given model identifier is registered.

    Args:
        model_id: Logical identifier for the model.

    Returns:
        True if the model is registered, False otherwise.
    """
    return model_id in _MODEL_REGISTRY


def get_model_metadata(model_id: str) -> ModelMetadata:
    """Retrieve structured metadata for a registered model.

    Args:
        model_id: Logical identifier for the model.

    Returns:
        ModelMetadata instance containing model specifications.

    Raises:
        ValueError: If model_id is not in the registry.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")
    return _MODEL_REGISTRY[model_id].metadata


def get_model_weights(model_id: str) -> Any:
    """Retrieve the official torchvision weights enum for a model.

    Args:
        model_id: Logical identifier for the model.

    Returns:
        The torchvision weights enum associated with the model.

    Raises:
        ValueError: If model_id is not in the registry.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")
    return _MODEL_REGISTRY[model_id].weights


def get_model_builder(model_id: str) -> Callable[..., Any]:
    """Retrieve the torchvision model builder function for a model.

    Args:
        model_id: Logical identifier for the model.

    Returns:
        Callable function to construct the torchvision model.

    Raises:
        ValueError: If model_id is not in the registry.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")
    return _MODEL_REGISTRY[model_id].builder
