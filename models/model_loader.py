"""
Model Loader for Automated Image Model Benchmarking.

Provides reusable functionality to load supported pretrained computer vision
models in evaluation mode, configure target execution devices, and retrieve
official preprocessing transforms.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
import torch
import torch.nn as nn

from models.model_registry import (
    ModelMetadata,
    get_model_builder,
    get_model_metadata,
    get_model_weights,
    is_model_supported,
    list_supported_models,
)


@dataclass
class LoadedModel:
    """Encapsulates a loaded PyTorch model, its metadata, weights, and transform."""

    model: nn.Module
    metadata: ModelMetadata
    weights: Any
    preprocess: Callable[..., Any]
    device: torch.device


def get_default_device() -> torch.device:
    """Determine the optimal default execution device.

    Prefers Apple Silicon MPS if available and built, otherwise defaults to CPU.

    Returns:
        Optimal torch.device instance.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve an optional device specification to a valid torch.device.

    Args:
        device: Device string (e.g. 'cpu', 'mps', 'cuda') or torch.device instance.
                If None, detects optimal default device.

    Returns:
        Resolved torch.device instance.

    Raises:
        TypeError: If device argument is of an unsupported type.
    """
    if device is None:
        return get_default_device()
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    raise TypeError(f"Expected str or torch.device, got {type(device).__name__}")


def load_model(
    model_id: str,
    device: Optional[Union[str, torch.device]] = None,
) -> LoadedModel:
    """Load a registered pretrained torchvision model in evaluation mode.

    Args:
        model_id: Identifier of the model to load.
        device: Target execution device. If None, auto-selects MPS (if available) or CPU.

    Returns:
        LoadedModel instance containing the initialized model, metadata, weights,
        official preprocessing transform, and active device.

    Raises:
        ValueError: If model_id is not recognized in the model registry.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    target_device = resolve_device(device)
    metadata = get_model_metadata(model_id)
    weights = get_model_weights(model_id)
    builder = get_model_builder(model_id)

    # Instantiate model with official pretrained weights
    model: nn.Module = builder(weights=weights)

    # Set to evaluation mode
    model.eval()

    # Move to target device
    model.to(target_device)

    # Obtain official preprocessing transforms
    preprocess = weights.transforms()

    return LoadedModel(
        model=model,
        metadata=metadata,
        weights=weights,
        preprocess=preprocess,
        device=target_device,
    )
