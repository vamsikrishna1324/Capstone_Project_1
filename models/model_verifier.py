"""
Model Verifier for Automated Image Model Benchmarking.

Performs structured verification of supported computer vision models:
- Validates model registration and loading
- Obtains official preprocessing transform
- Generates deterministic dummy image
- Transforms image and validates tensor shape
- Executes single forward pass in inference mode
- Verifies output classification logits
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from models.model_loader import LoadedModel, load_model
from models.model_registry import (
    get_model_metadata,
    is_model_supported,
    list_supported_models,
)


@dataclass
class VerificationResult:
    """Structured result of a model verification run."""

    model_id: str
    display_name: str
    success: bool
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    device: Optional[str] = None
    error_message: Optional[str] = None


def create_deterministic_dummy_image(
    size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Image.Image:
    """Generate a reproducible dummy RGB image for verification.

    Args:
        size: (width, height) tuple for the generated image.
        seed: Random seed for deterministic generation.

    Returns:
        PIL.Image instance in RGB mode.
    """
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def verify_model(
    model_id: str,
    device: Optional[Union[str, torch.device]] = None,
) -> VerificationResult:
    """Verify a supported model through loading, preprocessing, and inference.

    Args:
        model_id: Identifier of the model to verify.
        device: Target execution device. If None, auto-selects optimal device.

    Returns:
        Structured VerificationResult.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        return VerificationResult(
            model_id=model_id,
            display_name="Unknown",
            success=False,
            error_message=f"Model ID '{model_id}' is not supported. Supported models: {supported}",
        )

    metadata = get_model_metadata(model_id)

    try:
        loaded: LoadedModel = load_model(model_id, device=device)
    except Exception as exc:
        return VerificationResult(
            model_id=model_id,
            display_name=metadata.display_name,
            success=False,
            error_message=f"Failed to load model: {exc}",
        )

    # Validate preprocessing and deterministic image transformation
    try:
        dummy_img = create_deterministic_dummy_image(size=(224, 224), seed=42)
        transformed_tensor = loaded.preprocess(dummy_img)

        # Ensure batch dimension
        if transformed_tensor.ndim == 3:
            input_tensor = transformed_tensor.unsqueeze(0)
        elif transformed_tensor.ndim == 4:
            input_tensor = transformed_tensor
        else:
            return VerificationResult(
                model_id=model_id,
                display_name=metadata.display_name,
                success=False,
                error_message=f"Unexpected transformed tensor dimensions: {transformed_tensor.shape}",
            )

        if input_tensor.shape[1] != 3:
            return VerificationResult(
                model_id=model_id,
                display_name=metadata.display_name,
                success=False,
                error_message=f"Expected 3 channels, got shape {input_tensor.shape}",
            )

        input_shape = tuple(input_tensor.shape)
    except Exception as exc:
        return VerificationResult(
            model_id=model_id,
            display_name=metadata.display_name,
            success=False,
            error_message=f"Failed during image preprocessing: {exc}",
        )

    # Perform single forward inference pass
    try:
        input_device_tensor = input_tensor.to(loaded.device)
        with torch.no_grad():
            output_tensor = loaded.model(input_device_tensor)

        output_shape = tuple(output_tensor.shape)
        num_classes = output_tensor.shape[-1]

        if output_tensor.shape != torch.Size([1, metadata.num_classes]):
            return VerificationResult(
                model_id=model_id,
                display_name=metadata.display_name,
                success=False,
                input_shape=input_shape,
                output_shape=output_shape,
                num_classes=num_classes,
                device=str(loaded.device),
                error_message=f"Expected output shape [1, {metadata.num_classes}], got {output_shape}",
            )

        return VerificationResult(
            model_id=model_id,
            display_name=metadata.display_name,
            success=True,
            input_shape=input_shape,
            output_shape=output_shape,
            num_classes=num_classes,
            device=str(loaded.device),
            error_message=None,
        )

    except Exception as exc:
        return VerificationResult(
            model_id=model_id,
            display_name=metadata.display_name,
            success=False,
            input_shape=input_shape,
            device=str(loaded.device),
            error_message=f"Forward inference failed: {exc}",
        )


def verify_all_models(
    device: Optional[Union[str, torch.device]] = None,
) -> List[VerificationResult]:
    """Sequentially verify all registered models in the registry.

    Args:
        device: Target execution device. If None, auto-selects optimal device.

    Returns:
        List of VerificationResult for each registered model.
    """
    return [verify_model(model_id, device=device) for model_id in list_supported_models()]
