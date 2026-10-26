"""
ONNX Conversion and Verification Module for Edge Deployment.

Provides automated export of pretrained PyTorch computer vision models from the
model registry into ONNX format and runtime verification using ONNX Runtime.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings
import numpy as np
import onnxruntime as ort
import torch

from models.model_loader import load_model
from models.model_registry import (
    is_model_supported,
    list_supported_models,
)
from models.model_verifier import create_deterministic_dummy_image

DEFAULT_ONNX_OUTPUT_DIR = Path("results/conversions/onnx")
DEFAULT_OPSET_VERSION = 17


@dataclass
class ONNXConversionResult:
    """Structured result of an ONNX conversion and verification workflow."""

    model_id: str
    output_path: str
    conversion_success: bool
    verification_success: bool
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    opset_version: Optional[int] = None
    error_message: Optional[str] = None


def resolve_onnx_output_path(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve destination path for an ONNX export.

    Args:
        model_id: Identifier of the model.
        output_path: Optional user-specified destination path.

    Returns:
        Resolved Path instance.
    """
    if output_path is not None:
        target = Path(output_path)
    else:
        target = DEFAULT_ONNX_OUTPUT_DIR / f"{model_id}.onnx"

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def convert_to_onnx(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    """Export a supported PyTorch model from the registry to ONNX format.

    Loads the model in evaluation mode on CPU, generates a deterministic dummy input
    tensor [1, 3, 224, 224] using the official torchvision preprocessing transforms,
    and serializes the computational graph to the specified ONNX opset.

    Args:
        model_id: Identifier of the registered model to export.
        output_path: Optional destination path. Defaults to results/conversions/onnx/{model_id}.onnx.
        opset_version: Target ONNX opset version (default: 17).

    Returns:
        Resolved Path to the saved .onnx file.

    Raises:
        ValueError: If model_id is not registered.
        RuntimeError: If ONNX serialization fails.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    dest_path = resolve_onnx_output_path(model_id, output_path)

    try:
        # Load model on CPU in evaluation mode for export
        loaded = load_model(model_id, device="cpu")

        # Create deterministic dummy input using official model preprocessing
        dummy_img = create_deterministic_dummy_image(size=(224, 224), seed=42)
        dummy_tensor = loaded.preprocess(dummy_img).unsqueeze(0).to(torch.float32)

        # Export to ONNX using legacy TorchScript exporter for compatibility
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            torch.onnx.export(
                loaded.model,
                dummy_tensor,
                str(dest_path),
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
                opset_version=opset_version,
                dynamo=False,
            )

        if not dest_path.exists() or dest_path.stat().st_size == 0:
            raise RuntimeError(f"Export completed but output file is missing or empty: {dest_path}")

        return dest_path

    except Exception as exc:
        raise RuntimeError(f"Failed to export '{model_id}' to ONNX: {exc}") from exc


def verify_onnx(
    onnx_path: Union[str, Path],
    model_id: Optional[str] = None,
) -> ONNXConversionResult:
    """Verify an ONNX model file using ONNX Runtime.

    Confirms file existence, creates an InferenceSession, inspects input/output
    metadata, executes a forward inference pass using a deterministic dummy input,
    and asserts the output shape has 1000 classes ([1, 1000]).

    Args:
        onnx_path: Path to the .onnx file to verify.
        model_id: Optional logical model identifier.

    Returns:
        Structured ONNXConversionResult.
    """
    resolved_path = Path(onnx_path)
    assigned_model_id = model_id or resolved_path.stem

    if not resolved_path.exists():
        return ONNXConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=False,
            verification_success=False,
            error_message=f"ONNX file does not exist: {resolved_path}",
        )

    try:
        session = ort.InferenceSession(str(resolved_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if not inputs:
            return ONNXConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                error_message="ONNX model has no input tensors defined.",
            )

        input_name = inputs[0].name
        output_name = outputs[0].name

        # Create deterministic dummy input array [1, 3, 224, 224]
        rng = np.random.default_rng(42)
        dummy_input = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

        # Run inference
        ort_outputs = session.run([output_name], {input_name: dummy_input})
        output_tensor = ort_outputs[0]
        if not isinstance(output_tensor, np.ndarray):
            return ONNXConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                error_message=f"Expected numpy ndarray output, got {type(output_tensor).__name__}",
            )

        actual_input_shape = tuple(dummy_input.shape)
        actual_output_shape = tuple(output_tensor.shape)

        if actual_output_shape != (1, 1000):
            return ONNXConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                input_shape=actual_input_shape,
                output_shape=actual_output_shape,
                error_message=f"Expected output shape (1, 1000), got {actual_output_shape}",
            )

        return ONNXConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=True,
            verification_success=True,
            input_shape=actual_input_shape,
            output_shape=actual_output_shape,
            error_message=None,
        )

    except Exception as exc:
        return ONNXConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=True,
            verification_success=False,
            error_message=f"ONNX Runtime inference failed: {exc}",
        )


def convert_and_verify_onnx(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> ONNXConversionResult:
    """Convert a registered model to ONNX and verify it with ONNX Runtime.

    Args:
        model_id: Identifier of the supported model.
        output_path: Optional custom output destination.
        opset_version: Target ONNX opset version (default: 17).

    Returns:
        Structured ONNXConversionResult containing status and shapes.

    Raises:
        ValueError: If model_id is not registered.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    dest_path = resolve_onnx_output_path(model_id, output_path)

    try:
        exported_path = convert_to_onnx(
            model_id=model_id,
            output_path=dest_path,
            opset_version=opset_version,
        )
    except Exception as exc:
        return ONNXConversionResult(
            model_id=model_id,
            output_path=str(dest_path),
            conversion_success=False,
            verification_success=False,
            opset_version=opset_version,
            error_message=str(exc),
        )

    result = verify_onnx(exported_path, model_id=model_id)
    result.opset_version = opset_version
    return result
