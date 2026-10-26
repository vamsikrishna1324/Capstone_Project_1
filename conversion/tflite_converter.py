"""
TFLite Conversion and Verification Module for Edge Deployment.

Provides automated conversion of supported image classification models into
TensorFlow Lite (.tflite) format via ONNX and runtime verification using the
TensorFlow Lite Interpreter.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import tensorflow as tf

from conversion.onnx_converter import convert_to_onnx
from models.model_registry import (
    is_model_supported,
    list_supported_models,
)

DEFAULT_TFLITE_OUTPUT_DIR = Path("results/conversions/tflite")
DEFAULT_ONNX_SOURCE_DIR = Path("results/conversions/onnx")


def _patch_onnx2tf_test_data() -> None:
    """Patch onnx2tf dummy test image downloader to avoid external HTTP requests and unpickling bugs.

    onnx2tf downloads sample test data from an external S3 URL and unpickles it with np.load.
    Under NumPy 2.x and offline environments, this can raise UnpicklingError or network timeouts.
    Mocking this with deterministic local zero arrays ensures fully offline, reproducible execution.
    """
    try:
        import onnx2tf.onnx2tf as o2t_main
        import onnx2tf.utils.common_functions as o2t_cf

        dummy_data = np.zeros((20, 224, 224, 3), dtype=np.float32)
        o2t_main.download_test_image_data = lambda: dummy_data
        o2t_cf.download_test_image_data = lambda: dummy_data
    except (ImportError, AttributeError):
        pass


@dataclass
class TFLiteConversionResult:
    """Structured result of a TFLite conversion and verification workflow."""

    model_id: str
    output_path: str
    conversion_success: bool
    verification_success: bool
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    error_message: Optional[str] = None


def resolve_tflite_output_path(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve destination path for a TFLite export.

    Args:
        model_id: Identifier of the model.
        output_path: Optional user-specified destination path.

    Returns:
        Resolved Path instance with parent directories created.
    """
    if output_path is not None:
        target = Path(output_path)
    else:
        target = DEFAULT_TFLITE_OUTPUT_DIR / f"{model_id}.tflite"

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def convert_to_tflite(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
    onnx_source_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert a supported model to TFLite format via ONNX and onnx2tf.

    Reuses an existing verified ONNX export if present at results/conversions/onnx/{model_id}.onnx
    or exports one automatically if missing.

    Args:
        model_id: Identifier of the registered model to convert.
        output_path: Optional destination path for the .tflite flatbuffer.
                     Defaults to results/conversions/tflite/{model_id}.tflite.
        onnx_source_path: Optional path to an existing ONNX model file.

    Returns:
        Resolved Path to the saved .tflite file.

    Raises:
        ValueError: If model_id is not registered.
        RuntimeError: If ONNX conversion or TFLite translation fails.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    dest_path = resolve_tflite_output_path(model_id, output_path)

    # 1. Resolve ONNX source file
    if onnx_source_path is not None:
        onnx_path = Path(onnx_source_path)
        if not onnx_path.exists():
            raise RuntimeError(f"Specified ONNX source model does not exist: {onnx_path}")
    else:
        default_onnx = DEFAULT_ONNX_SOURCE_DIR / f"{model_id}.onnx"
        if default_onnx.exists():
            onnx_path = default_onnx
        else:
            onnx_path = convert_to_onnx(model_id, output_path=default_onnx)

    # 2. Convert ONNX to TFLite via onnx2tf in a temporary working folder
    _patch_onnx2tf_test_data()

    try:
        import onnx2tf.onnx2tf as o2t_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            o2t_main.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=tmp_dir,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True,
            )

            # Find the generated .tflite file in the output directory
            candidates = list(Path(tmp_dir).glob("*.tflite"))
            if not candidates:
                raise RuntimeError(
                    f"onnx2tf completed but no .tflite file was found in staging directory {tmp_dir}"
                )

            # Prefer the float32 file or model-named file
            chosen_file = candidates[0]
            for cand in candidates:
                if "float32" in cand.name or model_id in cand.name:
                    chosen_file = cand
                    break

            shutil.copy2(chosen_file, dest_path)

        if not dest_path.exists() or dest_path.stat().st_size == 0:
            raise RuntimeError(
                f"TFLite conversion completed but output file is missing or empty: {dest_path}"
            )

        return dest_path

    except Exception as exc:
        raise RuntimeError(f"Failed to convert '{model_id}' to TFLite: {exc}") from exc


def verify_tflite(
    tflite_path: Union[str, Path],
    model_id: Optional[str] = None,
) -> TFLiteConversionResult:
    """Verify a TFLite model file using the TensorFlow Lite Interpreter.

    Confirms file existence, creates an Interpreter, allocates tensors,
    inspects input/output shapes and dtypes, executes a forward inference
    pass using deterministic dummy input, and asserts the output shape has 1000 classes ([1, 1000]).

    Args:
        tflite_path: Path to the .tflite file to verify.
        model_id: Optional logical model identifier.

    Returns:
        Structured TFLiteConversionResult.
    """
    resolved_path = Path(tflite_path)
    assigned_model_id = model_id or resolved_path.stem.replace(".tflite", "")

    if not resolved_path.exists():
        return TFLiteConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=False,
            verification_success=False,
            error_message=f"TFLite file does not exist: {resolved_path}",
        )

    try:
        # Load interpreter with deprecation warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            interpreter = tf.lite.Interpreter(model_path=str(resolved_path))
            interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()

        if not in_details:
            return TFLiteConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                error_message="TFLite model has no input tensors defined.",
            )

        if not out_details:
            return TFLiteConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                error_message="TFLite model has no output tensors defined.",
            )

        primary_in = in_details[0]
        primary_out = out_details[0]

        in_shape = tuple(int(x) for x in primary_in["shape"])
        out_shape = tuple(int(x) for x in primary_out["shape"])
        in_dtype = np.dtype(primary_in["dtype"]).name
        out_dtype = np.dtype(primary_out["dtype"]).name

        # Create deterministic dummy input tensor
        rng = np.random.default_rng(42)
        if np.issubdtype(primary_in["dtype"], np.floating):
            dummy_input = rng.standard_normal(in_shape).astype(primary_in["dtype"])
        else:
            dummy_input = np.ones(in_shape, dtype=primary_in["dtype"])

        # Execute inference
        interpreter.set_tensor(primary_in["index"], dummy_input)
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(primary_out["index"])

        if not isinstance(output_tensor, np.ndarray):
            return TFLiteConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                input_shape=in_shape,
                output_shape=out_shape,
                input_dtype=in_dtype,
                output_dtype=out_dtype,
                error_message=f"Expected numpy ndarray output, got {type(output_tensor).__name__}",
            )

        actual_output_shape = tuple(int(x) for x in output_tensor.shape)

        if actual_output_shape != (1, 1000):
            return TFLiteConversionResult(
                model_id=assigned_model_id,
                output_path=str(resolved_path),
                conversion_success=True,
                verification_success=False,
                input_shape=in_shape,
                output_shape=actual_output_shape,
                input_dtype=in_dtype,
                output_dtype=out_dtype,
                error_message=f"Expected output shape (1, 1000), got {actual_output_shape}",
            )

        return TFLiteConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=True,
            verification_success=True,
            input_shape=in_shape,
            output_shape=actual_output_shape,
            input_dtype=in_dtype,
            output_dtype=out_dtype,
            error_message=None,
        )

    except Exception as exc:
        return TFLiteConversionResult(
            model_id=assigned_model_id,
            output_path=str(resolved_path),
            conversion_success=True,
            verification_success=False,
            error_message=f"TFLite Interpreter inference failed: {exc}",
        )


def convert_and_verify_tflite(
    model_id: str,
    output_path: Optional[Union[str, Path]] = None,
    onnx_source_path: Optional[Union[str, Path]] = None,
) -> TFLiteConversionResult:
    """Convert a registered model to TFLite and verify it with the TFLite Interpreter.

    Args:
        model_id: Identifier of the supported model.
        output_path: Optional custom output destination.
        onnx_source_path: Optional custom ONNX model input path.

    Returns:
        Structured TFLiteConversionResult containing status, shapes, and diagnostics.

    Raises:
        ValueError: If model_id is not registered.
    """
    if not is_model_supported(model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{model_id}'. Supported models: {supported}")

    dest_path = resolve_tflite_output_path(model_id, output_path)

    try:
        exported_path = convert_to_tflite(
            model_id=model_id,
            output_path=dest_path,
            onnx_source_path=onnx_source_path,
        )
    except Exception as exc:
        return TFLiteConversionResult(
            model_id=model_id,
            output_path=str(dest_path),
            conversion_success=False,
            verification_success=False,
            error_message=str(exc),
        )

    return verify_tflite(exported_path, model_id=model_id)
