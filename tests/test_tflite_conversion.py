"""
Test Suite for Step 3B: PyTorch -> TFLite Model Conversion and Verification.

Verifies:
- All 4 supported models can be converted to TFLite format via ONNX.
- Output .tflite flatbuffer files exist, are non-empty, and reside in the expected directory.
- TensorFlow Lite Interpreter loads the exported files.
- Deterministic forward inference succeeds via TFLite Interpreter.
- Output classification tensors match [1, 1000] logits.
- Invalid model IDs are rejected with clear ValueError exceptions.
- Verification of nonexistent or malformed files produces structured error results.
"""

from pathlib import Path
from typing import List
import pytest

from conversion.tflite_converter import (
    DEFAULT_TFLITE_OUTPUT_DIR,
    TFLiteConversionResult,
    convert_and_verify_tflite,
    convert_to_tflite,
    verify_tflite,
)
from models.model_registry import list_supported_models

SUPPORTED_MODELS: List[str] = [
    "mobilenet_v3_small",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
]


# ============================================================================
# 1. Conversion & File Creation Tests
# ============================================================================
@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_tflite_export_file_creation(model_id: str, tmp_path: Path) -> None:
    """Verify that each supported model converts to a valid non-empty .tflite file."""
    out_file = tmp_path / f"{model_id}.tflite"
    exported_path = convert_to_tflite(model_id, output_path=out_file)

    assert exported_path.exists(), f"Converted file does not exist: {exported_path}"
    assert exported_path.is_file(), f"Converted path is not a file: {exported_path}"
    assert exported_path.stat().st_size > 1024, f"File unexpectedly small: {exported_path.stat().st_size} bytes"


# ============================================================================
# 2. End-to-End Conversion and Verification
# ============================================================================
@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_tflite_conversion_and_verification(model_id: str, tmp_path: Path) -> None:
    """Verify that convert_and_verify_tflite converts and verifies each model successfully."""
    out_file = tmp_path / f"{model_id}.tflite"
    result: TFLiteConversionResult = convert_and_verify_tflite(model_id, output_path=out_file)

    assert result.conversion_success is True, f"Conversion failed: {result.error_message}"
    assert result.verification_success is True, f"Verification failed: {result.error_message}"
    assert result.error_message is None
    assert result.model_id == model_id
    assert result.output_shape == (1, 1000)
    assert Path(result.output_path).exists()


# ============================================================================
# 3. Default Results Directory Export
# ============================================================================
def test_tflite_default_results_directory() -> None:
    """Verify that default export destination results/conversions/tflite/ works as expected."""
    model_id = "mobilenet_v3_small"
    result = convert_and_verify_tflite(model_id)

    expected_path = DEFAULT_TFLITE_OUTPUT_DIR / f"{model_id}.tflite"
    assert result.conversion_success is True
    assert result.verification_success is True
    assert Path(result.output_path).resolve() == expected_path.resolve()
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


# ============================================================================
# 4. Error Handling: Invalid Model IDs
# ============================================================================
@pytest.mark.parametrize("invalid_id", ["unknown_model", "vgg16", ""])
def test_tflite_invalid_model_id(invalid_id: str) -> None:
    """Verify that attempting to convert an unsupported model ID raises a controlled ValueError."""
    with pytest.raises(ValueError, match="Unsupported model ID"):
        convert_to_tflite(invalid_id)

    with pytest.raises(ValueError, match="Unsupported model ID"):
        convert_and_verify_tflite(invalid_id)


# ============================================================================
# 5. Error Handling: Missing TFLite File Verification
# ============================================================================
def test_tflite_verify_nonexistent_file(tmp_path: Path) -> None:
    """Verify that verify_tflite gracefully returns structured failure for missing files."""
    missing_path = tmp_path / "does_not_exist.tflite"
    result = verify_tflite(missing_path, model_id="test_missing")

    assert result.conversion_success is False
    assert result.verification_success is False
    assert result.error_message is not None
    assert "does not exist" in result.error_message.lower()
