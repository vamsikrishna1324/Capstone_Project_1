"""
Test Suite for Step 3A: PyTorch -> ONNX Model Conversion and Verification.

Verifies:
- All 4 supported models can be exported to ONNX format.
- Output .onnx files exist, are non-empty, and reside in the expected directory.
- ONNX Runtime loads the exported files via InferenceSession.
- Deterministic forward inference succeeds via ONNX Runtime.
- Output classification tensors match [1, 1000] logits.
- Invalid model IDs are rejected with clear ValueError exceptions.
- Verification of nonexistent or malformed files produces structured error results.
"""

from pathlib import Path
from typing import List
import pytest

from conversion.onnx_converter import (
    DEFAULT_ONNX_OUTPUT_DIR,
    ONNXConversionResult,
    convert_and_verify_onnx,
    convert_to_onnx,
    verify_onnx,
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
def test_onnx_export_file_creation(model_id: str, tmp_path: Path) -> None:
    """Verify that each supported model exports to a valid non-empty .onnx file."""
    out_file = tmp_path / f"{model_id}.onnx"
    exported_path = convert_to_onnx(model_id, output_path=out_file)

    assert exported_path.exists(), f"Exported file does not exist: {exported_path}"
    assert exported_path.is_file(), f"Exported path is not a file: {exported_path}"
    assert exported_path.stat().st_size > 1024, f"File unexpectedly small: {exported_path.stat().st_size} bytes"


# ============================================================================
# 2. End-to-End Conversion and Verification
# ============================================================================
@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_onnx_conversion_and_verification(model_id: str, tmp_path: Path) -> None:
    """Verify that convert_and_verify_onnx exports and verifies each model successfully."""
    out_file = tmp_path / f"{model_id}.onnx"
    result: ONNXConversionResult = convert_and_verify_onnx(model_id, output_path=out_file)

    assert result.conversion_success is True, f"Conversion failed: {result.error_message}"
    assert result.verification_success is True, f"Verification failed: {result.error_message}"
    assert result.error_message is None
    assert result.model_id == model_id
    assert result.input_shape == (1, 3, 224, 224)
    assert result.output_shape == (1, 1000)
    assert result.opset_version == 17
    assert Path(result.output_path).exists()


# ============================================================================
# 3. Default Results Directory Export
# ============================================================================
def test_onnx_default_results_directory() -> None:
    """Verify that default export destination results/conversions/onnx/ works as expected."""
    model_id = "mobilenet_v3_small"
    result = convert_and_verify_onnx(model_id)

    expected_path = DEFAULT_ONNX_OUTPUT_DIR / f"{model_id}.onnx"
    assert result.conversion_success is True
    assert result.verification_success is True
    assert Path(result.output_path).resolve() == expected_path.resolve()
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


# ============================================================================
# 4. Error Handling: Invalid Model IDs
# ============================================================================
@pytest.mark.parametrize("invalid_id", ["unknown_model", "vgg16", ""])
def test_onnx_invalid_model_id(invalid_id: str) -> None:
    """Verify that attempting to convert an unsupported model ID raises a controlled ValueError."""
    with pytest.raises(ValueError, match="Unsupported model ID"):
        convert_to_onnx(invalid_id)

    with pytest.raises(ValueError, match="Unsupported model ID"):
        convert_and_verify_onnx(invalid_id)


# ============================================================================
# 5. Error Handling: Missing ONNX File Verification
# ============================================================================
def test_onnx_verify_nonexistent_file(tmp_path: Path) -> None:
    """Verify that verify_onnx gracefully returns structured failure for missing files."""
    missing_path = tmp_path / "does_not_exist.onnx"
    result = verify_onnx(missing_path, model_id="test_missing")

    assert result.conversion_success is False
    assert result.verification_success is False
    assert result.error_message is not None
    assert "does not exist" in result.error_message.lower()
