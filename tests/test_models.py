"""
Step 2 Test Suite: Model Selection & Verification.

Validates:
A. Registry listing: All 4 expected model IDs are present.
B. Registry lookup: Metadata retrieval for each model.
C. Supported model validation: Valid IDs return True, invalid return False.
D. Model loading: Each model loads successfully with official weights.
E. Evaluation mode: Loaded models are in eval mode.
F. Preprocessing: Official preprocessing transforms are callable and produce valid tensors.
G. Verification: Deterministic dummy inference succeeds.
H. Output shape: Each model produces [1, 1000] logits.
I. Invalid model handling: Nonexistent IDs raise clear controlled errors.
"""

from typing import List
import pytest
import torch
import torch.nn as nn

from models.model_loader import LoadedModel, load_model
from models.model_registry import (
    ModelMetadata,
    get_model_metadata,
    get_model_weights,
    is_model_supported,
    list_supported_models,
)
from models.model_verifier import (
    VerificationResult,
    create_deterministic_dummy_image,
    verify_model,
)

EXPECTED_MODEL_IDS: List[str] = [
    "mobilenet_v3_small",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
]


# ============================================================================
# A. Registry Listing
# ============================================================================
def test_registry_listing() -> None:
    """Verify that all four expected model IDs are registered."""
    supported = list_supported_models()
    for model_id in EXPECTED_MODEL_IDS:
        assert model_id in supported, f"Model ID '{model_id}' missing from registry"
    assert len(supported) == len(EXPECTED_MODEL_IDS)


# ============================================================================
# B. Registry Lookup
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_registry_lookup(model_id: str) -> None:
    """Verify metadata retrieval and schema correctness for each registered model."""
    meta: ModelMetadata = get_model_metadata(model_id)
    assert meta.model_id == model_id
    assert len(meta.display_name) > 0
    assert len(meta.architecture_name) > 0
    assert len(meta.torchvision_model_name) > 0
    assert len(meta.weights_identifier) > 0
    assert meta.num_classes == 1000
    assert meta.expected_input_size == (3, 224, 224)
    assert isinstance(meta.preprocessing_info, dict)
    assert "crop_size" in meta.preprocessing_info
    assert "mean" in meta.preprocessing_info
    assert "std" in meta.preprocessing_info


# ============================================================================
# C. Supported Model Validation
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_supported_model_validation_valid(model_id: str) -> None:
    """Verify that is_model_supported returns True for valid model IDs."""
    assert is_model_supported(model_id) is True


@pytest.mark.parametrize("invalid_id", ["", "unknown_model", "mobilenet", "resnet"])
def test_supported_model_validation_invalid(invalid_id: str) -> None:
    """Verify that is_model_supported returns False for invalid model IDs."""
    assert is_model_supported(invalid_id) is False


# ============================================================================
# D. Model Loading
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_model_loading(model_id: str) -> None:
    """Verify that each supported model can be loaded with official pretrained weights."""
    loaded: LoadedModel = load_model(model_id)
    assert loaded is not None
    assert isinstance(loaded.model, nn.Module)
    assert loaded.metadata.model_id == model_id
    assert loaded.weights is not None
    assert loaded.device is not None


# ============================================================================
# E. Evaluation Mode
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_model_evaluation_mode(model_id: str) -> None:
    """Verify that loaded models are explicitly set to evaluation mode."""
    loaded: LoadedModel = load_model(model_id)
    assert loaded.model.training is False


# ============================================================================
# F. Preprocessing
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_model_preprocessing(model_id: str) -> None:
    """Verify that official preprocessing transform is valid and produces [3, 224, 224] tensor."""
    loaded: LoadedModel = load_model(model_id)
    assert callable(loaded.preprocess)

    dummy_img = create_deterministic_dummy_image(size=(224, 224), seed=42)
    transformed = loaded.preprocess(dummy_img)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)
    assert transformed.dtype == torch.float32


# ============================================================================
# G. Verification (Deterministic Dummy Inference)
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_model_verification(model_id: str) -> None:
    """Verify that verify_model succeeds for all registered models."""
    result: VerificationResult = verify_model(model_id)
    assert result.success is True, f"Verification failed for {model_id}: {result.error_message}"
    assert result.error_message is None
    assert result.device is not None


# ============================================================================
# H. Output Shape
# ============================================================================
@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_output_shape(model_id: str) -> None:
    """Verify that output tensor has shape [1, 1000] representing ImageNet-1K logits."""
    result: VerificationResult = verify_model(model_id)
    assert result.input_shape == (1, 3, 224, 224)
    assert result.output_shape == (1, 1000)
    assert result.num_classes == 1000


# ============================================================================
# I. Invalid Model Handling
# ============================================================================
def test_invalid_model_handling() -> None:
    """Verify that requesting an unsupported model ID produces controlled errors."""
    with pytest.raises(ValueError, match="Unsupported model ID"):
        load_model("nonexistent_architecture")

    with pytest.raises(ValueError, match="Unsupported model ID"):
        get_model_metadata("nonexistent_architecture")

    with pytest.raises(ValueError, match="Unsupported model ID"):
        get_model_weights("nonexistent_architecture")

    result = verify_model("nonexistent_architecture")
    assert result.success is False
    assert "not supported" in result.error_message.lower()
