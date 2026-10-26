"""
Environment Verification Test for Step 1.

Automated Image Model Benchmarking for Edge Deployment.

Verifies:
- Python 3.11.x runtime
- PyTorch, Torchvision, NumPy, and Pillow imports and versions
- Apple Silicon MPS (Metal Performance Shaders) availability
- Loading MobileNetV3-Small via modern torchvision weights API
- Obtaining official preprocessing transform
- Generating a deterministic dummy RGB image
- Transforming image into input tensor [1, 3, 224, 224]
- Executing a single forward inference pass on the detected device
- Asserting 1000 output classification logits
"""

import platform
import sys


def main() -> int:
    print("=" * 60)
    print("STEP 1: ENVIRONMENT & RUNTIME VERIFICATION")
    print("=" * 60)

    # 1. Python version verification
    py_version = platform.python_version()
    major, minor, micro = sys.version_info[:3]
    if (major, minor) != (3, 11):
        print(f"[FAIL] Incompatible Python version: {py_version} (Expected 3.11.x)")
        return 1
    print(f"[*] Python Version:             {py_version} (OK)")

    # 2. Dependency imports & versions
    try:
        import numpy as np
        print(f"[*] NumPy Version:              {np.__version__}")
    except ImportError as exc:
        print(f"[FAIL] Could not import NumPy: {exc}")
        return 1

    try:
        import PIL
        from PIL import Image
        print(f"[*] Pillow Version:             {PIL.__version__}")
    except ImportError as exc:
        print(f"[FAIL] Could not import Pillow: {exc}")
        return 1

    try:
        import torch
        print(f"[*] PyTorch Version:            {torch.__version__}")
    except ImportError as exc:
        print(f"[FAIL] Could not import PyTorch: {exc}")
        return 1

    try:
        import torchvision
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        print(f"[*] Torchvision Version:        {torchvision.__version__}")
    except ImportError as exc:
        print(f"[FAIL] Could not import Torchvision: {exc}")
        return 1

    # 3. Execution Device & Apple Silicon MPS Detection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS (Metal Performance Shaders)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"[*] Execution Device:           {device_name}")

    # 4. Model Loading via Modern torchvision weights API (no pretrained=True)
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.to(device)
        model.eval()
        print(f"[*] MobileNetV3 Model Loading:  Loaded successfully ({weights})")
    except Exception as exc:
        print(f"[FAIL] Failed to load MobileNetV3-Small: {exc}")
        return 1

    # 5. Preprocessing transform from official weights
    try:
        preprocess = weights.transforms()
        print(f"[*] Preprocessing Transform:    {type(preprocess).__name__} obtained from weights")
    except Exception as exc:
        print(f"[FAIL] Failed to obtain preprocessing transform: {exc}")
        return 1

    # 6. Deterministic Dummy Image Creation
    # Use fixed seed and deterministic uniform pixel values
    np.random.seed(42)
    dummy_pixels = np.full((224, 224, 3), fill_value=128, dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_pixels, mode="RGB")

    # 7. Transform to Tensor and add batch dimension
    try:
        input_tensor = preprocess(dummy_image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        print(f"[*] Input Tensor Shape:         {list(input_tensor.shape)}")
    except Exception as exc:
        print(f"[FAIL] Preprocessing transform failed: {exc}")
        return 1

    # 8. Single Forward Inference Pass (No benchmarking, no latency/memory timing)
    try:
        with torch.no_grad():
            output_tensor = model(input_tensor.to(device))
        print(f"[*] Output Tensor Shape:        {list(output_tensor.shape)}")
    except Exception as exc:
        print(f"[FAIL] Forward inference pass failed: {exc}")
        return 1

    # 9. Output shape and class count validation
    if output_tensor.shape != torch.Size([1, 1000]):
        print(f"[FAIL] Unexpected output shape: {output_tensor.shape} (Expected [1, 1000])")
        return 1
    num_classes = output_tensor.shape[-1]
    print(f"[*] Output Classes Verified:    {num_classes} classes (ImageNet-1K)")

    # 10. Final Verification Status
    print("=" * 60)
    print("FINAL VERIFICATION STATUS:      PASSED")
    print("=" * 60)
    return 0


def test_environment() -> None:
    """Pytest test case executing full Step 1 environment verification."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
