"""
Model Conversion Package.

Provides conversion pipelines from PyTorch models into edge-deployment formats.
Step 3A implements PyTorch -> ONNX conversion and runtime verification.
Step 3B implements PyTorch -> TFLite conversion and runtime verification.
"""

from conversion.onnx_converter import (
    ONNXConversionResult,
    convert_and_verify_onnx,
    convert_to_onnx,
    verify_onnx,
)
from conversion.tflite_converter import (
    TFLiteConversionResult,
    convert_and_verify_tflite,
    convert_to_tflite,
    verify_tflite,
)

__all__ = [
    "ONNXConversionResult",
    "convert_to_onnx",
    "verify_onnx",
    "convert_and_verify_onnx",
    "TFLiteConversionResult",
    "convert_to_tflite",
    "verify_tflite",
    "convert_and_verify_tflite",
]
