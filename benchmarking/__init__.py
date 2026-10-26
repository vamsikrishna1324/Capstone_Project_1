"""
Benchmarking Package for Edge Deployment.

Provides configuration structures, result containers, foundational engine interfaces,
model-size measurements, and inference latency profiling for edge computer vision models.
"""

from benchmarking.accuracy import (
    EvaluationDataset,
    LabeledImageSample,
    compute_topk_accuracy,
    evaluate_onnx_accuracy,
    evaluate_pytorch_accuracy,
    evaluate_tflite_accuracy,
    measure_accuracy,
    resolve_dataset,
)
from benchmarking.imagenet_images import (
    extract_validation_images,
    generate_validation_manifest,
    scan_and_validate_val_archive,
    verify_extracted_images,
    verify_single_image_integrity,
)
from benchmarking.imagenet_manifest import (
    VALID_IMAGE_EXTENSIONS,
    create_dataset_from_manifest,
    load_imagenet_manifest,
    validate_wnid_mapping,
)
from benchmarking.imagenet_metadata import (
    EXPECTED_DEVKIT_MEMBERS,
    GroundTruthStatistics,
    derive_torchvision_class_mapping,
    generate_mapping_artifact,
    parse_and_validate_ground_truth,
    parse_meta_mat,
    safe_extract_devkit_metadata,
)
from benchmarking.config import (
    BenchmarkConfig,
    BenchmarkRuntime,
    SUPPORTED_RUNTIMES,
)
from benchmarking.engine import (
    BaseBenchmarkEngine,
    BenchmarkEngine,
    get_environment_info,
)
from benchmarking.latency import (
    compute_latency_statistics,
    measure_latency,
    measure_onnx_latency,
    measure_pytorch_latency,
    measure_tflite_latency,
)
from benchmarking.memory import (
    ProcessMemoryTracker,
    get_process_rss_mb,
    measure_memory,
    measure_onnx_memory,
    measure_pytorch_memory,
    measure_tflite_memory,
)
from benchmarking.model_size import (
    bytes_to_mb,
    get_model_parameter_count,
    measure_model_size,
    resolve_artifact_path,
)
from benchmarking.results import (
    AccuracyMetrics,
    BenchmarkResult,
    LatencyMetrics,
    MemoryMetrics,
    ModelSizeMetrics,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRuntime",
    "SUPPORTED_RUNTIMES",
    "BenchmarkResult",
    "AccuracyMetrics",
    "LatencyMetrics",
    "MemoryMetrics",
    "ModelSizeMetrics",
    "BaseBenchmarkEngine",
    "BenchmarkEngine",
    "get_environment_info",
    "bytes_to_mb",
    "get_model_parameter_count",
    "measure_model_size",
    "resolve_artifact_path",
    "compute_latency_statistics",
    "measure_latency",
    "measure_pytorch_latency",
    "measure_onnx_latency",
    "measure_tflite_latency",
    "get_process_rss_mb",
    "ProcessMemoryTracker",
    "measure_memory",
    "measure_pytorch_memory",
    "measure_onnx_memory",
    "measure_tflite_memory",
    "compute_topk_accuracy",
    "EvaluationDataset",
    "LabeledImageSample",
    "resolve_dataset",
    "measure_accuracy",
    "evaluate_pytorch_accuracy",
    "evaluate_onnx_accuracy",
    "evaluate_tflite_accuracy",
    "load_imagenet_manifest",
    "create_dataset_from_manifest",
    "validate_wnid_mapping",
    "VALID_IMAGE_EXTENSIONS",
    "safe_extract_devkit_metadata",
    "parse_and_validate_ground_truth",
    "parse_meta_mat",
    "derive_torchvision_class_mapping",
    "generate_mapping_artifact",
    "GroundTruthStatistics",
    "EXPECTED_DEVKIT_MEMBERS",
    "scan_and_validate_val_archive",
    "extract_validation_images",
    "verify_extracted_images",
    "verify_single_image_integrity",
    "generate_validation_manifest",
]
