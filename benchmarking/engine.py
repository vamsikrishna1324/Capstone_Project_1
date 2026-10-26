"""
Benchmarking Engine Foundation Module for Edge Deployment.

Provides base interfaces and the foundational benchmarking engine class for
orchestrating model profiling across accuracy, latency, memory footprint,
and model size.
"""

from abc import ABC, abstractmethod
import platform
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch

from benchmarking.accuracy import measure_accuracy
from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.latency import measure_latency
from benchmarking.memory import measure_memory
from benchmarking.model_size import (
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
from models.model_registry import (
    ModelMetadata,
    get_model_metadata,
    is_model_supported,
    list_supported_models,
)


def get_environment_info() -> Dict[str, Any]:
    """Collect hardware, operating system, and deep learning library metadata.

    Returns:
        Dictionary containing reproducible environment metadata.
    """
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if mps_available:
        accelerator = "Apple Silicon MPS"
    elif cuda_available:
        accelerator = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    else:
        accelerator = "CPU"

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torchvision_version": getattr(torch, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "tensorflow_version": tf.__version__,
        "device_accelerator": accelerator,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
    }


class BaseBenchmarkEngine(ABC):
    """Abstract base class defining the contract for model benchmarking engines."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmarking engine with a validated configuration.

        Args:
            config: BenchmarkConfig specifying model, runtime, and execution settings.
        """
        self.config = config
        self.environment: Dict[str, Any] = get_environment_info()
        self.model_metadata: ModelMetadata = get_model_metadata(config.model_id)

    def resolve_artifact_path(self) -> Path:
        """Resolve expected disk artifact path for the configured model and runtime."""
        return resolve_artifact_path(self.config.model_id, self.config.runtime, self.config.artifact_path)

    @abstractmethod
    def benchmark_accuracy(
        self,
        dataset: Optional[Any] = None,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[AccuracyMetrics]:
        """Profile classification accuracy metrics on a validation dataset."""

    @abstractmethod
    def benchmark_latency(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[LatencyMetrics]:
        """Profile inference latency, throughput, and distribution percentiles."""

    @abstractmethod
    def benchmark_memory(self) -> Optional[MemoryMetrics]:
        """Profile RAM footprint and peak memory consumption during inference."""

    @abstractmethod
    def benchmark_model_size(self) -> Optional[ModelSizeMetrics]:
        """Measure artifact disk storage size and parameter counts."""

    @abstractmethod
    def run_benchmark(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Execute the complete benchmarking pipeline and return structured results."""


class BenchmarkEngine(BaseBenchmarkEngine):
    """Foundational benchmarking engine for edge deployment models.

    Validates configuration, binds model metadata, collects environment details,
    and establishes the standard execution interface for subsequent metric implementations.
    """

    def __init__(self, config: Union[BenchmarkConfig, Dict[str, Any]]) -> None:
        """Initialize and validate the benchmarking engine.

        Args:
            config: BenchmarkConfig instance or raw configuration dictionary.

        Raises:
            ValueError: If configuration or model ID is invalid.
        """
        if isinstance(config, dict):
            resolved_config = BenchmarkConfig(**config)
        elif isinstance(config, BenchmarkConfig):
            resolved_config = config
        else:
            raise ValueError(
                f"config must be a BenchmarkConfig instance or dict, got {type(config).__name__}"
            )

        # Validate model registration
        if not is_model_supported(resolved_config.model_id):
            supported = ", ".join(list_supported_models())
            raise ValueError(
                f"Unsupported model ID '{resolved_config.model_id}'. Supported models: {supported}"
            )

        super().__init__(resolved_config)

    def benchmark_accuracy(
        self,
        dataset: Optional[Any] = None,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AccuracyMetrics:
        """Measure Top-1 and Top-5 classification accuracy on a validation dataset.

        Args:
            dataset: EvaluationDataset, directory path, or list of (image, label) tuples.
            max_samples: Optional maximum number of samples to evaluate.
            progress_callback: Optional intermediate progress callback.

        Returns:
            AccuracyMetrics containing Top-1 and Top-5 accuracy fractions.

        Raises:
            ValueError: If dataset is missing or model/runtime is invalid.
            FileNotFoundError: If an expected model artifact is missing from disk.
        """
        return measure_accuracy(self.config, dataset=dataset, max_samples=max_samples, progress_callback=progress_callback)

    def benchmark_latency(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> LatencyMetrics:
        """Measure inference latency and throughput for the configured model and runtime.

        Returns:
            LatencyMetrics containing distribution statistics and throughput (fps).

        Raises:
            FileNotFoundError: If an expected serialized artifact is missing from disk.
            ValueError: If model, runtime, or batch configuration is invalid.
        """
        return measure_latency(self.config, progress_callback=progress_callback)

    def benchmark_memory(self) -> MemoryMetrics:
        """Measure process memory consumption (RSS) for the configured model and runtime.

        Returns:
            MemoryMetrics containing baseline, post-load, peak RAM, and memory increase in MiB.

        Raises:
            FileNotFoundError: If an expected serialized artifact does not exist on disk.
            ValueError: If model, runtime, or batch configuration is invalid.
        """
        return measure_memory(self.config)

    def benchmark_model_size(self) -> ModelSizeMetrics:
        """Measure model disk footprint and parameter count for the configured model and runtime.

        Returns:
            ModelSizeMetrics containing file_size_bytes, file_size_mb, param_count, and artifact_path.

        Raises:
            FileNotFoundError: If an expected serialized artifact does not exist on disk.
            ValueError: If model or runtime is unsupported.
        """
        return measure_model_size(
            model_id=self.config.model_id,
            runtime=self.config.runtime,
            artifact_path=self.config.artifact_path,
        )

    def run_benchmark(
        self,
        dataset: Optional[Any] = None,
        measure_model_size: bool = True,
        measure_latency: bool = True,
        measure_memory: bool = True,
        measure_accuracy: bool = False,
        max_accuracy_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Execute the benchmarking pipeline.

        Measures model size and parameter count (Step 4B-1), inference latency /
        throughput (Step 4B-2), memory footprint (Step 4B-3), and classification
        accuracy (Step 4C).

        Args:
            dataset: EvaluationDataset, directory path, or list of (image, label) tuples.
            measure_model_size: Whether to execute model-size measurement during this run.
            measure_latency: Whether to execute latency and throughput measurement.
            measure_memory: Whether to execute memory footprint measurement.
            measure_accuracy: Whether to execute classification accuracy benchmarking.
            max_accuracy_samples: Optional upper bound on accuracy evaluation samples.
            progress_callback: Optional intermediate progress reporting callback.

        Returns:
            BenchmarkResult instance with populated metrics.
        """
        model_size_metrics: Optional[ModelSizeMetrics] = None
        latency_metrics: Optional[LatencyMetrics] = None
        memory_metrics: Optional[MemoryMetrics] = None
        accuracy_metrics: Optional[AccuracyMetrics] = None

        if measure_model_size:
            try:
                model_size_metrics = self.benchmark_model_size()
            except Exception as exc:
                return BenchmarkResult(
                    model_id=self.config.model_id,
                    runtime=self.config.runtime,
                    config=self.config,
                    accuracy=None,
                    latency=None,
                    memory_footprint=None,
                    model_size=None,
                    environment=self.environment,
                    success=False,
                    error_message=f"Model size measurement failed: {exc}",
                )

        if measure_latency:
            try:
                latency_metrics = self.benchmark_latency(progress_callback=progress_callback)
            except Exception as exc:
                return BenchmarkResult(
                    model_id=self.config.model_id,
                    runtime=self.config.runtime,
                    config=self.config,
                    accuracy=None,
                    latency=None,
                    memory_footprint=None,
                    model_size=model_size_metrics,
                    environment=self.environment,
                    success=False,
                    error_message=f"Latency benchmarking failed: {exc}",
                )

        if measure_memory:
            try:
                memory_metrics = self.benchmark_memory()
            except Exception as exc:
                return BenchmarkResult(
                    model_id=self.config.model_id,
                    runtime=self.config.runtime,
                    config=self.config,
                    accuracy=None,
                    latency=latency_metrics,
                    memory_footprint=None,
                    model_size=model_size_metrics,
                    environment=self.environment,
                    success=False,
                    error_message=f"Memory benchmarking failed: {exc}",
                )

        if measure_accuracy:
            try:
                accuracy_metrics = self.benchmark_accuracy(
                    dataset=dataset,
                    max_samples=max_accuracy_samples,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                return BenchmarkResult(
                    model_id=self.config.model_id,
                    runtime=self.config.runtime,
                    config=self.config,
                    accuracy=None,
                    latency=latency_metrics,
                    memory_footprint=memory_metrics,
                    model_size=model_size_metrics,
                    environment=self.environment,
                    success=False,
                    error_message=f"Accuracy benchmarking failed: {exc}",
                )

        return BenchmarkResult(
            model_id=self.config.model_id,
            runtime=self.config.runtime,
            config=self.config,
            accuracy=accuracy_metrics,
            latency=latency_metrics,
            memory_footprint=memory_metrics,
            model_size=model_size_metrics,
            environment=self.environment,
            success=True,
            error_message=None,
        )
