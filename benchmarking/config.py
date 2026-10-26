"""
Benchmarking Configuration Module for Edge Deployment.

Defines runtime enums, configuration structures, and validation rules for
standardized, reproducible benchmarking across PyTorch, ONNX, and TFLite.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.model_registry import (
    is_model_supported,
    list_supported_models,
)


class BenchmarkRuntime(str, Enum):
    """Supported execution runtimes/formats for benchmarking."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"

    @classmethod
    def from_string(cls, value: Union[str, "BenchmarkRuntime"]) -> "BenchmarkRuntime":
        """Convert a string or enum instance to BenchmarkRuntime.

        Args:
            value: Case-insensitive runtime name or BenchmarkRuntime instance.

        Returns:
            Normalized BenchmarkRuntime member.

        Raises:
            ValueError: If the runtime is not recognized.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        supported = [m.value for m in cls]
        raise ValueError(
            f"Unsupported benchmark runtime '{value}'. Supported runtimes: {supported}"
        )


SUPPORTED_RUNTIMES: List[str] = [m.value for m in BenchmarkRuntime]


@dataclass
class BenchmarkConfig:
    """Configuration specification for a reproducible model benchmarking run.

    Attributes:
        model_id: Logical identifier of the supported model in the registry.
        runtime: Target inference runtime/format (PyTorch, ONNX, TFLite).
        warmup_runs: Number of discarded warm-up inference iterations (>= 0).
        num_runs: Number of measured inference iterations (> 0).
        input_size: Expected input tensor dimensions as (channels, height, width).
        batch_size: Inference batch size (>= 1).
        device: Target execution device (e.g., 'cpu', 'mps', 'cuda').
        backend_provider: Optional execution provider or delegate (e.g., 'CPUExecutionProvider').
        artifact_path: Optional custom path to the serialized model file on disk.
        metadata: Optional dictionary for additional run metadata or user tags.
    """

    model_id: str
    runtime: Union[BenchmarkRuntime, str] = BenchmarkRuntime.PYTORCH
    warmup_runs: int = 10
    num_runs: int = 100
    input_size: Tuple[int, int, int] = (3, 224, 224)
    batch_size: int = 1
    device: str = "cpu"
    backend_provider: Optional[str] = None
    artifact_path: Optional[Union[str, Path]] = None
    dataset_path: Optional[Union[str, Path]] = None
    max_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters upon initialization."""
        self.validate()

    def validate(self) -> None:
        """Perform comprehensive validation of benchmarking configuration parameters.

        Raises:
            ValueError: If any configuration parameter fails validation.
        """
        # 1. Validate model_id
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string.")

        if not is_model_supported(self.model_id):
            supported = ", ".join(list_supported_models())
            raise ValueError(
                f"Unsupported model ID '{self.model_id}'. Supported models: {supported}"
            )

        # 2. Validate and normalize runtime
        self.runtime = BenchmarkRuntime.from_string(self.runtime)

        # 3. Validate iteration counts
        if not isinstance(self.warmup_runs, int) or isinstance(self.warmup_runs, bool) or self.warmup_runs < 0:
            raise ValueError(f"warmup_runs must be a non-negative integer (>= 0), got {self.warmup_runs}")

        if not isinstance(self.num_runs, int) or isinstance(self.num_runs, bool) or self.num_runs <= 0:
            raise ValueError(f"num_runs must be a positive integer (> 0), got {self.num_runs}")

        # 4. Validate batch size
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer (>= 1), got {self.batch_size}")

        # 5. Validate input size
        if (
            not isinstance(self.input_size, tuple)
            or len(self.input_size) != 3
            or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in self.input_size)
        ):
            raise ValueError(
                f"input_size must be a 3-tuple of positive integers (channels, height, width), got {self.input_size}"
            )

        # 6. Validate device
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        # 7. Validate artifact_path if provided
        if self.artifact_path is not None and not isinstance(self.artifact_path, (str, Path)):
            raise ValueError(f"artifact_path must be a string or Path, got {type(self.artifact_path).__name__}")

        # 8. Validate dataset_path if provided
        if self.dataset_path is not None and not isinstance(self.dataset_path, (str, Path)):
            raise ValueError(f"dataset_path must be a string or Path, got {type(self.dataset_path).__name__}")

        # 9. Validate max_samples if provided
        if self.max_samples is not None:
            if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 1:
                raise ValueError("max_samples must be a positive integer if specified.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a clean dictionary representation."""
        data = asdict(self)
        data["runtime"] = self.runtime.value if isinstance(self.runtime, BenchmarkRuntime) else str(self.runtime)
        data["artifact_path"] = str(self.artifact_path) if self.artifact_path is not None else None
        data["dataset_path"] = str(self.dataset_path) if self.dataset_path is not None else None
        return data
