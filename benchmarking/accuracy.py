"""
ImageNet Accuracy Benchmarking Module for Edge Deployment.

Evaluates Top-1 and Top-5 classification accuracy across PyTorch, ONNX Runtime,
and TensorFlow Lite using official torchvision preprocessing and runtime-specific
tensor layouts (NCHW for PyTorch and ONNX, NHWC for TFLite).
"""

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import onnxruntime as ort
from PIL import Image
import tensorflow as tf
import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.model_size import resolve_artifact_path
from benchmarking.results import AccuracyMetrics
from models.model_loader import load_model
from models.model_registry import (
    get_model_metadata,
    is_model_supported,
    list_supported_models,
)


def compute_topk_accuracy(
    predictions: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    targets: Union[np.ndarray, torch.Tensor, Sequence[int]],
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[int, float]:
    """Compute Top-k classification accuracies from logits/probabilities and targets.

    Args:
        predictions: Array-like of shape (N, num_classes) containing classification logits.
        targets: Array-like of shape (N,) containing ground-truth integer class labels.
        topk: Tuple of positive integers specifying the k values to evaluate.

    Returns:
        Dictionary mapping each k to its calculated accuracy fraction (0.0 to 1.0)
        rounded to 4 decimal places.

    Raises:
        ValueError: If predictions and targets have different sample counts (N),
                    or if any k in topk is <= 0.
    """
    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    if preds.ndim != 2:
        raise ValueError(f"predictions must be a 2D array of shape (N, num_classes), got ndim={preds.ndim}")
    if targs.ndim != 1:
        raise ValueError(f"targets must be a 1D array of shape (N,), got ndim={targs.ndim}")
    if len(preds) != len(targs):
        raise ValueError(f"Sample count mismatch: {len(preds)} predictions vs {len(targs)} targets.")

    num_samples = len(targs)
    if num_samples == 0:
        return {k: 0.0 for k in topk}

    num_classes = preds.shape[1]
    results: Dict[int, float] = {}

    # Sort class indices descending by logit score
    sorted_indices = np.argsort(preds, axis=-1)[:, ::-1]

    for k in topk:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        effective_k = min(k, num_classes)
        matches = [targs[i] in sorted_indices[i, :effective_k] for i in range(num_samples)]
        results[k] = round(float(np.mean(matches)), 4)

    return results


@dataclass
class LabeledImageSample:
    """A single labeled sample containing an image and ground-truth integer label."""

    image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]
    label: int


class EvaluationDataset(SequenceABC):
    """Container and lazy sequence iterator for evaluation images with ground-truth integer labels."""

    def __init__(
        self,
        samples: Optional[Sequence[Union[Tuple[Any, int], LabeledImageSample]]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        name: str = "custom",
    ) -> None:
        """Initialize the evaluation dataset from in-memory samples, an on-disk directory, or a manifest file.

        Args:
            samples: Optional list of (image, label) pairs or LabeledImageSample instances.
            dataset_path: Optional directory path or path to a CSV/JSON manifest file.
            manifest_path: Optional explicit path to a CSV/JSON manifest file.
            root_dir: Optional dataset root directory for resolving relative image paths.
            name: Descriptive name for the dataset.
        """
        self.name = name
        self.samples: List[Tuple[Any, int]] = []

        if samples is not None:
            for s in samples:
                if isinstance(s, LabeledImageSample):
                    self.samples.append((s.image, s.label))
                elif isinstance(s, (tuple, list)) and len(s) == 2:
                    self.samples.append((s[0], int(s[1])))
                else:
                    raise ValueError(f"Unsupported sample format: {s}")
        elif manifest_path is not None:
            from benchmarking.imagenet_manifest import load_imagenet_manifest
            loaded_samples = load_imagenet_manifest(manifest_path, root_dir=root_dir)
            for s in loaded_samples:
                self.samples.append((s.image, s.label))
        elif dataset_path is not None:
            p = Path(dataset_path)
            if p.is_file() and p.suffix.lower() in (".csv", ".json"):
                from benchmarking.imagenet_manifest import load_imagenet_manifest
                loaded_samples = load_imagenet_manifest(p, root_dir=root_dir)
                for s in loaded_samples:
                    self.samples.append((s.image, s.label))
            else:
                self._load_from_directory(p)

    def _load_from_directory(self, dir_path: Path) -> None:
        """Discover image files and infer integer labels from directory names."""
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dir_path}")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
        subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        if subdirs:
            for class_idx, subdir in enumerate(subdirs):
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix in valid_exts:
                        self.samples.append((img_path, class_idx))
        else:
            for img_path in sorted(dir_path.iterdir()):
                if img_path.suffix in valid_exts:
                    self.samples.append((img_path, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[Image.Image, int], List[Tuple[Image.Image, int]]]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self.samples)))
            return [self[i] for i in indices]

        img_source, label = self.samples[index]
        if isinstance(img_source, (str, Path)):
            img = Image.open(str(img_source)).convert("RGB")
        elif isinstance(img_source, Image.Image):
            img = img_source.convert("RGB")
        elif isinstance(img_source, np.ndarray):
            img = Image.fromarray(img_source)
        elif isinstance(img_source, torch.Tensor):
            arr = img_source.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8 and arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            img = Image.fromarray(arr.astype(np.uint8))
        else:
            raise TypeError(f"Cannot convert image source of type {type(img_source).__name__} to PIL.Image")
        return img, label

    def __iter__(self) -> Iterator[Tuple[Image.Image, int]]:
        for i in range(len(self.samples)):
            yield self[i]

    def cap_samples(self, max_samples: Optional[int]) -> "EvaluationDataset":
        """Return an EvaluationDataset capped at max_samples without eagerly decoding images."""
        if max_samples is not None and max_samples > 0:
            return EvaluationDataset(samples=self.samples[:max_samples], name=self.name)
        return self

    @classmethod
    def create_deterministic_synthetic(
        cls,
        num_samples: int = 10,
        num_classes: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
    ) -> "EvaluationDataset":
        """Generate reproducible synthetic images and integer labels without external downloads.

        Args:
            num_samples: Number of synthetic images to generate.
            num_classes: Upper bound of class indices (0 to num_classes - 1).
            image_size: (width, height) in pixels.
            seed: Random seed for deterministic generation.

        Returns:
            Populated EvaluationDataset instance.
        """
        rng = np.random.default_rng(seed)
        samples: List[Tuple[Any, int]] = []
        for _ in range(num_samples):
            pixels = rng.integers(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(pixels, mode="RGB")
            label = int(rng.integers(0, num_classes))
            samples.append((img, label))
        return cls(samples=samples, name=f"synthetic_{num_samples}")

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        name: str = "imagenet_manifest",
        wnid_to_idx: Optional[Union[Dict[str, int], str, Path]] = None,
    ) -> "EvaluationDataset":
        """Load evaluation dataset from a CSV or JSON manifest file.

        Args:
            manifest_path: Path to CSV or JSON manifest file.
            root_dir: Optional dataset root directory for resolving relative paths.
            name: Descriptive name for the dataset.
            wnid_to_idx: Optional WNID-to-index mapping if manifest uses synset/WNID labels.

        Returns:
            Populated EvaluationDataset instance.
        """
        from benchmarking.imagenet_manifest import create_dataset_from_manifest
        return create_dataset_from_manifest(
            manifest_path=manifest_path,
            root_dir=root_dir,
            name=name,
            wnid_to_idx=wnid_to_idx,
        )


def resolve_dataset(
    dataset_input: Any,
    max_samples: Optional[int] = None,
) -> EvaluationDataset:
    """Normalize various dataset inputs into a lazy EvaluationDataset, capped at max_samples.

    Avoids eager image decoding so memory usage remains bounded.

    Args:
        dataset_input: EvaluationDataset, Path, str, or list of (image, label) pairs.
        max_samples: Optional maximum number of samples to return.

    Returns:
        Lazy EvaluationDataset instance.

    Raises:
        ValueError: If dataset is None or empty.
        TypeError: If dataset_input is of an unsupported type.
    """
    if dataset_input is None:
        raise ValueError(
            "No dataset provided for accuracy benchmarking. "
            "Please provide an EvaluationDataset, directory path, or list of (image, label) pairs."
        )

    if isinstance(dataset_input, EvaluationDataset):
        ds = dataset_input
    elif isinstance(dataset_input, (str, Path)):
        p = Path(dataset_input)
        if p.is_file() and p.suffix.lower() in (".csv", ".json"):
            from benchmarking.imagenet_manifest import create_dataset_from_manifest
            ds = create_dataset_from_manifest(p)
        else:
            ds = EvaluationDataset(dataset_path=dataset_input)
    elif isinstance(dataset_input, (list, tuple)):
        ds = EvaluationDataset(samples=dataset_input)
    else:
        raise TypeError(
            f"Unsupported dataset type: {type(dataset_input).__name__}. "
            "Expected EvaluationDataset, Path, str, or list of (image, label) tuples."
        )

    if max_samples is not None and max_samples > 0:
        ds = ds.cap_samples(max_samples)

    if len(ds) == 0:
        raise ValueError("Dataset is empty; no samples available for accuracy evaluation.")

    return ds


def evaluate_pytorch_accuracy(
    model_id: str,
    config: BenchmarkConfig,
    dataset: Any,
    max_samples: Optional[int] = None,
    progress_interval: int = 2500,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AccuracyMetrics:
    """Evaluate Top-1 and Top-5 accuracy for a PyTorch model.

    Processes images in streaming batches of config.batch_size, releasing
    decoded images and batch tensors after each inference batch so memory
    remains strictly bounded.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        dataset: Dataset source (EvaluationDataset, list, or directory).
        max_samples: Optional upper bound on evaluated samples.
        progress_interval: Sample interval for periodic progress logging (default: 2,500).

    Returns:
        Populated AccuracyMetrics dataclass.
    """
    loaded = load_model(model_id, device=config.device)
    model = loaded.model
    model.eval()
    transform = loaded.preprocess
    device = config.device.strip().lower()

    samples = resolve_dataset(dataset, max_samples=max_samples or config.max_samples)
    predictions: List[np.ndarray] = []
    targets: List[int] = []

    batch_size = max(1, config.batch_size)
    total_samples = len(samples)
    eval_start_time = time.time()
    last_reported_count = 0

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_samples = samples[i : i + batch_size]
            tensors = [transform(img) for img, _ in batch_samples]
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            batch_targets = [lbl for _, lbl in batch_samples]

            logits = model(batch_tensor).detach().cpu().numpy()
            predictions.append(logits)
            targets.extend(batch_targets)

            # Explicitly release batch memory immediately after inference
            del batch_samples, tensors, batch_tensor, logits

            # Periodic progress reporting for long-running benchmarks
            processed_count = len(targets)
            if progress_callback:
                progress_callback(processed_count, total_samples)
            if total_samples >= progress_interval and (
                processed_count - last_reported_count >= progress_interval
                or processed_count == total_samples
            ):
                elapsed = time.time() - eval_start_time
                throughput = processed_count / elapsed if elapsed > 0 else 0.0
                remaining = total_samples - processed_count
                eta_s = remaining / throughput if throughput > 0 else 0.0
                eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s" if eta_s > 0 else "0s"
                pct = (processed_count / total_samples) * 100.0
                print(
                    f"   [Accuracy Progress] {processed_count:,}/{total_samples:,} samples ({pct:5.1f}%) "
                    f"| Throughput: {throughput:.1f} samples/s | Elapsed: {int(elapsed // 60)}m {int(elapsed % 60):02d}s | ETA: {eta_str}",
                    flush=True,
                )
                last_reported_count = processed_count

    all_predictions = np.concatenate(predictions, axis=0)
    topk_acc = compute_topk_accuracy(all_predictions, targets, topk=(1, 5))

    return AccuracyMetrics(
        top1=topk_acc[1],
        top5=topk_acc[5],
        num_samples=len(targets),
    )


def evaluate_onnx_accuracy(
    model_id: str,
    config: BenchmarkConfig,
    dataset: Any,
    max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AccuracyMetrics:
    """Evaluate Top-1 and Top-5 accuracy for an ONNX Runtime model.

    Respects the static batch size of 1 in exported ONNX models by running
    sample-by-sample inference with NCHW layout.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        dataset: Dataset source.
        max_samples: Optional upper bound on evaluated samples.
        progress_callback: Optional intermediate progress callback.

    Returns:
        Populated AccuracyMetrics dataclass.

    Raises:
        FileNotFoundError: If the ONNX artifact is missing from disk.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.ONNX, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"ONNX model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run ONNX conversion first."
        )

    providers = [config.backend_provider] if config.backend_provider else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(artifact_path), providers=providers)
    input_name = session.get_inputs()[0].name

    loaded = load_model(model_id, device="cpu")
    transform = loaded.preprocess

    samples = resolve_dataset(dataset, max_samples=max_samples or config.max_samples)
    predictions: List[np.ndarray] = []
    targets: List[int] = []

    for img, lbl in samples:
        t = transform(img).numpy()  # (3, 224, 224)
        batch_np = np.expand_dims(t, axis=0).astype(np.float32)  # (1, 3, 224, 224)
        logits = session.run(None, {input_name: batch_np})[0]
        predictions.append(logits)
        targets.append(lbl)
        if progress_callback:
            progress_callback(len(targets), len(samples))
        del img, t, batch_np, logits

    all_predictions = np.concatenate(predictions, axis=0)
    topk_acc = compute_topk_accuracy(all_predictions, targets, topk=(1, 5))

    return AccuracyMetrics(
        top1=topk_acc[1],
        top5=topk_acc[5],
        num_samples=len(targets),
    )


def evaluate_tflite_accuracy(
    model_id: str,
    config: BenchmarkConfig,
    dataset: Any,
    max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AccuracyMetrics:
    """Evaluate Top-1 and Top-5 accuracy for a TensorFlow Lite (.tflite) flatbuffer model.

    Transposes preprocessed NCHW tensors to NHWC layout (1, 224, 224, 3) as required
    by onnx2tf converted TFLite models.

    Args:
        model_id: Registered model identifier.
        config: Benchmarking configuration specification.
        dataset: Dataset source.
        max_samples: Optional upper bound on evaluated samples.
        progress_callback: Optional intermediate progress callback.

    Returns:
        Populated AccuracyMetrics dataclass.

    Raises:
        FileNotFoundError: If the TFLite artifact is missing from disk.
    """
    artifact_path = resolve_artifact_path(model_id, BenchmarkRuntime.TFLITE, config.artifact_path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"TFLite model artifact for '{model_id}' does not exist on disk at: {artifact_path}. "
            f"Please run TFLite conversion first."
        )

    loaded = load_model(model_id, device="cpu")
    transform = loaded.preprocess

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        interpreter = tf.lite.Interpreter(model_path=str(artifact_path))
        interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    in_idx = in_details["index"]
    out_idx = out_details["index"]

    samples = resolve_dataset(dataset, max_samples=max_samples or config.max_samples)
    predictions: List[np.ndarray] = []
    targets: List[int] = []

    for img, lbl in samples:
        t = transform(img).numpy()  # (3, 224, 224)
        nhwc = np.transpose(t, (1, 2, 0))  # (224, 224, 3)
        batch_np = np.expand_dims(nhwc, axis=0).astype(np.float32)  # (1, 224, 224, 3)

        interpreter.set_tensor(in_idx, batch_np)
        interpreter.invoke()
        logits = interpreter.get_tensor(out_idx)
        predictions.append(logits)
        targets.append(lbl)
        if progress_callback:
            progress_callback(len(targets), len(samples))
        del img, t, nhwc, batch_np, logits

    all_predictions = np.concatenate(predictions, axis=0)
    topk_acc = compute_topk_accuracy(all_predictions, targets, topk=(1, 5))

    return AccuracyMetrics(
        top1=topk_acc[1],
        top5=topk_acc[5],
        num_samples=len(targets),
    )


def measure_accuracy(
    config: BenchmarkConfig,
    dataset: Optional[Any] = None,
    max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AccuracyMetrics:
    """Orchestrate classification accuracy benchmarking across supported runtimes.

    Args:
        config: Validated BenchmarkConfig specifying model and runtime.
        dataset: EvaluationDataset, directory path, or list of (image, label) tuples.
        max_samples: Optional maximum number of samples to evaluate.
        progress_callback: Optional intermediate progress callback.

    Returns:
        Populated AccuracyMetrics dataclass.

    Raises:
        ValueError: If model ID or runtime is unsupported, or no dataset is provided.
        FileNotFoundError: If an expected model artifact is missing from disk.
    """
    if not is_model_supported(config.model_id):
        supported = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model ID '{config.model_id}'. Supported models: {supported}")

    effective_dataset = dataset
    if effective_dataset is None:
        effective_dataset = config.dataset_path or config.metadata.get("dataset")

    if effective_dataset is None:
        raise ValueError(
            "No evaluation dataset provided for accuracy benchmarking. "
            "Please provide an EvaluationDataset, directory path, or list of (image, label) tuples."
        )

    norm_runtime = BenchmarkRuntime.from_string(config.runtime)
    effective_max = max_samples or config.max_samples

    if norm_runtime == BenchmarkRuntime.PYTORCH:
        return evaluate_pytorch_accuracy(config.model_id, config, effective_dataset, max_samples=effective_max, progress_callback=progress_callback)
    elif norm_runtime == BenchmarkRuntime.ONNX:
        return evaluate_onnx_accuracy(config.model_id, config, effective_dataset, max_samples=effective_max, progress_callback=progress_callback)
    elif norm_runtime == BenchmarkRuntime.TFLITE:
        return evaluate_tflite_accuracy(config.model_id, config, effective_dataset, max_samples=effective_max, progress_callback=progress_callback)
    else:
        raise ValueError(f"Unhandled benchmark runtime '{norm_runtime}'")
