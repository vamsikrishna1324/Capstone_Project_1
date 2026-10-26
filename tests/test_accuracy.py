"""
Unit and Integration Tests for Step 4C: ImageNet Accuracy Benchmarking.

Verifies:
1. compute_topk_accuracy calculations (Top-1, Top-5, empty data, shape mismatch, bounds).
2. EvaluationDataset, LabeledImageSample, and resolve_dataset behavior and synthetic generation.
3. PyTorch accuracy evaluation (CPU and MPS) with batching and official preprocessing.
4. ONNX Runtime accuracy evaluation respecting static batch size 1 and NCHW layout.
5. TensorFlow Lite accuracy evaluation with NHWC transpose and float32 normalization.
6. Dispatcher (measure_accuracy) routing, parameter validation, and missing dataset errors.
7. Coverage across all 4 registered models (MobileNetV3, ResNet50, EfficientNet-B0, DenseNet121).
8. BenchmarkEngine and BenchmarkResult integration, serialization, and error capture.
"""

import csv
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

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
from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine
from benchmarking.results import AccuracyMetrics, BenchmarkResult
from models.model_registry import list_supported_models


# ============================================================================
# 1. Top-k Accuracy Numerical Calculation Tests
# ============================================================================
def test_compute_topk_accuracy_perfect() -> None:
    """Verify compute_topk_accuracy returns 1.0 when predictions match targets."""
    # 3 samples, 4 classes. Predictions strictly peak at target index.
    logits = np.array([
        [10.0, 1.0, 0.0, 0.0],  # argmax = 0
        [0.0, 10.0, 0.0, 1.0],  # argmax = 1
        [0.0, 0.0, 10.0, 1.0],  # argmax = 2
    ])
    targets = np.array([0, 1, 2])
    acc = compute_topk_accuracy(logits, targets, topk=(1, 2))

    assert acc[1] == 1.0
    assert acc[2] == 1.0


def test_compute_topk_accuracy_known_values() -> None:
    """Verify compute_topk_accuracy against known ground-truth ranking."""
    # 4 samples, 5 classes
    logits = np.array([
        [0.1, 0.9, 0.2, 0.0, 0.0],  # ranked: [1, 2, 0, 3, 4]
        [0.5, 0.1, 0.2, 0.8, 0.0],  # ranked: [3, 0, 2, 1, 4]
        [0.1, 0.2, 0.7, 0.3, 0.0],  # ranked: [2, 3, 1, 0, 4]
        [0.0, 0.0, 0.1, 0.2, 0.8],  # ranked: [4, 3, 2, 0, 1]
    ])
    # Targets:
    # Sample 0: target 1 -> Top-1 match
    # Sample 1: target 0 -> Top-2 match, not Top-1
    # Sample 2: target 1 -> Top-3 match, not Top-1 or Top-2
    # Sample 3: target 4 -> Top-1 match
    targets = np.array([1, 0, 1, 4])
    acc = compute_topk_accuracy(logits, targets, topk=(1, 2, 3, 5))

    # Top-1 matches: sample 0 and 3 -> 2/4 = 0.50
    assert acc[1] == 0.5
    # Top-2 matches: samples 0, 1, 3 -> 3/4 = 0.75
    assert acc[2] == 0.75
    # Top-3 matches: samples 0, 1, 2, 3 -> 4/4 = 1.0
    assert acc[3] == 1.0
    assert acc[5] == 1.0


def test_compute_topk_accuracy_empty_data() -> None:
    """Verify compute_topk_accuracy returns 0.0 for empty predictions and targets."""
    acc = compute_topk_accuracy(np.empty((0, 10)), np.empty((0,), dtype=int), topk=(1, 5))
    assert acc[1] == 0.0
    assert acc[5] == 0.0


def test_compute_topk_accuracy_shape_mismatches() -> None:
    """Verify compute_topk_accuracy raises ValueError on mismatched dimensions or counts."""
    # Mismatched sample count
    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_topk_accuracy(np.zeros((5, 10)), np.zeros((4,), dtype=int))

    # Non-2D predictions
    with pytest.raises(ValueError, match="predictions must be a 2D array"):
        compute_topk_accuracy(np.zeros((5, 10, 1)), np.zeros((5,), dtype=int))

    # Non-1D targets
    with pytest.raises(ValueError, match="targets must be a 1D array"):
        compute_topk_accuracy(np.zeros((5, 10)), np.zeros((5, 1), dtype=int))


def test_compute_topk_accuracy_invalid_k() -> None:
    """Verify compute_topk_accuracy raises ValueError if any k < 1."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        compute_topk_accuracy(np.zeros((2, 5)), np.zeros((2,), dtype=int), topk=(0, 5))


# ============================================================================
# 2. EvaluationDataset & resolve_dataset Tests
# ============================================================================
def test_evaluation_dataset_from_samples() -> None:
    """Verify EvaluationDataset instantiation and indexing from sample tuples."""
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    ds = EvaluationDataset(samples=[(img1, 10), (img2, 20)], name="test_ds")

    assert len(ds) == 2
    assert ds.name == "test_ds"

    out_img1, lbl1 = ds[0]
    assert isinstance(out_img1, Image.Image)
    assert lbl1 == 10

    out_img2, lbl2 = ds[1]
    assert lbl2 == 20


def test_evaluation_dataset_from_labeled_sample_dataclass() -> None:
    """Verify EvaluationDataset accepts LabeledImageSample objects."""
    img = Image.new("RGB", (50, 50))
    sample = LabeledImageSample(image=img, label=42)
    ds = EvaluationDataset(samples=[sample])
    assert len(ds) == 1
    _, lbl = ds[0]
    assert lbl == 42


def test_evaluation_dataset_synthetic_generation() -> None:
    """Verify deterministic synthetic dataset generation creates valid PIL samples."""
    ds = EvaluationDataset.create_deterministic_synthetic(num_samples=5, num_classes=1000, seed=123)
    assert len(ds) == 5

    for i in range(5):
        img, lbl = ds[i]
        assert isinstance(img, Image.Image)
        assert img.size == (224, 224)
        assert 0 <= lbl < 1000


def test_resolve_dataset_max_samples_capping() -> None:
    """Verify resolve_dataset caps samples properly when max_samples is given."""
    ds = EvaluationDataset.create_deterministic_synthetic(num_samples=10, seed=42)
    resolved = resolve_dataset(ds, max_samples=4)
    assert len(resolved) == 4


def test_resolve_dataset_errors() -> None:
    """Verify resolve_dataset raises ValueError for None or empty inputs, and TypeError for invalid types."""
    with pytest.raises(ValueError, match="No dataset provided"):
        resolve_dataset(None)

    with pytest.raises(ValueError, match="Dataset is empty"):
        resolve_dataset([])

    with pytest.raises(TypeError, match="Unsupported dataset type"):
        resolve_dataset(12345)


# ============================================================================
# 3. PyTorch Accuracy Benchmarking Tests
# ============================================================================
def test_evaluate_pytorch_accuracy_cpu() -> None:
    """Verify PyTorch accuracy evaluation on CPU with synthetic dataset."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        device="cpu",
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=4, seed=42)
    metrics = evaluate_pytorch_accuracy("mobilenet_v3_small", config, dataset)

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 4
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0
    assert metrics.top5 >= metrics.top1


def test_evaluate_pytorch_accuracy_batching() -> None:
    """Verify PyTorch accuracy evaluation works cleanly with batch_size > 1."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        batch_size=2,
        device="cpu",
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=4, seed=42)
    metrics = evaluate_pytorch_accuracy("mobilenet_v3_small", config, dataset)

    assert metrics.num_samples == 4
    assert 0.0 <= metrics.top1 <= 1.0


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="Apple Silicon MPS is not available on this host.",
)
def test_evaluate_pytorch_accuracy_mps() -> None:
    """Verify PyTorch accuracy evaluation runs on Apple Silicon MPS."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        device="mps",
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)
    metrics = evaluate_pytorch_accuracy("mobilenet_v3_small", config, dataset)

    assert metrics.num_samples == 2
    assert 0.0 <= metrics.top1 <= 1.0


# ============================================================================
# 4. ONNX Runtime Accuracy Benchmarking Tests
# ============================================================================
def test_evaluate_onnx_accuracy_mobilenet_v3_small() -> None:
    """Verify ONNX Runtime accuracy evaluation on real artifact respecting static batch size 1."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=3, seed=42)
    metrics = evaluate_onnx_accuracy("mobilenet_v3_small", config, dataset)

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 3
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0
    assert metrics.top5 >= metrics.top1


def test_evaluate_onnx_accuracy_missing_artifact_error(tmp_path: Path) -> None:
    """Verify ONNX raises FileNotFoundError when the artifact is missing."""
    missing_file = tmp_path / "missing.onnx"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        artifact_path=missing_file,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=1)
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        evaluate_onnx_accuracy("mobilenet_v3_small", config, dataset)


# ============================================================================
# 5. TensorFlow Lite Accuracy Benchmarking Tests
# ============================================================================
def test_evaluate_tflite_accuracy_mobilenet_v3_small() -> None:
    """Verify TFLite accuracy evaluation on real artifact using NHWC layout."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=3, seed=42)
    metrics = evaluate_tflite_accuracy("mobilenet_v3_small", config, dataset)

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 3
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0
    assert metrics.top5 >= metrics.top1


def test_evaluate_tflite_accuracy_missing_artifact_error(tmp_path: Path) -> None:
    """Verify TFLite raises FileNotFoundError when the artifact is missing."""
    missing_file = tmp_path / "missing.tflite"
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.TFLITE,
        artifact_path=missing_file,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=1)
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        evaluate_tflite_accuracy("mobilenet_v3_small", config, dataset)


# ============================================================================
# 6. Dispatcher (measure_accuracy) Validation Tests
# ============================================================================
def test_measure_accuracy_dispatcher_pytorch() -> None:
    """Verify measure_accuracy dispatches to PyTorch."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="pytorch")
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)
    metrics = measure_accuracy(config, dataset=dataset)
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 2


def test_measure_accuracy_dispatcher_onnx() -> None:
    """Verify measure_accuracy dispatches to ONNX."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="onnx")
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)
    metrics = measure_accuracy(config, dataset=dataset)
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 2


def test_measure_accuracy_dispatcher_tflite() -> None:
    """Verify measure_accuracy dispatches to TFLite."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="tflite")
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)
    metrics = measure_accuracy(config, dataset=dataset)
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 2


def test_measure_accuracy_missing_dataset_error() -> None:
    """Verify measure_accuracy raises ValueError when no dataset is provided."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="pytorch")
    with pytest.raises(ValueError, match="No evaluation dataset provided"):
        measure_accuracy(config, dataset=None)


def test_measure_accuracy_invalid_model() -> None:
    """Verify measure_accuracy raises ValueError for unsupported model ID."""
    config = BenchmarkConfig.__new__(BenchmarkConfig)
    config.model_id = "invalid_model_123"
    config.runtime = BenchmarkRuntime.PYTORCH

    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=1)
    with pytest.raises(ValueError, match="Unsupported model ID"):
        measure_accuracy(config, dataset=dataset)


# ============================================================================
# 7. Coverage Across All 4 Supported Models (ONNX)
# ============================================================================
@pytest.mark.parametrize("model_id", list_supported_models())
def test_measure_accuracy_all_four_models_onnx(model_id: str) -> None:
    """Verify accuracy evaluation works across all 4 supported models via ONNX."""
    config = BenchmarkConfig(
        model_id=model_id,
        runtime=BenchmarkRuntime.ONNX,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)
    metrics = measure_accuracy(config, dataset=dataset)

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 2
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0


# ============================================================================
# 8. BenchmarkEngine and BenchmarkResult Integration Tests
# ============================================================================
def test_benchmark_engine_accuracy_method() -> None:
    """Verify BenchmarkEngine.benchmark_accuracy executes and returns AccuracyMetrics."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="onnx")
    engine = BenchmarkEngine(config)
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=2, seed=42)

    metrics = engine.benchmark_accuracy(dataset=dataset)
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 2


def test_benchmark_engine_run_benchmark_with_accuracy() -> None:
    """Verify run_benchmark populates accuracy when measure_accuracy=True and dataset is given."""
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.ONNX,
        warmup_runs=1,
        num_runs=2,
    )
    engine = BenchmarkEngine(config)
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=3, seed=42)

    result = engine.run_benchmark(
        dataset=dataset,
        measure_model_size=True,
        measure_latency=True,
        measure_memory=True,
        measure_accuracy=True,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.error_message is None
    assert result.model_size is not None
    assert result.latency is not None
    assert result.memory_footprint is not None
    assert result.accuracy is not None
    assert result.accuracy.num_samples == 3
    assert 0.0 <= result.accuracy.top1 <= 1.0
    assert 0.0 <= result.accuracy.top5 <= 1.0

    # Serialization
    d = result.to_dict()
    assert d["accuracy"] is not None
    assert d["accuracy"]["num_samples"] == 3
    assert "Accuracy:" in result.summary()


def test_benchmark_engine_run_benchmark_captures_accuracy_error() -> None:
    """Verify run_benchmark captures missing dataset error in BenchmarkResult."""
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime="onnx")
    engine = BenchmarkEngine(config)

    # Calling run_benchmark with measure_accuracy=True but dataset=None
    result = engine.run_benchmark(
        dataset=None,
        measure_model_size=False,
        measure_latency=False,
        measure_memory=False,
        measure_accuracy=True,
    )

    assert result.success is False
    assert result.accuracy is None
    assert result.error_message is not None
    assert "Accuracy benchmarking failed" in result.error_message


# ============================================================================
# 9. Streaming & Lazy Accuracy Benchmarking Tests
# ============================================================================
def test_resolve_dataset_lazy_manifest_does_not_eagerly_open_images(tmp_path: Path) -> None:
    """Verify that resolving a manifest does not eagerly load or decode any images."""
    from unittest.mock import patch

    # Create 5 dummy images and a CSV manifest
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        for i in range(5):
            p = img_dir / f"img_{i}.jpg"
            Image.new("RGB", (32, 32), color="blue").save(p)
            writer.writerow([str(p.resolve()), str(i)])

    with patch.object(Image, "open", wraps=Image.open) as mock_open:
        # Resolving dataset must NOT open any images
        ds = resolve_dataset(csv_path)
        assert mock_open.call_count == 0
        assert len(ds) == 5

        # Indexing one sample should open only 1 image
        item = ds[0]
        assert isinstance(item[0], Image.Image)
        assert item[1] == 0
        assert mock_open.call_count == 1

        # Slicing 3 samples should open exactly 3 additional images
        batch = ds[1:4]
        assert len(batch) == 3
        assert mock_open.call_count == 4


def test_evaluation_dataset_sequence_and_slicing() -> None:
    """Verify EvaluationDataset adheres to collections.abc.Sequence with slicing and iteration."""
    from collections.abc import Sequence as SequenceABC

    ds = EvaluationDataset.create_deterministic_synthetic(num_samples=7, seed=42)
    assert isinstance(ds, SequenceABC)
    assert len(ds) == 7

    # Slicing tests
    slice_1 = ds[0:3]
    assert len(slice_1) == 3
    assert all(isinstance(img, Image.Image) for img, _ in slice_1)

    slice_2 = ds[3:6]
    assert len(slice_2) == 3

    slice_incomplete = ds[6:7]
    assert len(slice_incomplete) == 1

    empty_slice = ds[2:2]
    assert len(empty_slice) == 0

    # Iteration
    iter_items = list(ds)
    assert len(iter_items) == 7
    assert all(isinstance(img, Image.Image) for img, _ in iter_items)


def test_evaluate_pytorch_accuracy_batch_boundaries() -> None:
    """Verify PyTorch accuracy evaluation handles prime sample counts and incomplete final batches."""
    # 7 samples with batch_size=3 produces batches of size 3, 3, 1
    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        device="cpu",
        batch_size=3,
    )
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=7, seed=123)
    metrics = evaluate_pytorch_accuracy("mobilenet_v3_small", config, dataset)

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.num_samples == 7
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0
    assert metrics.top5 >= metrics.top1


def test_resolve_dataset_lazy_max_samples_capping(tmp_path: Path) -> None:
    """Verify max_samples caps samples without reading uncapped items."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        for i in range(10):
            p = img_dir / f"img_{i}.jpg"
            Image.new("RGB", (32, 32), color="green").save(p)
            writer.writerow([str(p.resolve()), str(i % 1000)])

    # Cap at 4
    resolved = resolve_dataset(csv_path, max_samples=4)
    assert len(resolved) == 4

    config = BenchmarkConfig(
        model_id="mobilenet_v3_small",
        runtime=BenchmarkRuntime.PYTORCH,
        device="cpu",
        batch_size=3,
    )
    metrics = evaluate_pytorch_accuracy("mobilenet_v3_small", config, resolved)
    assert metrics.num_samples == 4


def test_lazy_streaming_metric_parity_across_batch_sizes() -> None:
    """Verify that batch_size variation (1, 2, 6) produces numerically identical Top-1/Top-5 accuracy."""
    dataset = EvaluationDataset.create_deterministic_synthetic(num_samples=6, seed=999)

    cfg_b1 = BenchmarkConfig(model_id="mobilenet_v3_small", runtime=BenchmarkRuntime.PYTORCH, device="cpu", batch_size=1)
    cfg_b2 = BenchmarkConfig(model_id="mobilenet_v3_small", runtime=BenchmarkRuntime.PYTORCH, device="cpu", batch_size=2)
    cfg_b6 = BenchmarkConfig(model_id="mobilenet_v3_small", runtime=BenchmarkRuntime.PYTORCH, device="cpu", batch_size=6)

    m1 = evaluate_pytorch_accuracy("mobilenet_v3_small", cfg_b1, dataset)
    m2 = evaluate_pytorch_accuracy("mobilenet_v3_small", cfg_b2, dataset)
    m6 = evaluate_pytorch_accuracy("mobilenet_v3_small", cfg_b6, dataset)

    assert m1.num_samples == m2.num_samples == m6.num_samples == 6
    assert m1.top1 == m2.top1 == m6.top1
    assert m1.top5 == m2.top5 == m6.top5

