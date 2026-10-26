"""
Unit and Integration Tests for ImageNet Manifest Loader (Step 4D.2).

Verifies:
1. Valid CSV manifest parsing with standard and aliased column names.
2. Valid JSON manifest parsing across lists, sample objects, and key-value formats.
3. Path resolution against explicit root_dir and default manifest parent folder.
4. Validation and rejection of missing files, invalid labels, unsupported extensions,
   duplicate entries, malformed syntax, and empty manifests.
5. WNID mapping validation and translation.
6. Integration with EvaluationDataset, resolve_dataset, and BenchmarkEngine.
"""

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import pytest

from benchmarking.accuracy import EvaluationDataset, LabeledImageSample, resolve_dataset
from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine
from benchmarking.imagenet_manifest import (
    IMAGENET_NUM_CLASSES,
    MAX_CLASS_IDX,
    MIN_CLASS_IDX,
    VALID_IMAGE_EXTENSIONS,
    create_dataset_from_manifest,
    load_imagenet_manifest,
    validate_wnid_mapping,
)


def _create_dummy_image(path: Path) -> Path:
    """Helper to create a small valid RGB JPEG/PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    img.save(str(path))
    return path


# ============================================================================
# 1. Valid CSV Manifest Tests
# ============================================================================
def test_load_manifest_csv_standard(tmp_path: Path) -> None:
    """Verify loading standard CSV with image_relpath and true_label_idx."""
    img1 = _create_dummy_image(tmp_path / "images" / "img1.JPEG")
    img2 = _create_dummy_image(tmp_path / "images" / "img2.jpg")

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["images/img1.JPEG", "0"])
        writer.writerow(["images/img2.jpg", "999"])

    samples = load_imagenet_manifest(csv_path, root_dir=tmp_path)
    assert len(samples) == 2
    assert samples[0].image == img1.resolve()
    assert samples[0].label == 0
    assert samples[1].image == img2.resolve()
    assert samples[1].label == 999


def test_load_manifest_csv_aliases(tmp_path: Path) -> None:
    """Verify loading CSV with aliased column headers (image_path, class_idx)."""
    img = _create_dummy_image(tmp_path / "data" / "cat.png")

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_idx"])
        writer.writerow(["data/cat.png", "281"])

    samples = load_imagenet_manifest(csv_path, root_dir=tmp_path)
    assert len(samples) == 1
    assert samples[0].image == img.resolve()
    assert samples[0].label == 281


def test_load_manifest_default_root_dir(tmp_path: Path) -> None:
    """Verify relative paths resolve to manifest parent directory when root_dir is None."""
    img = _create_dummy_image(tmp_path / "sample.jpg")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["sample.jpg", "42"])

    samples = load_imagenet_manifest(csv_path)
    assert len(samples) == 1
    assert samples[0].image == img.resolve()
    assert samples[0].label == 42


# ============================================================================
# 2. Valid JSON Manifest Tests
# ============================================================================
def test_load_manifest_json_list(tmp_path: Path) -> None:
    """Verify loading JSON manifest structured as a list of record objects."""
    img1 = _create_dummy_image(tmp_path / "img1.png")
    img2 = _create_dummy_image(tmp_path / "img2.webp")

    json_path = tmp_path / "manifest.json"
    data = [
        {"image_relpath": "img1.png", "true_label_idx": 15},
        {"image_relpath": "img2.webp", "true_label_idx": 800},
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    samples = load_imagenet_manifest(json_path, root_dir=tmp_path)
    assert len(samples) == 2
    assert samples[0].label == 15
    assert samples[1].label == 800


def test_load_manifest_json_dict_samples(tmp_path: Path) -> None:
    """Verify loading JSON manifest with {'samples': [...]} structure."""
    img = _create_dummy_image(tmp_path / "sub" / "dog.jpeg")
    json_path = tmp_path / "manifest.json"
    data = {
        "samples": [
            {"image_path": "sub/dog.jpeg", "label": 263}
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    samples = load_imagenet_manifest(json_path, root_dir=tmp_path)
    assert len(samples) == 1
    assert samples[0].image == img.resolve()
    assert samples[0].label == 263


def test_load_manifest_json_key_value(tmp_path: Path) -> None:
    """Verify loading JSON manifest structured as a path -> label mapping."""
    img1 = _create_dummy_image(tmp_path / "1.jpg")
    img2 = _create_dummy_image(tmp_path / "2.jpg")

    json_path = tmp_path / "manifest.json"
    data = {
        "1.jpg": 100,
        "2.jpg": 200,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    samples = load_imagenet_manifest(json_path, root_dir=tmp_path)
    assert len(samples) == 2
    assert samples[0].label == 100
    assert samples[1].label == 200


# ============================================================================
# 3. Validation & Error Handling Tests
# ============================================================================
def test_manifest_missing_manifest_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError if manifest file does not exist."""
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_imagenet_manifest(tmp_path / "nonexistent.csv")


def test_manifest_missing_root_dir(tmp_path: Path) -> None:
    """Verify FileNotFoundError if explicit root_dir does not exist."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_relpath,true_label_idx\n")
    with pytest.raises(FileNotFoundError, match="Dataset root directory does not exist"):
        load_imagenet_manifest(manifest, root_dir=tmp_path / "missing_dir")


def test_manifest_empty_file(tmp_path: Path) -> None:
    """Verify ValueError when manifest file is empty."""
    manifest = tmp_path / "empty.csv"
    manifest.write_text("")
    with pytest.raises(ValueError, match="CSV manifest '.*' has no header row"):
        load_imagenet_manifest(manifest)

    json_manifest = tmp_path / "empty.json"
    json_manifest.write_text("[]")
    with pytest.raises(ValueError, match="is empty; no image records found"):
        load_imagenet_manifest(json_manifest)


def test_manifest_missing_image_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError if referenced image does not exist on disk."""
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["missing_image.JPEG", "5"])

    with pytest.raises(FileNotFoundError, match="Referenced image file does not exist"):
        load_imagenet_manifest(csv_path, root_dir=tmp_path)


def test_manifest_unsupported_image_extension(tmp_path: Path) -> None:
    """Verify ValueError when referenced image has unsupported extension (e.g. .txt)."""
    bad_file = tmp_path / "file.txt"
    bad_file.write_text("not an image")

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["file.txt", "10"])

    with pytest.raises(ValueError, match="Unsupported image extension '.txt'"):
        load_imagenet_manifest(csv_path, root_dir=tmp_path)


def test_manifest_duplicate_image_path(tmp_path: Path) -> None:
    """Verify ValueError when manifest references the same image file multiple times."""
    img = _create_dummy_image(tmp_path / "img.jpg")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["img.jpg", "10"])
        writer.writerow(["img.jpg", "20"])

    with pytest.raises(ValueError, match="Duplicate image entry found"):
        load_imagenet_manifest(csv_path, root_dir=tmp_path)


@pytest.mark.parametrize("invalid_label", [
    "-1",       # negative
    "1000",     # out of upper bound
    "5.5",      # decimal string
    "abc",      # non-numeric
    "",         # empty
])
def test_manifest_invalid_label_values(tmp_path: Path, invalid_label: str) -> None:
    """Verify ValueError on various invalid label formats and ranges."""
    _create_dummy_image(tmp_path / "img.JPEG")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["img.JPEG", invalid_label])

    with pytest.raises(ValueError):
        load_imagenet_manifest(csv_path, root_dir=tmp_path)


def test_manifest_boolean_label_rejected(tmp_path: Path) -> None:
    """Verify boolean label in JSON is rejected (bool is subclass of int in Python)."""
    _create_dummy_image(tmp_path / "img.JPEG")
    json_path = tmp_path / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"image_relpath": "img.JPEG", "true_label_idx": True}], f)

    with pytest.raises(ValueError, match="Label cannot be a boolean value"):
        load_imagenet_manifest(json_path, root_dir=tmp_path)


def test_manifest_missing_label_field(tmp_path: Path) -> None:
    """Verify ValueError when record does not specify a label or WNID."""
    _create_dummy_image(tmp_path / "img.JPEG")
    json_path = tmp_path / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"image_relpath": "img.JPEG"}], f)

    with pytest.raises(ValueError, match="missing ground-truth label"):
        load_imagenet_manifest(json_path, root_dir=tmp_path)


def test_manifest_malformed_json(tmp_path: Path) -> None:
    """Verify ValueError when JSON file has syntax errors."""
    json_path = tmp_path / "broken.json"
    json_path.write_text("{ broken json content")

    with pytest.raises(ValueError, match="Malformed JSON in manifest"):
        load_imagenet_manifest(json_path, root_dir=tmp_path)


# ============================================================================
# 4. WNID Mapping Tests
# ============================================================================
def test_wnid_mapping_dict(tmp_path: Path) -> None:
    """Verify resolving labels using an explicit WNID mapping dict."""
    img = _create_dummy_image(tmp_path / "tench.JPEG")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "wnid"])
        writer.writerow(["tench.JPEG", "n01440764"])

    wnid_map = {"n01440764": 0}
    samples = load_imagenet_manifest(csv_path, root_dir=tmp_path, wnid_to_idx=wnid_map)
    assert len(samples) == 1
    assert samples[0].label == 0


def test_wnid_mapping_missing_lookup(tmp_path: Path) -> None:
    """Verify KeyError when record specifies a WNID not present in the mapping."""
    _create_dummy_image(tmp_path / "shark.JPEG")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "wnid"])
        writer.writerow(["shark.JPEG", "n01484850"])

    wnid_map = {"n01440764": 0}  # missing n01484850
    with pytest.raises(KeyError, match="not found in provided WNID-to-index mapping"):
        load_imagenet_manifest(csv_path, root_dir=tmp_path, wnid_to_idx=wnid_map)


def test_wnid_mapping_no_mapping_provided(tmp_path: Path) -> None:
    """Verify ValueError when record uses WNID but wnid_to_idx is None."""
    _create_dummy_image(tmp_path / "img.JPEG")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "wnid"])
        writer.writerow(["img.JPEG", "n01440764"])

    with pytest.raises(ValueError, match="no wnid_to_idx mapping was provided"):
        load_imagenet_manifest(csv_path, root_dir=tmp_path, wnid_to_idx=None)


def test_validate_wnid_mapping_from_json(tmp_path: Path) -> None:
    """Verify validate_wnid_mapping loading from a JSON file."""
    json_path = tmp_path / "wnid_map.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n01440764": 0, "n01443537": 1}, f)

    validated = validate_wnid_mapping(json_path)
    assert validated == {"n01440764": 0, "n01443537": 1}


def test_validate_wnid_mapping_from_csv(tmp_path: Path) -> None:
    """Verify validate_wnid_mapping loading from a CSV file."""
    csv_path = tmp_path / "wnid_map.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wnid", "class_idx"])
        writer.writerow(["n01440764", "0"])
        writer.writerow(["n01443537", "1"])

    validated = validate_wnid_mapping(csv_path)
    assert validated == {"n01440764": 0, "n01443537": 1}


def test_validate_wnid_mapping_invalid_index() -> None:
    """Verify ValueError when mapping index is out of bounds or invalid."""
    with pytest.raises(ValueError, match="out of range"):
        validate_wnid_mapping({"n01440764": 1000})

    with pytest.raises(ValueError, match="out of range"):
        validate_wnid_mapping({"n01440764": -1})

    with pytest.raises(ValueError, match="cannot be boolean"):
        validate_wnid_mapping({"n01440764": True})


def test_validate_wnid_mapping_require_full_1000() -> None:
    """Verify require_full_1000 enforces all 1,000 classes."""
    incomplete = {f"n{i:08d}": i for i in range(10)}
    with pytest.raises(ValueError, match="Complete ImageNet mapping requires 1000 classes"):
        validate_wnid_mapping(incomplete, require_full_1000=True)

    complete = {f"n{i:08d}": i for i in range(1000)}
    validated = validate_wnid_mapping(complete, require_full_1000=True)
    assert len(validated) == 1000


# ============================================================================
# 5. Dataset Integration Tests
# ============================================================================
def test_create_dataset_from_manifest(tmp_path: Path) -> None:
    """Verify create_dataset_from_manifest returns an EvaluationDataset."""
    _create_dummy_image(tmp_path / "img1.JPEG")
    _create_dummy_image(tmp_path / "img2.JPEG")

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["img1.JPEG", "50"])
        writer.writerow(["img2.JPEG", "75"])

    dataset = create_dataset_from_manifest(csv_path, root_dir=tmp_path, name="test_ds")
    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) == 2
    assert dataset.name == "test_ds"

    # Verify items are loaded as PIL Images and correct labels
    img, label = dataset[0]
    assert isinstance(img, Image.Image)
    assert label == 50


def test_evaluation_dataset_from_manifest_classmethod(tmp_path: Path) -> None:
    """Verify EvaluationDataset.from_manifest() classmethod."""
    _create_dummy_image(tmp_path / "a.png")
    json_path = tmp_path / "m.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"image_relpath": "a.png", "true_label_idx": 12}], f)

    ds = EvaluationDataset.from_manifest(json_path, root_dir=tmp_path)
    assert len(ds) == 1
    assert ds[0][1] == 12


def test_evaluation_dataset_init_with_manifest_path(tmp_path: Path) -> None:
    """Verify EvaluationDataset(dataset_path=csv_path) automatically detects manifest."""
    _create_dummy_image(tmp_path / "sample.jpg")
    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["sample.jpg", "88"])

    ds = EvaluationDataset(dataset_path=csv_path, root_dir=tmp_path)
    assert len(ds) == 1
    assert ds[0][1] == 88


def test_resolve_dataset_with_manifest_path(tmp_path: Path) -> None:
    """Verify resolve_dataset correctly resolves a manifest file path."""
    _create_dummy_image(tmp_path / "cat.jpg")
    _create_dummy_image(tmp_path / "dog.jpg")
    json_path = tmp_path / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {"image_relpath": str((tmp_path / "cat.jpg").resolve()), "true_label_idx": 281},
            {"image_relpath": str((tmp_path / "dog.jpg").resolve()), "true_label_idx": 263},
        ], f)

    items = resolve_dataset(json_path, max_samples=1)
    assert len(items) == 1
    assert isinstance(items[0][0], Image.Image)
    assert items[0][1] == 281


def test_benchmark_engine_accuracy_with_manifest(tmp_path: Path) -> None:
    """Verify BenchmarkEngine.benchmark_accuracy() consuming a manifest file."""
    # Create 2 dummy images
    _create_dummy_image(tmp_path / "img1.JPEG")
    _create_dummy_image(tmp_path / "img2.JPEG")

    csv_path = tmp_path / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        writer.writerow(["img1.JPEG", "0"])
        writer.writerow(["img2.JPEG", "1"])

    ds = EvaluationDataset.from_manifest(csv_path, root_dir=tmp_path)
    config = BenchmarkConfig(model_id="mobilenet_v3_small", runtime=BenchmarkRuntime.PYTORCH, batch_size=2)
    engine = BenchmarkEngine(config)

    metrics = engine.benchmark_accuracy(dataset=ds, max_samples=2)
    assert metrics.num_samples == 2
    assert 0.0 <= metrics.top1 <= 1.0
    assert 0.0 <= metrics.top5 <= 1.0
