"""
Unit and Integration Tests for ImageNet Validation Image Extraction and Manifest Generation (Step 4D.6).

Verifies:
1. Member header validation, security checks (absolute paths, parent traversal, symlinks),
   regex matching, and count validation.
2. Safe chunk-by-chunk extraction, target folder sandboxing, and limit enforcement.
3. Post-extraction verification (file presence, non-zero sizes, JPEG magic bytes).
4. Canonical CSV and JSON manifest generation, ground-truth label alignment,
   and round-trip validation with load_imagenet_manifest.
5. CLI runner operation (--help, --dry-run, --limit, error handling).
"""

import csv
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

from benchmarking.accuracy import EvaluationDataset
from benchmarking.imagenet_images import (
    JPEG_MAGIC_BYTES,
    TOTAL_VAL_IMAGES,
    VAL_IMAGE_NAME_PATTERN,
    extract_validation_images,
    generate_validation_manifest,
    scan_and_validate_val_archive,
    verify_extracted_images,
    verify_single_image_integrity,
)
from benchmarking.imagenet_manifest import load_imagenet_manifest


def _create_dummy_jpeg_bytes(width: int = 16, height: int = 16, color: tuple = (100, 150, 200)) -> bytes:
    """Create in-memory bytes of a valid RGB JPEG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_synthetic_val_tar(tar_path: Path, members: Dict[str, bytes]) -> Path:
    """Create a synthetic uncompressed TAR archive with specified member files."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:") as tar:
        for name, data in members.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return tar_path


def _create_synthetic_mapping_json(
    json_path: Path,
    labels_0_indexed: List[int],
    num_classes: int = 1000,
) -> Path:
    """Create a synthetic devkit_to_torchvision_mapping.json artifact."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "num_classes": num_classes,
        "devkit_id_to_torchvision_idx": {"1": 65, "2": 970, "3": 230},
        "torchvision_idx_to_devkit_id": {"65": 1, "970": 2, "230": 3},
        "torchvision_idx_to_wnid": {"65": "n01751748", "970": "n09193705", "230": "n02106662"},
        "torchvision_idx_to_words": {"65": "sea snake", "970": "alp", "230": "collie"},
        "provenance": {"source": "synthetic_test"},
        "validation_ground_truth": {
            "total_samples": len(labels_0_indexed),
            "labels_0_indexed": labels_0_indexed,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return json_path


# ============================================================================
# 1. Archive Scanning & Security Validation Tests
# ============================================================================

def test_scan_and_validate_valid_archive(tmp_path: Path) -> None:
    """Verify scanning valid synthetic archive returns sorted member tuples."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000003.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000002.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "test_val.tar", members)

    scanned = scan_and_validate_val_archive(tar_path, expected_count=3)
    assert len(scanned) == 3
    # Check sorted numerically
    assert [m[0] for m in scanned] == [1, 2, 3]
    assert scanned[0][1] == "ILSVRC2012_val_00000001.JPEG"
    assert scanned[1][1] == "ILSVRC2012_val_00000002.JPEG"
    assert scanned[2][1] == "ILSVRC2012_val_00000003.JPEG"
    assert scanned[0][2] == len(jpeg_bytes)


def test_scan_archive_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError if validation archive does not exist."""
    with pytest.raises(FileNotFoundError, match="Validation archive not found"):
        scan_and_validate_val_archive(tmp_path / "nonexistent.tar")


def test_scan_archive_absolute_path_rejection(tmp_path: Path) -> None:
    """Verify rejection of members with absolute paths."""
    tar_path = tmp_path / "traversal.tar"
    with tarfile.open(tar_path, "w:") as tar:
        ti = tarfile.TarInfo(name="/etc/ILSVRC2012_val_00000001.JPEG")
        ti.size = 10
        tar.addfile(ti, io.BytesIO(b"0123456789"))

    with pytest.raises(ValueError, match="absolute path"):
        scan_and_validate_val_archive(tar_path, expected_count=None)


def test_scan_archive_parent_traversal_rejection(tmp_path: Path) -> None:
    """Verify rejection of members with parent traversal (..)."""
    tar_path = tmp_path / "traversal.tar"
    with tarfile.open(tar_path, "w:") as tar:
        ti = tarfile.TarInfo(name="../ILSVRC2012_val_00000001.JPEG")
        ti.size = 10
        tar.addfile(ti, io.BytesIO(b"0123456789"))

    with pytest.raises(ValueError, match="parent traversal"):
        scan_and_validate_val_archive(tar_path, expected_count=None)


def test_scan_archive_non_regular_file_rejection(tmp_path: Path) -> None:
    """Verify rejection of non-regular files (directories, symlinks)."""
    tar_path = tmp_path / "symlink.tar"
    with tarfile.open(tar_path, "w:") as tar:
        ti = tarfile.TarInfo(name="ILSVRC2012_val_00000001.JPEG")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "some_target"
        tar.addfile(ti)

    with pytest.raises(ValueError, match="is not a regular file"):
        scan_and_validate_val_archive(tar_path, expected_count=None)


def test_scan_archive_invalid_member_name_rejection(tmp_path: Path) -> None:
    """Verify rejection of members not matching ILSVRC2012_val_\\d{8}\\.JPEG."""
    tar_path = tmp_path / "invalid_name.tar"
    with tarfile.open(tar_path, "w:") as tar:
        ti = tarfile.TarInfo(name="cat.jpg")
        ti.size = 5
        tar.addfile(ti, io.BytesIO(b"hello"))

    with pytest.raises(ValueError, match="does not match ILSVRC2012 validation format"):
        scan_and_validate_val_archive(tar_path, expected_count=None)


def test_scan_archive_duplicate_index_rejection(tmp_path: Path) -> None:
    """Verify rejection if archive contains duplicate image indices."""
    tar_path = tmp_path / "duplicates.tar"
    with tarfile.open(tar_path, "w:") as tar:
        for name in ["ILSVRC2012_val_00000001.JPEG", "ILSVRC2012_val_00000001.JPEG"]:
            ti = tarfile.TarInfo(name=name)
            ti.size = 5
            tar.addfile(ti, io.BytesIO(b"12345"))

    with pytest.raises(ValueError, match="Duplicate image index 1"):
        scan_and_validate_val_archive(tar_path, expected_count=None)


def test_scan_archive_count_mismatch(tmp_path: Path) -> None:
    """Verify ValueError when expected count does not match scanned member count."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    tar_path = _create_synthetic_val_tar(
        tmp_path / "count_test.tar",
        {"ILSVRC2012_val_00000001.JPEG": jpeg_bytes},
    )

    with pytest.raises(ValueError, match="Expected 50000 members in validation archive, got 1"):
        scan_and_validate_val_archive(tar_path, expected_count=50_000)


# ============================================================================
# 2. Image Extraction Tests
# ============================================================================

def test_extract_validation_images_all(tmp_path: Path) -> None:
    """Verify safe extraction of all synthetic images into dest_dir."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000002.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"

    result = extract_validation_images(tar_path, dest_dir=dest_dir)
    assert result["extracted_count"] == 2
    assert result["total_bytes_written"] == len(jpeg_bytes) * 2
    assert (dest_dir / "ILSVRC2012_val_00000001.JPEG").exists()
    assert (dest_dir / "ILSVRC2012_val_00000002.JPEG").exists()
    assert (dest_dir / "ILSVRC2012_val_00000001.JPEG").read_bytes() == jpeg_bytes


def test_extract_validation_images_with_limit(tmp_path: Path) -> None:
    """Verify limit parameter extracts only the requested number of images."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        f"ILSVRC2012_val_{i:08d}.JPEG": jpeg_bytes
        for i in range(1, 6)
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"

    result = extract_validation_images(tar_path, dest_dir=dest_dir, limit=3)
    assert result["extracted_count"] == 3
    assert len(result["extracted_files"]) == 3
    assert (dest_dir / "ILSVRC2012_val_00000001.JPEG").exists()
    assert (dest_dir / "ILSVRC2012_val_00000002.JPEG").exists()
    assert (dest_dir / "ILSVRC2012_val_00000003.JPEG").exists()
    assert not (dest_dir / "ILSVRC2012_val_00000004.JPEG").exists()


def test_extract_progress_callback(tmp_path: Path) -> None:
    """Verify progress callback is invoked with count, total, and filename."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000002.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)

    calls = []

    def callback(count: int, total: int, name: str) -> None:
        calls.append((count, total, name))

    extract_validation_images(
        tar_path,
        dest_dir=tmp_path / "val",
        progress_callback=callback,
    )
    assert len(calls) == 2
    assert calls[0] == (1, 2, "ILSVRC2012_val_00000001.JPEG")
    assert calls[1] == (2, 2, "ILSVRC2012_val_00000002.JPEG")


def test_extract_nonexistent_archive_raises(tmp_path: Path) -> None:
    """Verify FileNotFoundError when extracting non-existent archive."""
    with pytest.raises(FileNotFoundError, match="Validation archive not found"):
        extract_validation_images(tmp_path / "missing.tar", dest_dir=tmp_path / "dest")


def test_extract_skips_valid_existing_image(tmp_path: Path) -> None:
    """Verify existing valid image is preserved and skipped, not rewritten."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000002.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000003.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"
    dest_dir.mkdir(parents=True)

    # Pre-populate image 1 with valid JPEG content matching member size
    img1_path = dest_dir / "ILSVRC2012_val_00000001.JPEG"
    img1_path.write_bytes(jpeg_bytes)
    # Set an older mtime
    os.utime(img1_path, (1000000000, 1000000000))
    initial_mtime = img1_path.stat().st_mtime

    result = extract_validation_images(tar_path, dest_dir=dest_dir)
    assert result["extracted_count"] == 3
    assert result["newly_extracted_count"] == 2
    assert result["skipped_count"] == 1
    # Verify image 1 was NOT rewritten (mtime unchanged)
    assert img1_path.stat().st_mtime == initial_mtime
    # Images 2 and 3 exist
    assert (dest_dir / "ILSVRC2012_val_00000002.JPEG").exists()
    assert (dest_dir / "ILSVRC2012_val_00000003.JPEG").exists()


def test_extract_rejects_wrong_sized_existing_image(tmp_path: Path) -> None:
    """Verify extraction halts with ValueError if an existing file has unexpected size."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"
    dest_dir.mkdir(parents=True)

    # Pre-create image with wrong byte size
    img1_path = dest_dir / "ILSVRC2012_val_00000001.JPEG"
    img1_path.write_bytes(b"TOO_SHORT")

    with pytest.raises(ValueError, match="size mismatch|invalid or wrong-sized"):
        extract_validation_images(tar_path, dest_dir=dest_dir)


def test_extract_rejects_corrupted_existing_image(tmp_path: Path) -> None:
    """Verify extraction halts if an existing file has matching size but corrupt payload."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"
    dest_dir.mkdir(parents=True)

    # Pre-create corrupted image of same length
    corrupt_bytes = JPEG_MAGIC_BYTES + b"X" * (len(jpeg_bytes) - len(JPEG_MAGIC_BYTES))
    img1_path = dest_dir / "ILSVRC2012_val_00000001.JPEG"
    img1_path.write_bytes(corrupt_bytes)

    with pytest.raises(ValueError, match="cannot be decoded|invalid or wrong-sized"):
        extract_validation_images(tar_path, dest_dir=dest_dir)


def test_extract_cleans_up_temporary_file_on_failure(tmp_path: Path) -> None:
    """Verify temporary files are cleaned up if verification fails during extraction."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"

    # Simulate verification failure during write
    with patch("benchmarking.imagenet_images.verify_single_image_integrity", side_effect=ValueError("Simulated decode failure")):
        with pytest.raises(ValueError, match="Simulated decode failure"):
            extract_validation_images(tar_path, dest_dir=dest_dir)

    # Destination file should not exist
    assert not (dest_dir / "ILSVRC2012_val_00000001.JPEG").exists()
    # No .tmp files should remain in dest_dir
    tmp_files = list(dest_dir.glob(".tmp_*"))
    assert len(tmp_files) == 0


def test_extract_safe_promotion_race_rejection(tmp_path: Path) -> None:
    """Verify FileExistsError is raised if destination file appears before promotion."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"

    # Simulate race condition where os.link fails because destination was created concurrently
    with patch("os.link", side_effect=FileExistsError("File exists")):
        with pytest.raises(FileExistsError, match="appeared unexpectedly"):
            extract_validation_images(tar_path, dest_dir=dest_dir)

    # Temporary file must be cleaned up
    tmp_files = list(dest_dir.glob(".tmp_*"))
    assert len(tmp_files) == 0


def test_extract_resumes_after_simulated_interruption(tmp_path: Path) -> None:
    """Verify extraction resumes by skipping existing images and extracting remaining."""
    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        f"ILSVRC2012_val_{i:08d}.JPEG": jpeg_bytes
        for i in range(1, 6)
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    dest_dir = tmp_path / "extracted_val"

    # Step 1: Initial partial extraction (limit=2)
    res1 = extract_validation_images(tar_path, dest_dir=dest_dir, limit=2)
    assert res1["extracted_count"] == 2
    assert res1["newly_extracted_count"] == 2
    assert res1["skipped_count"] == 0

    # Step 2: Resume extraction for full set (limit=5)
    res2 = extract_validation_images(tar_path, dest_dir=dest_dir, limit=5)
    assert res2["extracted_count"] == 5
    assert res2["newly_extracted_count"] == 3
    assert res2["skipped_count"] == 2

    # Verify all 5 files exist and can be verified
    verify_res = verify_extracted_images(dest_dir, expected_count=5)
    assert verify_res["is_valid"] is True
    assert verify_res["verified_count"] == 5


# ============================================================================
# 3. Post-Extraction Verification Tests
# ============================================================================

def test_verify_extracted_images_valid(tmp_path: Path) -> None:
    """Verify verification succeeds on valid extracted JPEG images."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    jpeg_bytes = _create_dummy_jpeg_bytes()

    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(jpeg_bytes)
    (val_dir / "ILSVRC2012_val_00000002.JPEG").write_bytes(jpeg_bytes)

    res = verify_extracted_images(val_dir, expected_count=2, check_jpeg_headers=True)
    assert res["is_valid"] is True
    assert res["verified_count"] == 2
    assert res["min_index"] == 1
    assert res["max_index"] == 2
    assert res["total_bytes"] == len(jpeg_bytes) * 2


def test_verify_extracted_images_empty_file_rejected(tmp_path: Path) -> None:
    """Verify 0-byte file raises ValueError."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"")

    with pytest.raises(ValueError, match="empty \\(0 bytes\\) file"):
        verify_extracted_images(val_dir, expected_count=1)


def test_verify_extracted_images_corrupt_header_rejected(tmp_path: Path) -> None:
    """Verify non-JPEG bytes raise ValueError when check_jpeg_headers=True."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"NOT_A_JPEG_FILE_PAYLOAD")

    with pytest.raises(ValueError, match="does not have valid JPEG magic bytes"):
        verify_extracted_images(val_dir, expected_count=1, check_jpeg_headers=True)


def test_verify_extracted_images_count_mismatch(tmp_path: Path) -> None:
    """Verify count mismatch raises ValueError."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(_create_dummy_jpeg_bytes())

    with pytest.raises(ValueError, match="Extracted image count mismatch: Expected 5, found 1"):
        verify_extracted_images(val_dir, expected_count=5)


def test_verify_extracted_images_missing_dir(tmp_path: Path) -> None:
    """Verify FileNotFoundError if extracted images directory does not exist."""
    with pytest.raises(FileNotFoundError, match="Extracted images directory does not exist"):
        verify_extracted_images(tmp_path / "does_not_exist")


def test_verify_extracted_images_missing_expected_indices(tmp_path: Path) -> None:
    """Verify missing expected indices raise ValueError."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(_create_dummy_jpeg_bytes())

    with pytest.raises(ValueError, match="indices mismatch|Missing expected image indices"):
        verify_extracted_images(val_dir, expected_count=None, expected_indices=[1, 2])


def test_verify_extracted_images_continuity_gap_detected(tmp_path: Path) -> None:
    """Verify ValueError is raised if image sequence has gaps."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    jpeg_bytes = _create_dummy_jpeg_bytes()
    # Image 1 and 3 present, Image 2 missing
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(jpeg_bytes)
    (val_dir / "ILSVRC2012_val_00000003.JPEG").write_bytes(jpeg_bytes)

    with pytest.raises(ValueError, match="Discontinuous image sequence|indices mismatch"):
        verify_extracted_images(val_dir, expected_count=2)


# ============================================================================
# 4. Manifest Generation Tests
# ============================================================================

def test_generate_validation_manifest(tmp_path: Path) -> None:
    """Verify manifest generation creates valid CSV and JSON and validates via load_imagenet_manifest."""
    val_dir = tmp_path / "dataset" / "val"
    val_dir.mkdir(parents=True)
    jpeg_bytes = _create_dummy_jpeg_bytes()

    # Create 3 images
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(jpeg_bytes)
    (val_dir / "ILSVRC2012_val_00000002.JPEG").write_bytes(jpeg_bytes)
    (val_dir / "ILSVRC2012_val_00000003.JPEG").write_bytes(jpeg_bytes)

    # Synthetic mapping: image 1 -> label 65, image 2 -> label 970, image 3 -> label 230
    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "metadata" / "mapping.json",
        labels_0_indexed=[65, 970, 230],
    )

    csv_out = tmp_path / "dataset" / "val_manifest.csv"
    json_out = tmp_path / "dataset" / "val_manifest.json"

    result_csv = generate_validation_manifest(
        images_dir=val_dir,
        mapping_json_path=mapping_path,
        csv_output_path=csv_out,
        json_output_path=json_out,
        root_dir=tmp_path / "dataset",
    )

    assert result_csv == csv_out
    assert csv_out.exists()
    assert json_out.exists()

    # Verify CSV rows
    with open(csv_out, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["image_relpath", "true_label_idx"]
    assert reader[1] == ["val/ILSVRC2012_val_00000001.JPEG", "65"]
    assert reader[2] == ["val/ILSVRC2012_val_00000002.JPEG", "970"]
    assert reader[3] == ["val/ILSVRC2012_val_00000003.JPEG", "230"]

    # Verify JSON structure
    with open(json_out, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["dataset_name"] == "ILSVRC2012_val"
    assert manifest_data["num_samples"] == 3
    assert len(manifest_data["samples"]) == 3
    sample0 = manifest_data["samples"][0]
    assert sample0["image_relpath"] == "val/ILSVRC2012_val_00000001.JPEG"
    assert sample0["true_label_idx"] == 65
    assert sample0["devkit_id"] == 1
    assert sample0["wnid"] == "n01751748"
    assert sample0["class_name"] == "sea snake"

    # Verify compatibility with load_imagenet_manifest()
    samples = load_imagenet_manifest(csv_out, root_dir=tmp_path / "dataset")
    assert len(samples) == 3
    assert samples[0].label == 65
    assert samples[1].label == 970
    assert samples[2].label == 230


def test_generate_manifest_with_limit(tmp_path: Path) -> None:
    """Verify limit parameter caps the manifest to specified count."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    jpeg_bytes = _create_dummy_jpeg_bytes()
    for i in range(1, 5):
        (val_dir / f"ILSVRC2012_val_{i:08d}.JPEG").write_bytes(jpeg_bytes)

    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "mapping.json",
        labels_0_indexed=[10, 20, 30, 40],
    )
    csv_out = tmp_path / "manifest.csv"

    generate_validation_manifest(
        images_dir=val_dir,
        mapping_json_path=mapping_path,
        csv_output_path=csv_out,
        json_output_path=None,
        root_dir=tmp_path,
        limit=2,
    )

    with open(csv_out, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # Header + 2 rows
    assert len(rows) == 3
    assert rows[1][1] == "10"
    assert rows[2][1] == "20"


def test_generate_manifest_missing_mapping_raises(tmp_path: Path) -> None:
    """Verify FileNotFoundError when mapping artifact is missing."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(_create_dummy_jpeg_bytes())

    with pytest.raises(FileNotFoundError, match="Mapping artifact not found"):
        generate_validation_manifest(
            images_dir=val_dir,
            mapping_json_path=tmp_path / "nonexistent.json",
            csv_output_path=tmp_path / "out.csv",
        )


def test_generate_manifest_no_images_raises(tmp_path: Path) -> None:
    """Verify ValueError when target directory contains no ILSVRC2012 validation images."""
    empty_dir = tmp_path / "empty_val"
    empty_dir.mkdir()
    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "mapping.json",
        labels_0_indexed=[1, 2],
    )

    with pytest.raises(ValueError, match="No extracted validation images found"):
        generate_validation_manifest(
            images_dir=empty_dir,
            mapping_json_path=mapping_path,
            csv_output_path=tmp_path / "out.csv",
        )


def test_generate_manifest_rejects_incomplete_dataset(tmp_path: Path) -> None:
    """Verify ValueError when extracted images count is less than expected_count."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(parents=True)
    jpeg_bytes = _create_dummy_jpeg_bytes()
    # Only 2 images present
    (val_dir / "ILSVRC2012_val_00000001.JPEG").write_bytes(jpeg_bytes)
    (val_dir / "ILSVRC2012_val_00000002.JPEG").write_bytes(jpeg_bytes)

    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "mapping.json",
        labels_0_indexed=[1, 2, 3, 4, 5],
    )

    with pytest.raises(ValueError, match="Expected 5 extracted images to generate manifest, but found 2"):
        generate_validation_manifest(
            images_dir=val_dir,
            mapping_json_path=mapping_path,
            csv_output_path=tmp_path / "out.csv",
            expected_count=5,
        )


# ============================================================================
# 5. CLI Runner Tests
# ============================================================================

def test_cli_dry_run_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI --dry-run option performs non-destructive pre-checks and exits 0."""
    from benchmarking.extract_validation_images import main

    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        "ILSVRC2012_val_00000001.JPEG": jpeg_bytes,
        "ILSVRC2012_val_00000002.JPEG": jpeg_bytes,
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "mapping.json",
        labels_0_indexed=[65, 970],
    )
    dest_dir = tmp_path / "val"

    test_args = [
        "extract_validation_images",
        "--val-tar", str(tar_path),
        "--dest-dir", str(dest_dir),
        "--mapping-json", str(mapping_path),
        "--limit", "2",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    ret = main()
    assert ret == 0
    # Confirm no files were extracted
    assert not dest_dir.exists()


def test_cli_full_flow_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI end-to-end extraction, verification, and manifest generation on synthetic data."""
    from benchmarking.extract_validation_images import main

    jpeg_bytes = _create_dummy_jpeg_bytes()
    members = {
        f"ILSVRC2012_val_{i:08d}.JPEG": jpeg_bytes
        for i in range(1, 4)
    }
    tar_path = _create_synthetic_val_tar(tmp_path / "val.tar", members)
    mapping_path = _create_synthetic_mapping_json(
        tmp_path / "mapping.json",
        labels_0_indexed=[100, 200, 300],
    )
    dest_dir = tmp_path / "dataset" / "val"
    csv_out = tmp_path / "dataset" / "val_manifest.csv"
    json_out = tmp_path / "dataset" / "val_manifest.json"

    test_args = [
        "extract_validation_images",
        "--val-tar", str(tar_path),
        "--dest-dir", str(dest_dir),
        "--mapping-json", str(mapping_path),
        "--csv-out", str(csv_out),
        "--json-out", str(json_out),
        "--root-dir", str(tmp_path / "dataset"),
        "--limit", "3",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    ret = main()
    assert ret == 0

    assert dest_dir.exists()
    assert len(list(dest_dir.glob("*.JPEG"))) == 3
    assert csv_out.exists()
    assert json_out.exists()


def test_cli_missing_tar_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI exits with code 1 if validation archive is missing."""
    from benchmarking.extract_validation_images import main

    test_args = [
        "extract_validation_images",
        "--val-tar", str(tmp_path / "missing.tar"),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    ret = main()
    assert ret == 1


def test_cli_missing_mapping_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI exits with code 1 if mapping JSON is missing."""
    from benchmarking.extract_validation_images import main

    tar_path = _create_synthetic_val_tar(
        tmp_path / "val.tar",
        {"ILSVRC2012_val_00000001.JPEG": _create_dummy_jpeg_bytes()},
    )

    test_args = [
        "extract_validation_images",
        "--val-tar", str(tar_path),
        "--mapping-json", str(tmp_path / "missing_map.json"),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    ret = main()
    assert ret == 1
