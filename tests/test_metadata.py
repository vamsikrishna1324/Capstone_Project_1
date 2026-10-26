"""
Unit and Integration Tests for ImageNet Metadata Extraction & Mapping (Step 4D.5).

Verifies:
1. Safe devkit archive extraction, path traversal rejection, link rejection,
   and member isolation.
2. Ground-truth label parsing, 50,000-sample count validation, 1..1000 ID range check,
   and distribution statistics.
3. Synset metadata parsing (with scipy availability checks and mock parsing).
4. Canonical torchvision 0-based class mapping derivation and bidirectional lookup.
5. Mapping artifact generation with complete provenance and validation ground truth.
6. Integration with validate_wnid_mapping in benchmarking.imagenet_manifest.
7. CLI runner parameter handling, help, and dry-run behavior.
"""

import io
import json
from pathlib import Path
import tarfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from benchmarking.imagenet_manifest import validate_wnid_mapping
from benchmarking.imagenet_metadata import (
    DEFAULT_DEVKIT_TAR,
    DEFAULT_METADATA_DIR,
    EXPECTED_DEVKIT_MEMBERS,
    GroundTruthStatistics,
    derive_torchvision_class_mapping,
    generate_mapping_artifact,
    parse_and_validate_ground_truth,
    parse_meta_mat,
    safe_extract_devkit_metadata,
)


def _create_synthetic_devkit_tar(
    tar_path: Path,
    members: Dict[str, bytes],
) -> Path:
    """Helper to create a temporary tar.gz archive with specified member contents."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in members.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return tar_path


def _generate_synthetic_ground_truth_bytes(num_samples: int = 50_000, num_classes: int = 1_000) -> bytes:
    """Generate 50,000 synthetic ground truth lines (50 samples per class, IDs 1..1000)."""
    samples_per_class = num_samples // num_classes
    labels: List[int] = []
    for cls_id in range(1, num_classes + 1):
        labels.extend([cls_id] * samples_per_class)
    # Remaining if not perfectly divisible
    while len(labels) < num_samples:
        labels.append(1)
    text = "\n".join(str(lbl) for lbl in labels) + "\n"
    return text.encode("utf-8")


# ============================================================================
# 1. Safe Devkit Extraction Tests
# ============================================================================
def test_safe_extract_devkit_metadata_valid(tmp_path: Path) -> None:
    """Verify safe extraction of exactly the two required metadata files."""
    gt_bytes = _generate_synthetic_ground_truth_bytes(num_samples=100, num_classes=10)
    meta_bytes = b"MATLAB_DUMMY_DATA"

    archive_path = tmp_path / "devkit.tar.gz"
    _create_synthetic_devkit_tar(
        archive_path,
        {
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt": gt_bytes,
            "ILSVRC2012_devkit_t12/data/meta.mat": meta_bytes,
            "ILSVRC2012_devkit_t12/readme.txt": b"ignored documentation",
        },
    )

    dest_dir = tmp_path / "metadata"
    extracted = safe_extract_devkit_metadata(archive_path, dest_dir=dest_dir)

    assert "ground_truth" in extracted
    assert "meta_mat" in extracted
    assert extracted["ground_truth"].name == "ILSVRC2012_validation_ground_truth.txt"
    assert extracted["meta_mat"].name == "meta.mat"

    # Verify only the 2 files exist in dest_dir (readme.txt was NOT extracted)
    extracted_filenames = sorted(p.name for p in dest_dir.iterdir())
    assert extracted_filenames == ["ILSVRC2012_validation_ground_truth.txt", "meta.mat"]
    assert extracted["ground_truth"].read_bytes() == gt_bytes
    assert extracted["meta_mat"].read_bytes() == meta_bytes


def test_safe_extract_devkit_missing_archive(tmp_path: Path) -> None:
    """Verify FileNotFoundError if devkit archive does not exist."""
    with pytest.raises(FileNotFoundError, match="Devkit archive not found"):
        safe_extract_devkit_metadata(tmp_path / "nonexistent.tar.gz", tmp_path / "dest")


def test_safe_extract_devkit_missing_required_member(tmp_path: Path) -> None:
    """Verify ValueError when devkit is missing one of the required members."""
    archive_path = tmp_path / "incomplete_devkit.tar.gz"
    _create_synthetic_devkit_tar(
        archive_path,
        {
            "ILSVRC2012_devkit_t12/data/meta.mat": b"data",
            # Missing ILSVRC2012_validation_ground_truth.txt
        },
    )

    with pytest.raises(ValueError, match="missing expected metadata members"):
        safe_extract_devkit_metadata(archive_path, tmp_path / "dest")


def test_safe_extract_devkit_rejects_parent_traversal(tmp_path: Path) -> None:
    """Verify ValueError when archive contains path traversal (..) components."""
    archive_path = tmp_path / "traversal_devkit.tar.gz"
    _create_synthetic_devkit_tar(
        archive_path,
        {
            "../etc/passwd": b"evil",
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt": b"gt",
            "ILSVRC2012_devkit_t12/data/meta.mat": b"meta",
        },
    )

    with pytest.raises(ValueError, match="Security violation: Archive member contains parent traversal"):
        safe_extract_devkit_metadata(archive_path, tmp_path / "dest")


def test_safe_extract_devkit_rejects_absolute_paths(tmp_path: Path) -> None:
    """Verify ValueError when archive member has an absolute path."""
    archive_path = tmp_path / "abs_devkit.tar.gz"
    _create_synthetic_devkit_tar(
        archive_path,
        {
            "/tmp/evil.txt": b"evil",
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt": b"gt",
            "ILSVRC2012_devkit_t12/data/meta.mat": b"meta",
        },
    )

    with pytest.raises(ValueError, match="Security violation: Archive member has absolute path"):
        safe_extract_devkit_metadata(archive_path, tmp_path / "dest")


def test_safe_extract_devkit_rejects_symlinks(tmp_path: Path) -> None:
    """Verify ValueError when an expected metadata member is a link."""
    archive_path = tmp_path / "symlink_devkit.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add a symlink for meta.mat
        ti = tarfile.TarInfo(name="ILSVRC2012_devkit_t12/data/meta.mat")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/shadow"
        tar.addfile(ti)

        ti_gt = tarfile.TarInfo(name="ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        ti_gt.size = 2
        tar.addfile(ti_gt, io.BytesIO(b"1\n"))

    with pytest.raises(ValueError, match="Security violation: Expected metadata member .* is a link"):
        safe_extract_devkit_metadata(archive_path, tmp_path / "dest")


# ============================================================================
# 2. Ground-Truth Parsing and Validation Tests
# ============================================================================
def test_parse_and_validate_ground_truth_valid(tmp_path: Path) -> None:
    """Verify parsing and distribution calculation of valid 50,000-sample ground truth."""
    gt_file = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    gt_file.write_bytes(_generate_synthetic_ground_truth_bytes(num_samples=50_000, num_classes=1_000))

    labels, stats = parse_and_validate_ground_truth(gt_file)
    assert len(labels) == 50_000
    assert stats.total_samples == 50_000
    assert stats.num_unique_classes == 1_000
    assert stats.min_label == 1
    assert stats.max_label == 1_000
    assert stats.samples_per_class_min == 50
    assert stats.samples_per_class_max == 50
    assert stats.is_balanced_50_per_class is True


def test_parse_ground_truth_missing_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError if ground truth file does not exist."""
    with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
        parse_and_validate_ground_truth(tmp_path / "nonexistent.txt")


def test_parse_ground_truth_invalid_sample_count(tmp_path: Path) -> None:
    """Verify ValueError when ground truth has fewer or more than 50,000 lines."""
    gt_file = tmp_path / "short_gt.txt"
    gt_file.write_text("1\n2\n3\n")

    with pytest.raises(ValueError, match="Expected 50000 lines, got 3"):
        parse_and_validate_ground_truth(gt_file)


@pytest.mark.parametrize("invalid_val", ["0", "1001", "-5", "abc", "5.5"])
def test_parse_ground_truth_invalid_label_values(tmp_path: Path, invalid_val: str) -> None:
    """Verify ValueError on out-of-range or non-integer ground truth labels."""
    lines = ["1"] * 49_999 + [invalid_val]
    gt_file = tmp_path / "bad_gt.txt"
    gt_file.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError):
        parse_and_validate_ground_truth(gt_file)


def test_parse_ground_truth_empty_line(tmp_path: Path) -> None:
    """Verify ValueError when file contains an empty line."""
    lines = ["1"] * 25_000 + [""] + ["1"] * 24_999
    gt_file = tmp_path / "empty_line_gt.txt"
    gt_file.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="contains an empty line"):
        parse_and_validate_ground_truth(gt_file)


# ============================================================================
# 3. Class Mapping Derivation & Validation Tests
# ============================================================================
def test_derive_torchvision_class_mapping() -> None:
    """Verify derivation of canonical 0-based torchvision class index mapping."""
    # Synthetic 1,000 synsets with reverse devkit IDs to ensure sorting is tested
    id_to_wnid = {i + 1: f"n{1000 - i:08d}" for i in range(1000)}
    wnid_to_words = {f"n{1000 - i:08d}": f"class_{1000 - i}" for i in range(1000)}

    mapping = derive_torchvision_class_mapping(id_to_wnid, wnid_to_words)
    assert mapping["num_classes"] == 1000
    assert len(mapping["devkit_id_to_torchvision_idx"]) == 1000
    assert len(mapping["torchvision_idx_to_devkit_id"]) == 1000
    assert len(mapping["torchvision_idx_to_wnid"]) == 1000
    assert len(mapping["wnid_to_torchvision_idx"]) == 1000

    # Sorted WNIDs: n00000001 (tv idx 0), n00000002 (tv idx 1), ..., n00001000 (tv idx 999)
    assert mapping["torchvision_idx_to_wnid"]["0"] == "n00000001"
    assert mapping["torchvision_idx_to_wnid"]["999"] == "n00001000"

    # Bidirectional consistency check
    for tv_idx_str, devkit_id in mapping["torchvision_idx_to_devkit_id"].items():
        assert mapping["devkit_id_to_torchvision_idx"][str(devkit_id)] == int(tv_idx_str)


def test_derive_torchvision_mapping_incomplete_synsets() -> None:
    """Verify ValueError when fewer than 1,000 synsets are provided."""
    id_to_wnid = {1: "n01440764", 2: "n01443537"}
    with pytest.raises(ValueError, match="Complete mapping requires 1000 classes"):
        derive_torchvision_class_mapping(id_to_wnid, {})


# ============================================================================
# 4. Meta.mat Parsing Tests (ImportError and Mocked Parsing)
# ============================================================================
def test_parse_meta_mat_missing_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError if meta.mat does not exist."""
    with pytest.raises(FileNotFoundError, match="meta.mat file not found"):
        parse_meta_mat(tmp_path / "nonexistent.mat")


def test_parse_meta_mat_missing_scipy(tmp_path: Path) -> None:
    """Verify informative ImportError if scipy is not installed."""
    dummy_mat = tmp_path / "meta.mat"
    dummy_mat.write_bytes(b"dummy")

    with patch.dict("sys.modules", {"scipy": None, "scipy.io": None}):
        with pytest.raises(ImportError, match="scipy is required to parse MATLAB v5 meta.mat files"):
            parse_meta_mat(dummy_mat)


def test_parse_meta_mat_mocked_success(tmp_path: Path) -> None:
    """Verify parsing when scipy.io.loadmat returns valid 1,000 leaf synset records."""
    dummy_mat = tmp_path / "meta.mat"
    dummy_mat.write_bytes(b"dummy")

    # Construct 1,000 leaf synset items + 10 non-leaf parent synsets (num_children > 0)
    mock_synsets = []
    # 1,000 leaf nodes: num_children = 0
    for i in range(1, 1001):
        mock_synsets.append([i, f"n{i:08d}", f"name_{i}", f"gloss_{i}", 0])
    # 10 non-leaf nodes: num_children = 2
    for j in range(1001, 1011):
        mock_synsets.append([j, f"n{j:08d}", f"parent_{j}", f"gloss_{j}", 2])

    mock_scipy = MagicMock()
    mock_io = MagicMock()
    mock_scipy.io = mock_io
    mock_io.loadmat.return_value = {"synsets": mock_synsets}

    with patch.dict("sys.modules", {"scipy": mock_scipy, "scipy.io": mock_io}):
        result = parse_meta_mat(dummy_mat)

    assert result["total_leaf_synsets"] == 1000
    assert len(result["id_to_wnid"]) == 1000
    assert result["id_to_wnid"][1] == "n00000001"
    assert result["id_to_wnid"][1000] == "n00001000"


def test_parse_meta_mat_mocked_invalid_leaf_count(tmp_path: Path) -> None:
    """Verify ValueError if filtered leaf count is not exactly 1,000."""
    dummy_mat = tmp_path / "meta.mat"
    dummy_mat.write_bytes(b"dummy")

    mock_synsets = [[1, "n00000001", "name", "gloss", 0]]  # Only 1 leaf
    mock_scipy = MagicMock()
    mock_io = MagicMock()
    mock_scipy.io = mock_io
    mock_io.loadmat.return_value = {"synsets": mock_synsets}

    with patch.dict("sys.modules", {"scipy": mock_scipy, "scipy.io": mock_io}):
        with pytest.raises(ValueError, match="Expected 1000 leaf classification synsets"):
            parse_meta_mat(dummy_mat)


# ============================================================================
# 5. Artifact Generation and Manifest Integration Tests
# ============================================================================
def test_generate_mapping_artifact_and_manifest_integration(tmp_path: Path) -> None:
    """Verify generating JSON mapping artifact and using it in validate_wnid_mapping."""
    dummy_mat = tmp_path / "meta.mat"
    dummy_mat.write_bytes(b"dummy")

    gt_file = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    gt_file.write_bytes(_generate_synthetic_ground_truth_bytes(num_samples=50_000, num_classes=1_000))

    # Mock parse_meta_mat to return 1,000 valid leaf synsets
    id_to_wnid = {i: f"n{i:08d}" for i in range(1, 1001)}
    wnid_to_id = {f"n{i:08d}": i for i in range(1, 1001)}
    wnid_to_words = {f"n{i:08d}": f"class_{i}" for i in range(1, 1001)}
    mock_parsed = {
        "id_to_wnid": id_to_wnid,
        "wnid_to_id": wnid_to_id,
        "wnid_to_words": wnid_to_words,
        "total_leaf_synsets": 1000,
    }

    with patch("benchmarking.imagenet_metadata.parse_meta_mat", return_value=mock_parsed):
        artifact_path = generate_mapping_artifact(
            meta_mat_path=dummy_mat,
            gt_file_path=gt_file,
            dest_dir=tmp_path / "metadata",
        )

    assert artifact_path.exists()
    assert artifact_path.is_file()

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["num_classes"] == 1000
    assert "provenance" in data
    assert "validation_ground_truth" in data
    assert len(data["validation_ground_truth"]["labels_0_indexed"]) == 50_000
    assert len(data["validation_ground_truth"]["first_10_samples"]) == 10
    assert data["validation_ground_truth"]["statistics"]["is_balanced_50_per_class"] is True

    # Test integration with validate_wnid_mapping from Step 4D.2
    # The generated artifact JSON can be loaded directly as a WNID mapping!
    validated_mapping = validate_wnid_mapping(artifact_path, require_full_1000=True)
    assert len(validated_mapping) == 1000
    assert validated_mapping["n00000001"] == 0
    assert validated_mapping["n00001000"] == 999


# ============================================================================
# 6. CLI Extraction Tool Tests
# ============================================================================
def test_cli_extract_metadata_dry_run(tmp_path: Path) -> None:
    """Verify CLI --dry-run exits with code 0 without creating output files."""
    dummy_tar = tmp_path / "fake_devkit.tar.gz"
    dummy_tar.write_bytes(b"dummy")

    from benchmarking.extract_metadata import main
    with patch("sys.argv", ["extract_metadata", "--devkit-tar", str(dummy_tar), "--dry-run"]):
        ret = main()
    assert ret == 0

    dest_dir = tmp_path / "metadata"
    assert not dest_dir.exists()


def test_cli_extract_metadata_missing_archive(tmp_path: Path) -> None:
    """Verify CLI returns exit code 1 if devkit archive does not exist."""
    from benchmarking.extract_metadata import main
    with patch("sys.argv", ["extract_metadata", "--devkit-tar", str(tmp_path / "missing.tar.gz")]):
        ret = main()
    assert ret == 1
