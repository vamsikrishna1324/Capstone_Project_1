"""
ImageNet Validation Metadata Extraction and Mapping Utility (Step 4D.5).

Safely extracts validation ground truth and synset metadata from the official
ILSVRC2012 development kit archive, validates ground-truth distribution, and
derives the explicit mapping between 1-based ILSVRC devkit IDs, WNIDs, and the
0-based class indices used by torchvision pretrained models.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tarfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union


EXPECTED_DEVKIT_MEMBERS: Set[str] = {
    "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
    "ILSVRC2012_devkit_t12/data/meta.mat",
}

DEFAULT_DEVKIT_TAR: Path = Path("dataset/raw/ILSVRC2012_devkit_t12.tar.gz")
DEFAULT_METADATA_DIR: Path = Path("dataset/metadata")

NUM_VAL_IMAGES: int = 50_000
NUM_CLASSES: int = 1_000
MIN_DEVKIT_ID: int = 1
MAX_DEVKIT_ID: int = 1_000


@dataclass
class GroundTruthStatistics:
    """Summary statistics for parsed ImageNet validation ground truth."""

    total_samples: int
    num_unique_classes: int
    min_label: int
    max_label: int
    samples_per_class_min: int
    samples_per_class_max: int
    is_balanced_50_per_class: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)


def safe_extract_devkit_metadata(
    devkit_tar_path: Union[str, Path] = DEFAULT_DEVKIT_TAR,
    dest_dir: Union[str, Path] = DEFAULT_METADATA_DIR,
) -> Dict[str, Path]:
    """Safely extract only the validation ground-truth file and meta.mat from the devkit archive.

    Strict security checks:
    - Only extracts members explicitly listed in EXPECTED_DEVKIT_MEMBERS.
    - Rejects absolute paths, parent directory traversal (..), symlinks, and special files.
    - Does not use unrestricted extractall().
    - Preserves clean filenames directly inside dest_dir.

    Args:
        devkit_tar_path: Path to ILSVRC2012_devkit_t12.tar.gz.
        dest_dir: Target directory for extracted metadata files.

    Returns:
        Dictionary mapping logical keys ('ground_truth', 'meta_mat') to resolved Path objects.

    Raises:
        FileNotFoundError: If the devkit archive does not exist.
        ValueError: If archive contains path traversal, unexpected members, or is corrupted.
    """
    tar_path = Path(devkit_tar_path)
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"Devkit archive not found: {tar_path}")

    target_dir = Path(dest_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: Dict[str, Path] = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        # Pre-scan and validate all members in archive
        member_map: Dict[str, tarfile.TarInfo] = {}
        for member in tar:
            name = member.name
            # Check for directory traversal or absolute paths
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Security violation: Archive member has absolute path '{name}'.")
            parts = Path(name).parts
            if ".." in parts:
                raise ValueError(f"Security violation: Archive member contains parent traversal '..' in '{name}'.")
            if member.islnk() or member.issym():
                if name in EXPECTED_DEVKIT_MEMBERS:
                    raise ValueError(f"Security violation: Expected metadata member '{name}' is a link.")

            if name in EXPECTED_DEVKIT_MEMBERS:
                member_map[name] = member

        # Verify all expected metadata members were found in the archive
        missing = EXPECTED_DEVKIT_MEMBERS - set(member_map.keys())
        if missing:
            raise ValueError(f"Devkit archive is missing expected metadata members: {sorted(missing)}")

        # Safely extract only the two expected files directly into dest_dir
        for member_name in EXPECTED_DEVKIT_MEMBERS:
            member = member_map[member_name]
            if not member.isreg():
                raise ValueError(f"Expected metadata member '{member_name}' is not a regular file.")

            clean_filename = Path(member_name).name
            destination_path = (target_dir / clean_filename).resolve()

            # Ensure destination remains strictly inside target_dir
            if not str(destination_path).startswith(str(target_dir)):
                raise ValueError(f"Security violation: Destination path '{destination_path}' escapes target directory '{target_dir}'.")

            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise ValueError(f"Failed to read archive member: {member_name}")

            with open(destination_path, "wb") as out_f:
                while chunk := fileobj.read(1024 * 1024):
                    out_f.write(chunk)

            if clean_filename == "ILSVRC2012_validation_ground_truth.txt":
                extracted_files["ground_truth"] = destination_path
            elif clean_filename == "meta.mat":
                extracted_files["meta_mat"] = destination_path

    return extracted_files


def parse_and_validate_ground_truth(
    gt_file_path: Union[str, Path],
) -> Tuple[List[int], GroundTruthStatistics]:
    """Parse and validate the official ImageNet validation ground-truth text file.

    Validation rules:
    - File must contain exactly 50,000 non-empty lines.
    - Each line must be a valid integer in the range [1, 1000] (official ILSVRC devkit IDs).
    - Checks distribution to confirm balance across all 1,000 classes.

    Args:
        gt_file_path: Path to ILSVRC2012_validation_ground_truth.txt.

    Returns:
        Tuple of (labels_list, statistics).

    Raises:
        FileNotFoundError: If the ground truth file does not exist.
        ValueError: If line count != 50,000, labels are out of bounds, or labels are non-integers.
    """
    path = Path(gt_file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    labels: List[int] = []
    class_counts: Dict[int, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            cleaned = line.strip()
            if not cleaned:
                raise ValueError(f"Ground truth file '{path}' contains an empty line at line {line_num}.")
            try:
                label_val = int(cleaned)
            except ValueError as exc:
                raise ValueError(
                    f"Ground truth line {line_num} contains non-integer value '{cleaned}'."
                ) from exc

            if label_val < MIN_DEVKIT_ID or label_val > MAX_DEVKIT_ID:
                raise ValueError(
                    f"Ground truth line {line_num}: Label {label_val} out of valid devkit ID range [{MIN_DEVKIT_ID}, {MAX_DEVKIT_ID}]."
                )

            labels.append(label_val)
            class_counts[label_val] = class_counts.get(label_val, 0) + 1

    if len(labels) != NUM_VAL_IMAGES:
        raise ValueError(
            f"Invalid ground truth sample count: Expected {NUM_VAL_IMAGES} lines, got {len(labels)}."
        )

    unique_classes = len(class_counts)
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    is_balanced = (unique_classes == NUM_CLASSES) and (min_count == 50) and (max_count == 50)

    stats = GroundTruthStatistics(
        total_samples=len(labels),
        num_unique_classes=unique_classes,
        min_label=min(labels) if labels else 0,
        max_label=max(labels) if labels else 0,
        samples_per_class_min=min_count,
        samples_per_class_max=max_count,
        is_balanced_50_per_class=is_balanced,
    )

    return labels, stats


def parse_meta_mat(meta_mat_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse ILSVRC devkit meta.mat file to extract classification synset metadata.

    Filters down from the full synset hierarchy (1,860 entries) to the exactly 1,000
    leaf synsets (num_children == 0) evaluated in ImageNet-1K classification.

    Requires scipy.io.loadmat. If scipy is not installed in the environment, raises
    an explicit ImportError without attempting fragile custom binary parsing.

    Args:
        meta_mat_path: Path to meta.mat file.

    Returns:
        Dictionary containing:
        - 'id_to_wnid': Dict[int, str] mapping devkit ID (1..1000) to WNID.
        - 'wnid_to_id': Dict[str, int] mapping WNID to devkit ID.
        - 'wnid_to_words': Dict[str, str] mapping WNID to class description.
        - 'total_leaf_synsets': int (1000).

    Raises:
        ImportError: If scipy is not available.
        FileNotFoundError: If meta_mat_path does not exist.
        ValueError: If structure is malformed or leaf count != 1000.
    """
    path = Path(meta_mat_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"meta.mat file not found: {path}")

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to parse MATLAB v5 meta.mat files. "
            "Please install scipy (e.g. via 'uv pip install scipy' or 'pip install scipy')."
        ) from exc

    try:
        mat_data = sio.loadmat(str(path), squeeze_me=True)
    except Exception as exc:
        raise ValueError(f"Failed to parse MATLAB file '{path}': {exc}") from exc

    if "synsets" not in mat_data:
        raise ValueError(f"meta.mat does not contain 'synsets' table. Found keys: {list(mat_data.keys())}")

    raw_synsets = mat_data["synsets"]

    # Filter to leaf classification synsets (num_children == 0)
    # Each row in ILSVRC meta.mat:
    # index 0: ILSVRC2012_ID (int)
    # index 1: WNID (str, e.g. 'n01440764')
    # index 2: words (str, comma-separated class name)
    # index 3: gloss (str, definition)
    # index 4: num_children (int)
    # index 5: children (array of int IDs)
    # index 6: wordnet_height (int)
    # index 7: num_train_images (int)
    leaf_synsets = []
    for item in raw_synsets:
        try:
            num_children = int(item[4])
            if num_children == 0:
                leaf_synsets.append(item)
        except (IndexError, TypeError, ValueError):
            continue

    if len(leaf_synsets) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} leaf classification synsets (num_children == 0) in meta.mat, "
            f"got {len(leaf_synsets)}."
        )

    id_to_wnid: Dict[int, str] = {}
    wnid_to_id: Dict[str, int] = {}
    wnid_to_words: Dict[str, str] = {}

    for item in leaf_synsets:
        devkit_id = int(item[0])
        wnid = str(item[1]).strip()
        words = str(item[2]).strip()

        if devkit_id < MIN_DEVKIT_ID or devkit_id > MAX_DEVKIT_ID:
            raise ValueError(f"Invalid devkit ID {devkit_id} in meta.mat (expected {MIN_DEVKIT_ID}..{MAX_DEVKIT_ID}).")
        if not wnid.startswith("n") or len(wnid) != 9:
            raise ValueError(f"Malformed WNID '{wnid}' for devkit ID {devkit_id}.")

        if devkit_id in id_to_wnid:
            raise ValueError(f"Duplicate devkit ID {devkit_id} found in meta.mat.")
        if wnid in wnid_to_id:
            raise ValueError(f"Duplicate WNID '{wnid}' found in meta.mat.")

        id_to_wnid[devkit_id] = wnid
        wnid_to_id[wnid] = devkit_id
        wnid_to_words[wnid] = words

    return {
        "id_to_wnid": id_to_wnid,
        "wnid_to_id": wnid_to_id,
        "wnid_to_words": wnid_to_words,
        "total_leaf_synsets": len(id_to_wnid),
    }


def derive_torchvision_class_mapping(
    id_to_wnid: Dict[int, str],
    wnid_to_words: Dict[str, str],
) -> Dict[str, Any]:
    """Derive the canonical 0-based torchvision class index mapping from synset definitions.

    Torchvision ImageNet pretrained weights order classes by the alphabetical sort
    of the 1,000 WordNet IDs (WNIDs). This function builds the bidirectional lookup
    between 1-based devkit IDs, WNIDs, class descriptions, and 0-based model indices.

    Args:
        id_to_wnid: Dict mapping devkit ID (1..1000) to WNID.
        wnid_to_words: Dict mapping WNID to class description string.

    Returns:
        Structured mapping dictionary ready for JSON serialization.
    """
    if len(id_to_wnid) != NUM_CLASSES:
        raise ValueError(f"Complete mapping requires {NUM_CLASSES} classes, got {len(id_to_wnid)}.")

    # Canonical torchvision sort order: sorted by WNID alphabetically
    sorted_wnids = sorted(id_to_wnid.values())

    wnid_to_tv_idx: Dict[str, int] = {wnid: idx for idx, wnid in enumerate(sorted_wnids)}
    tv_idx_to_wnid: Dict[int, str] = {idx: wnid for idx, wnid in enumerate(sorted_wnids)}
    tv_idx_to_words: Dict[int, str] = {idx: wnid_to_words.get(wnid, "") for idx, wnid in enumerate(sorted_wnids)}

    devkit_id_to_tv_idx: Dict[int, int] = {
        d_id: wnid_to_tv_idx[wnid] for d_id, wnid in id_to_wnid.items()
    }
    tv_idx_to_devkit_id: Dict[int, int] = {
        tv_idx: d_id for d_id, tv_idx in devkit_id_to_tv_idx.items()
    }

    return {
        "num_classes": NUM_CLASSES,
        "devkit_id_to_torchvision_idx": {str(k): v for k, v in devkit_id_to_tv_idx.items()},
        "torchvision_idx_to_devkit_id": {str(k): v for k, v in tv_idx_to_devkit_id.items()},
        "torchvision_idx_to_wnid": {str(k): v for k, v in tv_idx_to_wnid.items()},
        "torchvision_idx_to_words": {str(k): v for k, v in tv_idx_to_words.items()},
        "wnid_to_torchvision_idx": wnid_to_tv_idx,
        "provenance": {
            "source": "ILSVRC2012_devkit_t12 meta.mat",
            "class_ordering_rule": "Alphabetical sort of 1,000 leaf WNIDs (official torchvision convention)",
            "index_bounds": "0 to 999",
            "devkit_id_bounds": "1 to 1000",
        },
    }


def generate_mapping_artifact(
    meta_mat_path: Union[str, Path],
    gt_file_path: Optional[Union[str, Path]] = None,
    dest_dir: Union[str, Path] = DEFAULT_METADATA_DIR,
    artifact_filename: str = "devkit_to_torchvision_mapping.json",
) -> Path:
    """Generate and persist the validated class mapping artifact in dataset/metadata/.

    Args:
        meta_mat_path: Path to extracted meta.mat.
        gt_file_path: Optional path to extracted ILSVRC2012_validation_ground_truth.txt.
                      If provided, translates all 50,000 validation labels into 0-based
                      torchvision class indices and includes them in the artifact.
        dest_dir: Output directory for the JSON mapping artifact.
        artifact_filename: Name of output JSON file.

    Returns:
        Path to the saved JSON artifact.
    """
    parsed = parse_meta_mat(meta_mat_path)
    mapping = derive_torchvision_class_mapping(
        id_to_wnid=parsed["id_to_wnid"],
        wnid_to_words=parsed["wnid_to_words"],
    )

    if gt_file_path is not None:
        gt_labels, stats = parse_and_validate_ground_truth(gt_file_path)
        d_to_tv = {int(k): v for k, v in mapping["devkit_id_to_torchvision_idx"].items()}
        val_tv_indices = [d_to_tv[gt_id] for gt_id in gt_labels]

        mapping["validation_ground_truth"] = {
            "total_samples": len(val_tv_indices),
            "statistics": stats.to_dict(),
            "first_10_samples": [
                {
                    "image_filename": f"ILSVRC2012_val_{i+1:08d}.JPEG",
                    "devkit_id": gt_labels[i],
                    "torchvision_class_idx": val_tv_indices[i],
                    "wnid": mapping["torchvision_idx_to_wnid"][str(val_tv_indices[i])],
                    "words": mapping["torchvision_idx_to_words"][str(val_tv_indices[i])],
                }
                for i in range(min(10, len(val_tv_indices)))
            ],
            "labels_0_indexed": val_tv_indices,
        }

    out_dir = Path(dest_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / artifact_filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return out_path
