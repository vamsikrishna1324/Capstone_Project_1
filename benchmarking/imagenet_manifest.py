"""
ImageNet Validation Manifest Loader for Edge Deployment Benchmarking.

Provides parsing, strict validation, and dataset instantiation for explicitly
labelled image manifests in CSV and JSON formats, supporting class index (0-999)
and externally supplied WNID mappings.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from benchmarking.accuracy import EvaluationDataset, LabeledImageSample


VALID_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPEG",
    ".JPG",
    ".PNG",
}

MIN_CLASS_IDX: int = 0
MAX_CLASS_IDX: int = 999
IMAGENET_NUM_CLASSES: int = 1000

# Canonical field names and recognized aliases
PATH_FIELD_ALIASES: Sequence[str] = ("image_relpath", "image_path", "filepath", "image")
LABEL_FIELD_ALIASES: Sequence[str] = ("true_label_idx", "label_idx", "class_idx", "label")
WNID_FIELD_ALIASES: Sequence[str] = ("wnid", "synset")


def validate_wnid_mapping(
    mapping_source: Union[Dict[str, int], str, Path],
    require_full_1000: bool = False,
) -> Dict[str, int]:
    """Validate and normalize an externally supplied WNID-to-index mapping.

    Args:
        mapping_source: Dictionary mapping WNID string to int index,
                        or path to a JSON/CSV file.
        require_full_1000: Whether to require all 1,000 ImageNet-1K classes.

    Returns:
        Validated dictionary mapping WNID strings to integer class indices (0-999).

    Raises:
        FileNotFoundError: If the specified mapping file does not exist.
        ValueError: If mapping is empty, contains invalid indices, invalid keys,
                    or fails the completeness requirement.
        TypeError: If mapping_source is an unsupported type.
    """
    mapping: Dict[str, int] = {}

    if isinstance(mapping_source, (str, Path)):
        map_path = Path(mapping_source)
        if not map_path.exists() or not map_path.is_file():
            raise FileNotFoundError(f"WNID mapping file not found: {map_path}")

        if map_path.suffix.lower() == ".json":
            with open(map_path, "r", encoding="utf-8") as f:
                try:
                    raw_data = json.load(f)
                except Exception as exc:
                    raise ValueError(f"Malformed JSON in WNID mapping file '{map_path}': {exc}") from exc
                if not isinstance(raw_data, dict):
                    raise ValueError(f"JSON WNID mapping must be a key-value object, got {type(raw_data).__name__}")
                if "wnid_to_torchvision_idx" in raw_data and isinstance(raw_data["wnid_to_torchvision_idx"], dict):
                    mapping = raw_data["wnid_to_torchvision_idx"]
                elif "torchvision_idx_to_wnid" in raw_data and isinstance(raw_data["torchvision_idx_to_wnid"], dict):
                    mapping = {wnid: int(idx) for idx, wnid in raw_data["torchvision_idx_to_wnid"].items()}
                else:
                    mapping = raw_data
        elif map_path.suffix.lower() in (".csv", ".txt"):
            with open(map_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV mapping file '{map_path}' has no header.")
                wnid_col = next((c for c in reader.fieldnames if c.strip().lower() in ("wnid", "synset")), None)
                idx_col = next((c for c in reader.fieldnames if c.strip().lower() in ("class_idx", "true_label_idx", "label_idx", "label", "idx")), None)
                if not wnid_col or not idx_col:
                    raise ValueError(
                        f"CSV mapping file '{map_path}' must contain 'wnid' and 'class_idx' columns. "
                        f"Found: {reader.fieldnames}"
                    )
                for line_num, row in enumerate(reader, start=2):
                    w = row[wnid_col].strip()
                    val = row[idx_col].strip()
                    mapping[w] = val
        else:
            raise ValueError(f"Unsupported WNID mapping file extension: '{map_path.suffix}'. Expected .json or .csv.")
    elif isinstance(mapping_source, dict):
        mapping = dict(mapping_source)
    else:
        raise TypeError(f"mapping_source must be a dict, str, or Path, got {type(mapping_source).__name__}")

    if not mapping:
        raise ValueError("WNID mapping is empty; at least one class mapping is required.")

    validated: Dict[str, int] = {}
    for wnid, idx_val in mapping.items():
        if not isinstance(wnid, str) or not wnid.strip():
            raise ValueError(f"Invalid WNID key '{wnid}': must be a non-empty string.")
        clean_wnid = wnid.strip()

        # Parse and validate integer index
        if isinstance(idx_val, bool):
            raise ValueError(f"Class index for WNID '{clean_wnid}' cannot be boolean: {idx_val}")
        try:
            int_idx = int(idx_val)
            # Ensure float strings like "5.5" are rejected
            if isinstance(idx_val, str) and "." in idx_val:
                raise ValueError(f"Float string '{idx_val}' not allowed as class index.")
            if isinstance(idx_val, float) and not idx_val.is_integer():
                raise ValueError(f"Non-integer float {idx_val} not allowed as class index.")
        except Exception as exc:
            raise ValueError(
                f"Invalid class index '{idx_val}' for WNID '{clean_wnid}': must be integer in [{MIN_CLASS_IDX}, {MAX_CLASS_IDX}]."
            ) from exc

        if int_idx < MIN_CLASS_IDX or int_idx > MAX_CLASS_IDX:
            raise ValueError(
                f"Class index {int_idx} for WNID '{clean_wnid}' is out of range [{MIN_CLASS_IDX}, {MAX_CLASS_IDX}]."
            )
        validated[clean_wnid] = int_idx

    if require_full_1000:
        if len(validated) != IMAGENET_NUM_CLASSES:
            raise ValueError(
                f"Complete ImageNet mapping requires {IMAGENET_NUM_CLASSES} classes, but got {len(validated)}."
            )
        unique_indices = set(validated.values())
        if len(unique_indices) != IMAGENET_NUM_CLASSES or unique_indices != set(range(IMAGENET_NUM_CLASSES)):
            raise ValueError(
                f"Complete ImageNet mapping must contain all indices from {MIN_CLASS_IDX} to {MAX_CLASS_IDX}."
            )

    return validated


def _find_field(record: Dict[str, Any], aliases: Sequence[str]) -> Tuple[Optional[str], Optional[Any]]:
    """Look up the first matching key from aliases in a record dictionary."""
    record_lower = {k.strip().lower(): (k, v) for k, v in record.items() if isinstance(k, str)}
    for alias in aliases:
        if alias in record_lower:
            orig_key, val = record_lower[alias]
            return orig_key, val
    return None, None


def _parse_integer_label(val: Any, record_id: Any) -> int:
    """Validate and convert a label value to an integer in [0, 999]."""
    if isinstance(val, bool):
        raise ValueError(f"Record {record_id}: Label cannot be a boolean value ({val}).")
    if isinstance(val, float) and not val.is_integer():
        raise ValueError(f"Record {record_id}: Non-integer float {val} is not a valid class index.")
    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            raise ValueError(f"Record {record_id}: Label string is empty.")
        if "." in val_clean:
            raise ValueError(f"Record {record_id}: Decimal string '{val}' is not a valid integer class index.")
        try:
            parsed = int(val_clean)
        except ValueError as exc:
            raise ValueError(f"Record {record_id}: Cannot parse label '{val}' as integer.") from exc
    elif isinstance(val, (int, float)):
        parsed = int(val)
    else:
        raise ValueError(f"Record {record_id}: Unsupported label type {type(val).__name__}: {val}")

    if parsed < MIN_CLASS_IDX or parsed > MAX_CLASS_IDX:
        raise ValueError(
            f"Record {record_id}: Label index {parsed} is out of bounds. Expected integer in range [{MIN_CLASS_IDX}, {MAX_CLASS_IDX}]."
        )
    return parsed


def load_imagenet_manifest(
    manifest_path: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    wnid_to_idx: Optional[Union[Dict[str, int], str, Path]] = None,
) -> List[LabeledImageSample]:
    """Parse and validate an ImageNet validation manifest file into LabeledImageSamples.

    Supports CSV and JSON manifest files. Relative image paths are resolved against
    the supplied root_dir (or the manifest's parent directory if root_dir is None).

    Args:
        manifest_path: Path to the manifest file (.csv or .json).
        root_dir: Base directory to resolve relative image paths against.
                  Defaults to the manifest file's parent folder.
        wnid_to_idx: Optional WNID-to-index mapping (dict or file path) if records
                     specify synset/WNID instead of integer label.

    Returns:
        List of strictly validated LabeledImageSample instances.

    Raises:
        FileNotFoundError: If manifest file or any referenced image does not exist.
        ValueError: If manifest is malformed, empty, missing required columns,
                    contains out-of-range labels, duplicate image entries,
                    or unsupported file extensions.
    """
    path = Path(manifest_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    resolved_root = Path(root_dir) if root_dir is not None else path.parent
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {resolved_root}")

    validated_wnid_map: Optional[Dict[str, int]] = None
    if wnid_to_idx is not None:
        validated_wnid_map = validate_wnid_mapping(wnid_to_idx)

    # 1. Parse raw records from CSV or JSON
    raw_records: List[Tuple[Any, Dict[str, Any]]] = []
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV manifest '{path}' has no header row.")
            for line_idx, row in enumerate(reader, start=2):
                raw_records.append((f"line {line_idx}", row))
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as exc:
                raise ValueError(f"Malformed JSON in manifest '{path}': {exc}") from exc

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"JSON manifest item at index {idx} must be an object, got {type(item).__name__}")
                raw_records.append((f"index {idx}", item))
        elif isinstance(data, dict):
            if "samples" in data and isinstance(data["samples"], list):
                for idx, item in enumerate(data["samples"]):
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON sample at index {idx} must be an object, got {type(item).__name__}")
                    raw_records.append((f"sample {idx}", item))
            elif "records" in data and isinstance(data["records"], list):
                for idx, item in enumerate(data["records"]):
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON record at index {idx} must be an object, got {type(item).__name__}")
                    raw_records.append((f"record {idx}", item))
            else:
                # Key-value mapping of image_path -> label/dict
                for k, v in data.items():
                    if isinstance(v, dict):
                        row = dict(v)
                        row.setdefault("image_relpath", k)
                        raw_records.append((f"key '{k}'", row))
                    else:
                        raw_records.append((f"key '{k}'", {"image_relpath": k, "true_label_idx": v}))
        else:
            raise ValueError(f"JSON manifest must contain a list of objects or key-value mapping, got {type(data).__name__}")
    else:
        raise ValueError(f"Unsupported manifest file extension '{suffix}'. Expected .csv or .json.")

    if not raw_records:
        raise ValueError(f"Manifest file '{path}' is empty; no image records found.")

    # 2. Validate each record strictly
    seen_images: Set[Path] = set()
    samples: List[LabeledImageSample] = []

    for record_id, record in raw_records:
        # Resolve image path
        path_key, img_rel = _find_field(record, PATH_FIELD_ALIASES)
        if path_key is None or img_rel is None:
            raise ValueError(
                f"Record {record_id} missing required image path field. Expected one of: {PATH_FIELD_ALIASES}"
            )
        if not isinstance(img_rel, str) or not img_rel.strip():
            raise ValueError(f"Record {record_id} has empty or non-string image path: {img_rel}")

        rel_path = Path(img_rel.strip())
        full_img_path = (resolved_root / rel_path).resolve()

        # Check for duplicate image paths
        if full_img_path in seen_images:
            raise ValueError(f"Duplicate image entry found for '{full_img_path}' ({record_id}).")
        seen_images.add(full_img_path)

        # Check image file existence
        if not full_img_path.exists():
            raise FileNotFoundError(
                f"Referenced image file does not exist: '{full_img_path}' ({record_id})."
            )
        if not full_img_path.is_file():
            raise ValueError(
                f"Referenced image path is not a file: '{full_img_path}' ({record_id})."
            )

        # Check image file extension
        if full_img_path.suffix not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{full_img_path.suffix}' for '{full_img_path}' ({record_id}). "
                f"Supported extensions: {sorted(VALID_IMAGE_EXTENSIONS)}"
            )

        # Resolve ground-truth label
        label_key, label_val = _find_field(record, LABEL_FIELD_ALIASES)
        wnid_key, wnid_val = _find_field(record, WNID_FIELD_ALIASES)

        if label_key is not None and label_val is not None:
            final_label = _parse_integer_label(label_val, record_id)
        elif wnid_key is not None and wnid_val is not None:
            if validated_wnid_map is None:
                raise ValueError(
                    f"Record {record_id} specifies WNID '{wnid_val}', but no wnid_to_idx mapping was provided."
                )
            clean_wnid = str(wnid_val).strip()
            if clean_wnid not in validated_wnid_map:
                raise KeyError(
                    f"WNID '{clean_wnid}' from {record_id} not found in provided WNID-to-index mapping."
                )
            final_label = validated_wnid_map[clean_wnid]
        else:
            raise ValueError(
                f"Record {record_id} missing ground-truth label. "
                f"Expected one of label fields {LABEL_FIELD_ALIASES} or WNID fields {WNID_FIELD_ALIASES}."
            )

        samples.append(LabeledImageSample(image=full_img_path, label=final_label))

    return samples


def create_dataset_from_manifest(
    manifest_path: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    name: str = "imagenet_manifest",
    wnid_to_idx: Optional[Union[Dict[str, int], str, Path]] = None,
) -> EvaluationDataset:
    """Create an EvaluationDataset instance directly from a validated manifest file.

    Args:
        manifest_path: Path to CSV or JSON manifest file.
        root_dir: Optional dataset root directory for resolving relative paths.
        name: Identifier name for the dataset.
        wnid_to_idx: Optional WNID-to-index mapping if records use WNID/synset keys.

    Returns:
        EvaluationDataset populated with LabeledImageSample items.
    """
    samples = load_imagenet_manifest(
        manifest_path=manifest_path,
        root_dir=root_dir,
        wnid_to_idx=wnid_to_idx,
    )
    return EvaluationDataset(samples=samples, name=name)
