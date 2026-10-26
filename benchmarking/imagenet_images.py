"""
ImageNet Validation Images Extraction and Manifest Generation (Step 4D.6).

Safely extracts validation images from ILSVRC2012_img_val.tar into dataset/val/,
validates image files and headers, and generates the canonical, verified
validation manifests (val_manifest.csv and val_manifest.json) using the official
ground-truth mapping artifact.
"""

import csv
import json
import os
from pathlib import Path
import re
import tarfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from PIL import Image

from benchmarking.imagenet_manifest import (
    VALID_IMAGE_EXTENSIONS,
    load_imagenet_manifest,
)


VAL_IMAGE_NAME_PATTERN = re.compile(r"^ILSVRC2012_val_(\d{8})\.JPEG$")
JPEG_MAGIC_BYTES = b"\xff\xd8\xff"

DEFAULT_VAL_TAR: Path = Path("dataset/raw/ILSVRC2012_img_val.tar")
DEFAULT_VAL_DEST: Path = Path("dataset/val")
DEFAULT_MAPPING_JSON: Path = Path("dataset/metadata/devkit_to_torchvision_mapping.json")
DEFAULT_MANIFEST_CSV: Path = Path("dataset/val_manifest.csv")
DEFAULT_MANIFEST_JSON: Path = Path("dataset/val_manifest.json")
DEFAULT_DATASET_ROOT: Path = Path("dataset")

TOTAL_VAL_IMAGES: int = 50_000


def verify_single_image_integrity(
    file_path: Union[str, Path],
    expected_size: Optional[int] = None,
) -> None:
    """Verify that an image file exists, matches expected size, has JPEG magic bytes, and can be fully decoded by Pillow.

    Args:
        file_path: Path to the image file to verify.
        expected_size: Optional expected size in bytes.

    Raises:
        ValueError: If file does not exist, is empty, has size mismatch,
                    has invalid JPEG magic bytes, or cannot be fully decoded by Pillow.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Image file does not exist or is not a regular file: {p}")

    size = p.stat().st_size
    if size == 0:
        raise ValueError(f"Image '{p.name}' is an empty (0 bytes) file.")

    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"Image '{p.name}' size mismatch: found {size} bytes, expected {expected_size} bytes."
        )

    with open(p, "rb") as f:
        header = f.read(3)
        if header != JPEG_MAGIC_BYTES:
            raise ValueError(f"Image '{p.name}' does not have valid JPEG magic bytes (got {header!r}).")

    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            img.load()
    except Exception as exc:
        raise ValueError(f"Image '{p.name}' is corrupt or cannot be decoded by Pillow: {exc}") from exc


def scan_and_validate_val_archive(
    val_tar_path: Union[str, Path] = DEFAULT_VAL_TAR,
    expected_count: Optional[int] = TOTAL_VAL_IMAGES,
) -> List[Tuple[int, str, int]]:
    """Scan and validate all member headers in the validation tar archive.

    Security & Integrity checks:
    - Rejects path traversal (..) and absolute paths.
    - Rejects links, symlinks, directories, and non-regular files.
    - Every filename must match regex: ^ILSVRC2012_val_\\d{8}\\.JPEG$.
    - Verifies continuous sequence if expected_count == 50,000.

    Args:
        val_tar_path: Path to ILSVRC2012_img_val.tar.
        expected_count: Expected total number of image members (or None to skip count check).

    Returns:
        List of tuples: (image_index, member_name, member_size_bytes) sorted by index.

    Raises:
        FileNotFoundError: If the tar file does not exist.
        ValueError: If archive contains unsafe members, non-matching names, or invalid count.
    """
    tar_path = Path(val_tar_path)
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"Validation archive not found: {tar_path}")

    members_info: List[Tuple[int, str, int]] = []
    seen_indices: Set[int] = set()

    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            name = member.name
            # Security checks
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Security violation: Archive member has absolute path '{name}'.")
            if ".." in Path(name).parts:
                raise ValueError(f"Security violation: Archive member contains parent traversal in '{name}'.")
            if not member.isreg():
                raise ValueError(f"Security violation: Archive member '{name}' is not a regular file.")

            match = VAL_IMAGE_NAME_PATTERN.match(name)
            if not match:
                raise ValueError(f"Invalid member name '{name}'; does not match ILSVRC2012 validation format.")

            idx = int(match.group(1))
            if idx in seen_indices:
                raise ValueError(f"Duplicate image index {idx} in archive member '{name}'.")
            seen_indices.add(idx)
            members_info.append((idx, name, member.size))

    if expected_count is not None and len(members_info) != expected_count:
        raise ValueError(
            f"Expected {expected_count} members in validation archive, got {len(members_info)}."
        )

    # Sort numerically by image index
    members_info.sort(key=lambda x: x[0])

    if expected_count == TOTAL_VAL_IMAGES:
        min_idx = members_info[0][0]
        max_idx = members_info[-1][0]
        if min_idx != 1 or max_idx != TOTAL_VAL_IMAGES or len(seen_indices) != TOTAL_VAL_IMAGES:
            raise ValueError(
                f"Image indices must span continuously from 1 to {TOTAL_VAL_IMAGES}. Got range [{min_idx}, {max_idx}]."
            )

    return members_info


def extract_validation_images(
    val_tar_path: Union[str, Path] = DEFAULT_VAL_TAR,
    dest_dir: Union[str, Path] = DEFAULT_VAL_DEST,
    limit: Optional[int] = None,
    expected_count: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Safely extract validation images chunk-by-chunk without using extractall().

    Supports safe resuming and never blindly overwrites an existing image.
    If a destination image already exists:
    - If it matches member size, magic bytes, and passes full Pillow decoding, it is safely skipped.
    - If it is invalid, corrupt, or has unexpected size, extraction halts with a clear error.
    New images are extracted to a temporary file, fully verified, and atomically promoted
    without overwriting (using os.link to prevent race conditions).

    Args:
        val_tar_path: Path to ILSVRC2012_img_val.tar.
        dest_dir: Target output folder for images (e.g. dataset/val/).
        limit: Optional maximum number of images to extract (for subsets or testing).
        expected_count: Optional expected total count of members in archive.
        progress_callback: Optional callback(extracted_count, total_count, current_filename).

    Returns:
        Dictionary summarizing extraction results:
        - 'extracted_count': int (total verified files in target_dir)
        - 'newly_extracted_count': int
        - 'skipped_count': int
        - 'destination_dir': Path
        - 'total_bytes_written': int
        - 'extracted_files': List[Path]

    Raises:
        FileNotFoundError: If tar archive does not exist.
        FileExistsError: If a destination file unexpectedly appears before promotion.
        ValueError: On security violation, archive corruption, or invalid existing file.
    """
    tar_path = Path(val_tar_path)
    if not tar_path.exists() or not tar_path.is_file():
        raise FileNotFoundError(f"Validation archive not found: {tar_path}")

    target_dir = Path(dest_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Pre-validate headers and members
    members_info = scan_and_validate_val_archive(
        tar_path,
        expected_count=expected_count,
    )

    if limit is not None and limit > 0:
        members_to_extract = members_info[:limit]
    else:
        members_to_extract = members_info

    name_set = {name for _, name, _ in members_to_extract}
    total_to_extract = len(members_to_extract)

    extracted_files: List[Path] = []
    total_bytes = 0
    skipped_count = 0
    newly_extracted_count = 0

    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            if member.name not in name_set:
                continue

            destination_path = (target_dir / member.name).resolve()
            if not str(destination_path).startswith(str(target_dir)):
                raise ValueError(f"Security violation: Path '{destination_path}' escapes target '{target_dir}'.")

            # Check if destination file already exists
            if destination_path.exists():
                # Verify existing file; if corrupt or wrong size, halts with clear error
                try:
                    verify_single_image_integrity(destination_path, expected_size=member.size)
                except ValueError as exc:
                    raise ValueError(
                        f"Existing destination file '{destination_path.name}' is invalid or wrong-sized: {exc}. "
                        "Aborting extraction to prevent corrupt state."
                    ) from exc

                # Valid existing file -> safe resume / skip
                skipped_count += 1
                extracted_files.append(destination_path)
                count = len(extracted_files)
                if progress_callback is not None:
                    progress_callback(count, total_to_extract, member.name)
                if count >= total_to_extract:
                    break
                continue

            # Atomic temporary write in the same directory (same filesystem)
            temp_path = destination_path.with_name(f".tmp_{member.name}_{time.time_ns()}")
            try:
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    raise ValueError(f"Failed to read archive member: {member.name}")

                bytes_written = 0
                with open(temp_path, "wb") as out_f:
                    while chunk := fileobj.read(1024 * 1024):
                        out_f.write(chunk)
                        bytes_written += len(chunk)

                # Verify written temporary file integrity and Pillow decode
                verify_single_image_integrity(temp_path, expected_size=member.size)

                # Safe no-overwrite promotion:
                # os.link creates destination_path pointing to temp_path.
                # If destination_path already exists (even via a race), os.link raises FileExistsError!
                try:
                    os.link(temp_path, destination_path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Destination file '{destination_path.name}' appeared unexpectedly before temporary file promotion. "
                        "Refusing to overwrite existing file."
                    ) from exc

                total_bytes += bytes_written
                newly_extracted_count += 1
                extracted_files.append(destination_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            count = len(extracted_files)
            if progress_callback is not None:
                progress_callback(count, total_to_extract, member.name)

            if count >= total_to_extract:
                break

    return {
        "extracted_count": len(extracted_files),
        "newly_extracted_count": newly_extracted_count,
        "skipped_count": skipped_count,
        "destination_dir": target_dir,
        "total_bytes_written": total_bytes,
        "extracted_files": extracted_files,
    }


def verify_extracted_images(
    images_dir: Union[str, Path] = DEFAULT_VAL_DEST,
    expected_count: Optional[int] = TOTAL_VAL_IMAGES,
    check_jpeg_headers: bool = True,
    verify_decodable: bool = True,
    expected_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Verify that extracted image files exist, have valid sizes, valid headers, and can be fully decoded.

    Validates exact expected image indices for limited runs, and complete 1..50000 for full runs.

    Args:
        images_dir: Directory containing extracted JPEG files.
        expected_count: Expected total count of images (or None).
        check_jpeg_headers: Whether to inspect JPEG magic bytes.
        verify_decodable: Whether to verify full Pillow image stream and decoding.
        expected_indices: Optional explicit sequence of 1-based indices that must be present.

    Returns:
        Dictionary with verification results:
        - 'verified_count': int
        - 'total_bytes': int
        - 'is_valid': bool
        - 'min_index': int
        - 'max_index': int

    Raises:
        FileNotFoundError: If images_dir or expected images do not exist.
        ValueError: If count mismatch, index gaps, non-JPEG headers, or un-decodable files found.
    """
    dir_path = Path(images_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Extracted images directory does not exist: {dir_path}")

    found_files: List[Tuple[int, Path]] = []
    total_bytes = 0

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            match = VAL_IMAGE_NAME_PATTERN.match(file_path.name)
            if match:
                idx = int(match.group(1))
                found_files.append((idx, file_path))

    found_files.sort(key=lambda x: x[0])
    count = len(found_files)

    if count == 0:
        raise ValueError(f"No valid ILSVRC2012 validation images found in {dir_path}.")

    if expected_count is not None and count != expected_count:
        raise ValueError(f"Extracted image count mismatch: Expected {expected_count}, found {count}.")

    indices = [idx for idx, _ in found_files]
    if expected_indices is not None:
        expected_seq = list(expected_indices)
        if indices != expected_seq:
            missing = set(expected_seq) - set(indices)
            extra = set(indices) - set(expected_seq)
            err_msg = f"Extracted image indices mismatch in {dir_path}."
            if missing:
                err_msg += f" Missing {len(missing)} indices (e.g. {sorted(missing)[:5]})."
            if extra:
                err_msg += f" Unexpected {len(extra)} indices (e.g. {sorted(extra)[:5]})."
            raise ValueError(err_msg)
    elif expected_count is not None:
        expected_seq = list(range(1, expected_count + 1))
        if indices != expected_seq:
            missing = set(expected_seq) - set(indices)
            raise ValueError(
                f"Discontinuous image sequence in {dir_path}. "
                f"Expected continuous indices [1..{expected_count}], missing {len(missing)} indices."
            )

    # Full integrity and decoding verification
    for idx, p in found_files:
        if verify_decodable:
            verify_single_image_integrity(p)
        else:
            size = p.stat().st_size
            if size == 0:
                raise ValueError(f"Extracted image '{p.name}' is an empty (0 bytes) file.")
            if check_jpeg_headers:
                with open(p, "rb") as f:
                    if f.read(3) != JPEG_MAGIC_BYTES:
                        raise ValueError(f"Image '{p.name}' does not have valid JPEG magic bytes.")
        total_bytes += p.stat().st_size

    return {
        "verified_count": count,
        "total_bytes": total_bytes,
        "is_valid": True,
        "min_index": indices[0],
        "max_index": indices[-1],
    }


def generate_validation_manifest(
    images_dir: Union[str, Path] = DEFAULT_VAL_DEST,
    mapping_json_path: Union[str, Path] = DEFAULT_MAPPING_JSON,
    csv_output_path: Union[str, Path] = DEFAULT_MANIFEST_CSV,
    json_output_path: Optional[Union[str, Path]] = DEFAULT_MANIFEST_JSON,
    root_dir: Union[str, Path] = DEFAULT_DATASET_ROOT,
    limit: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> Path:
    """Generate canonical CSV and JSON validation manifests from extracted images and mapping metadata.

    Ground-truth mapping logic:
    - Reads 'validation_ground_truth.labels_0_indexed' from devkit_to_torchvision_mapping.json.
    - Each image 'ILSVRC2012_val_{i:08d}.JPEG' maps to labels_0_indexed[i - 1].
    - Relative paths are computed relative to root_dir (e.g. 'val/ILSVRC2012_val_00000001.JPEG').
    - Strictly validated by loading back through load_imagenet_manifest().

    Args:
        images_dir: Path to extracted images directory.
        mapping_json_path: Path to devkit_to_torchvision_mapping.json artifact.
        csv_output_path: Path for output CSV manifest.
        json_output_path: Optional path for output JSON manifest.
        root_dir: Dataset root directory for relative path resolution.
        limit: Optional maximum number of images to include in manifest.
        expected_count: Optional expected total count of images that must be present.

    Returns:
        Path to the generated CSV manifest.

    Raises:
        FileNotFoundError: If mapping JSON or images directory does not exist.
        ValueError: If ground truth labels are missing, image count does not match expected_count,
                    or manifest validation fails.
    """
    map_path = Path(mapping_json_path)
    if not map_path.exists() or not map_path.is_file():
        raise FileNotFoundError(f"Mapping artifact not found: {map_path}")

    with open(map_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    if "validation_ground_truth" not in mapping_data or "labels_0_indexed" not in mapping_data["validation_ground_truth"]:
        raise ValueError(f"Mapping file '{map_path}' is missing 'validation_ground_truth.labels_0_indexed'.")

    labels_0_indexed = mapping_data["validation_ground_truth"]["labels_0_indexed"]
    tv_to_wnid = mapping_data.get("torchvision_idx_to_wnid", {})
    tv_to_words = mapping_data.get("torchvision_idx_to_words", {})
    tv_to_devkit = mapping_data.get("torchvision_idx_to_devkit_id", {})

    img_dir = Path(images_dir).resolve()
    base_root = Path(root_dir).resolve()

    # Discover and sort existing image files
    image_files: List[Tuple[int, Path]] = []
    for p in img_dir.iterdir():
        if p.is_file():
            match = VAL_IMAGE_NAME_PATTERN.match(p.name)
            if match:
                idx = int(match.group(1))
                image_files.append((idx, p))

    image_files.sort(key=lambda x: x[0])
    if limit is not None and limit > 0:
        image_files = image_files[:limit]

    if not image_files:
        raise ValueError(f"No extracted validation images found in '{img_dir}'.")

    if expected_count is not None and len(image_files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} extracted images to generate manifest, but found {len(image_files)}. "
            "Refusing to generate manifest for incomplete or interrupted dataset."
        )

    # Build manifest records
    csv_rows: List[Tuple[str, int]] = []
    json_samples: List[Dict[str, Any]] = []

    for img_idx, abs_path in image_files:
        if img_idx < 1 or img_idx > len(labels_0_indexed):
            raise ValueError(f"Image index {img_idx} exceeds ground-truth labels length ({len(labels_0_indexed)}).")

        label_0_idx = labels_0_indexed[img_idx - 1]
        try:
            rel_path = abs_path.relative_to(base_root)
        except ValueError:
            rel_path = abs_path

        csv_rows.append((str(rel_path), label_0_idx))
        json_samples.append({
            "image_relpath": str(rel_path),
            "true_label_idx": label_0_idx,
            "image_index": img_idx,
            "devkit_id": tv_to_devkit.get(str(label_0_idx)),
            "wnid": tv_to_wnid.get(str(label_0_idx), ""),
            "class_name": tv_to_words.get(str(label_0_idx), ""),
        })

    # Write CSV manifest
    csv_path = Path(csv_output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_relpath", "true_label_idx"])
        for rel_p, lbl in csv_rows:
            writer.writerow([rel_p, lbl])

    # Write JSON manifest if requested
    if json_output_path is not None:
        json_path = Path(json_output_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_meta = {
            "dataset_name": "ILSVRC2012_val",
            "num_samples": len(json_samples),
            "num_classes": mapping_data.get("num_classes", 1000),
            "root_dir": str(root_dir),
            "provenance": {
                "archive_source": "ILSVRC2012_img_val.tar",
                "mapping_source": str(mapping_json_path),
                "ground_truth_rule": "0-based torchvision class index aligned with 1-based image filename numbers",
            },
            "samples": json_samples,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2)

    # Validate output manifest using existing manifest utility
    load_imagenet_manifest(manifest_path=csv_path, root_dir=base_root)

    return csv_path
