"""
CLI Tool for Extracting and Validating ImageNet Metadata (Step 4D.5).

Safely extracts validation ground truth and meta.mat from the official devkit
archive (dataset/raw/ILSVRC2012_devkit_t12.tar.gz) into dataset/metadata/,
validates ground-truth distribution, and generates the canonical devkit-to-torchvision
class mapping JSON artifact.

Usage:
    ./.venv311/bin/python -m benchmarking.extract_metadata
    ./.venv311/bin/python -m benchmarking.extract_metadata --help
"""

import argparse
from pathlib import Path
import sys

from benchmarking.imagenet_metadata import (
    DEFAULT_DEVKIT_TAR,
    DEFAULT_METADATA_DIR,
    generate_mapping_artifact,
    parse_and_validate_ground_truth,
    safe_extract_devkit_metadata,
)


def main() -> int:
    """Execute metadata extraction and mapping artifact generation."""
    parser = argparse.ArgumentParser(
        description="Extract and validate ImageNet validation metadata from the official devkit archive."
    )
    parser.add_argument(
        "--devkit-tar",
        type=Path,
        default=DEFAULT_DEVKIT_TAR,
        help=f"Path to devkit archive (default: {DEFAULT_DEVKIT_TAR})",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Output directory for extracted metadata (default: {DEFAULT_METADATA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and archive existence without extracting files.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ImageNet Validation Metadata Extraction & Mapping Tool")
    print("=" * 70)
    print(f"Devkit Archive: {args.devkit_tar}")
    print(f"Destination Dir: {args.dest_dir}")

    if not args.devkit_tar.exists():
        print(f"\n[ERROR] Devkit archive not found at: {args.devkit_tar}")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Devkit archive exists. No files extracted.")
        return 0

    # Check scipy availability before extracting
    try:
        import scipy.io  # noqa: F401
    except ImportError:
        print("\n[ERROR] Missing required dependency 'scipy' to parse MATLAB v5 meta.mat.")
        print("Please install scipy into the virtual environment:")
        print("    ./.venv311/bin/python -m pip install scipy  (or uv pip install scipy)")
        return 1

    # 1. Safely extract only the two expected files
    print("\n1. Safely extracting validation ground truth and meta.mat...")
    try:
        extracted = safe_extract_devkit_metadata(
            devkit_tar_path=args.devkit_tar,
            dest_dir=args.dest_dir,
        )
    except Exception as exc:
        print(f"[ERROR] Extraction failed: {exc}")
        return 1

    print(f"   Extracted ground truth: {extracted['ground_truth']}")
    print(f"   Extracted meta.mat:     {extracted['meta_mat']}")

    # 2. Parse and validate ground truth
    print("\n2. Validating validation ground-truth labels...")
    try:
        labels, stats = parse_and_validate_ground_truth(extracted["ground_truth"])
    except Exception as exc:
        print(f"[ERROR] Ground truth validation failed: {exc}")
        return 1

    print(f"   Total validation samples: {stats.total_samples}")
    print(f"   Unique classes:           {stats.num_unique_classes}")
    print(f"   Devkit ID range:          [{stats.min_label}, {stats.max_label}]")
    print(f"   Samples per class:        min={stats.samples_per_class_min}, max={stats.samples_per_class_max}")
    print(f"   Is balanced 50/class:     {stats.is_balanced_50_per_class}")

    # 3. Generate mapping artifact
    print("\n3. Deriving canonical torchvision class mapping...")
    try:
        artifact_path = generate_mapping_artifact(
            meta_mat_path=extracted["meta_mat"],
            gt_file_path=extracted["ground_truth"],
            dest_dir=args.dest_dir,
        )
    except Exception as exc:
        print(f"[ERROR] Mapping generation failed: {exc}")
        return 1

    print(f"   Mapping artifact saved to: {artifact_path}")
    print("\n[SUCCESS] ImageNet validation metadata extracted and verified successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
