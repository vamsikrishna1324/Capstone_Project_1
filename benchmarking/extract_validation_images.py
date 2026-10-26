"""
CLI Tool for ImageNet Validation Image Extraction and Manifest Generation (Step 4D.6).

Safely extracts validation images from ILSVRC2012_img_val.tar into dataset/val/,
validates image files and headers, and generates canonical validation manifests
(val_manifest.csv and val_manifest.json) using the official ground-truth mapping artifact.

Usage:
    ./.venv311/bin/python -m benchmarking.extract_validation_images --help
    ./.venv311/bin/python -m benchmarking.extract_validation_images --dry-run
    ./.venv311/bin/python -m benchmarking.extract_validation_images --limit 100
    ./.venv311/bin/python -m benchmarking.extract_validation_images
"""

import argparse
from pathlib import Path
import shutil
import sys
import time
from typing import Optional

from benchmarking.imagenet_images import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MANIFEST_JSON,
    DEFAULT_MAPPING_JSON,
    DEFAULT_VAL_DEST,
    DEFAULT_VAL_TAR,
    TOTAL_VAL_IMAGES,
    extract_validation_images,
    generate_validation_manifest,
    scan_and_validate_val_archive,
    verify_extracted_images,
)


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string (GiB / MiB)."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GiB ({bytes_val:,} bytes)"
    if bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MiB ({bytes_val:,} bytes)"
    return f"{bytes_val:,} bytes"


def main() -> int:
    """Execute ImageNet validation image extraction and manifest generation."""
    parser = argparse.ArgumentParser(
        description="Extract and verify ImageNet validation images and generate canonical manifests."
    )
    parser.add_argument(
        "--val-tar",
        type=Path,
        default=DEFAULT_VAL_TAR,
        help=f"Path to validation archive (default: {DEFAULT_VAL_TAR})",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_VAL_DEST,
        help=f"Output directory for extracted images (default: {DEFAULT_VAL_DEST})",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=DEFAULT_MAPPING_JSON,
        help=f"Path to class mapping JSON artifact (default: {DEFAULT_MAPPING_JSON})",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_MANIFEST_CSV,
        help=f"Path to output CSV manifest (default: {DEFAULT_MANIFEST_CSV})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_MANIFEST_JSON,
        help=f"Path to output JSON manifest (default: {DEFAULT_MANIFEST_JSON})",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory for manifest paths (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to extract (e.g. for subset testing)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction if images already exist on disk, only verify and generate manifests.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip JPEG magic byte verification of extracted images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect archive headers, mapping artifact, and disk space without extracting files.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("ImageNet Validation Image Extraction & Manifest Generator")
    print("=" * 75)
    print(f"Validation Archive: {args.val_tar}")
    print(f"Destination Dir:    {args.dest_dir}")
    print(f"Mapping Artifact:   {args.mapping_json}")
    print(f"CSV Manifest Out:   {args.csv_out}")
    print(f"JSON Manifest Out:  {args.json_out}")
    print(f"Dataset Root Dir:   {args.root_dir}")
    if args.limit:
        print(f"Image Limit:        {args.limit}")

    # Check input files
    if not args.val_tar.exists():
        print(f"\n[ERROR] Validation archive not found at: {args.val_tar}")
        return 1

    if not args.mapping_json.exists():
        print(f"\n[ERROR] Mapping artifact not found at: {args.mapping_json}")
        print("Please run Step 4D.5 metadata extraction first:")
        print("    ./.venv311/bin/python -m benchmarking.extract_metadata")
        return 1

    # Check disk space on existing directory ancestor
    dest_path = args.dest_dir.resolve()
    existing_ancestor = dest_path
    while not existing_ancestor.exists() and existing_ancestor != existing_ancestor.parent:
        existing_ancestor = existing_ancestor.parent
    disk_usage = shutil.disk_usage(existing_ancestor)
    free_gib = disk_usage.free / (1024**3)
    print(f"\nAvailable disk space on {existing_ancestor}: {free_gib:.2f} GiB")
    if free_gib < 10.0 and not args.dry_run and not args.skip_extract:
        print("[WARNING] Less than 10 GiB free storage available. ImageNet validation requires ~6.7 GiB.")

    # 1. Archive Header Scan & Validation
    print("\n1. Scanning archive member headers...")
    start_time = time.time()
    expected_members = TOTAL_VAL_IMAGES if (args.limit is None and args.val_tar.resolve() == DEFAULT_VAL_TAR.resolve()) else None
    try:
        members = scan_and_validate_val_archive(
            val_tar_path=args.val_tar,
            expected_count=expected_members,
        )
    except Exception as exc:
        print(f"[ERROR] Archive validation failed: {exc}")
        return 1

    total_uncompressed_bytes = sum(m[2] for m in members)
    scan_time = time.time() - start_time
    print(f"   Validated {len(members):,} archive members in {scan_time:.2f}s.")
    print(f"   Total uncompressed payload: {format_bytes(total_uncompressed_bytes)}")
    print(f"   First image: {members[0][1]} ({members[0][2]:,} bytes)")
    print(f"   Last image:  {members[-1][1]} ({members[-1][2]:,} bytes)")

    if args.dry_run:
        print("\n[DRY RUN] All pre-checks passed successfully. No files were written.")
        return 0

    # 2. Image Extraction
    if not args.skip_extract:
        total_to_extract = len(members) if args.limit is None else min(args.limit, len(members))
        print(f"\n2. Extracting {total_to_extract:,} validation images into {args.dest_dir}...")
        log_interval = 5000 if total_to_extract > 500 else max(1, total_to_extract // 10)
        extract_start = time.time()

        def on_progress(count: int, total: int, current_name: str) -> None:
            if count == 1 or count % log_interval == 0 or count == total:
                elapsed = time.time() - extract_start
                rate = count / elapsed if elapsed > 0 else 0
                pct = (count / total) * 100
                print(f"   [{pct:5.1f}%] Extracted {count:,}/{total:,} ({rate:.1f} imgs/s) — {current_name}")

        try:
            extract_result = extract_validation_images(
                val_tar_path=args.val_tar,
                dest_dir=args.dest_dir,
                limit=args.limit,
                expected_count=expected_members,
                progress_callback=on_progress,
            )
        except Exception as exc:
            print(f"[ERROR] Extraction failed: {exc}")
            return 1

        extract_time = time.time() - extract_start
        print(
            f"   Extraction complete: {extract_result['newly_extracted_count']:,} newly extracted, "
            f"{extract_result['skipped_count']:,} pre-verified & skipped "
            f"({extract_result['extracted_count']:,} total valid on disk) in {extract_time:.2f}s."
        )
        print(f"   Total bytes written: {format_bytes(extract_result['total_bytes_written'])}")
    else:
        print("\n2. Skipping image extraction (--skip-extract specified).")

    # 3. Verification of Extracted Images
    if not args.skip_verify:
        print(f"\n3. Verifying extracted image files in {args.dest_dir}...")
        verify_start = time.time()
        expected_cnt = args.limit if args.limit is not None else TOTAL_VAL_IMAGES
        expected_seq = list(range(1, expected_cnt + 1))
        try:
            verify_result = verify_extracted_images(
                images_dir=args.dest_dir,
                expected_count=expected_cnt,
                check_jpeg_headers=True,
                verify_decodable=True,
                expected_indices=expected_seq,
            )
        except Exception as exc:
            print(f"[ERROR] Image verification failed: {exc}")
            return 1

        verify_time = time.time() - verify_start
        print(f"   Verification passed in {verify_time:.2f}s:")
        print(f"   Verified count: {verify_result['verified_count']:,} images (all fully decoded via Pillow)")
        print(f"   Index range:    [{verify_result['min_index']}, {verify_result['max_index']}]")
        print(f"   Total on-disk:  {format_bytes(verify_result['total_bytes'])}")
    else:
        print("\n3. Skipping image verification (--skip-verify specified).")

    # 4. Generate Validation Manifests
    print("\n4. Generating validation manifests...")
    manifest_start = time.time()
    try:
        csv_path = generate_validation_manifest(
            images_dir=args.dest_dir,
            mapping_json_path=args.mapping_json,
            csv_output_path=args.csv_out,
            json_output_path=args.json_out,
            root_dir=args.root_dir,
            limit=args.limit,
            expected_count=args.limit if args.limit is not None else TOTAL_VAL_IMAGES,
        )
    except Exception as exc:
        print(f"[ERROR] Manifest generation failed: {exc}")
        return 1

    manifest_time = time.time() - manifest_start
    print(f"   Generated CSV manifest: {csv_path} ({format_bytes(csv_path.stat().st_size)})")
    if args.json_out and Path(args.json_out).exists():
        json_p = Path(args.json_out)
        print(f"   Generated JSON manifest: {json_p} ({format_bytes(json_p.stat().st_size)})")
    print(f"   Manifest verified with load_imagenet_manifest() in {manifest_time:.2f}s.")

    print("\n" + "=" * 75)
    print("SUCCESS: ImageNet validation dataset and manifests are ready for benchmarking!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
