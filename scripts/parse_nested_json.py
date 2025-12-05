#!/usr/bin/env python3
"""
Custom Data Loader for Nested JSON Structure
Handles STag-based train/valid/test splitting
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def split_from_stag(stag):
    """
    Determine split from STag field.

    Args:
        stag: STag string (e.g., "Tx", "Ty", "Tz")

    Returns:
        "train", "valid", "test", or None
    """
    if not stag:
        return None
    if "Tx" in stag:
        return "train"
    if "Ty" in stag:
        return "valid"
    if "Tz" in stag:
        return "test"
    return None


def flatten_records(data):
    """
    Flatten nested JSON structure into list of records.

    Args:
        data: Nested JSON structure

    Returns:
        List of flattened records
    """
    records = []
    for level1 in data:
        if isinstance(level1, list):
            for level2 in level1:
                if isinstance(level2, dict):
                    for study_key, recs in level2.items():
                        if isinstance(recs, list):
                            records.extend(recs)
        elif isinstance(level1, dict):
            for study_key, recs in level1.items():
                if isinstance(recs, list):
                    records.extend(recs)
    return records


def group_files_by_case(records, base_filename="base.nii",
                        mask1_filename="mask1.nii.gz",
                        mask2_filename="mask2.nii.gz"):
    """
    Group files into cases based on patient/study/subfolder.

    Args:
        records: List of flattened records
        base_filename: Name of base/CT file
        mask1_filename: Name of first mask (bone)
        mask2_filename: Name of second mask (tumor)

    Returns:
        Dictionary of grouped files by case key
    """
    groups = defaultdict(dict)

    for r in records:
        split = split_from_stag(r.get("STag"))
        if split is None:
            continue

        # Create unique key for this case
        key = (
            r.get("patients_id"),
            r.get("studies_id") or r.get("StudyID"),
            r.get("SubFolder"),
            split,
        )

        fname = r.get("Filename")
        path = r.get("absFilePath")

        if fname and path:
            groups[key][fname] = path

    return groups, (base_filename, mask1_filename, mask2_filename)


def build_samples(groups, filenames):
    """
    Build train/valid/test samples from grouped files.

    Args:
        groups: Dictionary of grouped files
        filenames: Tuple of (base, mask1, mask2) filenames

    Returns:
        Dictionary with train, valid, test lists
    """
    BASE, M1, M2 = filenames

    out = {"train": [], "valid": [], "test": []}

    skipped = {"train": 0, "valid": 0, "test": 0}

    for key, files in groups.items():
        pid, sid, subf, split = key

        # Check if all required files exist
        if BASE in files and M1 in files and M2 in files:
            sample = {
                "ct": files[BASE],           # Base CT scan
                "mask": files[M1],           # Bone mask
                "mask1": files[M2],          # Tumor mask
                # Optional metadata
                "patients_id": pid,
                "study_id": sid,
                "SubFolder": subf,
            }
            out[split].append(sample)
        else:
            # Track incomplete cases
            skipped[split] += 1
            missing = []
            if BASE not in files:
                missing.append(BASE)
            if M1 not in files:
                missing.append(M1)
            if M2 not in files:
                missing.append(M2)
            print(f"⚠ Skipping case {pid}/{sid}/{subf} ({split}): missing {', '.join(missing)}")

    return out, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse nested JSON data structure into train/valid/test splits"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with nested structure"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_config.json",
        help="Output JSON file path (default: data_config.json)"
    )
    parser.add_argument(
        "--base-filename",
        type=str,
        default="base.nii",
        help="Filename for base/CT scan (default: base.nii)"
    )
    parser.add_argument(
        "--mask1-filename",
        type=str,
        default="mask1.nii.gz",
        help="Filename for bone mask (default: mask1.nii.gz)"
    )
    parser.add_argument(
        "--mask2-filename",
        type=str,
        default="mask2.nii.gz",
        help="Filename for tumor mask (default: mask2.nii.gz)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Custom Data Loader - Nested JSON Parser")
    print("="*60)
    print()

    # Load input JSON
    print(f"Loading JSON from: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Handle different top-level structures
    if isinstance(data, dict) and "json" in data:
        data = data["json"]

    # Flatten records
    print("Flattening nested structure...")
    records = flatten_records(data)
    print(f"✓ Found {len(records)} total records")

    # Group by case
    print("\nGrouping files by case...")
    groups, filenames = group_files_by_case(
        records,
        args.base_filename,
        args.mask1_filename,
        args.mask2_filename
    )
    print(f"✓ Found {len(groups)} unique cases")

    # Build samples
    print("\nBuilding train/valid/test samples...")
    samples, skipped = build_samples(groups, filenames)

    # Print statistics
    print("\n" + "="*60)
    print("Data Split Summary:")
    print("="*60)
    print(f"  Training:   {len(samples['train']):4d} cases")
    print(f"  Validation: {len(samples['valid']):4d} cases")
    print(f"  Test:       {len(samples['test']):4d} cases")
    print(f"  Total:      {len(samples['train']) + len(samples['valid']) + len(samples['test']):4d} cases")

    if sum(skipped.values()) > 0:
        print("\nSkipped (incomplete):")
        print(f"  Training:   {skipped['train']} cases")
        print(f"  Validation: {skipped['valid']} cases")
        print(f"  Test:       {skipped['test']} cases")

    # Validate splits
    if len(samples['train']) == 0:
        print("\n❌ ERROR: No training samples found!")
        print("   Check your STag values. Expected 'Tx' for training.")
        return

    if len(samples['valid']) == 0:
        print("\n⚠ WARNING: No validation samples found!")
        print("   Check your STag values. Expected 'Ty' for validation.")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Saved data configuration to: {args.output}")
    print("="*60)
    print()
    print("Next steps:")
    print(f"  1. Review the generated file: {args.output}")
    print(f"  2. Train model: python scripts/train.py --data {args.output}")
    print(f"  3. Test model:  python scripts/test.py --data {args.output} --model results/best_model.pth")
    print()


if __name__ == "__main__":
    main()
