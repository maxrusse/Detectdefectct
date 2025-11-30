#!/usr/bin/env python3
"""
Data Preparation Utility
Creates JSON data configuration from directory structure
"""

import argparse
import json
import os
from pathlib import Path


def find_cases(data_dir, pattern="*.nii.gz"):
    """
    Scan directory for CT/mask pairs.

    Expected structure:
        data_dir/
            case001/
                ct.nii.gz
                bone.nii.gz
                tumor.nii.gz
            case002/
                ...

    Returns:
        List of dictionaries with ct, mask, mask1 paths
    """
    cases = []
    data_path = Path(data_dir)

    for case_dir in sorted(data_path.iterdir()):
        if not case_dir.is_dir():
            continue

        ct_file = case_dir / "ct.nii.gz"
        bone_file = case_dir / "bone.nii.gz"
        tumor_file = case_dir / "tumor.nii.gz"

        # Check if all files exist
        if ct_file.exists() and bone_file.exists() and tumor_file.exists():
            cases.append({
                "ct": str(ct_file.absolute()),
                "mask": str(bone_file.absolute()),
                "mask1": str(tumor_file.absolute())
            })
        else:
            print(f"⚠ Skipping {case_dir.name}: missing files")

    return cases


def split_data(cases, train_ratio=0.7, val_ratio=0.15):
    """
    Split cases into train/val/test sets.

    Args:
        cases: List of case dictionaries
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)

    Returns:
        Dictionary with train, valid, test keys
    """
    n_total = len(cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = cases[:n_train]
    valid = cases[n_train:n_train + n_val]
    test = cases[n_train + n_val:]

    return {
        "train": train,
        "valid": valid,
        "test": test
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data configuration JSON")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing case subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_config.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    args = parser.parse_args()

    print(f"Scanning directory: {args.data_dir}")
    cases = find_cases(args.data_dir)

    if len(cases) == 0:
        print("❌ No valid cases found!")
        print("\nExpected structure:")
        print("  data_dir/")
        print("    case001/")
        print("      ct.nii.gz")
        print("      bone.nii.gz")
        print("      tumor.nii.gz")
        return

    print(f"✓ Found {len(cases)} valid cases")

    # Split data
    data_dict = split_data(cases, args.train_ratio, args.val_ratio)

    print(f"\nData split:")
    print(f"  Training:   {len(data_dict['train'])} cases")
    print(f"  Validation: {len(data_dict['valid'])} cases")
    print(f"  Test:       {len(data_dict['test'])} cases")

    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print(f"\n✓ Saved configuration to {args.output}")


if __name__ == "__main__":
    main()
