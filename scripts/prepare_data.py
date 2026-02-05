#!/usr/bin/env python3
"""
Data Preparation Utility
Creates JSON data configuration from directory structure
"""

import argparse
import json
import os
import random
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


def split_data(cases, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split cases into train/val/test sets with shuffling.

    Args:
        cases: List of case dictionaries
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with train, valid, test keys
    """
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    # Shuffle cases to avoid bias from directory ordering
    shuffled_cases = cases.copy()
    random.seed(seed)
    random.shuffle(shuffled_cases)

    n_total = len(shuffled_cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = shuffled_cases[:n_train]
    valid = shuffled_cases[n_train:n_train + n_val]
    test = shuffled_cases[n_train + n_val:]

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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42)"
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

    # Split data (with shuffling for unbiased splits)
    data_dict = split_data(cases, args.train_ratio, args.val_ratio, args.seed)

    print(f"\nData split:")
    print(f"  Training:   {len(data_dict['train'])} cases")
    print(f"  Validation: {len(data_dict['valid'])} cases")
    print(f"  Test:       {len(data_dict['test'])} cases")

    if len(data_dict['train']) == 0:
        print("⚠ Training split is empty. Adjust ratios or add more cases.")
    if len(data_dict['valid']) == 0:
        print("⚠ Validation split is empty. Consider lowering train_ratio or adding cases.")
    if len(data_dict['test']) == 0:
        print("⚠ Test split is empty. Consider lowering train_ratio/val_ratio or adding cases.")

    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print(f"\n✓ Saved configuration to {args.output}")


if __name__ == "__main__":
    main()
