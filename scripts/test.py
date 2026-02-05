#!/usr/bin/env python3
"""
Testing Script for CT Jaw Defect Segmentation
Usage: python scripts/test.py --model best_model.pth --data data_config.json --config config/default_config.yaml
"""

import argparse
import json
import sys
import os

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.data import MultiMaskDataset, get_transforms
from src.inference import Predictor
from src.utils import load_config, get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Test CT Jaw Defect Segmentation Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSON data configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="swin",
        help="Model architecture name: swin, segresnet, unet"
    )
    return parser.parse_args()


def load_data_config(data_path):
    """Load data configuration from JSON file."""
    with open(data_path, 'r') as f:
        data_dict = json.load(f)

    # Use test set if available, otherwise validation set
    if 'test' in data_dict and len(data_dict['test']) > 0:
        test_list = data_dict['test']
        print(f"✓ Using test set: {len(test_list)} samples")
    elif 'valid' in data_dict:
        test_list = data_dict['valid']
        print(f"⚠ No test set found, using validation set: {len(test_list)} samples")
    else:
        raise ValueError("Data config must contain 'test' or 'valid' key")

    return test_list


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()

    # Load data configuration
    test_list = load_data_config(args.data)

    # Create test dataset
    print("\nInitializing test dataset...")
    test_dataset = MultiMaskDataset(
        test_list,
        transforms=get_transforms("val", config)
    )

    # Create model
    print(f"\nInitializing model: {args.model_name}")
    model = get_model(
        model_name=args.model_name,
        n_classes=config['n_classes'],
        roi_size=tuple(config['roi_size'])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained weights
    print(f"Loading weights from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Create predictor
    predictor = Predictor(model, config)

    # Run evaluation
    print("\nRunning evaluation...")
    results = predictor.evaluate(test_dataset, verbose=True)

    print(f"\n{'='*60}")
    print(f"Evaluation completed successfully!")
    print(f"  Bone Dice:  {results['bone_dice']:.4f}")
    print(f"  Tumor Dice: {results['tumor_dice']:.4f}")
    print(f"  Mean Dice:  {results['mean_dice']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
