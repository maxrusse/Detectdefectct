#!/usr/bin/env python3
"""
Training Script for CT Jaw Defect Segmentation
Usage: python scripts/train.py --config config/default_config.yaml --data data_config.json
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.data import MultiMaskDataset, get_transforms
from src.training import Trainer
from src.utils import load_config, get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train CT Jaw Defect Segmentation Model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional, uses defaults if not provided)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSON data configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name: swin, segresnet, unet (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    return parser.parse_args()


def load_data_config(data_path):
    """Load data configuration from JSON file."""
    with open(data_path, 'r') as f:
        data_dict = json.load(f)

    # Validate structure
    required_keys = ['train', 'valid']
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"Data config missing required key: {key}")

    print(f"✓ Loaded data configuration:")
    print(f"  Training samples: {len(data_dict['train'])}")
    print(f"  Validation samples: {len(data_dict['valid'])}")

    return data_dict


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()

    # Override with command-line arguments
    if args.model:
        config['model_name'] = args.model
    if args.output:
        config['output_dir'] = args.output

    # Load data configuration
    data_dict = load_data_config(args.data)

    # Create datasets
    print("\nInitializing datasets...")
    train_dataset = MultiMaskDataset(
        data_dict['train'],
        transforms=get_transforms("train", config)
    )
    val_dataset = MultiMaskDataset(
        data_dict['valid'],
        transforms=get_transforms("val", config)
    )

    # Create model
    print(f"\nInitializing model: {config['model_name']}")
    model = get_model(
        model_name=config['model_name'],
        n_classes=config['n_classes'],
        roi_size=tuple(config['roi_size'])
    )

    # Create trainer
    trainer = Trainer(model, config, output_dir=config['output_dir'])

    # Run training
    print("\nStarting training...")
    best_model_path = trainer.train(train_dataset, val_dataset)

    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
