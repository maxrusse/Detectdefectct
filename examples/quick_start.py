#!/usr/bin/env python3
"""
Quick Start Example for CT Jaw Defect Segmentation
Demonstrates basic usage of the pipeline
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.data import MultiMaskDataset, get_transforms
from src.training import Trainer
from src.inference import Predictor
from src.utils import get_default_config


def main():
    """
    Quick start example workflow.

    IMPORTANT: Replace the data_dict below with your actual data paths!
    """

    print("="*60)
    print("CT Jaw Defect Segmentation - Quick Start Example")
    print("="*60)

    # STEP 1: Define your data
    # Replace these paths with your actual NIfTI file paths
    data_dict = {
        'train': [
            {
                'ct': '/path/to/case001/ct.nii.gz',
                'mask': '/path/to/case001/bone.nii.gz',
                'mask1': '/path/to/case001/tumor.nii.gz'
            },
            # Add more training cases...
        ],
        'valid': [
            {
                'ct': '/path/to/case_val/ct.nii.gz',
                'mask': '/path/to/case_val/bone.nii.gz',
                'mask1': '/path/to/case_val/tumor.nii.gz'
            },
        ],
        'test': [
            {
                'ct': '/path/to/case_test/ct.nii.gz',
                'mask': '/path/to/case_test/bone.nii.gz',
                'mask1': '/path/to/case_test/tumor.nii.gz'
            },
        ]
    }

    # STEP 2: Get configuration
    config = get_default_config()

    # Optional: Customize configuration
    config['max_epochs'] = 10  # Reduce for quick test
    config['model_name'] = 'swin'
    config['output_dir'] = './quick_start_results'

    print(f"\nConfiguration:")
    print(f"  Model: {config['model_name']}")
    print(f"  ROI Size: {config['roi_size']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['max_epochs']}")

    # STEP 3: Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiMaskDataset(
        data_dict['train'],
        transforms=get_transforms("train", config)
    )
    val_dataset = MultiMaskDataset(
        data_dict['valid'],
        transforms=get_transforms("val", config)
    )
    test_dataset = MultiMaskDataset(
        data_dict['test'],
        transforms=get_transforms("val", config)
    )

    # STEP 4: Create model
    print("\nCreating model...")
    model = get_model(
        model_name=config['model_name'],
        n_classes=config['n_classes'],
        roi_size=tuple(config['roi_size'])
    )

    # STEP 5: Train model
    print("\nStarting training...")
    trainer = Trainer(model, config, output_dir=config['output_dir'])
    best_model_path = trainer.train(train_dataset, val_dataset)

    # STEP 6: Evaluate on test set
    print("\nEvaluating on test set...")
    predictor = Predictor(model, config)
    results = predictor.evaluate(test_dataset)

    print("\n" + "="*60)
    print("Quick Start Complete!")
    print(f"Best Model: {best_model_path}")
    print(f"Test Results:")
    print(f"  Bone Dice:  {results['bone_dice']:.4f}")
    print(f"  Tumor Dice: {results['tumor_dice']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
