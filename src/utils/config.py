"""
Configuration Management Utilities
Supports YAML and dictionary-based configurations
"""

import yaml
import os


def get_default_config():
    """
    Get default A6000-optimized configuration.

    Returns:
        Dictionary with default hyperparameters
    """
    return {
        # Hardware Optimization
        "roi_size": (128, 128, 128),  # Massive patch size for A6000
        "batch_size": 4,              # Stable gradient for transformers
        "workers": 8,                 # CPU workers for data loading

        # Training
        "lr": 1e-4,                   # Learning rate
        "min_lr": 1e-6,               # Minimum LR for cosine annealing
        "weight_decay": 1e-5,         # AdamW regularization
        "max_epochs": 300,
        "val_interval": 5,            # Validate every N epochs
        "warmup_epochs": 10,          # LR warmup epochs
        "grad_clip": 1.0,             # Gradient clipping max norm

        # Loss function
        "loss_type": "dice_ce",       # dice_ce | dice_focal
        # "class_weights": [1.0, 1.0, 2.0],  # Optional: weight tumor class higher
        # "focal_weight": 1.0,        # For dice_focal loss

        # Early stopping (set to None to disable)
        "early_stopping_patience": 50,   # Stop if no improvement for N validations
        "early_stopping_delta": 0.001,   # Minimum improvement threshold

        # Reproducibility
        "seed": 42,                   # Random seed for deterministic training

        # Data Preprocessing
        "spacing": (0.4, 0.4, 0.4),   # High-res isotropic voxels
        "hu_range": (-150, 2000),     # Soft-tissue tumor + bone window
        "num_samples": 2,             # Number of crops per sample in training

        # Augmentation probabilities (CT-SAFE defaults)
        # Conservative values to preserve HU relationships
        "affine_prob": 0.2,           # Small affine transforms
        "noise_prob": 0.2,            # Subtle Gaussian noise
        "metal_artifact_prob": 0.1,   # Metal artifact simulation

        # Inference
        "sw_batch_size": 4,           # Sliding window batch size
        "sw_overlap": 0.5,            # Sliding window overlap

        # Model
        "model_name": "swin",         # swin | segresnet | unet
        "n_classes": 3,               # Background | Bone | Tumor

        # Output
        "output_dir": "./results",
        "save_interval": 50           # Save checkpoint every N epochs
    }


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge with defaults (user config overrides defaults)
    default_config = get_default_config()
    default_config.update(config)

    return default_config


def save_config(config, output_path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Config saved to {output_path}")
