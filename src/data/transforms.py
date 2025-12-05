"""
MONAI Transform Pipelines with Physics-Based Preprocessing
Optimized for jaw tumor detection with proper HU windowing
"""

from monai.transforms import (
    Compose, Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, RandRotate90d,
    RandShiftIntensityd, RandFlipd, RandCoarseDropoutd, EnsureTyped,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandAffined, RandAdjustContrastd
)


def get_transforms(mode="train", config=None):
    """
    Create MONAI transform pipeline based on mode.

    Args:
        mode: "train", "val", or "test"
        config: Configuration dictionary with keys:
            - spacing: tuple of target voxel spacing (default: (0.4, 0.4, 0.4))
            - hu_range: tuple of HU window (default: (-150, 2000))
            - roi_size: tuple of patch size (default: (128, 128, 128))

    Returns:
        MONAI Compose transform
    """
    if config is None:
        config = {
            "spacing": (0.4, 0.4, 0.4),
            "hu_range": (-150, 2000),
            "roi_size": (128, 128, 128)
        }

    keys = ["image", "label"]

    # Base transforms (applied to all modes)
    base_transforms = [
        # Ensure proper tensor type
        EnsureTyped(keys=keys),

        # Standardize orientation
        Orientationd(keys=keys, axcodes="RAS"),

        # Resample to isotropic high-resolution
        Spacingd(
            keys=keys,
            pixdim=config["spacing"],
            mode=("bilinear", "nearest")
        ),

        # CRITICAL: Physics-based HU windowing
        # Range: [-150, 2000] captures soft-tissue tumors (~40 HU) + bone
        # Standard bone windows (300+) make lytic tumors invisible!
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["hu_range"][0],
            a_max=config["hu_range"][1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),

        # Remove background air
        CropForegroundd(keys=keys, source_key="image"),
    ]

    if mode == "train":
        # Training augmentations
        train_transforms = base_transforms + [
            # Spatial cropping (balanced pos/neg sampling)
            RandCropByPosNegLabeld(
                keys=keys,
                spatial_size=config["roi_size"],
                label_key="label",
                pos=1,  # 50% positive samples
                neg=1,  # 50% negative samples
                num_samples=config.get("num_samples", 2),
                image_key="image",
                image_threshold=0
            ),

            # === SPATIAL AUGMENTATIONS ===
            # Rotation before flip is standard practice
            RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),

            # Random affine for more spatial variation (scale, rotation, shear)
            RandAffined(
                keys=keys,
                prob=config.get("affine_prob", 0.3),
                rotate_range=(0.1, 0.1, 0.1),  # Small rotations in radians
                scale_range=(0.1, 0.1, 0.1),   # ±10% scaling
                mode=("bilinear", "nearest"),
                padding_mode="border"
            ),

            # === INTENSITY AUGMENTATIONS ===
            # Gaussian noise - simulates scanner noise
            RandGaussianNoised(
                keys=["image"],
                prob=config.get("noise_prob", 0.3),
                mean=0.0,
                std=0.05
            ),

            # Gaussian smoothing - simulates lower resolution / motion blur
            RandGaussianSmoothd(
                keys=["image"],
                prob=config.get("blur_prob", 0.2),
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            ),

            # Contrast adjustment - simulates different scanner settings
            RandAdjustContrastd(
                keys=["image"],
                prob=config.get("contrast_prob", 0.3),
                gamma=(0.8, 1.2)
            ),

            # Scale intensity - random brightness adjustment
            RandScaleIntensityd(
                keys=["image"],
                factors=0.15,
                prob=0.3
            ),

            # Intensity shift
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

            # === ARTIFACT SIMULATION ===
            # Metal artifact simulation (dental fillings/implants)
            # Creates streak artifacts that can cause false positives
            RandCoarseDropoutd(
                keys=["image"],
                holes=2,
                spatial_size=(10, 10, 10),
                fill_value=1.0,  # Bright spots like metal
                prob=config.get("metal_artifact_prob", 0.15)
            ),
        ]
        return Compose(train_transforms)

    else:
        # Validation/Test: only base transforms
        return Compose(base_transforms)
