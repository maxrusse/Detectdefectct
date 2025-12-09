"""
MONAI Transform Pipelines with Physics-Based Preprocessing
Optimized for jaw tumor detection with proper HU windowing

CT-SAFE augmentation policy:
- Spatial transforms (flips, rotations, small affine) are safe
- Intensity transforms must be conservative to preserve HU relationships
- Avoid gamma/contrast changes that break linear HU scale
- Noise should be subtle (CT reconstruction already reduces noise)
"""

from monai.transforms import (
    Compose, Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, RandRotate90d,
    RandShiftIntensityd, RandFlipd, RandCoarseDropoutd, EnsureTyped,
    RandGaussianNoised, RandAffined
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
        # Training augmentations - CT-SAFE transforms only
        # CT data has physical HU meaning, so we avoid transforms that break this
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

            # === SPATIAL AUGMENTATIONS (CT-SAFE) ===
            # These are safe because anatomy can appear in different orientations
            RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),

            # Small affine transforms (CT-SAFE: simulates patient positioning variation)
            RandAffined(
                keys=keys,
                prob=config.get("affine_prob", 0.2),
                rotate_range=(0.05, 0.05, 0.05),  # Very small rotations (~3 degrees)
                scale_range=(0.05, 0.05, 0.05),   # ±5% scaling (anatomical variation)
                mode=("bilinear", "nearest"),
                padding_mode="border"
            ),

            # === INTENSITY AUGMENTATIONS (CT-CONSERVATIVE) ===
            # Small Gaussian noise - simulates realistic scanner noise
            # CT noise is typically low after reconstruction
            RandGaussianNoised(
                keys=["image"],
                prob=config.get("noise_prob", 0.2),
                mean=0.0,
                std=0.02  # Reduced from 0.05 - CT noise is subtle
            ),

            # Very mild intensity shift - simulates scanner calibration differences
            # Keep small to preserve HU relationships
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.05,  # Reduced from 0.1
                prob=0.3
            ),

            # === ARTIFACT SIMULATION (CT-REALISTIC) ===
            # Metal artifact simulation (dental fillings/implants)
            # This is realistic - metal creates bright streaks in CT
            RandCoarseDropoutd(
                keys=["image"],
                holes=2,
                spatial_size=(8, 8, 8),  # Slightly smaller
                fill_value=1.0,  # Bright like metal
                prob=config.get("metal_artifact_prob", 0.1)  # Reduced probability
            ),
        ]
        return Compose(train_transforms)

    else:
        # Validation/Test: only base transforms
        return Compose(base_transforms)
