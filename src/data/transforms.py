"""
MONAI Transform Pipelines with Physics-Based Preprocessing
Optimized for jaw tumor detection with proper HU windowing
"""

from monai.transforms import (
    Compose, Orientation, Spacing, ScaleIntensityRange,
    CropForeground, RandCropByPosNegLabel, RandRotate90,
    RandShiftIntensity, RandFlip, RandCoarseDropout
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
        # Standardize orientation
        Orientation(axcodes="RAS", keys=keys),

        # Resample to isotropic high-resolution
        Spacing(
            pixdim=config["spacing"],
            mode=("bilinear", "nearest"),
            keys=keys
        ),

        # CRITICAL: Physics-based HU windowing
        # Range: [-150, 2000] captures soft-tissue tumors (~40 HU) + bone
        # Standard bone windows (300+) make lytic tumors invisible!
        ScaleIntensityRange(
            a_min=config["hu_range"][0],
            a_max=config["hu_range"][1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
            keys=["image"]
        ),

        # Remove background air
        CropForeground(keys=keys, source_key="image"),
    ]

    if mode == "train":
        # Training augmentations
        train_transforms = base_transforms + [
            # Spatial cropping (balanced pos/neg sampling)
            RandCropByPosNegLabel(
                spatial_size=config["roi_size"],
                label_key="label",
                pos=1,  # 50% positive samples
                neg=1,  # 50% negative samples
                num_samples=2,
                image_key="image",
                image_threshold=0
            ),

            # Spatial augmentations
            RandFlip(prob=0.5, spatial_axis=0, keys=keys),
            RandRotate90(prob=0.5, max_k=3, keys=keys),

            # CRITICAL: Metal artifact simulation
            # Simulates dental fillings/implants (streak artifacts)
            # Prevents false positives from metal artifacts
            RandCoarseDropout(
                keys=["image"],
                holes=2,
                spatial_size=(10, 10, 10),
                fill_value=1.0,  # Bright spots like metal
                prob=0.15
            ),

            # Intensity augmentation
            RandShiftIntensity(offsets=0.1, prob=0.5, keys=["image"]),
        ]
        return Compose(train_transforms)

    else:
        # Validation/Test: only base transforms
        return Compose(base_transforms)
