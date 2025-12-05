"""
Custom Dataset Handler for Multi-Mask CT Jaw Segmentation
Handles bone and tumor masks with proper priority merging
"""

import numpy as np
import nibabel as nib
from monai.data import Dataset, CacheDataset, PersistentDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Lambda


def load_and_merge_masks(data_dict):
    """
    Load NIfTI files and merge masks with priority.

    Args:
        data_dict: Dictionary with 'ct', 'mask', 'mask1' paths

    Returns:
        Dictionary with 'image' and 'label' arrays
    """
    # Load NIfTI data
    ct_data = nib.load(data_dict['ct']).get_fdata().astype(np.float32)
    bone_data = nib.load(data_dict['mask']).get_fdata()
    tumor_data = nib.load(data_dict['mask1']).get_fdata()

    # CRITICAL: Merge masks with priority (Tumor > Bone > Background)
    label = np.zeros_like(bone_data, dtype=np.uint8)
    label[bone_data > 0] = 1   # Healthy bone
    label[tumor_data > 0] = 2  # Tumor (overwrites bone)

    return {
        "image": ct_data[np.newaxis, ...],  # Shape: (1, H, W, D)
        "label": label[np.newaxis, ...]     # Shape: (1, H, W, D)
    }


class MultiMaskDataset(Dataset):
    """
    Custom dataset for CT jaw defect segmentation with multiple masks.

    Handles merging of bone and tumor masks with proper priority:
    - Background: 0
    - Healthy Bone: 1
    - Tumor: 2 (overwrites bone)

    Args:
        data_list: List of dictionaries with keys 'ct', 'mask' (bone), 'mask1' (tumor)
        transforms: MONAI transforms to apply
    """

    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Load and process a single sample.

        Returns:
            dict: Dictionary with 'image' and 'label' keys
        """
        try:
            item = load_and_merge_masks(self.data_list[index])
        except Exception as e:
            raise RuntimeError(
                f"Error loading sample {index}: {self.data_list[index].get('ct', 'unknown')}"
            ) from e

        # Apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item


def create_cached_dataset(data_list, transforms=None, cache_rate=1.0, num_workers=4):
    """
    Create a cached dataset for faster training.

    Uses MONAI's CacheDataset to pre-load and cache data in memory,
    significantly speeding up training by avoiding repeated disk I/O.

    Args:
        data_list: List of dictionaries with keys 'ct', 'mask', 'mask1'
        transforms: MONAI transforms to apply
        cache_rate: Fraction of data to cache (0.0 to 1.0, default: 1.0)
        num_workers: Number of workers for caching (default: 4)

    Returns:
        CacheDataset instance
    """
    # Create a loading transform that handles our custom multi-mask format
    load_transform = Lambda(func=load_and_merge_masks)

    # Combine loading with user transforms
    if transforms is not None:
        full_transforms = Compose([load_transform, transforms])
    else:
        full_transforms = load_transform

    return CacheDataset(
        data=data_list,
        transform=full_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True
    )


def create_persistent_dataset(data_list, transforms=None, cache_dir="./cache"):
    """
    Create a persistent cached dataset for very large datasets.

    Uses MONAI's PersistentDataset to cache transformed data to disk,
    useful when data doesn't fit in memory.

    Args:
        data_list: List of dictionaries with keys 'ct', 'mask', 'mask1'
        transforms: MONAI transforms to apply
        cache_dir: Directory to store cached data (default: './cache')

    Returns:
        PersistentDataset instance
    """
    # Create a loading transform that handles our custom multi-mask format
    load_transform = Lambda(func=load_and_merge_masks)

    # Combine loading with user transforms
    if transforms is not None:
        full_transforms = Compose([load_transform, transforms])
    else:
        full_transforms = load_transform

    return PersistentDataset(
        data=data_list,
        transform=full_transforms,
        cache_dir=cache_dir
    )
