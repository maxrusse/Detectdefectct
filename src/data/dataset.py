"""
Custom Dataset Handler for Multi-Mask CT Jaw Segmentation
Handles bone and tumor masks with proper priority merging
"""

import numpy as np
import nibabel as nib
from monai.data import Dataset


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
        # Load file paths
        ct_path = self.data_list[index]['ct']
        bone_path = self.data_list[index]['mask']   # Bone mask
        tumor_path = self.data_list[index]['mask1']  # Tumor mask

        # Load NIfTI data
        ct_data = nib.load(ct_path).get_fdata().astype(np.float32)
        bone_data = nib.load(bone_path).get_fdata()
        tumor_data = nib.load(tumor_path).get_fdata()

        # CRITICAL: Merge masks with priority (Tumor > Bone > Background)
        label = np.zeros_like(bone_data, dtype=np.uint8)
        label[bone_data > 0] = 1   # Healthy bone
        label[tumor_data > 0] = 2  # Tumor (overwrites bone)

        # Create batch with channel dimension
        item = {
            "image": ct_data[np.newaxis, ...],  # Shape: (1, H, W, D)
            "label": label[np.newaxis, ...]     # Shape: (1, H, W, D)
        }

        # Apply transforms
        if self.transforms:
            item = self.transforms(item)

        return item
