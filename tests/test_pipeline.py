"""
Basic unit tests for pipeline components
Run with: pytest tests/
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.utils import get_default_config


class TestModelFactory:
    """Test model creation."""

    def test_swin_creation(self):
        """Test SwinUNETR model creation."""
        model = get_model("swin", n_classes=3, roi_size=(128, 128, 128))
        assert model is not None
        assert next(model.parameters()).device.type == "cuda" or next(model.parameters()).device.type == "cpu"

    def test_segresnet_creation(self):
        """Test SegResNet model creation."""
        model = get_model("segresnet", n_classes=3, roi_size=(128, 128, 128))
        assert model is not None

    def test_unet_creation(self):
        """Test U-Net model creation."""
        model = get_model("unet", n_classes=3, roi_size=(128, 128, 128))
        assert model is not None

    def test_invalid_model(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model("invalid_model")


class TestConfig:
    """Test configuration utilities."""

    def test_default_config(self):
        """Test default configuration has required keys."""
        config = get_default_config()
        required_keys = ['roi_size', 'batch_size', 'lr', 'max_epochs', 'spacing', 'hu_range']
        for key in required_keys:
            assert key in config

    def test_config_values(self):
        """Test default configuration has correct types."""
        config = get_default_config()
        assert isinstance(config['roi_size'], tuple) or isinstance(config['roi_size'], list)
        assert isinstance(config['batch_size'], int)
        assert isinstance(config['lr'], float)


class TestDataPipeline:
    """Test data handling components."""

    def test_label_merging_logic(self):
        """Test that tumor overwrites bone in label merging."""
        # Simulate merging logic
        bone = np.zeros((10, 10, 10))
        tumor = np.zeros((10, 10, 10))

        bone[2:5, 2:5, 2:5] = 1  # Bone region
        tumor[3:6, 3:6, 3:6] = 1  # Overlapping tumor region

        # Merge (tumor priority)
        label = np.zeros_like(bone, dtype=np.uint8)
        label[bone > 0] = 1
        label[tumor > 0] = 2

        # Check overlap region has tumor label
        assert label[4, 4, 4] == 2  # Tumor overwrites bone

    def test_hu_windowing_range(self):
        """Test HU windowing range is correct."""
        config = get_default_config()
        hu_range = config['hu_range']

        # Critical: range must capture soft-tissue tumors
        assert hu_range[0] <= -150, "Lower bound should capture soft tissue"
        assert hu_range[1] >= 2000, "Upper bound should capture bone"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
