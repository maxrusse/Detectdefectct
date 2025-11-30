"""
Model Factory for Exchangeable Backbone Architectures
Supports SwinUNETR (Transformer) and SegResNet (CNN) for A6000
"""

import torch
from monai.networks.nets import SwinUNETR, SegResNet, UNet


def get_model(model_name, n_classes=3, roi_size=(128, 128, 128), device=None):
    """
    Factory function to create segmentation models.

    Args:
        model_name: Model architecture name
            - "swin": SwinUNETR (Vision Transformer, best for global context)
            - "segresnet": SegResNet (Optimized CNN, best for boundary precision)
            - "unet": Baseline 3D U-Net
        n_classes: Number of output classes (default: 3 for background/bone/tumor)
        roi_size: Input patch size (default: (128, 128, 128) for A6000)
        device: Target device (default: auto-detect CUDA)

    Returns:
        PyTorch model on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.lower() == "swin":
        # SwinUNETR: Vision Transformer Architecture
        # Best for: Global context, complex geometric relationships
        # Trade-off: Higher memory usage, slower training
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=1,
            out_channels=n_classes,
            feature_size=48,           # Base feature dimension
            use_checkpoint=True,       # Gradient checkpointing to save VRAM
            spatial_dims=3
        )
        print(f"✓ Loaded SwinUNETR (Transformer) for ROI {roi_size}")

    elif model_name.lower() == "segresnet":
        # SegResNet: Optimized Residual CNN
        # Best for: Boundary precision, faster training
        # Trade-off: Less global context than Transformer
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            init_filters=32,           # Initial filter count
            blocks_down=[1, 2, 2, 4],  # Encoder depth
            blocks_up=[1, 1, 1],       # Decoder depth
            dropout_prob=0.2           # Regularization
        )
        print(f"✓ Loaded SegResNet (CNN) for {n_classes} classes")

    elif model_name.lower() == "unet":
        # Baseline 3D U-Net
        # Best for: Quick experiments, baseline comparisons
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        print(f"✓ Loaded 3D U-Net (Baseline)")

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: 'swin', 'segresnet', 'unet'"
        )

    model = model.to(device)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {n_params:,}")

    return model
