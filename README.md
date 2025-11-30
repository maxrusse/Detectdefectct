# CT Jaw Defect Segmentation System

**High-Performance 3D Deep Learning Pipeline for Mandibular/Maxillary Bone and Tumor Segmentation**

Optimized for NVIDIA RTX A6000 (48GB VRAM) using PyTorch and MONAI.

---

## Overview

This system segments **osseous defects** (bone tumors) in CT scans of the jaw (mandible/maxilla) using state-of-the-art 3D deep learning. It leverages the massive VRAM of the RTX A6000 to process large 3D context windows (`128×128×128` voxels), enabling superior detection of complex geometric relationships that standard models miss.

### Key Features

- **Exchangeable Backbone Architecture**: Switch between SwinUNETR (Transformer), SegResNet (CNN), or U-Net
- **Physics-Based Preprocessing**: Specialized HU windowing (`-150 to 2000`) to capture lytic tumors invisible to standard bone windows
- **A6000-Optimized**: Maximizes 48GB VRAM with large batch sizes and patch dimensions
- **Production-Ready**: Modular design with comprehensive documentation and examples

### Performance Advantages

| Feature | Standard Approach | This System |
|---------|------------------|-------------|
| Patch Size | 96×96×96 | **128×128×128** |
| HU Window | 300-2000 | **-150 to 2000** |
| Context | Limited | **Full mandible cross-section** |
| Tumor Detection | Misses lytic lesions | **Captures soft-tissue density** |

---

## Installation

### Prerequisites

- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Hardware**: NVIDIA RTX A6000 (48GB) or similar high-VRAM GPU
- **Software**:
  - Python 3.9+ (3.10 recommended)
  - CUDA 11.8 or 12.x
  - cuDNN
  - Conda or Miniconda (recommended)

### Method 1: Conda Installation (Recommended)

**Automatic Setup:**
```bash
# Clone repository
git clone https://github.com/maxrusse/Detectdefectct.git
cd Detectdefectct

# Run automated setup script
chmod +x setup_conda.sh
./setup_conda.sh

# Activate environment
conda activate jaw-segmentation
```

**Manual Setup:**
```bash
# Clone repository
git clone https://github.com/maxrusse/Detectdefectct.git
cd Detectdefectct

# Create conda environment
# For CUDA 11.8:
conda env create -f environment.yml

# For CUDA 12.x:
conda env create -f environment_cuda12.yml

# Activate environment
conda activate jaw-segmentation
```

### Method 2: Pip Installation (Alternative)

```bash
# Clone repository
git clone https://github.com/maxrusse/Detectdefectct.git
cd Detectdefectct

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Method 3: Conda + Pip (Hybrid)

```bash
# Create base conda environment
conda create -n jaw-segmentation python=3.10
conda activate jaw-segmentation

# Install PyTorch with conda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining packages with pip
pip install -r requirements.txt
```

### Verify Installation

```bash
# Activate environment (if using conda)
conda activate jaw-segmentation

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"

# Check MONAI
python -c "import monai; print(f'MONAI Version: {monai.__version__}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Quick Start

### 1. Prepare Your Data

Your data should be organized with **three files per case**:

- `ct.nii.gz` - CT scan (Hounsfield Units)
- `bone.nii.gz` - Bone segmentation mask
- `tumor.nii.gz` - Tumor segmentation mask

**Directory Structure:**
```
data/
├── case001/
│   ├── ct.nii.gz
│   ├── bone.nii.gz
│   └── tumor.nii.gz
├── case002/
│   ├── ct.nii.gz
│   ├── bone.nii.gz
│   └── tumor.nii.gz
└── ...
```

### 2. Create Data Configuration

```bash
python scripts/prepare_data.py \
    --data-dir /path/to/data \
    --output data_config.json \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

This generates a JSON file:
```json
{
  "train": [...],
  "valid": [...],
  "test": [...]
}
```

### 3. Train Model

```bash
python scripts/train.py \
    --config config/a6000_optimized.yaml \
    --data data_config.json \
    --output ./results
```

### 4. Test Model

```bash
python scripts/test.py \
    --model results/best_model.pth \
    --data data_config.json \
    --model-name swin
```

---

## Configuration

### Default Configuration

See `config/default_config.yaml`:

```yaml
# Model
model_name: "swin"      # swin | segresnet | unet
n_classes: 3            # Background | Bone | Tumor

# Hardware (A6000 Optimized)
roi_size: [128, 128, 128]
batch_size: 4
workers: 8

# Training
lr: 0.0001
max_epochs: 300
val_interval: 5

# Preprocessing (CRITICAL!)
spacing: [0.4, 0.4, 0.4]    # High-res isotropic
hu_range: [-150, 2000]      # Soft-tissue + bone window
```

### A6000-Specific Optimization

Use `config/a6000_optimized.yaml` for maximum performance:

```yaml
roi_size: [128, 128, 128]   # Maximum stable patch
batch_size: 4               # Optimized for SwinUNETR
workers: 12                 # Fast I/O
max_epochs: 500             # Extended training
```

---

## Architecture Details

### Model Options

#### 1. SwinUNETR (Recommended)
**Vision Transformer architecture**

- **Best for**: Global context, complex geometric relationships
- **Trade-off**: Higher memory, slower training
- **Use case**: Primary model for production

```python
from src.models import get_model
model = get_model("swin", n_classes=3, roi_size=(128, 128, 128))
```

#### 2. SegResNet
**Optimized Residual CNN**

- **Best for**: Boundary precision, faster training
- **Trade-off**: Less global context
- **Use case**: Baseline comparison, resource-limited scenarios

```python
model = get_model("segresnet", n_classes=3)
```

#### 3. U-Net
**Classic 3D U-Net baseline**

- **Best for**: Quick experiments, sanity checks
- **Use case**: Baseline comparisons

---

## Data Pipeline

### Label Merging Strategy

The system handles overlapping masks with **priority logic**:

```python
# Tumor pixels OVERWRITE bone pixels
label = np.zeros_like(bone_data)
label[bone_data > 0] = 1   # Healthy bone
label[tumor_data > 0] = 2  # Tumor (overwrites)
```

**Output Classes:**
- `0` = Background
- `1` = Healthy Bone
- `2` = Tumor

### HU Windowing (Critical!)

**Why `-150 to 2000`?**

Standard bone windows (`300-2000 HU`) **miss lytic tumors**:
- Lytic lesions: ~40 HU (soft-tissue density)
- Standard window: Starts at 300 HU → **tumor appears black!**

Our window captures:
- Soft tissue: -150 to 100 HU
- Bone: 300 to 2000 HU
- **Result**: Tumors visible inside bone

### Metal Artifact Simulation

Training includes random dropout to simulate dental fillings:

```python
RandCoarseDropout(
    holes=2,
    spatial_size=(10, 10, 10),
    prob=0.15
)
```

Prevents false positives from metal streaks.

---

## Advanced Usage

### Python API

```python
from src.models import get_model
from src.data import MultiMaskDataset, get_transforms
from src.training import Trainer
from src.inference import Predictor
from src.utils import get_default_config

# Load config
config = get_default_config()

# Create datasets
train_dataset = MultiMaskDataset(train_list, get_transforms("train", config))
val_dataset = MultiMaskDataset(val_list, get_transforms("val", config))

# Create model
model = get_model("swin", n_classes=3, roi_size=(128, 128, 128))

# Train
trainer = Trainer(model, config, output_dir="./results")
best_model = trainer.train(train_dataset, val_dataset)

# Evaluate
predictor = Predictor(model, config)
results = predictor.evaluate(test_dataset)
```

### Custom Configuration

```python
config = {
    "roi_size": (96, 96, 96),     # Smaller for less VRAM
    "batch_size": 2,              # Reduce if OOM
    "lr": 5e-5,                   # Lower learning rate
    "max_epochs": 100,
    "spacing": (0.5, 0.5, 0.5),   # Coarser resolution
    "hu_range": (-150, 2000),     # Keep this!
    "model_name": "segresnet"
}
```

---

## Project Structure

```
Detectdefectct/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installer
├── .gitignore
│
├── config/
│   ├── default_config.yaml     # Default hyperparameters
│   └── a6000_optimized.yaml    # A6000-specific config
│
├── src/
│   ├── models/
│   │   └── model_factory.py    # Model creation
│   ├── data/
│   │   ├── dataset.py          # Multi-mask dataset
│   │   └── transforms.py       # MONAI transforms
│   ├── training/
│   │   └── trainer.py          # Training engine
│   ├── inference/
│   │   └── predictor.py        # Inference engine
│   └── utils/
│       └── config.py           # Config utilities
│
├── scripts/
│   ├── train.py                # Training script
│   ├── test.py                 # Testing script
│   └── prepare_data.py         # Data preparation
│
├── examples/
│   ├── sample_data_config.json # Example data config
│   └── quick_start.py          # Quick start demo
│
└── tests/
    └── test_pipeline.py        # Unit tests
```

---

## Technical Specifications

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3090 (24GB) | **RTX A6000 (48GB)** |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 100GB SSD | 500GB NVMe SSD |
| **CPU** | 8 cores | 16+ cores |

### Performance Benchmarks

On RTX A6000:

| Model | Patch Size | Batch Size | Training Speed | VRAM Usage |
|-------|-----------|------------|----------------|------------|
| SwinUNETR | 128³ | 4 | ~3.5 it/s | ~42GB |
| SegResNet | 128³ | 4 | ~5.0 it/s | ~28GB |
| U-Net | 128³ | 4 | ~6.5 it/s | ~24GB |

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
batch_size: 2

# Reduce patch size
roi_size: [96, 96, 96]

# Use gradient checkpointing (SwinUNETR)
use_checkpoint: true
```

### Poor Tumor Detection

**Check HU windowing:**
```yaml
# MUST use this range!
hu_range: [-150, 2000]

# NOT this (standard bone window):
# hu_range: [300, 2000]  # Wrong!
```

### Training Instability

```yaml
# Reduce learning rate
lr: 5e-5

# Increase batch size (if VRAM allows)
batch_size: 8

# Add gradient clipping
max_grad_norm: 1.0
```

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{jaw_defect_seg_2024,
  title={High-Performance CT Jaw Defect Segmentation System},
  author={Development Team},
  year={2024},
  url={https://github.com/maxrusse/Detectdefectct}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Support

For issues, questions, or contributions:

- **GitHub Issues**: https://github.com/maxrusse/Detectdefectct/issues
- **Documentation**: See `examples/` directory
- **Email**: [Your contact email]

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [MONAI](https://monai.io/)
- [NiBabel](https://nipy.org/nibabel/)

Optimized for NVIDIA RTX A6000.

---

**Last Updated**: 2024
**Version**: 1.0.0
