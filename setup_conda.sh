#!/bin/bash
# Quick setup script for conda environment

set -e

echo "=================================================="
echo "CT Jaw Defect Segmentation - Environment Setup"
echo "=================================================="
echo ""

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "✓ Detected CUDA Version: $CUDA_VERSION"
else
    echo "⚠ nvidia-smi not found, defaulting to CUDA 11.8"
    CUDA_VERSION=11
fi

# Select environment file
if [ "$CUDA_VERSION" -ge 12 ]; then
    ENV_FILE="environment_cuda12.yml"
    echo "Using environment_cuda12.yml for CUDA 12.x"
else
    ENV_FILE="environment.yml"
    echo "Using environment.yml for CUDA 11.x"
fi

# Create conda environment
echo ""
echo "Creating conda environment 'jaw-segmentation'..."
conda env create -f $ENV_FILE

echo ""
echo "=================================================="
echo "✓ Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate jaw-segmentation"
echo ""
echo "To verify installation:"
echo "  conda activate jaw-segmentation"
echo "  python -c 'import torch; print(f\"CUDA Available: {torch.cuda.is_available()}\")'"
echo "  python -c 'import monai; print(f\"MONAI Version: {monai.__version__}\")'"
echo ""
