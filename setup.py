"""
Setup script for CT Jaw Defect Segmentation System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jaw-defect-segmentation",
    version="1.0.0",
    author="Development Team",
    description="High-Performance 3D Deep Learning Pipeline for CT Jaw Defect Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxrusse/Detectdefectct",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "monai[all]>=1.3.0",
        "nibabel>=5.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
