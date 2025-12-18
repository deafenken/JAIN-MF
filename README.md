# JAIN-MF: Tri-Modal Action Recognition via Skeleton-Guided Sparse Subspace Alignment

This repository implements the JAIN-MF framework for action recognition using skeleton-guided sparse subspace alignment across RGB, optical flow, and skeleton modalities.

## Overview

JAIN-MF addresses two key challenges in multi-modal action recognition:
1. **Incomplete Modalities**: Handles scenarios with missing RGB or skeleton data
2. **Complementary Features**: Effectively combines information from RGB, optical flow, and skeleton modalities

## Key Components

1. **Skeleton-guided Sparse Subspace Alignment (S3A)**: Extracts pose-consistent regions using sparse subspace clustering
2. **CorrNet-based Cross-modal Attention**: Aligns and enhances features between modalities
3. **Tri-modal Fusion**: Combines RGB, optical flow, and skeleton features using attention mechanisms

## Project Structure

```
JAIN_MF/
├── models/
│   ├── backbone.py        # ResNet-50 backbone
│   ├── s3a_module.py      # Skeleton-guided Sparse Subspace Alignment
│   ├── corrnet.py         # CorrNet-based Cross-modal Attention
│   └── jain_mf.py         # Main JAIN-MF model
├── data/
│   ├── datasets.py        # Dataset loaders
│   └── transforms.py      # Data preprocessing and augmentation
├── utils/
│   ├── optical_flow.py    # TV-L1 optical flow computation
│   ├── visualization.py   # Visualization tools
│   └── metrics.py         # Evaluation metrics
├── configs/
│   └── config.yaml        # Configuration file
├── scripts/
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   └── extract_features.py  # Feature extraction
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- tensorboard
- Open3D (for visualization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Testing
```bash
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

### Feature Extraction
```bash
python scripts/extract_features.py --config configs/config.yaml
```
