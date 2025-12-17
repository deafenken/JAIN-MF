# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU support)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/JAIN-MF.git
cd JAIN-MF
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n jainmf python=3.8
conda activate jainmf

# Or using venv
python -m venv jainmf_env
source jainmf_env/bin/activate  # On Windows: jainmf_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Package

```bash
# Development installation
pip install -e .

# Or regular installation
pip install .
```

## Dependencies

### Core Dependencies
- `torch>=1.12.0` - PyTorch deep learning framework
- `torchvision>=0.13.0` - Computer vision utilities
- `opencv-python>=4.6.0` - Computer vision operations
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.1.0` - Machine learning utilities

### Visualization and Utilities
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `open3d>=0.15.0` - 3D data processing and visualization
- `tqdm>=4.64.0` - Progress bars
- `tensorboard>=2.9.0` - Training visualization

### Data Processing
- `Pillow>=9.0.0` - Image processing
- `scipy>=1.8.0` - Scientific computing
- `einops>=0.6.0` - Tensor operations

## Optional: GPU Support

If you have a CUDA-enabled GPU, install the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Test your installation:

```python
python examples/quickstart.py
```

If the script runs without errors, your installation is successful.

## Dataset Setup

### NTU RGB+D 120 Dataset

1. Download the dataset from the [official website](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. Extract and organize the data as follows:

```
data/NTU_RGB_D_120/
├── rgb/
│   ├── S001/
│   │   └── C001/
│   │       ├── 00001.avi
│   │       └── ...
│   └── ...
├── skeleton/
│   ├── S001/
│   │   └── C001/
│   │       └── skeleton.txt
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### PKU-MMD Dataset

1. Download the dataset from the [official repository](https://github.com/PKU-MMD/PKU-MMD)
2. Organize the data in a similar structure

## Common Issues

### Issue: CUDA out of memory
- Reduce batch size in `config.yaml`
- Use gradient checkpointing

### Issue: Cannot import OpenCV
- Install with: `pip install opencv-python-headless`

### Issue: Missing TV-L1 optical flow
- Install with: `pip install opencv-contrib-python`

## Troubleshooting

For additional help:

1. Check the [FAQ](faq.md)
2. Browse [Issues](https://github.com/yourusername/JAIN-MF/issues)
3. Create a new issue with details about your problem