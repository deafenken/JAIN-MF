"""
JAIN-MF: Tri-Modal Action Recognition via Skeleton-Guided Sparse Subspace Alignment

This package implements the JAIN-MF framework for action recognition using
RGB, optical flow, and skeleton modalities with skeleton-guided sparse
subspace alignment.

Main Components:
- Skeleton-guided Sparse Subspace Alignment (S3A) module
- CorrNet-based cross-modal attention
- Multi-modal fusion framework
- Comprehensive training and evaluation pipeline
"""

__version__ = "1.0.0"
__author__ = "JAIN-MF Implementation"

from .models.jain_mf import JAINMFModel, JAINMFConfig
from .models.backbone import ResNetBackbone, FlowBackbone, SkeletonBackbone
from .models.s3a_module import SkeletonGuidedS3A
from .models.corrnet import CorrNetCrossModalAttention

__all__ = [
    'JAINMFModel',
    'JAINMFConfig',
    'ResNetBackbone',
    'FlowBackbone',
    'SkeletonBackbone',
    'SkeletonGuidedS3A',
    'CorrNetCrossModalAttention'
]