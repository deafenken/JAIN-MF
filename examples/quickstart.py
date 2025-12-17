"""
JAIN-MF Quickstart Example

This script demonstrates how to use JAIN-MF for action recognition.
"""

import torch
import numpy as np
from models.jain_mf import JAINMFModel, JAINMFConfig

def main():
    # Configuration
    config = JAINMFConfig()
    config.num_classes = 120
    config.num_frames = 32

    # Initialize model
    model = JAINMFModel(**config.__dict__)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dummy input data
    batch_size = 4
    rgb_frames = torch.randn(batch_size, 32, 3, 224, 224)  # RGB frames
    flow_frames = torch.randn(batch_size, 31, 2, 224, 224)  # Optical flow
    skeleton_data = torch.randn(batch_size, 32, 25, 3)  # Skeleton joints (25 joints, 3D)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(rgb_frames, flow_frames, skeleton_data)
        print(f"Output shape: {outputs.shape}")
        print(f"Output logits range: [{outputs.min():.2f}, {outputs.max():.2f}]")

    # Test with missing modalities
    print("\nTesting with missing RGB modality...")
    rgb_missing = torch.zeros_like(rgb_frames)
    outputs_missing = model(rgb_missing, flow_frames, skeleton_data)
    print(f"Output with missing RGB shape: {outputs_missing.shape}")

    print("\nTesting with skeleton-only...")
    flow_missing = torch.zeros_like(flow_frames)
    outputs_skel_only = model(rgb_missing, flow_missing, skeleton_data)
    print(f"Output with skeleton-only shape: {outputs_skel_only.shape}")

if __name__ == "__main__":
    main()