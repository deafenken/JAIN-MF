import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange

class ResNetBackbone(nn.Module):
    """ResNet-50 backbone for feature extraction"""

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Remove classification layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        # Additional layers for better feature representation
        self.additional_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W)
        Returns:
            features: Feature maps of shape (B, T, 512, H/16, W/16)
        """
        B, T, C, H, W = x.shape

        # Reshape for processing
        x = x.view(B * T, C, H, W)

        # Extract features
        features = self.features(x)
        features = self.additional_conv(features)

        # Reshape back
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, T, C_out, H_out, W_out)

        return features


class FlowBackbone(nn.Module):
    """Backbone for optical flow processing"""

    def __init__(self, input_channels=2):
        super(FlowBackbone, self).__init__()

        # Custom CNN for optical flow
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: Optical flow of shape (B, T, 2, H, W)
        Returns:
            features: Flow features of shape (B, T, 512, H/32, W/32)
        """
        B, T, C, H, W = x.shape

        # Reshape for processing
        x = x.view(B * T, C, H, W)

        # Extract features
        features = self.conv_layers(x)

        # Reshape back
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, T, C_out, H_out, W_out)

        return features


class SkeletonBackbone(nn.Module):
    """Backbone for skeleton sequence processing"""

    def __init__(self, joint_dim=3, num_joints=25, hidden_dim=256):
        super(SkeletonBackbone, self).__init__()

        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # Linear projection for joint coordinates
        self.joint_embed = nn.Linear(joint_dim, hidden_dim)

        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * num_joints, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Spatial attention for joints
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: Skeleton data of shape (B, T, J, 3)
        Returns:
            features: Skeleton features of shape (B, T, hidden_dim)
        """
        B, T, J, C = x.shape

        # Reshape and embed joint coordinates
        x = x.view(B * T, J, C)
        x = self.joint_embed(x)  # (B*T, J, hidden_dim)

        # Apply spatial attention
        x, _ = self.spatial_attention(x, x, x)  # (B*T, J, hidden_dim)

        # Reshape for temporal convolution
        x = x.view(B * T, -1).transpose(1, 0).unsqueeze(0)  # (1, J*hidden_dim, B*T)

        # Apply temporal modeling
        x = self.temporal_conv(x)  # (1, hidden_dim, B*T)

        # Reshape back
        x = x.squeeze(0).transpose(0, 1)  # (B*T, hidden_dim)
        x = x.view(B, T, self.hidden_dim)

        return x