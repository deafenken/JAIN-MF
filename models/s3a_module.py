import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class SkeletonGuidedS3A(nn.Module):
    """Skeleton-guided Sparse Subspace Alignment (S3A) Module"""

    def __init__(self, feature_dim=512, num_subspaces=10, sparsity_weight=0.1):
        super(SkeletonGuidedS3A, self).__init__()

        self.feature_dim = feature_dim
        self.num_subspaces = num_subspaces
        self.sparsity_weight = sparsity_weight

        # Learnable dictionary for subspace representation
        self.dictionary = nn.Parameter(
            torch.randn(num_subspaces, feature_dim),
            requires_grad=True
        )

        # Skeleton feature projector
        self.skeleton_proj = nn.Sequential(
            nn.Linear(3 * 25, 512),  # 25 joints with 3 coordinates each
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.Tanh()
        )

        # Attention mechanism for skeleton guidance
        self.skeleton_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # Reconstruction layer
        self.reconstruction_layer = nn.Linear(feature_dim, feature_dim)

    def compute_skeleton_features(self, skeleton_data):
        """Compute skeleton-based features for guidance"""
        B, T, J, C = skeleton_data.shape

        # Flatten skeleton data
        skeleton_flat = skeleton_data.view(B * T, -1)  # (B*T, J*C)

        # Project to feature space
        skeleton_features = self.skeleton_proj(skeleton_flat)  # (B*T, feature_dim)

        # Reshape back
        skeleton_features = skeleton_features.view(B, T, self.feature_dim)

        return skeleton_features

    def sparse_representation(self, features, skeleton_features):
        """Compute sparse representation of features in subspaces"""

        B, T, H, W, C = features.shape

        # Reshape features for processing
        features_flat = features.view(B * T * H * W, C)  # (N, C)
        skeleton_features_expanded = skeleton_features.unsqueeze(2).unsqueeze(2)  # (B, T, 1, 1, C)
        skeleton_features_expanded = skeleton_features_expanded.expand(-1, -1, H, W, -1)  # (B, T, H, W, C)
        skeleton_features_flat = skeleton_features_expanded.view(B * T * H * W, C)  # (N, C)

        # Initialize representation coefficients
        N = features_flat.shape[0]
        coeffs = torch.zeros(N, self.num_subspaces, device=features.device)

        # Compute similarity with dictionary elements
        dict_normalized = F.normalize(self.dictionary, dim=1)
        features_norm = F.normalize(features_flat, dim=1)
        skeleton_norm = F.normalize(skeleton_features_flat, dim=1)

        # Compute weighted similarities
        for i in range(self.num_subspaces):
            dict_atom = dict_normalized[i:i+1]  # (1, C)

            # Feature similarity
            feat_sim = torch.mm(features_norm, dict_atom.t()).squeeze()  # (N,)

            # Skeleton-guided similarity
            skeleton_sim = torch.mm(skeleton_norm, dict_atom.t()).squeeze()  # (N,)

            # Combined similarity with skeleton guidance
            combined_sim = feat_sim + 0.5 * skeleton_sim

            coeffs[:, i] = combined_sim

        # Apply softmax for sparse representation
        coeffs = F.softmax(coeffs, dim=1)

        # Apply sparsity constraint (encourage few active subspaces)
        coeffs = self.apply_sparsity(coeffs)

        return coeffs

    def apply_sparsity(self, coeffs):
        """Apply sparsity constraint on representation coefficients"""
        # L1 regularization to encourage sparsity
        l1_penalty = torch.sign(coeffs) * self.sparsity_weight

        # Soft thresholding
        coeffs = torch.sign(coeffs) * torch.clamp(torch.abs(coeffs) - l1_penalty, min=0)

        # Re-normalize
        coeffs = F.normalize(coeffs, p=1, dim=1)

        return coeffs

    def reconstruct_features(self, coeffs, features):
        """Reconstruct features from sparse representation"""
        B, T, H, W, C = features.shape
        N = B * T * H * W

        # Reconstruct using sparse coefficients
        reconstructed = torch.mm(coeffs, self.dictionary)  # (N, C)
        reconstructed = reconstructed.view(B, T, H, W, C)

        # Apply reconstruction layer
        reconstructed = self.reconstruction_layer(reconstructed)

        return reconstructed

    def compute_alignment_loss(self, features, reconstructed, skeleton_features):
        """Compute alignment loss between RGB features and skeleton guidance"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, features)

        # Skeleton alignment loss
        B, T, H, W, C = features.shape
        skeleton_features_expanded = skeleton_features.unsqueeze(2).unsqueeze(2)
        skeleton_features_expanded = skeleton_features_expanded.expand(-1, -1, H, W, -1)

        # Compute attention between RGB and skeleton features
        rgb_flat = features.view(B * T * H * W, C).unsqueeze(1)  # (N, 1, C)
        skeleton_flat = skeleton_features_expanded.view(B * T * H * W, C).unsqueeze(1)  # (N, 1, C)

        aligned_features, attention_weights = self.skeleton_attention(
            rgb_flat, skeleton_flat, skeleton_flat
        )

        alignment_loss = F.mse_loss(aligned_features.squeeze(1), rgb_flat.squeeze(1))

        # Total loss
        total_loss = recon_loss + 0.1 * alignment_loss

        return total_loss, {
            'reconstruction_loss': recon_loss.item(),
            'alignment_loss': alignment_loss.item()
        }

    def forward(self, rgb_features, skeleton_data):
        """
        Args:
            rgb_features: RGB feature maps of shape (B, T, C, H, W)
            skeleton_data: Skeleton data of shape (B, T, J, 3)
        Returns:
            aligned_features: Aligned feature maps
            loss_dict: Dictionary of losses
        """
        B, T, C, H, W = rgb_features.shape

        # Permute features for processing
        rgb_features = rgb_features.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        # Compute skeleton features for guidance
        skeleton_features = self.compute_skeleton_features(skeleton_data)  # (B, T, C)

        # Compute sparse representation
        coeffs = self.sparse_representation(rgb_features, skeleton_features)

        # Reconstruct features using sparse representation
        reconstructed = self.reconstruct_features(coeffs, rgb_features)

        # Compute alignment loss
        loss, loss_dict = self.compute_alignment_loss(rgb_features, reconstructed, skeleton_features)

        # Return aligned features (reconstructed features)
        aligned_features = reconstructed.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        return aligned_features, loss, loss_dict


class S3AConfig:
    """Configuration for S3A module"""
    def __init__(self):
        self.feature_dim = 512
        self.num_subspaces = 10
        self.sparsity_weight = 0.1
        self.skeleton_joints = 25
        self.skeleton_dims = 3