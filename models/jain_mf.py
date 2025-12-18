import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .backbone import ResNetBackbone, FlowBackbone, SkeletonBackbone
from .s3a_module import SkeletonGuidedS3A
from .corrnet import CorrNetCrossModalAttention


class JAINMFModel(nn.Module):
    """JAIN-MF: Tri-Modal Action Recognition via Skeleton-Guided Sparse Subspace Alignment"""

    def __init__(self,
                 num_classes: int = 120,
                 feature_dim: int = 512,
                 num_frames: int = 32,
                 spatial_size: tuple = (224, 224),
                 use_s3a: bool = True,
                 dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of action classes
            feature_dim: Feature dimension for all modalities
            num_frames: Number of frames in input sequence
            spatial_size: Spatial size of input frames
            use_s3a: Whether to use Skeleton-guided S3A module
            dropout_rate: Dropout rate for classifier
        """
        super(JAINMFModel, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.use_s3a = use_s3a

        # Backbones for different modalities
        self.rgb_backbone = ResNetBackbone(pretrained=True)
        self.flow_backbone = FlowBackbone(input_channels=2)
        self.skeleton_backbone = SkeletonBackbone(joint_dim=3, num_joints=25)

        # Skeleton-guided S3A module
        if use_s3a:
            self.s3a = SkeletonGuidedS3A(
                feature_dim=feature_dim,
                num_subspaces=10,
                sparsity_weight=0.1
            )

        # CorrNet-based cross-modal attention
        self.corrnet = CorrNetCrossModalAttention(
            feature_dim=feature_dim,
            num_heads=8,
            num_layers=2
        )

        # Temporal modeling
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )

        # Multi-head attention for temporal fusion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )

        # Modality-specific classifiers for incomplete modality scenarios
        self.rgb_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )

        self.skeleton_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self,
                rgb_frames: torch.Tensor,
                flow_frames: torch.Tensor,
                skeleton_data: torch.Tensor,
                return_attention: bool = False):
        """
        Args:
            rgb_frames: RGB frames of shape (B, T, 3, H, W)
            flow_frames: Optical flow of shape (B, T-1, 2, H, W)
            skeleton_data: Skeleton data of shape (B, T, 25, 3)
            return_attention: Whether to return attention weights

        Returns:
            logits: Classification logits
            attention_weights: Dictionary of attention weights (if return_attention=True)
        """
        batch_size = rgb_frames.shape[0]

        # Check for missing modalities
        has_rgb = not torch.allclose(rgb_frames, torch.zeros_like(rgb_frames))
        has_flow = not torch.allclose(flow_frames, torch.zeros_like(flow_frames))
        has_skeleton = not torch.allclose(skeleton_data, torch.zeros_like(skeleton_data))

        # Extract features from each modality
        rgb_features = None
        flow_features = None
        skeleton_features = None

        if has_rgb:
            rgb_features = self.rgb_backbone(rgb_frames)  # (B, T, C, H, W)

            # Apply S3A if skeleton data is available
            if has_skeleton and self.use_s3a:
                rgb_features, s3a_loss, _ = self.s3a(rgb_features, skeleton_data)

            # Global average pooling
            rgb_features = rgb_features.mean(dim=[3, 4])  # (B, T, C)

        if has_flow:
            flow_features = self.flow_backbone(flow_frames)  # (B, T-1, C, H, W)
            flow_features = flow_features.mean(dim=[3, 4])  # (B, T-1, C)

        if has_skeleton:
            skeleton_features = self.skeleton_backbone(skeleton_data)  # (B, T, C)

        # Handle different scenarios based on available modalities
        attention_weights = {}

        if has_rgb and has_skeleton and has_flow:
            # Full tri-modal scenario with CorrNet
            # Apply CorrNet cross-modal attention on extracted features
            # Note: Currently simplified since CorrNet expects different input format
            # For now, use simple concatenation fusion
            min_t = min(rgb_features.shape[1], skeleton_features.shape[1], flow_features.shape[1])
            rgb_feat = rgb_features[:, :min_t]
            flow_feat = flow_features[:, :min_t]
            skel_feat = skeleton_features[:, :min_t]

            # Temporal fusion of features
            fused_features = torch.cat([rgb_feat, flow_feat, skel_feat], dim=-1)
            fused_features = fused_features.mean(dim=1)  # (B, C)

            # Additional fusion layer
            # Create fusion layer
            fusion_layer = nn.Linear(fused_features.shape[-1], self.feature_dim).to(fused_features.device)
            fused_features = fusion_layer(fused_features)
            fused_features = F.relu(fused_features)

        elif has_rgb and has_skeleton:
            # RGB + Skeleton only
            # Temporal alignment and fusion
            min_t = min(rgb_features.shape[1], skeleton_features.shape[1])
            rgb_feat = rgb_features[:, :min_t]
            skel_feat = skeleton_features[:, :min_t]

            # Concatenate and fuse
            concatenated = torch.cat([rgb_feat, skel_feat], dim=-1)
            fused_features = F.linear(concatenated, torch.randn(self.feature_dim, self.feature_dim * 2).to(rgb_feat.device))
            fused_features = F.relu(fused_features)

        elif has_rgb:
            # RGB only
            fused_features = rgb_features.mean(dim=1)  # (B, C)
            logits = self.rgb_classifier(fused_features)
            return logits

        elif has_skeleton:
            # Skeleton only
            fused_features = skeleton_features.mean(dim=1)  # (B, C)
            logits = self.skeleton_classifier(fused_features)
            return logits

        else:
            raise ValueError("At least one modality must be present")

        # Temporal encoding for fused features
        if len(fused_features.shape) == 3:  # (B, T, C)
            # Apply temporal transformer
            temporal_encoded = self.temporal_encoder(fused_features)

            # Global attention pooling
            pooled_features, temporal_attn = self.temporal_attention(
                temporal_encoded, temporal_encoded, temporal_encoded
            )
            attention_weights['temporal'] = temporal_attn

            # Average pooling
            final_features = pooled_features.mean(dim=1)  # (B, C)
        else:
            final_features = fused_features  # (B, C)

        # Classification
        logits = self.classifier(final_features)

        if return_attention:
            return logits, attention_weights
        else:
            return logits

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor,
                     rgb_frames: torch.Tensor,
                     flow_frames: torch.Tensor,
                     skeleton_data: torch.Tensor,
                     lambda_ce: float = 1.0,
                     lambda_s3a: float = 0.1):
        """
        Compute total loss including classification and alignment losses

        Args:
            logits: Classification logits
            labels: Ground truth labels
            rgb_frames: RGB frames
            flow_frames: Optical flow frames
            skeleton_data: Skeleton data
            lambda_ce: Weight for cross-entropy loss
            lambda_s3a: Weight for S3A alignment loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)

        total_loss = lambda_ce * ce_loss
        loss_dict = {'ce_loss': ce_loss.item()}

        # S3A alignment loss if applicable
        if self.use_s3a:
            has_rgb = not torch.allclose(rgb_frames, torch.zeros_like(rgb_frames))
            has_skeleton = not torch.allclose(skeleton_data, torch.zeros_like(skeleton_data))

            if has_rgb and has_skeleton:
                # Forward pass through S3A
                rgb_features = self.rgb_backbone(rgb_frames)
                _, s3a_loss, s3a_loss_dict = self.s3a(rgb_features, skeleton_data)

                total_loss += lambda_s3a * s3a_loss
                loss_dict.update({f's3a_{k}': v for k, v in s3a_loss_dict.items()})

        return total_loss, loss_dict


class JAINMFConfig:
    """Configuration for JAIN-MF model"""

    def __init__(self):
        # Model parameters
        self.num_classes = 120
        self.feature_dim = 512
        self.num_frames = 32
        self.spatial_size = (224, 224)
        self.use_s3a = True
        self.dropout_rate = 0.5

        # Loss weights
        self.lambda_ce = 1.0
        self.lambda_s3a = 0.1

        # Training parameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.num_epochs = 100

        # Data parameters
        self.missing_modalities_prob = 0.0
        self.modality_drop_rate = 0.1