import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class CorrNetBlock(nn.Module):
    """CorrNet block for cross-modal correlation learning"""

    def __init__(self, feature_dim=512, num_heads=8):
        super(CorrNetBlock, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        # Query, Key, Value projections for cross-modal attention
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.w_v = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.w_o = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def scaled_dot_product_attention(self, Q, K, V):
        """Compute scaled dot-product attention"""
        # Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        d_k = Q.size(-1)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query_modal, key_value_modal):
        """
        Args:
            query_modal: Query modality features of shape (B, T, N, C)
            key_value_modal: Key-Value modality features of shape (B, T, N, C)
        Returns:
            output: Attended features
            attention_weights: Attention weights
        """
        B, T, N, C = query_modal.shape

        # Reshape for attention computation
        query = query_modal.view(B * T, N, C)
        key_value = key_value_modal.view(B * T, N, C)

        # Apply layer normalization
        query_norm = self.norm1(query)
        key_value_norm = self.norm1(key_value)

        # Project to Q, K, V
        Q = self.w_q(query_norm)
        K = self.w_k(key_value_norm)
        V = self.w_v(key_value_norm)

        # Reshape for multi-head attention
        Q = Q.view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * T, N, C)
        attn_output = self.w_o(attn_output)

        # Add residual connection
        output = self.norm2(query + self.dropout(attn_output))

        # Apply feed-forward network
        ffn_output = self.ffn(output)
        output = output + self.dropout(ffn_output)

        # Reshape to original dimensions
        output = output.view(B, T, N, C)
        attention_weights = attention_weights.view(B, T, self.num_heads, N, N)

        return output, attention_weights


class CorrNetCrossModalAttention(nn.Module):
    """CorrNet-based Cross-modal Attention Module"""

    def __init__(self, feature_dim=512, num_heads=8, num_layers=2):
        super(CorrNetCrossModalAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # CorrNet blocks for different modality pairs
        self.rgb_flow_blocks = nn.ModuleList([
            CorrNetBlock(feature_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.rgb_skel_blocks = nn.ModuleList([
            CorrNetBlock(feature_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.flow_skel_blocks = nn.ModuleList([
            CorrNetBlock(feature_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Feature adaptation layers
        self.rgb_adapter = nn.Sequential(
            nn.Conv3d(3, feature_dim, kernel_size=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.flow_adapter = nn.Sequential(
            nn.Conv3d(2, feature_dim, kernel_size=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.skel_adapter = nn.Linear(3 * 25, feature_dim)  # 25 joints with 3 coordinates

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def adapt_features(self, rgb_features, flow_features, skeleton_data):
        """Adapt features to common dimension"""
        B = rgb_features.shape[0]

        # RGB features: (B, T, C, H, W) -> (B, T, H*W, C)
        _, T, C, H, W = rgb_features.shape
        rgb_flat = rgb_features.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, H*W, C)

        # Flow features: (B, T, 2, H, W) -> adapt -> (B, T, H*W, C)
        flow_adapted = self.flow_adapter(flow_features.permute(0, 2, 1, 3, 4))
        _, _, C_flow, H_flow, W_flow = flow_adapted.shape
        flow_flat = flow_adapted.view(B, T, C_flow, -1).permute(0, 1, 3, 2)  # (B, T, H*W, C)

        # Skeleton features: (B, T, J, 3) -> (B, T, C)
        B_skel, T_skel, J, C_skel = skeleton_data.shape
        skeleton_flat = skeleton_data.view(B_skel * T_skel, -1)
        skeleton_features = self.skel_adapter(skeleton_flat)
        skeleton_features = skeleton_features.view(B_skel, T_skel, -1)

        # Expand skeleton features to match spatial dimensions
        skeleton_expanded = repeat(skeleton_features, 'b t c -> b t n c', n=H*W)

        return rgb_flat, flow_flat, skeleton_expanded

    def forward(self, rgb_features, flow_features, skeleton_data):
        """
        Args:
            rgb_features: RGB features of shape (B, T, 3, H, W)
            flow_features: Flow features of shape (B, T, 2, H, W)
            skeleton_data: Skeleton data of shape (B, T, J, 3)
        Returns:
            fused_features: Fused multi-modal features
            attention_weights: Dictionary of attention weights
        """
        # Adapt features to common dimension
        rgb_feat, flow_feat, skel_feat = self.adapt_features(rgb_features, flow_features, skeleton_data)

        # Cross-modal attention
        rgb_from_flow, attn_rf = self.rgb_flow_blocks[0](rgb_feat, flow_feat)
        rgb_from_skel, attn_rs = self.rgb_skel_blocks[0](rgb_feat, skel_feat)

        flow_from_rgb, attn_fr = self.rgb_flow_blocks[1](flow_feat, rgb_feat)
        flow_from_skel, attn_fs = self.flow_skel_blocks[0](flow_feat, skel_feat)

        skel_from_rgb, attn_sr = self.rgb_skel_blocks[1](skel_feat, rgb_feat)
        skel_from_flow, attn_sf = self.flow_skel_blocks[1](skel_feat, flow_feat)

        # Enhanced features
        rgb_enhanced = rgb_feat + rgb_from_flow + rgb_from_skel
        flow_enhanced = flow_feat + flow_from_rgb + flow_from_skel
        skel_enhanced = skel_feat + skel_from_rgb + skel_from_flow

        # Global average pooling over spatial dimension
        rgb_pooled = rgb_enhanced.mean(dim=2)  # (B, T, C)
        flow_pooled = flow_enhanced.mean(dim=2)  # (B, T, C)
        skel_pooled = skel_enhanced.mean(dim=2)  # (B, T, C)

        # Concatenate and fuse
        concatenated = torch.cat([rgb_pooled, flow_pooled, skel_pooled], dim=-1)  # (B, T, 3*C)
        fused = self.fusion_layer(concatenated)  # (B, T, C)

        # Store attention weights for visualization
        attention_weights = {
            'rgb_flow': attn_rf,
            'rgb_skel': attn_rs,
            'flow_rgb': attn_fr,
            'flow_skel': attn_fs,
            'skel_rgb': attn_sr,
            'skel_flow': attn_sf
        }

        return fused, attention_weights


class CorrNetConfig:
    """Configuration for CorrNet module"""
    def __init__(self):
        self.feature_dim = 512
        self.num_heads = 8
        self.num_layers = 2
        self.dropout = 0.1