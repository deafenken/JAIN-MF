import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.animation import FuncAnimation
import cv2
from typing import Dict, List, Tuple, Optional
import open3d as o3d


def plot_attention_weights(attention_weights: Dict, save_path: str = None):
    """
    Visualize attention weights from different modalities

    Args:
        attention_weights: Dictionary of attention weights
        save_path: Path to save the visualization
    """
    num_modalities = len(attention_weights)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    idx = 0
    for modality, attn in attention_weights.items():
        if idx >= 6:  # Limit to 6 plots
            break

        # Average across heads and take first sample
        if len(attn.shape) == 5:  # (B, T, heads, N, N)
            attn_avg = attn[0].mean(dim=0).cpu().numpy()  # (N, N)
        elif len(attn.shape) == 4:  # (B, heads, N, N)
            attn_avg = attn[0].mean(dim=0).cpu().numpy()  # (N, N)
        else:
            attn_avg = attn.cpu().numpy()

        # Plot heatmap
        sns.heatmap(attn_avg, ax=axes[idx], cmap='viridis', cbar=True)
        axes[idx].set_title(f'{modality.replace("_", " ").title()} Attention')
        axes[idx].set_xlabel('Key')
        axes[idx].set_ylabel('Query')

        idx += 1

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_skeleton_sequence(skeleton_data: np.ndarray, save_path: str = None):
    """
    Visualize skeleton sequence

    Args:
        skeleton_data: Skeleton data of shape (T, 25, 3)
        save_path: Path to save the visualization
    """
    # NTU skeleton connections
    skeleton_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right hand
        (1, 5), (5, 6), (6, 7),  # Left arm
        (1, 8), (8, 9), (9, 10),  # Right leg
        (1, 11), (11, 12), (12, 13),  # Left leg
        (0, 14), (14, 15),  # Spine
        (15, 16), (15, 17),  # Shoulders
        (3, 17), (7, 16)  # Hands
    ]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select every nth frame for visualization
    frame_step = max(1, len(skeleton_data) // 10)
    selected_frames = skeleton_data[::frame_step]

    # Color map for different frames
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_frames)))

    for frame_idx, frame in enumerate(selected_frames):
        # Plot joints
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2],
                  c=[colors[frame_idx]], s=50, alpha=0.7)

        # Plot connections
        for connection in skeleton_connections:
            if connection[0] < len(frame) and connection[1] < len(frame):
                joint1 = frame[connection[0]]
                joint2 = frame[connection[1]]
                ax.plot([joint1[0], joint2[0]],
                       [joint1[1], joint2[1]],
                       [joint1[2], joint2[2]],
                       c=colors[frame_idx], alpha=0.5, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Skeleton Sequence Visualization')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_optical_flow(flow_data: np.ndarray, frame_idx: int = 0, save_path: str = None):
    """
    Visualize optical flow

    Args:
        flow_data: Flow data of shape (T, 2, H, W)
        frame_idx: Frame index to visualize
        save_path: Path to save the visualization
    """
    if frame_idx >= len(flow_data):
        frame_idx = 0

    flow = flow_data[frame_idx]  # (2, H, W)
    flow_u = flow[0]
    flow_v = flow[1]

    # Convert to HSV for visualization
    H, W = flow_u.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    # Hue corresponds to direction
    magnitude = np.sqrt(flow_u**2 + flow_v**2)
    angle = np.arctan2(flow_v, flow_u) * 180 / np.pi

    hsv[:, :, 0] = (angle + 180) / 2  # Normalize to 0-180
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.minimum(magnitude * 10, 255)  # Scale magnitude

    # Convert to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add quiver plot for better visualization
    step = 20  # Sample every 20th pixel
    Y, X = np.mgrid[0:H:step, 0:W:step]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Show HSV visualization
    ax1.imshow(flow_vis)
    ax1.set_title('Optical Flow (HSV Visualization)')
    ax1.axis('off')

    # Show quiver plot
    ax2.imshow(np.zeros((H, W)), cmap='gray')
    ax2.quiver(X, Y, flow_u[::step, ::step], flow_v[::step, ::step],
              color='red', alpha=0.7, scale=50)
    ax2.set_title('Optical Flow (Quiver Plot)')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_attention_video(rgb_frames: torch.Tensor, attention_weights: Dict,
                          save_path: str = None):
    """
    Create video showing attention maps overlaid on RGB frames

    Args:
        rgb_frames: RGB frames of shape (T, 3, H, W)
        attention_weights: Attention weights
        save_path: Path to save the video
    """
    # Convert to numpy
    frames_np = rgb_frames.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = (frames_np * 255).astype(np.uint8)

    T, H, W, C = frames_np.shape

    # Get attention maps (use first modality as example)
    if 'corrnet' in attention_weights:
        attn = attention_weights['corrnet'][0]  # (T, heads, N, N)
        attn = attn.mean(dim=1)  # (T, N, N)
        attn_np = attn.cpu().numpy()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (W * 2, H))

    for t in range(T):
        # Get current frame
        frame = frames_np[t].copy()

        # Create attention visualization
        if t < len(attn_np):
            # Resize attention map to frame size
            attn_map = attn_np[t].mean(axis=0)  # Average over queries
            attn_resized = cv2.resize(attn_map, (W, H))
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            attn_colored = plt.cm.jet(attn_resized)[:, :, :3] * 255

            # Convert to uint8
            attn_colored = attn_colored.astype(np.uint8)
            attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_RGB2BGR)

            # Blend with original frame
            blended = cv2.addWeighted(frame, 0.7, attn_colored, 0.3, 0)
        else:
            blended = frame

        # Combine original and attention
        combined = np.hstack([frame, blended])

        # Add frame number
        cv2.putText(combined, f'Frame {t}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write frame
        out.write(combined)

    out.release()


def plot_feature_distribution(features: torch.Tensor, labels: torch.Tensor,
                             class_names: List[str] = None, save_path: str = None):
    """
    Plot distribution of learned features

    Args:
        features: Feature embeddings of shape (N, feature_dim)
        labels: Class labels
        class_names: Names of classes
        save_path: Path to save the plot
    """
    # Use PCA for visualization if feature dimension > 2
    if features.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features.cpu().numpy())
    else:
        features_2d = features.cpu().numpy()

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    unique_labels = torch.unique(labels)

    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=class_names[i], alpha=0.7)

    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    plt.title('Feature Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_s3a_subspaces(s3a_module, skeleton_data: torch.Tensor, rgb_features: torch.Tensor,
                           save_path: str = None):
    """
    Visualize S3A subspace assignment

    Args:
        s3a_module: S3A module instance
        skeleton_data: Skeleton data
        rgb_features: RGB features
        save_path: Path to save the visualization
    """
    # Get subspace assignments
    B, T, C, H, W = rgb_features.shape
    skeleton_features = s3a_module.compute_skeleton_features(skeleton_data)
    coeffs = s3a_module.sparse_representation(
        rgb_features.permute(0, 1, 3, 4, 2),  # (B, T, H, W, C)
        skeleton_features
    )

    # Reshape coefficients for visualization
    coeffs_np = coeffs.cpu().numpy()  # (N, num_subspaces)
    N, num_subspaces = coeffs_np.shape

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot subspace distribution
    subspace_counts = np.sum(coeffs_np > 0.1, axis=0)  # Count active assignments
    ax1.bar(range(num_subspaces), subspace_counts)
    ax1.set_xlabel('Subspace Index')
    ax1.set_ylabel('Number of Pixels')
    ax1.set_title('Pixel Assignment to Subspaces')

    # Plot coefficient heatmap
    sample_size = min(1000, N)  # Sample pixels for visualization
    indices = np.random.choice(N, sample_size, replace=False)
    sample_coeffs = coeffs_np[indices]

    im = ax2.imshow(sample_coeffs.T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Pixel Index')
    ax2.set_ylabel('Subspace Index')
    ax2.set_title('Sparse Coefficients Heatmap')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(metrics: Dict, save_path: str = None):
    """
    Plot training curves for loss and accuracy

    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    if 'train_loss' in metrics and 'val_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # Accuracy curves
    if 'train_acc' in metrics and 'val_acc' in metrics:
        axes[0, 1].plot(metrics['train_acc'], label='Train Acc')
        axes[0, 1].plot(metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # S3A losses
    if 's3a_reconstruction_loss' in metrics:
        axes[1, 0].plot(metrics['s3a_reconstruction_loss'], label='S3A Recon Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('S3A Reconstruction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Learning rate
    if 'lr' in metrics:
        axes[1, 1].plot(metrics['lr'], label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def log_attention_maps(attention_weights: Dict, writer, step: int):
    """Log attention maps to TensorBoard"""
    for modality, attn in attention_weights.items():
        # Average across batch and heads
        if len(attn.shape) >= 4:
            attn_avg = attn.mean(dim=[0, 1]) if len(attn.shape) == 4 else attn[0].mean(dim=0)
        else:
            attn_avg = attn

        # Convert to numpy and normalize
        attn_np = attn_avg.cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

        writer.add_image(f'attention/{modality}', attn_np, step, dataformats='HW')