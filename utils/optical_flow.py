import cv2
import numpy as np
import torch
from torch.nn import functional as F
from typing import List, Tuple, Optional

class TVL1OpticalFlow:
    """TV-L1 Optical Flow Computation"""

    def __init__(self, scale_factor=0.5, iterations=100):
        """
        Args:
            scale_factor: Factor to scale down images for faster computation
            iterations: Number of iterations for TV-L1 algorithm
        """
        self.scale_factor = scale_factor
        self.iterations = iterations

    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two consecutive frames

        Args:
            prev_frame: Previous frame (H, W, 3) in BGR format
            curr_frame: Current frame (H, W, 3) in BGR format

        Returns:
            flow: Optical flow field (H, W, 2) with (dx, dy) components
        """
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Scale down if specified
        if self.scale_factor != 1.0:
            h, w = prev_gray.shape
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            prev_gray = cv2.resize(prev_gray, (new_w, new_h))
            curr_gray = cv2.resize(curr_gray, (new_w, new_h))

        # Create TV-L1 optical flow instance
        try:
            # Use OpenCV's TV-L1 implementation if available
            flow = cv2.optflow.DualTVL1OpticalFlow_create(
                nscales=5,
                warps=5,
                Epsilon=0.01,
                innerIterations=self.iterations,
                outerIterations=10,
                scaleStep=0.5,
                gamma=0.002,
                medianFiltering=5,
                useInitialFlow=False
            )

            # Compute flow
            flow_computed = flow.calc(prev_gray, curr_gray, None)

        except AttributeError:
            # Fallback to Farneback if TV-L1 is not available
            flow_computed = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

        # Scale back up if we scaled down
        if self.scale_factor != 1.0:
            h, w = prev_gray.shape
            flow_computed = cv2.resize(
                flow_computed,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            ) / self.scale_factor

        return flow_computed

    def compute_flow_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute optical flow for a sequence of frames

        Args:
            frames: List of frames (T, H, W, 3) in BGR format

        Returns:
            flows: Optical flow sequence (T-1, H, W, 2)
        """
        if len(frames) < 2:
            return np.array([])

        flows = []
        for i in range(len(frames) - 1):
            flow = self.compute_flow(frames[i], frames[i + 1])
            flows.append(flow)

        return np.stack(flows, axis=0)

    def visualize_flow(self, flow: np.ndarray, hsv_max_norm: bool = True) -> np.ndarray:
        """
        Visualize optical flow using HSV color representation

        Args:
            flow: Optical flow field (H, W, 2)
            hsv_max_norm: Normalize by maximum value in HSV space

        Returns:
            flow_vis: Flow visualization (H, W, 3) in BGR format
        """
        h, w = flow.shape[:2]

        # Convert to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue corresponds to direction
        hsv[..., 0] = ang * 180 / np.pi / 2

        # Normalize magnitude
        if hsv_max_norm:
            mag_norm = mag / (mag.max() + 1e-8)
        else:
            mag_norm = mag / 255.0
        mag_norm = np.clip(mag_norm, 0, 1)

        # Saturation and value
        hsv[..., 1] = mag_norm * 255
        hsv[..., 2] = 255

        # Convert to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return flow_vis


class FlowProcessor:
    """Processor for optical flow in PyTorch tensors"""

    def __init__(self, device='cpu'):
        self.device = device
        self.tv_l1 = TVL1OpticalFlow()

    def process_frames_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of video sequences to compute optical flow

        Args:
            frames: Video frames (B, T, 3, H, W) in RGB format

        Returns:
            flows: Optical flow (B, T-1, 2, H, W)
        """
        B, T, C, H, W = frames.shape

        if T < 2:
            return torch.empty(B, 0, 2, H, W, device=self.device)

        flows = []

        for b in range(B):
            batch_flows = []

            # Convert frames to numpy and RGB->BGR
            frames_np = frames[b].permute(0, 2, 3, 1).cpu().numpy()
            frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_np]

            # Compute optical flow
            flow_seq = self.tv_l1.compute_flow_sequence(frames_bgr)

            # Convert to tensor and normalize
            for flow in flow_seq:
                # Normalize flow to [-1, 1]
                flow_norm = flow / (np.abs(flow).max() + 1e-8)
                flow_tensor = torch.from_numpy(flow_norm).float()
                flow_tensor = flow_tensor.permute(2, 0, 1)  # (2, H, W)
                batch_flows.append(flow_tensor)

            # Stack flows for this batch
            batch_flows = torch.stack(batch_flows, dim=0)  # (T-1, 2, H, W)
            flows.append(batch_flows)

        # Stack across batches
        flows = torch.stack(flows, dim=0)  # (B, T-1, 2, H, W)

        return flows.to(self.device)

    def compute_flow_statistics(self, flows: torch.Tensor) -> dict:
        """
        Compute statistics of optical flow

        Args:
            flows: Optical flow (B, T, 2, H, W)

        Returns:
            stats: Dictionary of flow statistics
        """
        B, T, C, H, W = flows.shape

        # Magnitude of flow
        flow_mag = torch.norm(flows, dim=2)  # (B, T, H, W)

        stats = {
            'mean_magnitude': flow_mag.mean().item(),
            'max_magnitude': flow_mag.max().item(),
            'std_magnitude': flow_mag.std().item(),
            'mean_x': flows[:, :, 0].mean().item(),
            'mean_y': flows[:, :, 1].mean().item(),
            'std_x': flows[:, :, 0].std().item(),
            'std_y': flows[:, :, 1].std().item()
        }

        return stats


def augment_flow(flow: torch.Tensor, horizontal_flip: bool = False,
                vertical_flip: bool = False, rotation: int = 0) -> torch.Tensor:
    """
    Apply data augmentation to optical flow

    Args:
        flow: Optical flow (B, T, 2, H, W)
        horizontal_flip: Apply horizontal flip
        vertical_flip: Apply vertical flip
        rotation: Rotation angle (must be 90, 180, or 270)

    Returns:
        augmented_flow: Augmented optical flow
    """
    augmented_flow = flow.clone()

    if horizontal_flip:
        # Flip and reverse x component
        augmented_flow = torch.flip(augmented_flow, dims=[-1])
        augmented_flow[:, :, 0] = -augmented_flow[:, :, 0]

    if vertical_flip:
        # Flip and reverse y component
        augmented_flow = torch.flip(augmented_flow, dims=[-2])
        augmented_flow[:, :, 1] = -augmented_flow[:, :, 1]

    if rotation > 0:
        # Rotate flow vectors accordingly
        k = rotation // 90
        augmented_flow = torch.rot90(augmented_flow, k=k, dims=[-2, -1])

        # Rotate flow components
        flow_x = augmented_flow[:, :, 0].clone()
        flow_y = augmented_flow[:, :, 1].clone()

        if k == 1:  # 90 degrees
            augmented_flow[:, :, 0] = -flow_y
            augmented_flow[:, :, 1] = flow_x
        elif k == 2:  # 180 degrees
            augmented_flow[:, :, 0] = -flow_x
            augmented_flow[:, :, 1] = -flow_y
        elif k == 3:  # 270 degrees
            augmented_flow[:, :, 0] = flow_y
            augmented_flow[:, :, 1] = -flow_x

    return augmented_flow