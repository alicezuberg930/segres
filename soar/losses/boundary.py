from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


class BoundaryBCELoss(BaseLoss):
    """Boundary-aware loss using Sobel edge detection."""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_f32 = x.float().view(B * C, 1, H, W)
        sx = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=x.device, dtype=torch.float32)
        gx = F.conv2d(x_f32, sx, padding=1)
        gy = F.conv2d(x_f32, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8).view(B, C, H, W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()
        with torch.no_grad():
            edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)

        loss = F.l1_loss(edge_pred, edge_target, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class BoundaryDiceLoss(BaseLoss):
    """Boundary Dice loss on spatial gradient maps."""

    def __init__(self, weight: float = 1.0, smooth: float = 1.0):
        super().__init__(weight)
        self.smooth = smooth
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_f32 = x.float().view(B * C, 1, H, W)
        sx = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=x.device, dtype=torch.float32)
        gx = F.conv2d(x_f32, sx, padding=1)
        gy = F.conv2d(x_f32, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8).view(B, C, H, W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()
        with torch.no_grad():
            edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)

        if valid_mask is not None:
            vmask = valid_mask.float().expand_as(edge_pred) if valid_mask.shape != edge_pred.shape else valid_mask.float()
            edge_pred = edge_pred * vmask
            edge_target = edge_target * vmask

        dims = (-2, -1)
        inter = torch.sum(edge_pred * edge_target, dim=dims)
        cardinality = torch.sum(edge_pred, dim=dims) + torch.sum(edge_target, dim=dims)
        dice = (2.0 * inter + self.smooth) / (cardinality + self.smooth).clamp_min(1e-7)
        return self.weight * (1.0 - dice.mean())


class BoundaryDistLoss(BaseLoss):
    """
    Differentiable Signed Distance Transform Boundary Loss (Kervadec et al.).
    Computes distance transforms on target ground truth under no_grad and integrates with
    predicted probability fields to preserve strict differentiability.
    Optimized with GPU-native morphological pooling to eliminate CPU host synchronizations.
    """

    def __init__(self, weight: float = 1.0, max_iter: int = 15):
        super().__init__(weight)
        self.max_iter = max_iter

    @staticmethod
    def _compute_sdf_gpu(target: torch.Tensor, max_iter: int = 15) -> torch.Tensor:
        """GPU-accelerated signed distance field approximation via morphological pooling cascades."""
        pos = (target > 0.5).float()
        neg = 1.0 - pos
        d_out = torch.zeros_like(target)
        d_in = torch.zeros_like(target)
        curr_pos = pos
        curr_neg = neg
        for _ in range(max_iter):
            curr_pos = F.max_pool2d(curr_pos, kernel_size=3, stride=1, padding=1)
            curr_neg = F.max_pool2d(curr_neg, kernel_size=3, stride=1, padding=1)
            d_out = d_out + neg * (curr_pos > 0.5).float()
            d_in = d_in + pos * (curr_neg > 0.5).float()
        return d_out - d_in

    @staticmethod
    def _compute_sdf_cpu(target_np: np.ndarray) -> np.ndarray:
        """CPU fallback signed distance field where foreground boundary is zero."""
        pos = target_np > 0.5
        neg = ~pos
        if not np.any(pos):
            return np.ones_like(target_np, dtype=np.float32)
        if not np.any(neg):
            return -np.ones_like(target_np, dtype=np.float32)
        d_out = distance_transform_edt(neg)
        d_in = distance_transform_edt(pos)
        return (d_out - d_in).astype(np.float32)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()

        with torch.no_grad():
            if target.is_cuda:
                # Fast GPU-native path: zero CPU sync, <1ms
                sdf_tensor = self._compute_sdf_gpu(target.float(), max_iter=self.max_iter)
            else:
                target_cpu = target.detach().cpu().numpy()
                batch_size, c_dim = target.shape[:2]
                sdfs = [
                    np.stack([self._compute_sdf_cpu(target_cpu[b, c]) for c in range(c_dim)], axis=0)
                    for b in range(batch_size)
                ]
                sdf_tensor = torch.from_numpy(np.stack(sdfs, axis=0)).to(device=pred.device, dtype=torch.float32)

        boundary_penalty = prob * sdf_tensor

        if valid_mask is not None:
            vmask = valid_mask.float().expand_as(boundary_penalty) if valid_mask.shape != boundary_penalty.shape else valid_mask.float()
            boundary_penalty = boundary_penalty * vmask
            return self.weight * (boundary_penalty.sum() / vmask.sum().clamp_min(1.0))

        return self.weight * boundary_penalty.mean()
