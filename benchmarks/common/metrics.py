from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple


def extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour band of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return F.relu(mask - eroded)


def soft_skeletonize(x: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
    """
    Differentiable continuous morphological skeletonization via iterative min-max pooling.
    Extracts medial axis topology preserving thin curvilinear continuity.
    """
    skel = torch.zeros_like(x)
    for _ in range(n_iter):
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        delta = F.relu(x - opened)
        skel = skel + delta * (1.0 - skel)
        x = eroded
    return skel.clamp(0.0, 1.0)


class BenchmarkMetricAccumulator:
    """
    High-performance GPU metric accumulator for semantic segmentation benchmarks.
    Computes:
      - mIoU (Mean Intersection-over-Union)
      - Dice Score (F1 Overlap)
      - Boundary IoU (bIoU, contour sharpness with dilation band d=2)
      - Centerline Dice (clDice, topological continuity preservation)
    """

    def __init__(self, num_classes: int, class_names: Optional[Dict[int, str]] = None, device: torch.device = torch.device("cpu")):
        self.num_classes = max(1, num_classes)
        self.class_names = class_names or {i: f"Class_{i}" for i in range(self.num_classes)}
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset internal accumulator terms."""
        # 10 terms per class:
        # [0: inter, 1: union, 2: pred_card, 3: gt_card,
        #  4: b_inter, 5: b_union, 6: skel_prec_inter, 7: skel_prec_total,
        #  8: skel_sens_inter, 9: skel_sens_total]
        self.accum = torch.zeros((10, self.num_classes), dtype=torch.float64, device=self.device)
        self.sample_count = 0

    @torch.no_grad()
    def update(self, pred_probs: torch.Tensor, target_masks: torch.Tensor, valid_masks: Optional[torch.Tensor] = None) -> None:
        """
        Update accumulator with batch predictions and targets.
        Args:
            pred_probs: [B, C, H, W] in [0, 1] (or after sigmoid)
            target_masks: [B, C, H, W] in {0, 1}
            valid_masks: [B, 1, H, W] or [B, C, H, W] optional validity mask
        """
        pred_probs = pred_probs.to(self.device)
        target_masks = target_masks.to(self.device)
        bin_preds = (pred_probs >= 0.5).to(dtype=torch.float32)
        gt = target_masks.to(dtype=torch.float32)

        if valid_masks is not None:
            vmask = valid_masks.to(self.device, dtype=torch.float32).expand_as(bin_preds)
            bin_preds = bin_preds * vmask
            gt = gt * vmask

        # Volumetric overlap
        inter = (bin_preds * gt).sum(dim=(0, 2, 3))
        union = (bin_preds + gt).clamp_max(1.0).sum(dim=(0, 2, 3))
        pred_card = bin_preds.sum(dim=(0, 2, 3))
        gt_card = gt.sum(dim=(0, 2, 3))

        # Boundary IoU
        b_pred = extract_boundary(bin_preds, d=2)
        b_gt = extract_boundary(gt, d=2)
        b_inter = (b_pred * b_gt).sum(dim=(0, 2, 3))
        b_union = (b_pred + b_gt).clamp_max(1.0).sum(dim=(0, 2, 3))

        # clDice (Centerline Dice)
        skel_pred = soft_skeletonize(bin_preds, n_iter=2)
        skel_true = soft_skeletonize(gt, n_iter=2)
        skel_prec_inter = (skel_pred * gt).sum(dim=(0, 2, 3))
        skel_prec_total = skel_pred.sum(dim=(0, 2, 3))
        skel_sens_inter = (skel_true * bin_preds).sum(dim=(0, 2, 3))
        skel_sens_total = skel_true.sum(dim=(0, 2, 3))

        self.accum[0] += inter
        self.accum[1] += union
        self.accum[2] += pred_card
        self.accum[3] += gt_card
        self.accum[4] += b_inter
        self.accum[5] += b_union
        self.accum[6] += skel_prec_inter
        self.accum[7] += skel_prec_total
        self.accum[8] += skel_sens_inter
        self.accum[9] += skel_sens_total
        self.sample_count += pred_probs.shape[0]

    def compute(self) -> Dict[str, Any]:
        """Compute summary benchmark metrics across all classes."""
        eps = 1e-8
        inter = self.accum[0]
        union = self.accum[1]
        pred_card = self.accum[2]
        gt_card = self.accum[3]
        b_inter = self.accum[4]
        b_union = self.accum[5]
        s_p_inter = self.accum[6]
        s_p_total = self.accum[7]
        s_s_inter = self.accum[8]
        s_s_total = self.accum[9]

        ious = (inter / (union + eps)).cpu().numpy()
        dices = ((2.0 * inter) / (pred_card + gt_card + eps)).cpu().numpy()
        b_ious = (b_inter / (b_union + eps)).cpu().numpy()

        t_prec = s_p_inter / (s_p_total + eps)
        t_sens = s_s_inter / (s_s_total + eps)
        cldices = ((2.0 * t_prec * t_sens) / (t_prec + t_sens + eps)).cpu().numpy()

        per_class: Dict[str, Dict[str, float]] = {}
        for c in range(self.num_classes):
            c_name = self.class_names.get(c, f"Class_{c}")
            per_class[c_name] = {
                "iou": float(ious[c]),
                "dice": float(dices[c]),
                "biou": float(b_ious[c]),
                "cldice": float(cldices[c]),
            }

        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        mean_biou = float(np.mean(b_ious))
        mean_cldice = float(np.mean(cldices))

        return {
            "mIoU": mean_iou,
            "Dice": mean_dice,
            "bIoU": mean_biou,
            "clDice": mean_cldice,
            "per_class": per_class,
            "sample_count": self.sample_count,
        }
