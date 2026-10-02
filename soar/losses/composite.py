from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from .base import BaseLoss
from .region import DiceBCELoss, DiceFocalLoss, BCELoss, DiceLoss, FocalLoss, TverskyLoss, FocalTverskyLoss
from .boundary import BoundaryBCELoss, BoundaryDiceLoss, BoundaryDistLoss
from .structure import CLDiceLoss, SkeletonLoss


class SegmentationLoss(nn.Module):
    """
    High-Resolution Domain-Specific Composite Segmentation Loss Framework.

    Engineered specifically for high-resolution imagery (1024x1024 to 4096+x4096+)
    characterized by:
      1. Extreme foreground-background imbalance (<1% target pixels).
      2. Critical fine boundary precision (1-5 pixel wide targets/edges).
      3. Topological connectivity preservation for thin curvilinear structures.
      4. Balanced gradient scale dynamics between dense pixel losses and geometric contour losses.

    Combines:
      - Multi-domain region supervision: Scale-invariant Soft Dice + Focal gradient modulation.
      - High-frequency edge supervision: Boundary Dice / Boundary Signed Distance transform.
      - Topological structure supervision: Differentiable soft skeletonization clDice.
    """

    def __init__(
        self,
        region: Optional[BaseLoss] = None,
        boundary: Optional[Union[BaseLoss, bool]] = None,
        structure: Optional[Union[BaseLoss, bool]] = None,
        region_type: str = "dice_focal",
        boundary_type: Optional[str] = "boundary_dice",
        structure_type: Optional[str] = "cldice",
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        bce_weight: float = 1.0,
        pos_weight: float = 3.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        boundary_weight: float = 0.3,
        cldice_weight: float = 0.2,
        cldice_warmup_epochs: int = 3,
        deep_supervision: bool = False,
        deep_supervision_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        # 1. Setup High-Resolution Region Loss
        if region is not None:
            self.region = region
        else:
            rt = (region_type or "dice_focal").lower()
            if rt in ("dice_focal", "focal_dice"):
                self.region = DiceFocalLoss(
                    weight=1.0,
                    dice_weight=dice_weight,
                    focal_weight=focal_weight,
                    gamma=focal_gamma,
                    alpha=focal_alpha,
                )
            elif rt == "dice_bce":
                self.region = DiceBCELoss(
                    weight=1.0,
                    dice_weight=dice_weight,
                    bce_weight=bce_weight,
                    bce_pos_weight=pos_weight,
                )
            elif rt == "focal":
                self.region = FocalLoss(weight=focal_weight, gamma=focal_gamma, alpha=focal_alpha)
            elif rt == "dice":
                self.region = DiceLoss(weight=dice_weight)
            elif rt == "focal_tversky":
                self.region = FocalTverskyLoss(weight=1.0, gamma=focal_gamma)
            elif rt == "tversky":
                self.region = TverskyLoss(weight=1.0)
            else:
                self.region = DiceFocalLoss(weight=1.0, dice_weight=dice_weight, focal_weight=focal_weight)

        # 2. Setup High-Frequency Boundary Loss
        if boundary is False or boundary_weight <= 0.0:
            self.boundary = None
        elif isinstance(boundary, BaseLoss):
            self.boundary = boundary
        else:
            bt = (boundary_type or "boundary_dice").lower() if boundary_type else None
            if bt == "boundary_dice":
                self.boundary = BoundaryDiceLoss(weight=boundary_weight)
            elif bt == "boundary_dist":
                self.boundary = BoundaryDistLoss(weight=boundary_weight)
            elif bt == "boundary_bce":
                self.boundary = BoundaryBCELoss(weight=boundary_weight)
            else:
                self.boundary = None

        # 3. Setup Topological Structure Loss with Warmup
        if structure is False or cldice_weight <= 0.0:
            self.structure = None
        elif isinstance(structure, BaseLoss):
            self.structure = structure
        else:
            st = (structure_type or "cldice").lower() if structure_type else None
            if st == "cldice":
                self.structure = CLDiceLoss(weight=cldice_weight)
            elif st == "skeleton":
                self.structure = SkeletonLoss(weight=cldice_weight)
            else:
                self.structure = None

        self.cldice_warmup_epochs = max(0, int(cldice_warmup_epochs))
        self.target_structure_weight = self.structure.weight if self.structure is not None else 0.0

        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.3, 0.2, 0.1]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        auxiliary: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss_parts: Dict[str, Any] = {}

        # Safe logit stabilization against extreme high-resolution edge gradients in AMP
        pred_clamped = pred.float().clamp(-30.0, 30.0)
        target_f32 = target.float()

        # 1. Region Loss (Focal + Dice for foreground sparsity)
        region_loss = self.region(pred_clamped, target_f32, valid_mask)
        total = region_loss
        loss_parts["region"] = region_loss.detach()
        loss_parts["region_loss"] = loss_parts["region"]

        # 2. Boundary Loss (High-frequency edge sharpness)
        if self.boundary is not None:
            bnd_loss = self.boundary(pred_clamped, target_f32, valid_mask)
            total = total + bnd_loss
            loss_parts["boundary"] = bnd_loss.detach()
            loss_parts["boundary_loss"] = loss_parts["boundary"]
        else:
            zero_tensor = torch.tensor(0.0, device=pred.device)
            loss_parts["boundary"] = zero_tensor
            loss_parts["boundary_loss"] = zero_tensor

        # 3. Topological Structure Loss with Warmup Scheduling
        if self.structure is not None:
            if epoch < self.cldice_warmup_epochs:
                scale = 0.0
            else:
                scale = min(1.0, (epoch - self.cldice_warmup_epochs + 1) / max(1, self.cldice_warmup_epochs))
            self.structure.weight = self.target_structure_weight * scale

            if self.structure.weight > 0.0:
                struct_loss = self.structure(pred_clamped, target_f32, valid_mask)
                total = total + struct_loss
                loss_parts["cldice"] = struct_loss.detach()
                loss_parts["cldice_loss"] = struct_loss.detach()
            else:
                zero_tensor = torch.tensor(0.0, device=pred.device)
                loss_parts["cldice"] = zero_tensor
                loss_parts["cldice_loss"] = zero_tensor
        else:
            zero_tensor = torch.tensor(0.0, device=pred.device)
            loss_parts["cldice"] = zero_tensor
            loss_parts["cldice_loss"] = zero_tensor

        # 4. Multi-Scale Deep Supervision
        if self.deep_supervision and auxiliary is not None:
            ds_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            for key, weight in zip(sorted(auxiliary.keys()), self.deep_supervision_weights):
                aux_pred = auxiliary[key].float().clamp(-30.0, 30.0)
                ds_loss = ds_loss + weight * self.region(aux_pred, target_f32, valid_mask)
            total = total + ds_loss
            loss_parts["deep_supervision"] = ds_loss.detach()

        loss_parts["total"] = total.detach()
        loss_parts["total_loss"] = total.detach()
        return total, loss_parts

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SegmentationLoss":
        """Instantiate loss configuration from YAML dictionary supporting nested or flat structures."""
        # Check if nested 'loss' key exists
        if "loss" in config and isinstance(config["loss"], dict):
            config = config["loss"]

        region_cfg = config.get("region", {})
        boundary_cfg = config.get("boundary", {})
        structure_cfg = config.get("structure", {})

        region_type = region_cfg.get("type", config.get("region_type", "dice_focal"))
        dice_w = float(region_cfg.get("dice_weight", config.get("dice_weight", 1.0)))
        focal_w = float(region_cfg.get("focal_weight", config.get("focal_weight", 1.0)))
        bce_w = float(region_cfg.get("bce_weight", config.get("bce_weight", 1.0)))
        pos_w = float(region_cfg.get("bce_pos_weight", config.get("pos_weight", 3.0)))
        focal_gamma = float(region_cfg.get("gamma", config.get("focal_gamma", 2.0)))
        focal_alpha = float(region_cfg.get("alpha", config.get("focal_alpha", 0.25)))

        boundary_enabled = boundary_cfg.get("enabled", True)
        boundary_type = boundary_cfg.get("type", "boundary_dice") if boundary_enabled else None
        boundary_w = float(boundary_cfg.get("weight", config.get("boundary_weight", 0.3))) if boundary_enabled else 0.0

        structure_enabled = structure_cfg.get("enabled", True)
        structure_type = structure_cfg.get("type", "cldice") if structure_enabled else None
        cldice_w = float(structure_cfg.get("weight", config.get("cldice_weight", 0.2))) if structure_enabled else 0.0
        warmup = int(structure_cfg.get("warmup_epochs", config.get("cldice_warmup_epochs", 3)))

        return cls(
            region_type=region_type,
            boundary_type=boundary_type,
            structure_type=structure_type,
            dice_weight=dice_w,
            focal_weight=focal_w,
            bce_weight=bce_w,
            pos_weight=pos_w,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            boundary_weight=boundary_w,
            cldice_weight=cldice_w,
            cldice_warmup_epochs=warmup,
            deep_supervision=bool(config.get("deep_supervision", False)),
            deep_supervision_weights=config.get("deep_supervision_weights"),
        )
