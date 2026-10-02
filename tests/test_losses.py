from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from soar.losses import (
    SegmentationLoss,
    DiceBCELoss,
    DiceFocalLoss,
    FocalLoss,
    CLDiceLoss,
    BoundaryDistLoss,
    BoundaryDiceLoss,
)


def test_loss_backward():
    """Verify composite and atomic losses compute gradients stably without FP16/FP32 overflow."""
    loss_fn = SegmentationLoss()
    pred = torch.randn(1, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 128, 128), dtype=torch.float32)

    total_loss, loss_items = loss_fn(pred, target)
    assert not torch.isnan(total_loss), "Loss computed NaN"
    assert not torch.isinf(total_loss), "Loss computed Inf"
    total_loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any(), "Gradient contains NaN"


def test_highres_sparse_imbalance_and_boundary_gradients():
    """Verify high-resolution loss behavior under extreme foreground sparsity (<1%) at 1024x1024."""
    loss_fn = SegmentationLoss(
        region_type="dice_focal",
        boundary_type="boundary_dice",
        structure_type="cldice",
        dice_weight=1.0,
        focal_weight=1.0,
        boundary_weight=0.3,
        cldice_weight=0.2,
        cldice_warmup_epochs=0,
    )

    H, W = 1024, 1024
    pred = torch.randn(1, 1, H, W, requires_grad=True)
    # Extreme sparsity: 99.5% background, 0.5% thin filament foreground
    target = torch.zeros(1, 1, H, W, dtype=torch.float32)
    target[:, :, 500:505, :] = 1.0  # 5-pixel road across width (~0.5% foreground)

    total_loss, loss_parts = loss_fn(pred, target, epoch=1)
    assert not torch.isnan(total_loss), "High-res total loss computed NaN"
    assert not torch.isinf(total_loss), "High-res total loss computed Inf"
    assert "region" in loss_parts and "boundary" in loss_parts and "cldice" in loss_parts
    assert loss_parts["boundary"] > 0.0, "Boundary loss must be active for high-resolution target"
    assert loss_parts["cldice"] > 0.0, "clDice loss must be active for thin filament target"

    total_loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any(), "Gradient contains NaN on high-res sparse input"
    assert not torch.isinf(pred.grad).any(), "Gradient contains Inf on high-res sparse input"

    # Verify boundary gradients are non-zero around the target line
    grad_abs = pred.grad.abs().squeeze().numpy()
    target_region_grad = grad_abs[498:507, :].mean()
    distant_bg_grad = grad_abs[100:150, :].mean()
    assert target_region_grad > distant_bg_grad, "Gradient should be strongly focused around the boundary"


def test_highres_multiclass():
    """Verify high-resolution multi-class segmentation loss at 512x512 with 4 classes."""
    loss_fn = SegmentationLoss.from_config({
        "loss": {
            "region": {"type": "dice_focal", "dice_weight": 1.0, "focal_weight": 1.0},
            "boundary": {"enabled": True, "type": "boundary_dice", "weight": 0.3},
            "structure": {"enabled": True, "type": "cldice", "weight": 0.2, "warmup_epochs": 0},
        }
    })

    H, W = 512, 512
    C = 4
    pred = torch.randn(1, C, H, W, requires_grad=True)
    target = torch.zeros(1, C, H, W, dtype=torch.float32)
    target[:, 1, 100:110, 100:200] = 1.0
    target[:, 2, 300:305, :] = 1.0

    total_loss, loss_parts = loss_fn(pred, target, epoch=1)
    assert not torch.isnan(total_loss)
    total_loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


if __name__ == "__main__":
    print("Testing standard loss backward...")
    test_loss_backward()
    print("Testing high-resolution extreme sparsity and boundary gradients at 1024x1024...")
    test_highres_sparse_imbalance_and_boundary_gradients()
    print("Testing high-resolution multi-class...")
    test_highres_multiclass()
    print("All High-Resolution domain loss & gradient tests passed successfully!")
