from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .ema import ModelEMA


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Save training checkpoint with optimizer, scheduler, EMA, and AMP scaler state."""
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "model_name": model_name or getattr(model, "name", model.__class__.__name__),
        "num_classes": num_classes,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if ema_model is not None:
        ckpt["ema_state_dict"] = ema_model.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint."""
    ckpt = torch.load(filepath, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, nn.Module):
        model.load_state_dict(ckpt.state_dict())
    else:
        model.load_state_dict(ckpt)

    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ema_model is not None and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"])
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and isinstance(ckpt, dict) and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
        "loss": ckpt.get("loss", float("inf")) if isinstance(ckpt, dict) else float("inf"),
        "model_name": ckpt.get("model_name") if isinstance(ckpt, dict) else None,
        "num_classes": ckpt.get("num_classes") if isinstance(ckpt, dict) else None,
        "class_names": ckpt.get("class_names") if isinstance(ckpt, dict) else None,
        "checkpoint": ckpt,
    }
