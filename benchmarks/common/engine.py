from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from soar.data.dataset import SegmentationDataset, collate_fn
from soar.data.dataset_config import DatasetConfig
from .metrics import BenchmarkMetricAccumulator


class SoftDiceLoss(nn.Module):
    """Numerically stable multi-class soft Dice loss."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * targets, dim=dims)
        card = torch.sum(probs + targets, dim=dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - torch.mean(dice)


class CombinedSegmentationLoss(nn.Module):
    """Equilibrated BCE + Dice objective for semantic segmentation baselines."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets.float())
        loss_dice = self.dice(logits, targets.float())
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class BenchmarkTrainer:
    """
    Unified training and validation engine for benchmark baseline models.
    Supports:
      - Training from scratch on high-resolution datasets (1024x1024 or 2048x2048)
      - Epoch-by-epoch evaluation of mIoU, Dice, bIoU, and clDice
      - Automatic checkpoint tracking (best.pt, last.pt)
      - Mixed-precision training (torch.cuda.amp)
      - History logging in structured JSON
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        data_root: str | Path,
        annotation_file: Optional[str | Path] = None,
        mask_dir: Optional[str | Path] = None,
        img_size: Tuple[int, int] = (1024, 1024),
        epochs: int = 50,
        batch_size: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        amp: bool = True,
        num_workers: int = 2,
        checkpoint_dir: str | Path = "checkpoints/benchmarks",
        val_interval: int = 1,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        loss_type: str = "soar",
    ):
        self.model = model
        self.model_name = model_name
        self.data_root = Path(data_root).resolve()
        self.annotation_file = Path(annotation_file).resolve() if annotation_file else None
        self.mask_dir = Path(mask_dir).resolve() if mask_dir else None
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.amp = amp and (self.device.type == "cuda")
        self.num_workers = num_workers
        self.checkpoint_dir = Path(checkpoint_dir).resolve() / self.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.val_interval = max(1, val_interval)
        self.loss_type = loss_type.lower()

        # Dataset configuration
        self.cfg: Optional[DatasetConfig] = None
        if self.data_root.is_file() and self.data_root.suffix.lower() in (".yaml", ".yml"):
            self.cfg = DatasetConfig.resolve(self.data_root)
            self.num_classes = num_classes or (self.cfg.nc if self.cfg else 1)
            self.class_names = class_names or (self.cfg.names if self.cfg else {})
        else:
            self.num_classes = num_classes or 1
            self.class_names = class_names or {}

        # Setup model and optimizers
        self.model.to(self.device)
        if self.loss_type == "soar":
            from soar.losses import SegmentationLoss as CompositeSegmentationLoss
            self.criterion = CompositeSegmentationLoss()
        else:
            self.criterion = CombinedSegmentationLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.setup_dataloaders()
        self.history: List[Dict[str, Any]] = []
        self.best_metric: float = -1.0
        self.best_epoch: int = 0

    def setup_dataloaders(self) -> None:
        """Initialize training and validation datasets."""
        train_image_dir = self.cfg.train_images if self.cfg else None
        train_ann = self.cfg.annotation_files.get("train") if (self.cfg and self.cfg.annotation_files) else str(self.annotation_file) if self.annotation_file else None
        train_masks = self.cfg.mask_dirs.get("train") if (self.cfg and self.cfg.mask_dirs) else str(self.mask_dir) if self.mask_dir else None

        self.train_dataset = SegmentationDataset(
            data_root=self.cfg.root_path if self.cfg else self.data_root,
            annotation_file=str(train_ann) if train_ann else None,
            mask_dir=str(train_masks) if train_masks else None,
            split="train",
            img_size=self.img_size,
            num_classes=self.num_classes,
            augment=True,
            image_dir=train_image_dir,
            image_files=self.cfg.train_image_list if self.cfg else None,
            names=self.class_names,
        )

        val_image_dir = self.cfg.val_images if self.cfg else None
        val_ann = self.cfg.annotation_files.get("val") if (self.cfg and self.cfg.annotation_files) else None
        val_masks = self.cfg.mask_dirs.get("val") if (self.cfg and self.cfg.mask_dirs) else None

        self.val_dataset = SegmentationDataset(
            data_root=self.cfg.root_path if self.cfg else self.data_root,
            annotation_file=str(val_ann) if val_ann else None,
            mask_dir=str(val_masks) if val_masks else None,
            split="val",
            img_size=self.img_size,
            num_classes=self.num_classes,
            augment=False,
            image_dir=val_image_dir,
            image_files=self.cfg.val_image_list if self.cfg else None,
            names=self.class_names,
        )

        if not self.class_names and hasattr(self.train_dataset, "class_names") and self.train_dataset.class_names:
            self.class_names = self.train_dataset.class_names
            self.num_classes = max(self.num_classes, len(self.class_names))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=(self.device.type == "cuda"),
            drop_last=(len(self.train_dataset) > self.batch_size),
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

    def train_epoch(self, epoch: int) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            valid_masks = batch.get("valid_mask")
            if valid_masks is not None:
                valid_masks = valid_masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp):
                outputs = self.model(images)
                if self.loss_type == "soar":
                    loss, _ = self.criterion(outputs, masks, valid_masks, epoch)
                else:
                    loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = float(loss.detach().item())
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        self.scheduler.step()
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Run full evaluation on validation split computing mIoU, Dice, bIoU, and clDice."""
        self.model.eval()
        accumulator = BenchmarkMetricAccumulator(
            num_classes=self.num_classes,
            class_names=self.class_names,
            device=self.device,
        )

        pbar = tqdm(self.val_loader, desc="Validating", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            valid_masks = batch.get("valid_mask")
            if valid_masks is not None:
                valid_masks = valid_masks.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(images)
                probs = torch.sigmoid(logits)

            accumulator.update(probs, masks, valid_masks)

        return accumulator.compute()

    def train(self) -> Dict[str, Any]:
        """Execute end-to-end training loop across all epochs."""
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print("=" * 70)
        print(f"Starting Benchmark Training: {self.model_name}")
        print(f"  - Model Parameters:  {num_params:.2f} M")
        print(f"  - Input Resolution:  {self.img_size[0]}x{self.img_size[1]}")
        print(f"  - Loss Framework:    {'SOAR Composite (Focal+Dice+Boundary+clDice)' if self.loss_type == 'soar' else 'Standard (BCE+Dice)'}")
        print(f"  - Epochs:            {self.epochs}")
        print(f"  - Batch Size:        {self.batch_size}")
        print(f"  - Learning Rate:     {self.lr}")
        print(f"  - Mixed Precision:   {self.amp}")
        print(f"  - Classes ({self.num_classes}):       {self.class_names}")
        print(f"  - Train Samples:     {len(self.train_dataset)}")
        print(f"  - Val Samples:       {len(self.val_dataset)}")
        print(f"  - Checkpoint Dir:    {self.checkpoint_dir}")
        print("=" * 70)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            time_epoch = time.time() - t0

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "time_sec": time_epoch,
            }

            if epoch % self.val_interval == 0 or epoch == self.epochs:
                val_metrics = self.validate()
                epoch_record.update({
                    "val_mIoU": val_metrics["mIoU"],
                    "val_Dice": val_metrics["Dice"],
                    "val_bIoU": val_metrics["bIoU"],
                    "val_clDice": val_metrics["clDice"],
                    "per_class": val_metrics["per_class"],
                })

                score = val_metrics["mIoU"]
                is_best = score > self.best_metric
                if is_best:
                    self.best_metric = score
                    self.best_epoch = epoch
                    self.save_checkpoint("best.pt", epoch, val_metrics)

                print(
                    f"Epoch [{epoch:03d}/{self.epochs:03d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"mIoU: {val_metrics['mIoU']*100:.2f}% | "
                    f"Dice: {val_metrics['Dice']*100:.2f}% | "
                    f"bIoU: {val_metrics['bIoU']*100:.2f}% | "
                    f"clDice: {val_metrics['clDice']*100:.2f}% "
                    f"({'* BEST' if is_best else ''})"
                )
            else:
                print(f"Epoch [{epoch:03d}/{self.epochs:03d}] Train Loss: {train_loss:.4f} ({time_epoch:.1f}s)")

            self.history.append(epoch_record)
            self.save_checkpoint("last.pt", epoch, epoch_record)

            # Dump history log
            with open(self.checkpoint_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)

        print("=" * 70)
        print(f"Training Complete! Best mIoU: {self.best_metric*100:.2f}% at Epoch {self.best_epoch}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 70)

        return {
            "best_epoch": self.best_epoch,
            "best_mIoU": self.best_metric,
            "checkpoint_dir": str(self.checkpoint_dir),
            "history": self.history,
        }

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint with optimizer and metadata."""
        ckpt_path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "img_size": self.img_size,
            "metrics": metrics,
        }, ckpt_path)
