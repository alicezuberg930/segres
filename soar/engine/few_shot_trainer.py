from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.few_shot import FewShotSOAR
from ..data.dataset import SegmentationDataset
from ..data.dataset_config import DatasetConfig
from ..data.few_shot import FewShotEpisodeDataset, few_shot_collate_fn
from ..losses import SegmentationLoss
from ..losses.structure import soft_skeletonize


def _extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -torch.nn.functional.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return torch.nn.functional.relu(mask - eroded)


class FewShotTrainer:
    """
    Episodic Trainer for Few-Shot and One-Shot Segmentation (FS-SOAR).
    
    Supports:
    - 1-Shot and N-Shot episodic meta-learning.
    - K-fold cross-validation class partitioning.
    - Full-resolution topology preservation (clDice + Boundary IoU).
    - FP16 Automatic Mixed Precision (AMP).
    - Virtual batch gradient accumulation.
    """

    def __init__(
        self,
        model_cfg: str = "configs/models/soar_nano1.yaml",
        data_root: str = "data",
        shots: int = 1,
        fold: int = 0,
        total_folds: int = 4,
        train_episodes: int = 1000,
        val_episodes: int = 200,
        img_size: Tuple[int, int] = (1024, 1024),
        in_channels: int = 3,
        num_classes: int = 1,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        accumulate_grad_batches: int = 1,
        amp: bool = True,
        device: str = "cuda",
        workers: int = 2,
        checkpoint_dir: str = "checkpoints_few_shot",
        annotation_file: Optional[str] = None,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.shots = max(1, shots)
        self.fold = fold
        self.total_folds = total_folds
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes
        self.img_size = tuple(img_size)
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)
        self.use_amp = amp and (self.device.type == "cuda")
        self.workers = workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resolve dataset configuration
        self.dataset_cfg = DatasetConfig.resolve(data_root)
        self.data_root = self.dataset_cfg.root_path
        if (self.dataset_cfg.nc > 1 or len(self.dataset_cfg.names) > 1) and num_classes == 1:
            self.num_classes = self.dataset_cfg.nc
        else:
            self.num_classes = num_classes
        self.class_names = self.dataset_cfg.names

        # 1. Initialize Few-Shot Model
        print(f"\n[Few-Shot SOAR] Initializing {shots}-Shot Segmentation Framework (Fold {fold}/{total_folds})...")
        self.model = FewShotSOAR(
            backbone_cfg=model_cfg,
            in_channels=in_channels,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        # 2. Setup Optimizer & Scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        # 3. Setup Loss & AMP
        self.criterion = SegmentationLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 4. Setup Data
        self._setup_datasets(self.data_root, annotation_file)

        self.best_miou = 0.0

    def _setup_datasets(self, data_root: Path, annotation_file: Optional[str]):
        """Construct base datasets and wrap them into Episodic Few-Shot Loaders."""
        train_image_dir = self.dataset_cfg.train_images
        val_image_dir = self.dataset_cfg.val_images
        train_image_files = self.dataset_cfg.train_image_list
        val_image_files = self.dataset_cfg.val_image_list

        train_ann = (
            str(self.dataset_cfg.annotation_files["train"])
            if "train" in self.dataset_cfg.annotation_files
            else annotation_file
        )
        val_ann = (
            str(self.dataset_cfg.annotation_files["val"])
            if "val" in self.dataset_cfg.annotation_files
            else annotation_file
        )

        train_mask_dir = (
            str(self.dataset_cfg.mask_dirs["train"])
            if "train" in self.dataset_cfg.mask_dirs
            else None
        )
        val_mask_dir = (
            str(self.dataset_cfg.mask_dirs["val"])
            if "val" in self.dataset_cfg.mask_dirs
            else None
        )

        base_train_ds = SegmentationDataset(
            data_root=data_root,
            split="train",
            img_size=self.img_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            names=self.class_names,
            augment=True,
            annotation_file=train_ann,
            mask_dir=train_mask_dir,
            image_dir=train_image_dir,
            image_files=train_image_files,
        )

        try:
            base_val_ds = SegmentationDataset(
                data_root=data_root,
                split="val",
                img_size=self.img_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                names=self.class_names,
                augment=False,
                annotation_file=val_ann,
                mask_dir=val_mask_dir,
                image_dir=val_image_dir,
                image_files=val_image_files,
            )
            if len(base_val_ds) == 0:
                base_val_ds = base_train_ds
        except Exception:
            base_val_ds = base_train_ds

        # Create episodic samplers
        self.train_episode_ds = FewShotEpisodeDataset(
            base_dataset=base_train_ds,
            shots=self.shots,
            episodes=self.train_episodes,
            fold=self.fold,
            total_folds=self.total_folds,
            is_train=True,
        )

        self.val_episode_ds = FewShotEpisodeDataset(
            base_dataset=base_val_ds,
            shots=self.shots,
            episodes=self.val_episodes,
            fold=self.fold,
            total_folds=self.total_folds,
            is_train=False,
            seed=42,
        )

        self.train_loader = DataLoader(
            self.train_episode_ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=few_shot_collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

        self.val_loader = DataLoader(
            self.val_episode_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=few_shot_collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

        print(f" - Train Base Classes: {self.train_episode_ds.active_classes}")
        print(f" - Val Novel Classes:   {self.val_episode_ds.active_classes}")
        print(f" - Train Episodes/epoch: {len(self.train_episode_ds)}")
        print(f" - Val Episodes/epoch:   {len(self.val_episode_ds)}\n")

    def train_epoch(self, epoch: int) -> float:
        """Run one episodic training epoch."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1:02d}/{self.epochs:02d}")

        for step, batch in pbar:
            supp_imgs = batch["support_images"].to(self.device, non_blocking=True)  # (B, K, C, H, W)
            supp_masks = batch["support_masks"].to(self.device, non_blocking=True)  # (B, K, 1, H, W)
            query_imgs = batch["query_images"].to(self.device, non_blocking=True)   # (B, C, H, W)
            query_masks = batch["query_masks"].to(self.device, non_blocking=True)   # (B, 1, H, W)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                query_logits, aux = self.model(
                    query_image=query_imgs,
                    support_images=supp_imgs,
                    support_masks=supp_masks,
                    return_aux=True,
                )

                valid_mask = torch.ones_like(query_masks)
                loss_query, loss_parts = self.criterion(query_logits, query_masks, valid_mask, epoch=epoch)

                # Cycle-consistency support reconstruction loss
                supp_logits = aux["supp_logits"]
                supp_first_mask = supp_masks[:, 0]
                loss_supp, _ = self.criterion(supp_logits, supp_first_mask, torch.ones_like(supp_first_mask), epoch=epoch)

                loss = loss_query + 0.2 * loss_supp
                scaled_loss = loss / self.accumulate_grad_batches

            self.scaler.scale(scaled_loss).backward()

            if (step + 1) % self.accumulate_grad_batches == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "bnd": f"{loss_parts.get('boundary_loss', 0.0):.3f}",
                "cldice": f"{loss_parts.get('cldice_loss', 0.0):.3f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        self.scheduler.step()
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on unseen novel classes over fixed episodes."""
        self.model.eval()

        class_ious: Dict[int, List[float]] = {}
        fg_ious = []
        bg_ious = []
        cldice_scores = []
        bnd_ious = []

        pbar = tqdm(self.val_loader, desc="Validating Novel Classes", leave=False)

        for batch in pbar:
            supp_imgs = batch["support_images"].to(self.device, non_blocking=True)
            supp_masks = batch["support_masks"].to(self.device, non_blocking=True)
            query_imgs = batch["query_images"].to(self.device, non_blocking=True)
            query_masks = batch["query_masks"].to(self.device, non_blocking=True)
            cid = batch["class_ids"][0]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(
                    query_image=query_imgs,
                    support_images=supp_imgs,
                    support_masks=supp_masks,
                    return_aux=False,
                )

            preds = (torch.sigmoid(logits) > 0.5).float()
            targets = query_masks

            # Foreground IoU
            intersection = (preds * targets).sum().item()
            union = (preds + targets).clamp(max=1.0).sum().item()
            fg_iou = intersection / max(union, 1e-6)

            # Background IoU
            bg_pred = 1.0 - preds
            bg_tgt = 1.0 - targets
            bg_inter = (bg_pred * bg_tgt).sum().item()
            bg_union = (bg_pred + bg_tgt).clamp(max=1.0).sum().item()
            bg_iou = bg_inter / max(bg_union, 1e-6)

            # clDice (Topological skeleton precision & sensitivity)
            skel_pred = soft_skeletonize(preds, n_iter=2)
            skel_true = soft_skeletonize(targets, n_iter=2)
            skel_prec_inter = (skel_pred * targets).sum().item()
            skel_prec_total = skel_pred.sum().item()
            skel_sens_inter = (skel_true * preds).sum().item()
            skel_sens_total = skel_true.sum().item()
            tprec = skel_prec_inter / max(skel_prec_total, 1e-6)
            tsens = skel_sens_inter / max(skel_sens_total, 1e-6)
            cld = (2.0 * tprec * tsens) / max(tprec + tsens, 1e-6)

            # Boundary IoU
            b_pred = _extract_boundary(preds, d=2)
            b_gt = _extract_boundary(targets, d=2)
            b_inter = (b_pred * b_gt).sum().item()
            b_union = (b_pred + b_gt).clamp_max(1.0).sum().item()
            bnd = b_inter / max(b_union, 1e-6)

            fg_ious.append(fg_iou)
            bg_ious.append(bg_iou)
            cldice_scores.append(max(0.0, cld))
            bnd_ious.append(bnd)

            if cid not in class_ious:
                class_ious[cid] = []
            class_ious[cid].append(fg_iou)

        # Compute aggregate metrics
        mean_fg_iou = float(np.mean(fg_ious)) if fg_ious else 0.0
        mean_bg_iou = float(np.mean(bg_ious)) if bg_ious else 0.0
        fb_iou = 0.5 * (mean_fg_iou + mean_bg_iou)
        mean_cldice = float(np.mean(cldice_scores)) if cldice_scores else 0.0
        mean_bnd_iou = float(np.mean(bnd_ious)) if bnd_ious else 0.0

        # Class mIoU (mean across unique novel classes)
        novel_class_ious = [np.mean(vals) for vals in class_ious.values()]
        novel_miou = float(np.mean(novel_class_ious)) if novel_class_ious else mean_fg_iou

        return {
            "mIoU": novel_miou,
            "FB-IoU": fb_iou,
            "fg_iou": mean_fg_iou,
            "bg_iou": mean_bg_iou,
            "cldice": mean_cldice,
            "boundary_iou": mean_bnd_iou,
        }

    def train(self) -> None:
        """Run full episodic training loop."""
        print(f"\nStarting {self.shots}-Shot Training for {self.epochs} epochs...")
        print(f"{'Epoch':>6} {'TrainLoss':>11} {'Novel mIoU':>12} {'FB-IoU':>10} {'clDice':>10} {'Bnd-IoU':>10}")
        print("-" * 65)

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()

            novel_miou = metrics["mIoU"]
            fb_iou = metrics["FB-IoU"]
            cldice = metrics["cldice"]
            bnd_iou = metrics["boundary_iou"]

            print(
                f"{epoch+1:6d} {train_loss:11.4f} {novel_miou:12.4f} {fb_iou:10.4f} "
                f"{cldice:10.4f} {bnd_iou:10.4f}"
            )

            # Save checkpoint
            ckpt = {
                "epoch": epoch + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
                "shots": self.shots,
                "fold": self.fold,
            }
            torch.save(ckpt, self.checkpoint_dir / "last_model.pt")

            if novel_miou > self.best_miou:
                self.best_miou = novel_miou
                torch.save(ckpt, self.checkpoint_dir / "best_model.pt")
                print(f" -> Best novel class mIoU improved to {self.best_miou:.4f}! Saved to {self.checkpoint_dir / 'best_model.pt'}")

        print(f"\nTraining completed! Peak novel class mIoU ({self.shots}-Shot): {self.best_miou:.4f}")
