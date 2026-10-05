from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..models import SegmentationModel
from ..data import SegmentationDataset, collate_fn
from ..losses import SegmentationLoss as CompositeSegmentationLoss
from ..losses.structure import soft_skeletonize


def _extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return F.relu(mask - eroded)


class BaseValidator:
    """
    Validation engine for high-resolution segmentation models.
    Implements O(1) host memory streaming metric accumulation and multi-rank distributed reduction.
    """

    def __init__(
        self,
        model: nn.Module,
        data_root: str,
        img_size: tuple = (1024, 1024),
        device: str = "cuda",
        num_workers: int = 2,
        save_dir: Optional[str] = None,
        dataloader: Optional[DataLoader] = None,
        num_classes: int = 1,
        class_names: Optional[Dict[int, str]] = None,
        annotation_file: Optional[str] = None,
        cache_ram: bool = False,
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.annotation_file = annotation_file
        self.img_size = img_size
        self.batch_size = 1
        self.num_classes = max(1, int(num_classes))
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.num_workers = num_workers
        self.save_dir = Path(save_dir) if save_dir else None
        self._dataloader = dataloader
        self.class_names = class_names or {}
        self.cache_ram = bool(cache_ram)
        if not self.class_names and dataloader is not None:
            ds = getattr(dataloader, "dataset", None)
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            if hasattr(ds, "class_names") and ds.class_names:
                self.class_names = ds.class_names

        self.model.to(self.device)
        self.model.eval()

        self.criterion = CompositeSegmentationLoss()
        self.metrics: Dict[str, Any] = {}
        self._cached_vis: List[Dict[str, torch.Tensor]] = []
        self.num_vis_samples: int = 4

    @property
    def dataloader(self) -> Optional[DataLoader]:
        return self._dataloader

    @dataloader.setter
    def dataloader(self, loader: Optional[DataLoader]):
        self._dataloader = loader
        if not self.class_names and loader is not None:
            ds = getattr(loader, "dataset", None)
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            if hasattr(ds, "class_names") and ds.class_names:
                self.class_names = ds.class_names

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self._dataloader

    @val_loader.setter
    def val_loader(self, loader: Optional[DataLoader]):
        self.dataloader = loader

    def setup_data(self, split: str = "val"):
        """Setup validation data loader if not externally provided."""
        self.dataset = SegmentationDataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            num_classes=self.num_classes,
            annotation_file=self.annotation_file,
            augment=False,
            use_cache=True,
            auto=True,
            cache_ram=self.cache_ram,
        )

        if self.cache_ram:
            self.dataset.preload_cache(verbose=True)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

    @staticmethod
    def _ensure_4d_tensor(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Run validation with streaming O(1) memory metric computation across arbitrary classes."""
        if self.dataloader is None:
            self.setup_data(split="val")

        self.model.eval()
        self._cached_vis = []
        accum = None
        total_loss = 0.0
        n_batches = 0

        is_rank_zero = (not dist.is_initialized()) or dist.get_rank() == 0
        pbar = (
            tqdm(self.dataloader, desc="Validating", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")
            if is_rank_zero
            else self.dataloader
        )

        for batch in pbar:
            images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
            valid_masks = self._ensure_4d_tensor(batch["valid_mask"].to(self.device, non_blocking=True))
            masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))

            preds = self.model(images)
            loss, _ = self.criterion(preds, masks, valid_masks, 0)
            total_loss += float(loss.detach().item())
            n_batches += 1

            probs = torch.sigmoid(preds)

            if len(self._cached_vis) < self.num_vis_samples and is_rank_zero:
                needed = self.num_vis_samples - len(self._cached_vis)
                take = min(needed, images.shape[0])
                preview_size = (512, 512)
                sub_imgs = F.interpolate(images[:take], size=preview_size, mode="bilinear", align_corners=False).cpu()
                sub_masks = F.interpolate(masks[:take].float(), size=preview_size, mode="nearest").cpu()
                sub_probs = F.interpolate(probs[:take], size=preview_size, mode="nearest").cpu()
                for i in range(take):
                    self._cached_vis.append({
                        "image": sub_imgs[i],
                        "mask": sub_masks[i],
                        "prob": sub_probs[i],
                    })

            bin_preds = (probs >= 0.5).to(dtype=torch.float32)
            gt = masks.to(dtype=torch.float32)

            if valid_masks is not None:
                vmask = valid_masks.to(dtype=torch.float32).expand_as(bin_preds)
                bin_preds = bin_preds * vmask
                gt = gt * vmask

            c_dim = preds.shape[1]
            if accum is None:
                # 10 terms per channel:
                # [0: inter, 1: union, 2: pred_card, 3: gt_card,
                #  4: b_inter, 5: b_union, 6: skel_prec_inter, 7: skel_prec_total,
                #  8: skel_sens_inter, 9: skel_sens_total]
                accum = torch.zeros((10, c_dim), dtype=torch.float64, device=self.device)

            inter = (bin_preds * gt).sum(dim=(0, 2, 3))
            union = (bin_preds + gt).clamp_max(1.0).sum(dim=(0, 2, 3))
            pred_card = bin_preds.sum(dim=(0, 2, 3))
            gt_card = gt.sum(dim=(0, 2, 3))

            # Boundary IoU
            b_pred = _extract_boundary(bin_preds, d=2)
            b_gt = _extract_boundary(gt, d=2)
            b_inter = (b_pred * b_gt).sum(dim=(0, 2, 3))
            b_union = (b_pred + b_gt).clamp_max(1.0).sum(dim=(0, 2, 3))

            # clDice (Topological skeleton precision & sensitivity)
            skel_pred = soft_skeletonize(bin_preds, n_iter=2)
            skel_true = soft_skeletonize(gt, n_iter=2)
            skel_prec_inter = (skel_pred * gt).sum(dim=(0, 2, 3))
            skel_prec_total = skel_pred.sum(dim=(0, 2, 3))
            skel_sens_inter = (skel_true * bin_preds).sum(dim=(0, 2, 3))
            skel_sens_total = skel_true.sum(dim=(0, 2, 3))

            accum[0] += inter
            accum[1] += union
            accum[2] += pred_card
            accum[3] += gt_card
            accum[4] += b_inter
            accum[5] += b_union
            accum[6] += skel_prec_inter
            accum[7] += skel_prec_total
            accum[8] += skel_sens_inter
            accum[9] += skel_sens_total

        # Synchronize metrics across distributed ranks
        if dist.is_initialized():
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)
            loss_tensor = torch.tensor([total_loss, float(n_batches)], dtype=torch.float64, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = float(loss_tensor[0].item())
            n_batches = int(loss_tensor[1].item())

        if accum is None:
            return {}

        avg_loss = total_loss / max(n_batches, 1)

        total_inter = accum[0]
        total_union = accum[1]
        total_pred = accum[2]
        total_gt = accum[3]
        total_b_inter = accum[4]
        total_b_union = accum[5]
        total_skel_prec_inter = accum[6]
        total_skel_prec_total = accum[7]
        total_skel_sens_inter = accum[8]
        total_skel_sens_total = accum[9]

        class_ious = (total_inter / total_union.clamp_min(1e-7)).cpu().tolist()
        class_dices = ((2.0 * total_inter) / (total_pred + total_gt).clamp_min(1e-7)).cpu().tolist()
        class_prec = (total_inter / total_pred.clamp_min(1e-7)).cpu().tolist()
        class_recall = (total_inter / total_gt.clamp_min(1e-7)).cpu().tolist()
        class_biou = (total_b_inter / total_b_union.clamp_min(1e-7)).cpu().tolist()

        t_prec = (total_skel_prec_inter + 1e-7) / (total_skel_prec_total + 1e-7)
        t_sens = (total_skel_sens_inter + 1e-7) / (total_skel_sens_total + 1e-7)
        class_cldice = ((2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-7)).cpu().tolist()

        present_mask = (total_gt > 0).cpu().numpy()
        c_dim = len(class_ious)

        if present_mask.any():
            iou = float(np.mean([class_ious[c] for c in range(c_dim) if present_mask[c]]))
            dice = float(np.mean([class_dices[c] for c in range(c_dim) if present_mask[c]]))
            prec = float(np.mean([class_prec[c] for c in range(c_dim) if present_mask[c]]))
            recall = float(np.mean([class_recall[c] for c in range(c_dim) if present_mask[c]]))
            boundary_iou = float(np.mean([class_biou[c] for c in range(c_dim) if present_mask[c]]))
            cldice = float(np.mean([class_cldice[c] for c in range(c_dim) if present_mask[c]]))
        else:
            iou = float(np.mean(class_ious))
            dice = float(np.mean(class_dices))
            prec = float(np.mean(class_prec))
            recall = float(np.mean(class_recall))
            boundary_iou = float(np.mean(class_biou))
            cldice = float(np.mean(class_cldice))

        self.metrics = {
            "loss": float(avg_loss),
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(prec),
            "recall": float(recall),
            "boundary_iou": float(boundary_iou),
            "cldice": float(cldice),
            "class_ious": class_ious,
            "class_dices": class_dices,
            "class_gt": total_gt.cpu().tolist(),
        }
        return self.metrics

    def print_results(self, epoch: Optional[int] = None):
        """Print validation metrics summary in clean tabular format with per-class breakdown."""
        ep_str = f"Epoch {epoch}" if epoch is not None else "Summary"
        metric_label = "mIoU" if len(self.metrics.get("class_ious", [])) > 1 else "IoU"
        print(f"\n{'-'*95}")
        print(
            f"{'Stage / Metric':<16} {'Samples':<8} {'Loss':<10} {metric_label:<10} {'Dice':<10} {'Prec':<10} {'Recall':<10} {'bIoU':<10} {'clDice':<10}"
        )
        print(f"{'-'*95}")

        num_images = len(self.dataloader.dataset) if self.dataloader is not None else 0
        loss_val = self.metrics.get("loss", 0.0)
        iou_val = self.metrics.get("iou", 0.0)
        dice_val = self.metrics.get("dice", 0.0)
        prec_val = self.metrics.get("precision", 0.0)
        recall_val = self.metrics.get("recall", 0.0)
        biou_val = self.metrics.get("boundary_iou", 0.0)
        cldice_val = self.metrics.get("cldice", 0.0)

        print(
            f"{ep_str:<16} {num_images:<8} {loss_val:<10.4f} {iou_val:<10.4f} {dice_val:<10.4f} {prec_val:<10.4f} {recall_val:<10.4f} {biou_val:<10.4f} {cldice_val:<10.4f}"
        )
        print(f"{'-'*95}")

        class_ious = self.metrics.get("class_ious", [])
        class_gt = self.metrics.get("class_gt", [])
        if len(class_ious) > 1:
            print("Per-class IoU breakdown:")
            names_list = [str(self.class_names.get(c, f"Class_{c}")) for c in range(len(class_ious))]
            max_len = max(len(name) for name in names_list) if names_list else 10
            max_len = max(max_len, 10)

            for c_idx, c_iou in enumerate(class_ious):
                label_name = str(self.class_names.get(c_idx, f"Class_{c_idx}"))
                gt_note = " (no GT in split)" if (c_idx < len(class_gt) and class_gt[c_idx] == 0) else ""
                print(f"  {label_name:<{max_len}} : {c_iou:.4f}{gt_note}")
            print(f"{'-'*95}\n")
        else:
            print()

    def save_visualizations(self, num_samples: int = 4):
        """Save sample validation qualitative comparisons using lightning-fast OpenCV rendering."""
        if self.save_dir is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        samples = list(self._cached_vis)
        if len(samples) < num_samples and self.dataloader is not None:
            with torch.no_grad():
                for batch in self.dataloader:
                    if len(samples) >= num_samples:
                        break
                    images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
                    masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))
                    preds = self.model(images)
                    probs = torch.sigmoid(preds)
                    preview_size = (512, 512)
                    needed = num_samples - len(samples)
                    take = min(needed, images.shape[0])
                    sub_imgs = F.interpolate(images[:take], size=preview_size, mode="bilinear", align_corners=False).cpu()
                    sub_masks = F.interpolate(masks[:take].float(), size=preview_size, mode="nearest").cpu()
                    sub_probs = F.interpolate(probs[:take], size=preview_size, mode="nearest").cpu()
                    for i in range(take):
                        samples.append({
                            "image": sub_imgs[i],
                            "mask": sub_masks[i],
                            "prob": sub_probs[i],
                        })

        for idx, item in enumerate(samples[:num_samples]):
            self._save_sample_fast(item, idx)

        self._cached_vis = []

    def _save_sample_fast(self, item: Dict[str, torch.Tensor], idx: int):
        """Render multi-class colored comparison strip with real class legend in sub-10ms via OpenCV."""
        img_tensor = item["image"]
        mask_tensor = item["mask"]
        prob_tensor = item["prob"]

        try:
            img_np = img_tensor.float().numpy()
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=0)
            elif img_np.shape[0] == 1:
                img_np = np.repeat(img_np, 3, axis=0)
            elif img_np.shape[0] > 3:
                img_np = img_np[:3]

            img_np = np.transpose(img_np, (1, 2, 0))
            v_min = float(img_np.min())
            v_max = float(img_np.max())
            if v_max <= 1.5 and v_min >= -0.5:
                scale = 255.0 / max(v_max - v_min, 1e-6)
                img_np = np.clip((img_np - v_min) * scale, 0, 255).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            h, w = img_bgr.shape[:2]
            mask_np = mask_tensor.float().numpy()
            prob_np = prob_tensor.float().numpy()
            k_classes = mask_np.shape[0]

            palette = [
                (0, 0, 255),    # Red
                (0, 255, 0),    # Green
                (255, 255, 0),  # Cyan
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 165, 255),  # Orange
                (255, 128, 0),  # Blue
                (128, 255, 0),  # Light green
            ]

            gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
            pred_colored = np.zeros((h, w, 3), dtype=np.uint8)

            for c in range(k_classes):
                color = palette[c % len(palette)]
                m_c = mask_np[c] > 0.5
                p_c = prob_np[c] >= 0.5
                for ch in range(3):
                    gt_colored[:, :, ch] = np.maximum(gt_colored[:, :, ch], m_c * color[ch])
                    pred_colored[:, :, ch] = np.maximum(pred_colored[:, :, ch], p_c * color[ch])

            overlay = img_bgr.copy()
            p_any = np.any(pred_colored > 0, axis=-1)
            overlay[p_any] = cv2.addWeighted(img_bgr, 0.5, pred_colored, 0.5, 0)[p_any]

            panels = [img_bgr, gt_colored, pred_colored, overlay]
            titles = ["Input Image", "Ground Truth", "Prediction (p >= 0.5)", "Overlay"]
            for p, title in zip(panels, titles):
                cv2.rectangle(p, (0, 0), (p.shape[1], 26), (30, 30, 30), -1)
                cv2.putText(p, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            strip = np.hstack(panels)

            # Legend banner at the bottom with real class labels
            legend_h = 32
            legend_bar = np.full((legend_h, strip.shape[1], 3), 25, dtype=np.uint8)
            x_offset = 15
            for c in range(k_classes):
                color = palette[c % len(palette)]
                name = self.class_names[c] if (self.class_names and c < len(self.class_names)) else f"Class {c}"
                # Draw color swatch
                cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), color, -1)
                cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), (200, 200, 200), 1)
                # Draw label
                text = f" {name} "
                cv2.putText(legend_bar, text, (x_offset + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                x_offset += 24 + text_size[0] + 15

            strip = np.vstack([strip, legend_bar])

            out_path = self.save_dir / f"val_sample_{idx}.png"
            cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        except Exception:
            pass