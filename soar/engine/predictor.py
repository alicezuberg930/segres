from __future__ import annotations

import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

from ..models import SegmentationModel
from ..data import SegmentationDataset, collate_fn
from ..data.preprocess_config import PreprocessConfig
from ..losses.structure import soft_skeletonize
from ..utils import binary_mask_to_rle


def _extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return F.relu(mask - eroded)


class BasePredictor:
    """
    Dedicated inference pipeline for SOAR models.
    Executes native-resolution predictions, exports standardized multi-class semantic masks,
    generates 4-panel visual qualitative strips, and computes peer-reviewed benchmark metrics
    if ground truth annotations are present.
    """

    def __init__(
        self,
        model: nn.Module,
        data_root: str | Path,
        annotation_file: Optional[str | Path] = None,
        img_size: tuple = (2048, 2048),
        num_classes: int = 1,
        device: str = "cuda",
        num_workers: int = 2,
        threshold: float = 0.5,
        min_area: int = 0,
        close_kernel: int = 0,
        output_dir: str | Path = "predictions/soar",
        class_names: Optional[Dict[int, str]] = None,
        weights_name: str = "soar",
        preprocess_config: Optional[PreprocessConfig] = None,
    ):
        self.model = model
        self.data_root = Path(data_root).resolve()
        self.annotation_file = Path(annotation_file).resolve() if annotation_file else None
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.num_workers = num_workers
        self.threshold = threshold
        self.min_area = min_area
        self.close_kernel = close_kernel
        self.output_dir = Path(output_dir).resolve()
        self.class_names = class_names or {}
        self.weights_name = weights_name
        self.preprocess_config = preprocess_config

        self.model.to(self.device)
        self.model.eval()

        self.palette = [
            (0, 0, 255),    # Class 0: Red
            (0, 255, 0),    # Class 1: Green
            (255, 255, 0),  # Class 2: Cyan
            (0, 255, 255),  # Class 3: Yellow
            (255, 0, 255),  # Class 4: Magenta
            (0, 165, 255),  # Class 5: Orange
            (255, 128, 0),  # Class 6: Blue
            (128, 255, 0),  # Class 7: Light green
        ]

        self.dataset: Optional[SegmentationDataset] = None
        self.dataloader: Optional[DataLoader] = None

    def setup_data(self, split: str = "test", samples: Optional[int] = None) -> None:
        """Setup inference data loader with optional ground truth annotations."""
        prep_cfg = self.preprocess_config
        if prep_cfg is None:
            prep_cfg = PreprocessConfig.standard_validation(img_size=self.img_size)

        self.dataset = SegmentationDataset(
            data_root=self.data_root,
            annotation_file=str(self.annotation_file) if self.annotation_file else None,
            split=split,
            img_size=self.img_size,
            num_classes=self.num_classes,
            preprocess_config=prep_cfg,
            samples=samples,
            augment=False,
            use_cache=False,
        )

        if not self.class_names and hasattr(self.dataset, "class_names") and self.dataset.class_names:
            self.class_names = dict(self.dataset.class_names)
        if not self.class_names:
            self.class_names = {i: f"Class {i}" for i in range(self.num_classes)}

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

    @torch.no_grad()
    def predict(self, save_vis: bool = True, num_vis: int = 10) -> Dict[str, Any]:
        """
        Execute full inference pass across the dataset:
        1. Measure precise inference latency and FPS.
        2. Postprocess and export semantic mask PNGs and color overlays.
        3. If ground truth is present, accumulate and display dense semantic metrics.
        """
        if self.dataloader is None:
            self.setup_data(split="test")

        self.model.eval()

        # Output subdirectories matching the standard benchmark contract
        masks_dir = self.output_dir / "masks"
        color_masks_dir = self.output_dir / "masks_color"
        vis_dir = self.output_dir / "visualizations"
        masks_dir.mkdir(parents=True, exist_ok=True)
        color_masks_dir.mkdir(parents=True, exist_ok=True)
        if save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)

        k = self.num_classes
        has_gt = bool(getattr(self.dataset, "has_gt", False))
        accum = torch.zeros(10, k, dtype=torch.float64, device=self.device) if has_gt else None

        latencies_ms: List[float] = []
        vis_count = 0

        # Model parameter count
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        print(f"\n[SOAR Inference] Executing predictions on {len(self.dataset)} samples...")
        print(f"  - Model:         {self.weights_name} ({num_params:.2f}M params)")
        print(f"  - Resolution:    {self.img_size[0]}x{self.img_size[1]}")
        print(f"  - Classes ({k}):   {list(self.class_names.values())}")
        print(f"  - Output Dir:    {self.output_dir}")
        print(f"  - Evaluate GT:   {'Yes' if has_gt else 'No'}")

        # Warmup GPU
        if self.device.type == "cuda" and len(self.dataloader) > 0:
            warmup_img = torch.zeros((1, 3, self.img_size[0], self.img_size[1]), device=self.device)
            for _ in range(5):
                _ = self.model(warmup_img)
            torch.cuda.synchronize()

        pbar = tqdm(self.dataloader, desc="SOAR Inference", bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            valid_masks = batch.get("valid_mask")
            if valid_masks is not None:
                valid_masks = valid_masks.to(self.device, non_blocking=True)
            image_ids = batch["image_id"]
            gt_masks = batch.get("mask")
            if gt_masks is not None:
                gt_masks = gt_masks.to(self.device, non_blocking=True)

            # Precise latency measurement
            if self.device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                preds = self.model(images)
                end_evt.record()
                torch.cuda.synchronize()
                latencies_ms.append(float(start_evt.elapsed_time(end_evt)))
            else:
                t0 = time.perf_counter()
                preds = self.model(images)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            probs = torch.sigmoid(preds)
            bin_preds = (probs >= self.threshold).float()

            if valid_masks is not None:
                vmask = valid_masks.float().expand_as(bin_preds)
                bin_preds = bin_preds * vmask

            # Postprocessing & Metric Accumulation per sample in batch
            for idx, img_id in enumerate(image_ids):
                pred_np = bin_preds[idx].cpu().numpy()  # (K, H, W)
                prob_np = probs[idx].cpu().numpy()      # (K, H, W)
                img_t = images[idx].cpu()
                h, w = pred_np.shape[1], pred_np.shape[2]

                # 1. Generate Categorical 2D Mask (0 background, 1..K foreground)
                cat_mask = np.zeros((h, w), dtype=np.uint8)
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)

                for c in range(k):
                    c_mask = pred_np[c] > 0
                    if self.close_kernel > 0 or self.min_area > 0:
                        c_mask = self._morph_postprocess(c_mask)
                        pred_np[c] = c_mask.astype(np.float32)

                    cat_mask[c_mask] = c + 1
                    color = self.palette[c % len(self.palette)]
                    for ch in range(3):
                        color_mask[:, :, ch] = np.maximum(color_mask[:, :, ch], c_mask * color[ch])

                # Save raw mask PNG (categorical: pixel values 0..K)
                cv2.imwrite(str(masks_dir / f"{img_id}.png"), cat_mask)

                # Save color-coded mask PNG
                cv2.imwrite(str(color_masks_dir / f"{img_id}.png"), color_mask)

                # 2. Accumulate Ground Truth Metrics if available
                gt_np = None
                if has_gt and gt_masks is not None:
                    gt_np = gt_masks[idx].cpu().numpy()  # (K, H, W)
                    s_pred = bin_preds[idx:idx+1]
                    s_gt = gt_masks[idx:idx+1].float()
                    if valid_masks is not None:
                        s_gt = s_gt * valid_masks[idx:idx+1].float().expand_as(s_gt)

                    inter = (s_pred * s_gt).sum(dim=(0, 2, 3))
                    union = (s_pred + s_gt).clamp_max(1.0).sum(dim=(0, 2, 3))
                    pred_card = s_pred.sum(dim=(0, 2, 3))
                    gt_card = s_gt.sum(dim=(0, 2, 3))

                    b_pred = _extract_boundary(s_pred)
                    b_gt = _extract_boundary(s_gt)
                    b_inter = (b_pred * b_gt).sum(dim=(0, 2, 3))
                    b_union = (b_pred + b_gt).clamp_max(1.0).sum(dim=(0, 2, 3))

                    skel_pred = soft_skeletonize(s_pred)
                    skel_true = soft_skeletonize(s_gt)

                    accum[0] += inter
                    accum[1] += union
                    accum[2] += pred_card
                    accum[3] += gt_card
                    accum[4] += b_inter
                    accum[5] += b_union
                    accum[6] += (skel_pred * s_gt).sum(dim=(0, 2, 3))
                    accum[7] += skel_pred.sum(dim=(0, 2, 3))
                    accum[8] += (skel_true * s_pred).sum(dim=(0, 2, 3))
                    accum[9] += skel_true.sum(dim=(0, 2, 3))

                # 3. Export Qualitative Comparison Strips
                if save_vis and vis_count < num_vis:
                    self._save_vis_strip(
                        img_tensor=img_t,
                        pred_np=pred_np,
                        gt_np=gt_np,
                        color_pred=color_mask,
                        img_id=img_id,
                        out_dir=vis_dir,
                    )
                    vis_count += 1

        # Summary Metrics & Statistics
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        summary: Dict[str, Any] = {
            "model_type": "SOAR",
            "weights_name": self.weights_name,
            "params_m": round(num_params, 2),
            "resolution": f"{self.img_size[0]}x{self.img_size[1]}",
            "samples_processed": len(self.dataset),
            "latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1),
            "classes": self.class_names,
        }

        if has_gt and accum is not None:
            total_inter = accum[0]
            total_union = accum[1]
            total_pred = accum[2]
            total_gt = accum[3]
            total_b_inter = accum[4]
            total_b_union = accum[5]

            class_ious = (total_inter / total_union.clamp_min(1e-7)).cpu().tolist()
            class_dices = ((2.0 * total_inter) / (total_pred + total_gt).clamp_min(1e-7)).cpu().tolist()
            class_prec = (total_inter / total_pred.clamp_min(1e-7)).cpu().tolist()
            class_recall = (total_inter / total_gt.clamp_min(1e-7)).cpu().tolist()
            class_biou = (total_b_inter / total_b_union.clamp_min(1e-7)).cpu().tolist()

            t_prec = (accum[6] + 1e-7) / (accum[7] + 1e-7)
            t_sens = (accum[8] + 1e-7) / (accum[9] + 1e-7)
            class_cldice = ((2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-7)).cpu().tolist()

            present_mask = (total_gt > 0).cpu().numpy()
            if present_mask.any():
                m_iou = float(np.mean([class_ious[c] for c in range(k) if present_mask[c]]))
                m_dice = float(np.mean([class_dices[c] for c in range(k) if present_mask[c]]))
                m_prec = float(np.mean([class_prec[c] for c in range(k) if present_mask[c]]))
                m_rec = float(np.mean([class_recall[c] for c in range(k) if present_mask[c]]))
                m_biou = float(np.mean([class_biou[c] for c in range(k) if present_mask[c]]))
                m_cldice = float(np.mean([class_cldice[c] for c in range(k) if present_mask[c]]))
            else:
                m_iou = float(np.mean(class_ious))
                m_dice = float(np.mean(class_dices))
                m_prec = float(np.mean(class_prec))
                m_rec = float(np.mean(class_recall))
                m_biou = float(np.mean(class_biou))
                m_cldice = float(np.mean(class_cldice))

            summary.update({
                "mIoU": round(m_iou * 100.0, 2),
                "Dice": round(m_dice * 100.0, 2),
                "clDice": round(m_cldice * 100.0, 2),
                "Boundary_IoU": round(m_biou * 100.0, 2),
                "Precision": round(m_prec * 100.0, 2),
                "Recall": round(m_rec * 100.0, 2),
                "per_class": {
                    self.class_names.get(c, f"Class {c}"): {
                        "IoU": round(class_ious[c] * 100.0, 2),
                        "Dice": round(class_dices[c] * 100.0, 2),
                        "clDice": round(class_cldice[c] * 100.0, 2),
                        "GT_Pixels": int(total_gt[c].item()),
                    }
                    for c in range(k)
                },
            })

            # Print formatted evaluation table
            self._print_tabular_summary(summary)

        # Dump summary.json
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[SOAR Inference Complete] Results saved to: {self.output_dir}")
        print(f"  - Semantic Masks:    {masks_dir}")
        print(f"  - Color Overlays:    {color_masks_dir}")
        if save_vis:
            print(f"  - Strip Previews:    {vis_dir}")
        print(f"  - Summary JSON:      {self.output_dir / 'summary.json'}")

        return summary

    def _morph_postprocess(self, bin_mask: np.ndarray) -> np.ndarray:
        """Morphological closing and small-object filtering."""
        u8 = bin_mask.astype(np.uint8)
        if self.close_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
        if self.min_area > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA]
            valid = np.where((areas >= self.min_area) & (np.arange(n) > 0))[0]
            u8 = np.isin(labels, valid).astype(np.uint8)
        return u8 > 0

    def _save_vis_strip(
        self,
        img_tensor: torch.Tensor,
        pred_np: np.ndarray,
        gt_np: Optional[np.ndarray],
        color_pred: np.ndarray,
        img_id: str,
        out_dir: Path,
    ) -> None:
        """Render standard 4-panel visual strip with semantic legend bar."""
        preview_size = (512, 512)
        k = pred_np.shape[0]

        # Convert input image to BGR uint8
        img_raw = img_tensor.float().numpy()
        if img_raw.ndim == 2:
            img_raw = np.stack([img_raw] * 3, axis=0)
        elif img_raw.shape[0] == 1:
            img_raw = np.repeat(img_raw, 3, axis=0)
        img_raw = np.transpose(img_raw, (1, 2, 0))
        if img_raw.max() <= 1.5 and img_raw.min() >= -0.5:
            img_raw = np.clip((img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-6) * 255.0, 0, 255)
        img_u8 = img_raw.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        img_small = cv2.resize(img_bgr, preview_size, interpolation=cv2.INTER_AREA)

        # Ground Truth panel
        gt_small = np.zeros_like(img_small)
        if gt_np is not None:
            for c in range(k):
                color = self.palette[c % len(self.palette)]
                m_gt = cv2.resize((gt_np[c] > 0).astype(np.uint8), preview_size, interpolation=cv2.INTER_NEAREST) > 0
                for ch in range(3):
                    gt_small[:, :, ch] = np.maximum(gt_small[:, :, ch], m_gt * color[ch])

        # Prediction panel
        pred_small = cv2.resize(color_pred, preview_size, interpolation=cv2.INTER_NEAREST)

        # Overlay panel
        overlay = img_small.copy()
        p_active = np.any(pred_small > 0, axis=-1)
        overlay[p_active] = cv2.addWeighted(img_small, 0.45, pred_small, 0.55, 0)[p_active]

        panels = [img_small, gt_small, pred_small, overlay]
        titles = ["Input Image", "Ground Truth", f"SOAR ({self.weights_name})", "Overlay"]
        for p, title in zip(panels, titles):
            cv2.rectangle(p, (0, 0), (p.shape[1], 26), (30, 30, 30), -1)
            cv2.putText(p, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        strip = np.hstack(panels)

        # Bottom legend banner
        legend_h = 32
        legend_bar = np.full((legend_h, strip.shape[1], 3), 25, dtype=np.uint8)
        x_offset = 15
        for c in range(k):
            color = self.palette[c % len(self.palette)]
            name = self.class_names.get(c, f"Class {c}")
            cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), color, -1)
            cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), (200, 200, 200), 1)
            text = f" {name} "
            cv2.putText(legend_bar, text, (x_offset + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_offset += 24 + text_size[0] + 15

        strip = np.vstack([strip, legend_bar])
        cv2.imwrite(str(out_dir / f"strip_{img_id}.png"), strip, [cv2.IMWRITE_PNG_COMPRESSION, 2])

    def _print_tabular_summary(self, summary: Dict[str, Any]) -> None:
        """Print paper-ready comparison summary."""
        print(f"\n{'='*105}")
        print(f"{'Model Architecture':<22} {'Samples':<8} {'mIoU (%)':<10} {'Dice (%)':<10} {'clDice (%)':<12} {'bIoU (%)':<10} {'Latency':<10} {'FPS':<8}")
        print(f"{'-'*105}")
        m_name = f"SOAR ({summary['weights_name']})"
        print(f"{m_name:<22} {summary['samples_processed']:<8} {summary.get('mIoU', 0.0):<10.2f} {summary.get('Dice', 0.0):<10.2f} {summary.get('clDice', 0.0):<12.2f} {summary.get('Boundary_IoU', 0.0):<10.2f} {summary['latency_ms']:<8.2f}ms {summary['fps']:<8.1f}")
        print(f"{'='*105}")

        print("Per-Class Semantic Breakdown:")
        for name, stats in summary.get("per_class", {}).items():
            gt_px = stats["GT_Pixels"]
            tag = f"({gt_px} px)" if gt_px > 0 else "(no GT in split)"
            print(f"  - {name:<16}: IoU = {stats['IoU']:5.2f}% | Dice = {stats['Dice']:5.2f}% | clDice = {stats['clDice']:5.2f}% {tag}")
        print()
