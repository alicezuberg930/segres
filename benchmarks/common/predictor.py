from __future__ import annotations

import cv2
import json
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


class BenchmarkPredictor:
    """
    Dedicated inference pipeline for benchmark baseline models.
    Produces identical deliverables and metrics as SOAR:
      1. Raw predicted semantic masks (PNGs)
      2. Color overlay visualizations (PNGs)
      3. 4-panel qualitative visual strips [Input | GT | Prediction | Overlay]
      4. Standardized benchmark_summary.json with mIoU, Dice, bIoU, clDice, Latency, FPS, and LaTeX row.
    """

    def __init__(
        self,
        model: nn.Module,
        weights_path: str | Path,
        model_name: str,
        data_root: str | Path,
        annotation_file: Optional[str | Path] = None,
        mask_dir: Optional[str | Path] = None,
        img_size: Tuple[int, int] = (1024, 1024),
        num_classes: Optional[int] = None,
        device: str = "cuda",
        output_dir: str | Path = "predictions/benchmarks",
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model = model
        self.weights_path = Path(weights_path).resolve()
        self.model_name = model_name
        self.data_root = Path(data_root).resolve()
        self.annotation_file = Path(annotation_file).resolve() if annotation_file else None
        self.mask_dir = Path(mask_dir).resolve() if mask_dir else None
        self.img_size = img_size
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.output_dir = Path(output_dir).resolve() / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output folders
        self.masks_dir = self.output_dir / "masks"
        self.overlays_dir = self.output_dir / "overlays"
        self.vis_dir = self.output_dir / "visualizations"
        for d in (self.masks_dir, self.overlays_dir, self.vis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if available
        self.class_names = class_names or {}
        self.num_classes = num_classes or 1
        if self.weights_path.is_file():
            self.load_weights()

        self.model.to(self.device)
        self.model.eval()

        # Color palette for classes (BGR)
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

    def load_weights(self) -> None:
        """Load trained model weights from checkpoint."""
        ckpt = torch.load(self.weights_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "num_classes" in ckpt:
                self.num_classes = ckpt["num_classes"]
            if "class_names" in ckpt and ckpt["class_names"]:
                self.class_names = ckpt["class_names"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, nn.Module):
            self.model = ckpt
        else:
            self.model.load_state_dict(ckpt)
        print(f"[{self.model_name}] Successfully loaded weights from: {self.weights_path}")

    def setup_data(self, split: str = "val", samples: Optional[int] = None) -> None:
        """Initialize inference dataset."""
        cfg = None
        try:
            cfg = DatasetConfig.resolve(self.data_root)
            if self.num_classes is None and cfg and cfg.nc:
                self.num_classes = cfg.nc
            if not self.class_names and cfg and cfg.names:
                self.class_names = cfg.names
        except Exception:
            cfg = None



        ann_path = cfg.annotation_files.get(split) if (cfg and cfg.annotation_files) else str(self.annotation_file) if self.annotation_file else None
        masks_path = cfg.mask_dirs.get(split) if (cfg and cfg.mask_dirs) else str(self.mask_dir) if self.mask_dir else None
        img_dir = getattr(cfg, f"{split}_images", None) if cfg else None
        img_list = getattr(cfg, f"{split}_image_list", None) if cfg else None

        self.dataset = SegmentationDataset(
            data_root=cfg.root_path if cfg else self.data_root,
            annotation_file=str(ann_path) if ann_path else None,
            mask_dir=str(masks_path) if masks_path else None,
            split=split,
            img_size=self.img_size,
            num_classes=self.num_classes,
            samples=samples,
            augment=False,
            image_dir=img_dir,
            image_files=img_list,
            names=self.class_names,
        )

        if not self.class_names and hasattr(self.dataset, "class_names") and self.dataset.class_names:
            self.class_names = self.dataset.class_names
        if not self.class_names:
            self.class_names = {i: f"Class_{i}" for i in range(self.num_classes)}

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

    @torch.no_grad()
    def predict(self, save_strips: int = 10) -> Dict[str, Any]:
        """
        Execute full inference pass across the dataset:
          - Measures latency and FPS
          - Saves raw masks and overlays
          - Saves visual strips
          - Computes mIoU, Dice, bIoU, clDice
        """
        if not hasattr(self, "dataloader") or self.dataloader is None:
            self.setup_data()

        accumulator = BenchmarkMetricAccumulator(
            num_classes=self.num_classes,
            class_names=self.class_names,
            device=self.device,
        )

        print(f"\n[{self.model_name}] Starting Inference on {len(self.dataset)} samples...")

        # Warm-up GPU
        dummy = torch.zeros((1, 3, *self.img_size), device=self.device)
        for _ in range(5):
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_times: List[float] = []
        pbar = tqdm(self.dataloader, desc=f"Predicting [{self.model_name}]", bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")

        for idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            if "image_id" in batch and batch["image_id"]:
                raw_id = batch["image_id"][0] if isinstance(batch["image_id"], (list, tuple)) else batch["image_id"]
                stem = Path(str(raw_id)).stem
            else:
                stem = f"sample_{idx:04d}"

            # Latency measurement
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = self.model(images)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                logits = self.model(images)
                t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000.0)
            probs = torch.sigmoid(logits)

            # Update benchmark metrics
            accumulator.update(probs, masks)

            # Rasterize prediction
            bin_preds = (probs[0] >= 0.5).cpu().numpy().astype(np.uint8)  # [C, H, W]
            gt_masks = masks[0].cpu().numpy().astype(np.uint8)            # [C, H, W]
            orig_img = (images[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

            # Save multi-class semantic mask
            h, w = self.img_size
            pred_canvas = np.zeros((h, w), dtype=np.uint8)
            for c in range(self.num_classes):
                pred_canvas[bin_preds[c] == 1] = (c + 1)
            cv2.imwrite(str(self.masks_dir / f"{stem}.png"), pred_canvas)

            # Save color overlay
            overlay = orig_img.copy()
            for c in range(self.num_classes):
                color = self.palette[c % len(self.palette)]
                c_mask = (bin_preds[c] == 1)
                overlay[c_mask] = (overlay[c_mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
            cv2.imwrite(str(self.overlays_dir / f"{stem}.png"), overlay)

            # Save visual strips for first N samples
            if idx < save_strips:
                self.save_visual_strip(orig_img, gt_masks, bin_preds, overlay, stem)

        # Compute summary
        bench_metrics = accumulator.compute()
        mean_latency = float(np.mean(inference_times[5:])) if len(inference_times) > 5 else float(np.mean(inference_times))
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        summary: Dict[str, Any] = {
            "model_name": self.model_name,
            "weights": str(self.weights_path),
            "params_m": round(num_params, 2),
            "img_size": list(self.img_size),
            "mIoU": round(bench_metrics["mIoU"] * 100, 2),
            "Dice": round(bench_metrics["Dice"] * 100, 2),
            "bIoU": round(bench_metrics["bIoU"] * 100, 2),
            "clDice": round(bench_metrics["clDice"] * 100, 2),
            "latency_ms": round(mean_latency, 2),
            "fps": round(fps, 1),
            "per_class": bench_metrics["per_class"],
            "latex_row": (
                f"{self.model_name} & Baseline & {num_params:.2f} & -- & "
                f"{bench_metrics['mIoU']*100:.1f} & {bench_metrics['Dice']*100:.1f} & "
                f"{bench_metrics['bIoU']*100:.1f} & {bench_metrics['clDice']*100:.1f} & "
                f"{mean_latency:.1f} & {fps:.1f} \\\\"
            ),
        }

        # Save summary JSON
        with open(self.output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULTS: {self.model_name}")
        print(f"  - Parameters: {num_params:.2f} M")
        print(f"  - Resolution: {self.img_size[0]}x{self.img_size[1]}")
        print(f"  - mIoU:       {summary['mIoU']}%")
        print(f"  - Dice:       {summary['Dice']}%")
        print(f"  - bIoU:       {summary['bIoU']}%")
        print(f"  - clDice:     {summary['clDice']}%")
        print(f"  - Latency:    {summary['latency_ms']} ms/frame ({summary['fps']} FPS)")
        print(f"  - LaTeX Row:  {summary['latex_row']}")
        print(f"Deliverables exported to: {self.output_dir}")
        print("=" * 70 + "\n")

        return summary

    def save_visual_strip(self, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, overlay: np.ndarray, stem: str) -> None:
        """Render 4-panel visual qualitative strip: [Input | Ground Truth | Prediction | Overlay]."""
        h, w = img.shape[:2]

        gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for c in range(self.num_classes):
            color = self.palette[c % len(self.palette)]
            gt_rgb[gt[c] == 1] = color
            pred_rgb[pred[c] == 1] = color

        panels = [img, gt_rgb, pred_rgb, overlay]
        titles = ["Input Image", "Ground Truth", f"Prediction ({self.model_name})", "Overlay"]

        # Add title headers
        labeled_panels = []
        for p, title in zip(panels, titles):
            panel_img = p.copy()
            cv2.rectangle(panel_img, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(panel_img, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            labeled_panels.append(panel_img)

        # Concatenate horizontally
        strip = np.hstack(labeled_panels)

        # Downsample visual strip to 2048 width if huge
        if strip.shape[1] > 2048:
            scale = 2048 / strip.shape[1]
            new_h = int(strip.shape[0] * scale)
            strip = cv2.resize(strip, (2048, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(self.vis_dir / f"strip_{stem}.png"), strip)
