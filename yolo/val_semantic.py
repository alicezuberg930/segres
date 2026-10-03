from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from soar.losses.structure import soft_skeletonize


def _extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return F.relu(mask - eroded)


def evaluate_yolo_semantic(
    weights_path: str | Path,
    data_yaml_path: str | Path,
    img_size: int = 2048,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    device: str = "cuda",
    save_dir: Optional[str | Path] = None,
    num_vis_samples: int = 4,
) -> Dict[str, Any]:
    """
    Evaluate a trained YOLO segmentation model on Dense Semantic Segmentation metrics
    (mIoU, Dice, Precision, Recall, Boundary IoU, clDice) for direct, fair comparison with SOAR.

    Args:
        weights_path: Path to trained YOLO weights (e.g. best.pt).
        data_yaml_path: Path to dataset data.yaml.
        img_size: Native inference image resolution.
        conf_thresh: Confidence threshold for instance detections.
        iou_thresh: NMS IoU threshold.
        device: 'cuda' or 'cpu'.
        save_dir: Optional directory to save qualitative comparison strips.
        num_vis_samples: Number of sample visualization strips to generate.

    Returns:
        Dictionary containing benchmark metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for YOLO evaluation. Install it with: pip install ultralytics"
        )

    import yaml
    weights_path = Path(weights_path).resolve()
    data_yaml_path = Path(data_yaml_path).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    raw_names = data_cfg.get("names", {0: "filament"})
    if isinstance(raw_names, list):
        class_names = {idx: name for idx, name in enumerate(raw_names)}
    else:
        class_names = {int(k): str(v) for k, v in raw_names.items()}
    num_classes = len(class_names)

    # Resolve validation image paths from manifest or directory
    val_source = data_cfg.get("val")
    val_image_paths: List[Path] = []
    if isinstance(val_source, str) and val_source.endswith(".txt"):
        val_txt = Path(val_source)
        if not val_txt.is_absolute():
            val_txt = Path(data_cfg.get("path", ".")) / val_source
        with open(val_txt, "r", encoding="utf-8") as f:
            val_image_paths = [Path(line.strip()) for line in f if line.strip()]
    elif val_source:
        v_dir = Path(val_source)
        if not v_dir.is_absolute():
            v_dir = Path(data_cfg.get("path", ".")) / val_source
        val_image_paths = sorted([p for p in v_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    if not val_image_paths:
        raise RuntimeError("No validation images found in data.yaml specification.")

    print(f"\n[YOLO Semantic Benchmark] Initializing evaluation on {len(val_image_paths)} samples...")
    print(f"  - Model:         {weights_path.name}")
    print(f"  - Resolution:    {img_size}x{img_size}")
    print(f"  - Classes ({num_classes}): {list(class_names.values())}")

    model = YOLO(str(weights_path))

    # Metric accumulators: shape (10, num_classes)
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    accum = torch.zeros(10, num_classes, dtype=torch.float64, device=dev)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    vis_saved = 0
    palette = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
    ]

    pbar = tqdm(val_image_paths, desc="Benchmarking YOLO Semantic", bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")
    for img_path in pbar:
        # 1. Run YOLO inference
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )
        res = results[0]

        # 2. Reconstruct Ground Truth mask from YOLO label txt
        # Path convention: /images/ -> /labels/, .jpg -> .txt
        img_stem = img_path.stem
        parent_dir = img_path.parent
        label_candidates = [
            parent_dir.parent / "labels" / "val" / f"{img_stem}.txt",
            parent_dir.parent / "labels" / "train" / f"{img_stem}.txt",
            parent_dir.parent / "labels" / f"{img_stem}.txt",
            Path(data_cfg.get("path", ".")) / "labels" / "val" / f"{img_stem}.txt",
            Path(data_cfg.get("path", ".")) / "labels" / "train" / f"{img_stem}.txt",
        ]
        label_file = next((cand for cand in label_candidates if cand.is_file()), None)

        orig_h, orig_w = res.orig_shape if hasattr(res, "orig_shape") else (img_size, img_size)
        gt_mask = np.zeros((num_classes, orig_h, orig_w), dtype=np.float32)

        if label_file and label_file.is_file():
            with open(label_file, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 7:  # class + at least 3 points
                        cid = int(parts[0])
                        if 0 <= cid < num_classes:
                            coords = [float(p) for p in parts[1:]]
                            pts = []
                            for i in range(0, len(coords) - 1, 2):
                                px = int(coords[i] * orig_w)
                                py = int(coords[i + 1] * orig_h)
                                pts.append([px, py])
                            if len(pts) >= 3:
                                pts_np = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                                cv2.fillPoly(gt_mask[cid], [pts_np], color=1.0)

        # 3. Rasterize YOLO predicted instances into multi-class semantic tensor
        pred_mask = np.zeros((num_classes, orig_h, orig_w), dtype=np.float32)
        if res.masks is not None and res.boxes is not None:
            classes = res.boxes.cls.cpu().numpy().astype(int)
            masks_data = res.masks.data.cpu().numpy()  # shape (N, H_mask, W_mask)

            for i, cid in enumerate(classes):
                if 0 <= cid < num_classes:
                    m = masks_data[i]
                    if m.shape[:2] != (orig_h, orig_w):
                        m = cv2.resize(m.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    pred_mask[cid] = np.maximum(pred_mask[cid], (m >= 0.5).astype(np.float32))

        # 4. Compute metrics using PyTorch streaming accumulation
        gt_t = torch.from_numpy(gt_mask).unsqueeze(0).to(dev)        # (1, C, H, W)
        pred_t = torch.from_numpy(pred_mask).unsqueeze(0).to(dev)    # (1, C, H, W)

        inter = (pred_t * gt_t).sum(dim=(0, 2, 3))
        union = (pred_t + gt_t).clamp_max(1.0).sum(dim=(0, 2, 3))
        pred_card = pred_t.sum(dim=(0, 2, 3))
        gt_card = gt_t.sum(dim=(0, 2, 3))

        b_gt = _extract_boundary(gt_t)
        b_pred = _extract_boundary(pred_t)
        b_inter = (b_pred * b_gt).sum(dim=(0, 2, 3))
        b_union = (b_pred + b_gt).clamp_max(1.0).sum(dim=(0, 2, 3))

        skel_pred = soft_skeletonize(pred_t)
        skel_true = soft_skeletonize(gt_t)
        skel_prec_inter = (skel_pred * gt_t).sum(dim=(0, 2, 3))
        skel_prec_total = skel_pred.sum(dim=(0, 2, 3))
        skel_sens_inter = (skel_true * pred_t).sum(dim=(0, 2, 3))
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

        # 5. Optional qualitative visual strip export
        if save_dir and vis_saved < num_vis_samples:
            _save_yolo_comparison_strip(
                img_path=img_path,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                class_names=class_names,
                palette=palette,
                save_path=save_dir / f"yolo_sample_{vis_saved}.png",
            )
            vis_saved += 1

    # 6. Aggregate final benchmark metrics
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

    metrics = {
        "mIoU": float(iou),
        "Dice": float(dice),
        "Precision": float(prec),
        "Recall": float(recall),
        "Boundary_IoU": float(boundary_iou),
        "clDice": float(cldice),
        "class_ious": class_ious,
        "class_dices": class_dices,
        "class_gt": total_gt.cpu().tolist(),
        "class_names": class_names,
    }

    # 7. Print identical tabular summary for paper benchmarking
    print(f"\n{'-'*95}")
    print(f"{'Model / Architecture':<22} {'Samples':<8} {'mIoU':<10} {'Dice':<10} {'Prec':<10} {'Recall':<10} {'bIoU':<10} {'clDice':<10}")
    print(f"{'-'*95}")
    m_name = f"YOLO ({weights_path.stem})"
    print(f"{m_name:<22} {len(val_image_paths):<8} {iou:<10.4f} {dice:<10.4f} {prec:<10.4f} {recall:<10.4f} {boundary_iou:<10.4f} {cldice:<10.4f}")
    print(f"{'-'*95}")
    print("Per-class IoU breakdown:")
    for c in range(num_classes):
        name = class_names.get(c, f"Class {c}")
        gt_px = int(total_gt[c].item())
        gt_info = f" ({gt_px} px)" if gt_px > 0 else " (no GT in split)"
        print(f"  {name:<15}: {class_ious[c]:.4f}{gt_info}")
    print(f"{'-'*95}\n")

    return metrics


def _save_yolo_comparison_strip(
    img_path: Path,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_names: Dict[int, str],
    palette: List[Tuple[int, int, int]],
    save_path: Path,
):
    """Render 4-panel comparison strip with class swatches via OpenCV."""
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        return

    preview_size = (512, 512)
    img_bgr = cv2.resize(raw_img, preview_size, interpolation=cv2.INTER_AREA)

    k_classes = gt_mask.shape[0]
    gt_colored = np.zeros_like(img_bgr)
    pred_colored = np.zeros_like(img_bgr)

    for c in range(k_classes):
        color = palette[c % len(palette)]
        m_c = cv2.resize(gt_mask[c], preview_size, interpolation=cv2.INTER_NEAREST) > 0.5
        p_c = cv2.resize(pred_mask[c], preview_size, interpolation=cv2.INTER_NEAREST) > 0.5
        for ch in range(3):
            gt_colored[:, :, ch] = np.maximum(gt_colored[:, :, ch], m_c * color[ch])
            pred_colored[:, :, ch] = np.maximum(pred_colored[:, :, ch], p_c * color[ch])

    overlay = img_bgr.copy()
    p_any = np.any(pred_colored > 0, axis=-1)
    overlay[p_any] = cv2.addWeighted(img_bgr, 0.5, pred_colored, 0.5, 0)[p_any]

    panels = [img_bgr, gt_colored, pred_colored, overlay]
    titles = ["Input Image", "Ground Truth", "YOLO Prediction", "Overlay"]
    for p, title in zip(panels, titles):
        cv2.rectangle(p, (0, 0), (p.shape[1], 26), (30, 30, 30), -1)
        cv2.putText(p, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    strip = np.hstack(panels)

    # Legend banner at the bottom
    legend_h = 32
    legend_bar = np.full((legend_h, strip.shape[1], 3), 25, dtype=np.uint8)
    x_offset = 15
    for c in range(k_classes):
        color = palette[c % len(palette)]
        name = class_names.get(c, f"Class {c}")
        cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), color, -1)
        cv2.rectangle(legend_bar, (x_offset, 8), (x_offset + 16, 24), (200, 200, 200), 1)
        text = f" {name} "
        cv2.putText(legend_bar, text, (x_offset + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_offset += 24 + text_size[0] + 15

    strip = np.vstack([strip, legend_bar])
    cv2.imwrite(str(save_path), strip, [cv2.IMWRITE_PNG_COMPRESSION, 2])
