from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .predict import BaseYOLOPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo",
        description="YOLO Inference & Semantic Benchmark Evaluation Pipeline for Ultra-High-Resolution Vision",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")

    # Predict Subparser (Dedicated inference pipeline matching SOAR identically)
    p_pred = subparsers.add_parser("predict", help="Run YOLO inference with identical outputs and semantic benchmarks as SOAR")
    p_pred.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights (e.g. best.pt)")
    p_pred.add_argument("--data", type=str, required=True, help="Path to test images or dataset directory")
    p_pred.add_argument("--annotation-file", type=str, default=None, help="Optional COCO JSON annotation file for ground-truth semantic metric evaluation")
    p_pred.add_argument("--imgsz", "--img-size", type=int, nargs="+", default=[2048, 2048], dest="img_size", help="Inference resolution [H, W] or single int")
    p_pred.add_argument("--split", type=str, default="test", help="Dataset split ('test', 'val', or 'train')")
    p_pred.add_argument("--samples", type=int, default=None, help="Limit to first N samples for quick benchmarking")
    p_pred.add_argument("--device", type=str, default="cuda", help="Target device ('cuda' or 'cpu')")
    p_pred.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for instance predictions")
    p_pred.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p_pred.add_argument("--output-dir", type=str, default="predictions/yolo", help="Directory to save masks, color overlays, visual strips, and summary.json")
    p_pred.add_argument("--num-classes", type=int, default=4, help="Number of foreground classes")
    p_pred.add_argument("--num-vis", type=int, default=10, help="Number of 4-panel visual comparison strips to save")
    p_pred.add_argument("--no-vis", action="store_true", help="Disable visual strip generation")

    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "predict":
        img_size = parsed_args.img_size
        if isinstance(img_size, (list, tuple)):
            img_sz = (int(img_size[0]), int(img_size[1])) if len(img_size) >= 2 else (int(img_size[0]), int(img_size[0]))
        else:
            img_sz = (int(img_size), int(img_size))

        predictor = BaseYOLOPredictor(
            weights=parsed_args.weights,
            data_root=parsed_args.data,
            annotation_file=parsed_args.annotation_file,
            img_size=img_sz,
            num_classes=parsed_args.num_classes,
            device=parsed_args.device,
            num_workers=parsed_args.workers,
            conf_threshold=parsed_args.conf,
            iou_threshold=parsed_args.iou,
            output_dir=parsed_args.output_dir,
        )
        predictor.setup_data(split=parsed_args.split, samples=parsed_args.samples)
        predictor.predict(save_vis=not parsed_args.no_vis, num_vis=parsed_args.num_vis)


if __name__ == "__main__":
    main()
