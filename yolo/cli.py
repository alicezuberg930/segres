from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .converter import convert_coco_to_yolo_seg
from .train import train_yolo
from .val_semantic import evaluate_yolo_semantic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo",
        description="YOLO Instance & Semantic Segmentation Benchmark Pipeline for Ultra-High-Resolution Vision",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")

    # 1. Prepare Dataset Subparser
    p_prep = subparsers.add_parser("prepare", help="Convert COCO JSON annotations to YOLO segmentation format")
    p_prep.add_argument("--annotation-file", type=str, required=True, help="Path to COCO JSON annotation file")
    p_prep.add_argument("--data", type=str, required=True, help="Directory containing images")
    p_prep.add_argument("--output-dir", type=str, default="dataset_yolo", help="Output directory for YOLO labels and data.yaml")
    p_prep.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction (0.0 to 1.0)")
    p_prep.add_argument("--samples", type=int, default=None, help="Limit to first N samples for few-shot/sanity benchmarking")
    p_prep.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # 2. Train Subparser
    p_train = subparsers.add_parser("train", help="Train a YOLO segmentation model")
    p_train.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLO model checkpoint or config (e.g. yolov8n-seg.pt, yolo11n-seg.pt)")
    p_train.add_argument("--data", type=str, default="dataset_yolo/data.yaml", help="Path to dataset data.yaml")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_train.add_argument("--imgsz", "--img-size", type=int, default=2048, dest="imgsz", help="Input image resolution")
    p_train.add_argument("--batch", type=int, default=1, help="Physical batch size")
    p_train.add_argument("--lr", type=float, default=1e-4, dest="lr0", help="Initial learning rate")
    p_train.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    p_train.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    p_train.add_argument("--no-augment", action="store_true", help="Disable all data augmentations for fair comparison with SOAR")
    p_train.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    p_train.add_argument("--no-amp", action="store_false", dest="amp", help="Disable Automatic Mixed Precision")
    p_train.add_argument("--project", type=str, default="checkpoints/yolo_benchmark", help="Output directory for checkpoints")
    p_train.add_argument("--name", type=str, default="exp", help="Run experiment name")

    # 3. Validate Subparser (Dense Semantic Metric Evaluation)
    p_val = subparsers.add_parser("val", help="Evaluate trained YOLO model on dense semantic metrics (mIoU, clDice, bIoU)")
    p_val.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights (e.g. best.pt)")
    p_val.add_argument("--data", type=str, default="dataset_yolo/data.yaml", help="Path to dataset data.yaml")
    p_val.add_argument("--imgsz", "--img-size", type=int, default=2048, dest="imgsz", help="Input image resolution")
    p_val.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    p_val.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for instance predictions")
    p_val.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p_val.add_argument("--save-dir", type=str, default=None, help="Directory to save visual comparison strips")
    p_val.add_argument("--num-vis", type=int, default=4, help="Number of qualitative visual strips to save")

    # 4. Predict Subparser (Dedicated inference pipeline matching SOAR identically)
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

    # 5. End-to-End Benchmark Subparser
    p_bench = subparsers.add_parser("benchmark", help="End-to-end one-command prepare, train, and semantic evaluation")
    p_bench.add_argument("--annotation-file", type=str, required=True, help="Path to COCO JSON annotation file")
    p_bench.add_argument("--data", type=str, required=True, help="Directory containing raw images")
    p_bench.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLO model checkpoint (e.g. yolov8n-seg.pt, yolo11n-seg.pt)")
    p_bench.add_argument("--output-dir", type=str, default="dataset_yolo", help="Directory to generate YOLO dataset")
    p_bench.add_argument("--samples", type=int, default=None, help="Limit to first N samples for exact fair comparison with SOAR")
    p_bench.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p_bench.add_argument("--imgsz", "--img-size", type=int, default=2048, dest="imgsz", help="Input image resolution")
    p_bench.add_argument("--batch", type=int, default=1, help="Physical batch size")
    p_bench.add_argument("--lr", type=float, default=1e-4, dest="lr0", help="Initial learning rate")
    p_bench.add_argument("--device", type=str, default="cuda", help="Target device")
    p_bench.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p_bench.add_argument("--no-augment", action="store_true", help="Disable all augmentations")
    p_bench.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    p_bench.add_argument("--project", type=str, default="checkpoints/yolo_benchmark", help="Output directory")
    p_bench.add_argument("--name", type=str, default="exp", help="Run experiment name")

    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "prepare":
        convert_coco_to_yolo_seg(
            coco_json_path=parsed_args.annotation_file,
            image_dir=parsed_args.data,
            output_dir=parsed_args.output_dir,
            val_split=parsed_args.val_split,
            samples=parsed_args.samples,
            seed=parsed_args.seed,
        )

    elif parsed_args.command == "train":
        train_yolo(
            model=parsed_args.model,
            data=parsed_args.data,
            epochs=parsed_args.epochs,
            imgsz=parsed_args.imgsz,
            batch=parsed_args.batch,
            lr0=parsed_args.lr0,
            device=parsed_args.device,
            workers=parsed_args.workers,
            amp=parsed_args.amp,
            augment=not parsed_args.no_augment,
            project=parsed_args.project,
            name=parsed_args.name,
            eval_semantic=True,
        )

    elif parsed_args.command == "val":
        evaluate_yolo_semantic(
            weights_path=parsed_args.weights,
            data_yaml_path=parsed_args.data,
            img_size=parsed_args.imgsz,
            conf_thresh=parsed_args.conf,
            iou_thresh=parsed_args.iou,
            device=parsed_args.device,
            save_dir=parsed_args.save_dir,
            num_vis_samples=parsed_args.num_vis,
        )

    elif parsed_args.command == "predict":
        from .predict import BaseYOLOPredictor

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

    elif parsed_args.command == "benchmark":
        print("\n=== Step 1: Converting Dataset to YOLO Segmentation Format ===")
        yaml_path = convert_coco_to_yolo_seg(
            coco_json_path=parsed_args.annotation_file,
            image_dir=parsed_args.data,
            output_dir=parsed_args.output_dir,
            val_split=0.0 if parsed_args.samples else 0.1,
            samples=parsed_args.samples,
        )

        print("\n=== Step 2: Training YOLO Model ===")
        train_yolo(
            model=parsed_args.model,
            data=yaml_path,
            epochs=parsed_args.epochs,
            imgsz=parsed_args.imgsz,
            batch=parsed_args.batch,
            lr0=parsed_args.lr0,
            device=parsed_args.device,
            workers=parsed_args.workers,
            amp=parsed_args.amp,
            augment=not parsed_args.no_augment,
            project=parsed_args.project,
            name=parsed_args.name,
            eval_semantic=True,
        )


if __name__ == "__main__":
    main()
