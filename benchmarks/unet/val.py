from __future__ import annotations

import argparse
from pathlib import Path
from .model import UNet
from ..common.predictor import BenchmarkPredictor


def validate(
    weights: str,
    data: str,
    split: str = "val",
    imgsz: int = 1024,
    num_classes: int = 1,
    device: str = "cuda",
    output_dir: str = "evaluations/benchmarks",
    base_channels: int = 64,
):
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
    predictor = BenchmarkPredictor(
        model=model,
        weights_path=weights,
        model_name="UNet",
        data_root=data,
        img_size=(imgsz, imgsz),
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
    )
    predictor.setup_data(split=split)
    return predictor.predict(save_strips=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate U-Net Benchmark Baseline")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory or data.yaml manifest")
    parser.add_argument("--split", type=str, default="val", help="Split to validate (val)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Validation resolution (1024 or 2048)")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of foreground classes")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="evaluations/benchmarks", help="Output directory")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channel capacity")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        num_classes=args.num_classes,
        device=args.device,
        output_dir=args.output_dir,
        base_channels=args.base_channels,
    )
