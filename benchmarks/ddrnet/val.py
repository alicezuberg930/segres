from __future__ import annotations

import argparse
from .model import DDRNet
from ..common.predictor import BenchmarkPredictor


def validate(
    weights: str,
    data: str,
    split: str = "val",
    imgsz: int = 1024,
    num_classes: int = 1,
    device: str = "cuda",
    output_dir: str = "evaluations/benchmarks",
    variant: str = "slim",
):
    model = DDRNet(in_channels=3, num_classes=num_classes, variant=variant)
    model_name = f"DDRNet_23_{variant}"
    predictor = BenchmarkPredictor(
        model=model,
        weights_path=weights,
        model_name=model_name,
        data_root=data,
        img_size=(imgsz, imgsz),
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
    )
    predictor.setup_data(split=split)
    return predictor.predict(save_strips=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DDRNet Benchmark Baseline")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory or data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Split")
    parser.add_argument("--imgsz", type=int, default=1024, help="Resolution")
    parser.add_argument("--num-classes", type=int, default=1, help="Classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output-dir", type=str, default="evaluations/benchmarks", help="Output dir")
    parser.add_argument("--variant", type=str, default="slim", choices=["slim", "standard"], help="Variant: slim or standard")
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
        variant=args.variant,
    )
