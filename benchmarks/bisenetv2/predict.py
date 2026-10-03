from __future__ import annotations

import argparse
from .model import BiSeNetV2
from ..common.predictor import BenchmarkPredictor


def predict(
    weights: str,
    data: str,
    split: str = "val",
    imgsz: int = 1024,
    num_classes: int = 1,
    device: str = "cuda",
    output_dir: str = "predictions/benchmarks",
    save_strips: int = 10,
):
    model = BiSeNetV2(in_channels=3, num_classes=num_classes)
    predictor = BenchmarkPredictor(
        model=model,
        weights_path=weights,
        model_name="BiSeNetV2",
        data_root=data,
        img_size=(imgsz, imgsz),
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
    )
    predictor.setup_data(split=split)
    return predictor.predict(save_strips=save_strips)


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiSeNet V2 Benchmark Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory or data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split (val or test)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution")
    parser.add_argument("--num-classes", type=int, default=1, help="Foreground classes")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="predictions/benchmarks", help="Output directory")
    parser.add_argument("--save-strips", type=int, default=10, help="Visual strips to save")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        num_classes=args.num_classes,
        device=args.device,
        output_dir=args.output_dir,
        save_strips=args.save_strips,
    )
