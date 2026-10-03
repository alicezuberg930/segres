from __future__ import annotations

import argparse
from .model import SegFormer
from ..common.engine import BenchmarkTrainer


def train(
    data: str,
    imgsz: int = 1024,
    epochs: int = 50,
    batch: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    amp: bool = True,
    project: str = "checkpoints/benchmarks",
    num_classes: int = 1,
    num_workers: int = 2,
    variant: str = "b0",
):
    model = SegFormer(in_channels=3, num_classes=num_classes, variant=variant)
    model_name = f"SegFormer_{variant.upper()}"
    trainer = BenchmarkTrainer(
        model=model,
        model_name=model_name,
        data_root=data,
        img_size=(imgsz, imgsz),
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        device=device,
        amp=amp,
        checkpoint_dir=project,
        num_classes=num_classes,
        num_workers=num_workers,
    )
    return trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer Benchmark Baseline")
    parser.add_argument("--data", type=str, required=True, help="Data directory or data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input resolution")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--project", type=str, default="checkpoints/benchmarks", help="Checkpoint dir")
    parser.add_argument("--num-classes", type=int, default=1, help="Num classes")
    parser.add_argument("--workers", type=int, default=2, help="Workers")
    parser.add_argument("--variant", type=str, default="b0", choices=["b0", "b1"], help="b0 (SegFormer-B0) or b1 (SegFormer-B1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        amp=args.amp,
        project=args.project,
        num_classes=args.num_classes,
        num_workers=args.workers,
        variant=args.variant,
    )
