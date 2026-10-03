from __future__ import annotations

import argparse
from pathlib import Path
from .model import UNet
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
    base_channels: int = 64,
    loss_type: str = "soar",
):
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
    trainer = BenchmarkTrainer(
        model=model,
        model_name="UNet",
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
        loss_type=loss_type,
    )
    return trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net Benchmark Baseline")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory or data.yaml manifest")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input resolution (e.g. 1024 or 2048)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP mixed precision")
    parser.add_argument("--project", type=str, default="checkpoints/benchmarks", help="Output checkpoint directory")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of foreground target classes")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channel capacity")
    parser.add_argument("--loss", type=str, default="soar", choices=["soar", "standard"], help="soar (Focal+Dice+Boundary+clDice) or standard (BCE+Dice)")
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
        base_channels=args.base_channels,
        loss_type=args.loss,
    )
