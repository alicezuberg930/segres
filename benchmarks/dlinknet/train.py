from __future__ import annotations

import argparse
from .model import DLinkNet
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
    num_classes: int | None = None,
    num_workers: int = 2,
    loss_type: str = "soar",
):
    if num_classes is None:
        try:
            from soar.data.dataset_config import DatasetConfig
            cfg = DatasetConfig.resolve(data)
            num_classes = cfg.nc if (cfg and cfg.nc) else 1
        except Exception:
            num_classes = 1

    model = DLinkNet(in_channels=3, num_classes=num_classes)
    trainer = BenchmarkTrainer(
        model=model,
        model_name="DLinkNet",
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
    parser = argparse.ArgumentParser(description="Train D-LinkNet Benchmark Baseline")
    parser.add_argument("--data", type=str, required=True, help="Data directory or data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input resolution")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--project", type=str, default="checkpoints/benchmarks", help="Checkpoint dir")
    parser.add_argument("--num-classes", type=int, default=None, help="Num classes (auto-detected if omitted)")

    parser.add_argument("--workers", type=int, default=2, help="Workers")
    parser.add_argument("--loss", type=str, default="soar", choices=["soar", "standard"], help="soar or standard")
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
        loss_type=args.loss,
    )
