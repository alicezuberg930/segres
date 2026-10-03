from __future__ import annotations

import argparse
from .train import train
from .val import validate
from .predict import predict


def main():
    parser = argparse.ArgumentParser(prog="benchmarks.segformer", description="SegFormer Benchmark Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train SegFormer from scratch")
    p_train.add_argument("--data", type=str, required=True, help="Data directory or data.yaml")
    p_train.add_argument("--imgsz", type=int, default=1024, help="Image resolution")
    p_train.add_argument("--epochs", type=int, default=50, help="Epochs")
    p_train.add_argument("--batch", type=int, default=1, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p_train.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p_train.add_argument("--amp", action="store_true", default=True, help="AMP")
    p_train.add_argument("--project", type=str, default="checkpoints/benchmarks", help="Checkpoint dir")
    p_train.add_argument("--num-classes", type=int, default=None, help="Num classes (auto-detected if omitted)")
    p_train.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    p_train.add_argument("--variant", type=str, default="b0", choices=["b0", "b1"], help="b0 or b1")
    p_train.add_argument("--loss", type=str, default="soar", choices=["soar", "standard"], help="Loss type: soar or standard")

    # Val
    p_val = subparsers.add_parser("val", help="Validate SegFormer checkpoint")
    p_val.add_argument("--weights", type=str, required=True, help="Path to weights (.pt)")
    p_val.add_argument("--data", type=str, required=True, help="Data directory or data.yaml")
    p_val.add_argument("--split", type=str, default="val", help="Split")
    p_val.add_argument("--imgsz", type=int, default=1024, help="Resolution")
    p_val.add_argument("--num-classes", type=int, default=None, help="Num classes (auto-detected if omitted)")
    p_val.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p_val.add_argument("--output-dir", type=str, default="evaluations/benchmarks", help="Output dir")
    p_val.add_argument("--variant", type=str, default="b0", choices=["b0", "b1"], help="b0 or b1")

    # Predict
    p_pred = subparsers.add_parser("predict", help="Run inference with SegFormer")
    p_pred.add_argument("--weights", type=str, required=True, help="Path to weights (.pt)")
    p_pred.add_argument("--data", type=str, required=True, help="Data directory or data.yaml")
    p_pred.add_argument("--split", type=str, default="val", help="Split")
    p_pred.add_argument("--imgsz", type=int, default=1024, help="Resolution")
    p_pred.add_argument("--num-classes", type=int, default=None, help="Num classes (auto-detected if omitted)")

    p_pred.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p_pred.add_argument("--output-dir", type=str, default="predictions/benchmarks", help="Output dir")
    p_pred.add_argument("--save-strips", type=int, default=10, help="Visual strips to save")
    p_pred.add_argument("--variant", type=str, default="b0", choices=["b0", "b1"], help="b0 or b1")

    args = parser.parse_args()
    if args.command == "train":
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
            loss_type=args.loss,
        )
    elif args.command == "val":
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
    elif args.command == "predict":
        predict(
            weights=args.weights,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            num_classes=args.num_classes,
            device=args.device,
            output_dir=args.output_dir,
            save_strips=args.save_strips,
            variant=args.variant,
        )


if __name__ == "__main__":
    main()
