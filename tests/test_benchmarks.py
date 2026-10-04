from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml

from benchmarks import (
    UNet,
    DLinkNet,
    CSNet,
    BiSeNetV2,
    DDRNet,
    PIDNet,
    SegFormer,
)

from benchmarks.common.metrics import BenchmarkMetricAccumulator
from benchmarks.common.engine import BenchmarkTrainer
from benchmarks.common.predictor import BenchmarkPredictor


def test_model_forward_passes():
    """Verify all 7 baseline architectures can perform forward pass and output matching tensor shapes."""
    models = {
        "UNet": UNet(in_channels=3, num_classes=2, base_channels=32),
        "DLinkNet": DLinkNet(in_channels=3, num_classes=2),
        "CSNet": CSNet(in_channels=3, num_classes=2, base_channels=32),
        "BiSeNetV2": BiSeNetV2(in_channels=3, num_classes=2),
        "DDRNet_slim": DDRNet(in_channels=3, num_classes=2, variant="slim"),
        "DDRNet_std": DDRNet(in_channels=3, num_classes=2, variant="standard"),
        "PIDNet_s": PIDNet(in_channels=3, num_classes=2, variant="s"),
        "PIDNet_m": PIDNet(in_channels=3, num_classes=2, variant="m"),
        "SegFormer_b0": SegFormer(in_channels=3, num_classes=2, variant="b0"),
        "SegFormer_b1": SegFormer(in_channels=3, num_classes=2, variant="b1"),
    }

    dummy = torch.randn(1, 3, 64, 64)
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 2, 64, 64), f"Failed for {name}: expected (1, 2, 64, 64), got {out.shape}"
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [PASS] {name:15s}: Forward shape {tuple(out.shape)} | Params: {params_m:.2f} M")


def test_metric_accumulator():
    """Verify GPU/CPU benchmark metric accumulator computes mIoU, Dice, bIoU, clDice."""
    accum = BenchmarkMetricAccumulator(num_classes=2, class_names={0: "filament", 1: "background"})
    
    # Perfect prediction test
    pred = torch.zeros(1, 2, 32, 32)
    pred[:, 0, 10:20, 10:20] = 1.0
    gt = pred.clone()

    accum.update(pred, gt)
    metrics = accum.compute()

    assert metrics["mIoU"] > 0.49, f"Expected high mIoU, got {metrics['mIoU']}"
    assert metrics["Dice"] > 0.49, f"Expected high Dice, got {metrics['Dice']}"
    assert "clDice" in metrics
    assert "bIoU" in metrics
    print(f"  [PASS] Metric Accumulator: mIoU={metrics['mIoU']:.4f}, Dice={metrics['Dice']:.4f}, clDice={metrics['clDice']:.4f}")


def create_synthetic_dataset(root: Path) -> Path:
    """Create minimal synthetic dataset manifest."""
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_lbl = root / "labels" / "train"
    val_lbl = root / "labels" / "val"

    for d in (train_dir, val_dir, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(train_dir / "sample1.jpg"), img)
    cv2.imwrite(str(val_dir / "sample2.jpg"), img)

    with open(train_lbl / "sample1.txt", "w") as f:
        f.write("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    with open(val_lbl / "sample2.txt", "w") as f:
        f.write("0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n")

    yaml_content = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "filament"},
    }
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f)
    return yaml_path


def test_benchmark_training_and_inference():
    """Verify BenchmarkTrainer and BenchmarkPredictor execute on synthetic data."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        yaml_path = create_synthetic_dataset(root)

        model = UNet(in_channels=3, num_classes=1, base_channels=16)
        ckpt_dir = root / "checkpoints"

        # Train 1 epoch with SOAR composite loss
        trainer_soar = BenchmarkTrainer(
            model=model,
            model_name="UNet_Test",
            data_root=yaml_path,
            img_size=(64, 64),
            epochs=1,
            batch_size=1,
            device="cpu",
            amp=False,
            checkpoint_dir=ckpt_dir,
            num_workers=0,
            loss_type="soar",
        )
        res = trainer_soar.train()
        assert res["best_epoch"] == 1
        assert (ckpt_dir / "UNet_Test" / "best.pt").is_file()

        # Train 1 epoch with standard loss
        model_std = UNet(in_channels=3, num_classes=1, base_channels=16)
        trainer_std = BenchmarkTrainer(
            model=model_std,
            model_name="UNet_Std_Test",
            data_root=yaml_path,
            img_size=(64, 64),
            epochs=1,
            batch_size=1,
            device="cpu",
            amp=False,
            checkpoint_dir=ckpt_dir,
            num_workers=0,
            loss_type="standard",
        )
        res_std = trainer_std.train()
        assert res_std["best_epoch"] == 1
        assert (ckpt_dir / "UNet_Std_Test" / "best.pt").is_file()


        # Run inference
        predictor = BenchmarkPredictor(
            model=model,
            weights_path=ckpt_dir / "UNet_Test" / "best.pt",
            model_name="UNet_Test",
            data_root=yaml_path,
            img_size=(64, 64),
            device="cpu",
            output_dir=root / "predictions",
        )
        pred_res = predictor.predict(save_strips=1)
        assert pred_res["mIoU"] >= 0.0
        assert (root / "predictions" / "UNet_Test" / "benchmark_summary.json").is_file()
        assert (root / "predictions" / "UNet_Test" / "visualizations" / "strip_sample2.png").is_file()
        print(f"  [PASS] End-to-end Train & Predict: Summary and visual strips generated successfully!")


if __name__ == "__main__":
    print("Testing Model Forward Passes...")
    test_model_forward_passes()
    print("Testing Benchmark Metric Accumulator...")
    test_metric_accumulator()
    print("Testing End-to-End Training & Inference Pipeline...")
    test_benchmark_training_and_inference()
    print("All Benchmark Tests Passed Successfully!")
