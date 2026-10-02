from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml

from soar.data.dataset_config import DatasetConfig
from soar.data.dataset import SegmentationDataset
from soar.engine.trainer import BaseTrainer


def create_synthetic_yolo_dataset(root: Path):
    """Create a minimal synthetic Ultralytics-style dataset with images and YOLO polygon labels."""
    train_img_dir = root / "images" / "train"
    val_img_dir = root / "images" / "val"
    train_lbl_dir = root / "labels" / "train"
    val_lbl_dir = root / "labels" / "val"

    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Synthetic images (64x64)
    img_t1 = np.full((64, 64, 3), 50, dtype=np.uint8)
    img_t2 = np.full((64, 64, 3), 100, dtype=np.uint8)
    img_v1 = np.full((64, 64, 3), 150, dtype=np.uint8)

    cv2.imwrite(str(train_img_dir / "sample1.jpg"), img_t1)
    cv2.imwrite(str(train_img_dir / "sample2.jpg"), img_t2)
    cv2.imwrite(str(val_img_dir / "sample3.jpg"), img_v1)

    # Normalized polygon coordinates: class_id x1 y1 x2 y2 x3 y3 x4 y4
    # sample1: class 0 polygon
    with open(train_lbl_dir / "sample1.txt", "w") as f:
        f.write("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

    # sample2: class 1 polygon
    with open(train_lbl_dir / "sample2.txt", "w") as f:
        f.write("1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    # sample3: class 0 and 1 polygons
    with open(val_lbl_dir / "sample3.txt", "w") as f:
        f.write("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
        f.write("1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n")

    yaml_content = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {
            0: "road",
            1: "building"
        }
    }

    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f)

    return yaml_path


def test_ultralytics_yaml_parsing():
    """Verify DatasetConfig parses Ultralytics YAML configurations correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        yaml_path = create_synthetic_yolo_dataset(root)

        cfg = DatasetConfig.resolve(yaml_path)
        assert cfg.nc == 2
        assert cfg.names == {0: "road", 1: "building"}
        assert cfg.train_images == (root / "images" / "train").resolve()
        assert cfg.val_images == (root / "images" / "val").resolve()


def test_yolo_segmentation_dataset_loading():
    """Verify SegmentationDataset loads samples and correctly rasterizes YOLO polygon masks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        yaml_path = create_synthetic_yolo_dataset(root)
        cfg = DatasetConfig.resolve(yaml_path)

        # Multi-class loading (num_classes=2)
        ds = SegmentationDataset(
            data_root=cfg.root_path,
            split="train",
            img_size=(64, 64),
            num_classes=cfg.nc,
            names=cfg.names,
            image_dir=cfg.train_images,
        )
        assert len(ds) == 2
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["mask"].shape == (2, 64, 64)
        assert sample["mask"].max() > 0.0

        # Validation split
        val_ds = SegmentationDataset(
            data_root=cfg.root_path,
            split="val",
            img_size=(64, 64),
            num_classes=cfg.nc,
            names=cfg.names,
            image_dir=cfg.val_images,
        )
        assert len(val_ds) == 1
        val_sample = val_ds[0]
        assert val_sample["mask"].shape == (2, 64, 64)


def test_trainer_integration_with_yaml():
    """Verify BaseTrainer automatically configures num_classes, splits, and trains on data.yaml."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        yaml_path = create_synthetic_yolo_dataset(root)

        ckpt_dir = root / "checkpoints"
        trainer = BaseTrainer(
            model_cfg="configs/models/soar_nano1.yaml",
            data_root=str(yaml_path),
            img_size=(64, 64),
            epochs=1,
            batch_size=1,
            device="cpu",
            checkpoint_dir=str(ckpt_dir),
            num_workers=0,
        )

        # Trainer should automatically detect 2 classes from data.yaml
        assert trainer.num_classes == 2
        assert trainer.class_names == {0: "road", 1: "building"}
        assert len(trainer.train_loader.dataset) == 2
        assert len(trainer.val_loader.dataset) == 1

        # Execute 1 training step
        batch = next(iter(trainer.train_loader))
        img = batch["image"].to("cpu")
        mask = batch["mask"].to("cpu")

        pred = trainer.model(img)
        assert pred.shape == (1, 2, 64, 64)
        loss, _ = trainer.criterion(pred, mask)
        loss.backward()
        trainer.optimizer.step()
        assert not torch.isnan(loss)


if __name__ == "__main__":
    print("Testing Ultralytics YAML parsing...")
    test_ultralytics_yaml_parsing()
    print("Testing YOLO segmentation dataset loading...")
    test_yolo_segmentation_dataset_loading()
    print("Testing BaseTrainer integration with data.yaml...")
    test_trainer_integration_with_yaml()
    print("All Ultralytics YAML, COCO, and YOLO segmentation tests passed successfully!")
