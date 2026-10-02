from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soar.models.few_shot import FewShotSOAR
from soar.data.dataset import SegmentationDataset
from soar.data.few_shot import FewShotEpisodeDataset, few_shot_collate_fn
from soar.engine.few_shot_trainer import FewShotTrainer


def test_few_shot_model_forward_backward():
    """Verify FewShotSOAR forward and backward pass for 1-shot and 5-shot episodes."""
    print("Testing FewShotSOAR architecture...")
    model = FewShotSOAR(
        backbone_cfg="configs/models/soar_nano1.yaml",
        in_channels=3,
    )
    model.train()

    # 1-Shot Test
    b, k, c, h, w = 1, 1, 3, 128, 128
    supp_imgs = torch.randn(b, k, c, h, w)
    supp_masks = (torch.rand(b, k, 1, h, w) > 0.8).float()
    query_imgs = torch.randn(b, c, h, w)
    query_masks = (torch.rand(b, 1, h, w) > 0.8).float()

    logits, aux = model(query_imgs, supp_imgs, supp_masks, return_aux=True)
    assert logits.shape == (b, 1, h, w), f"Expected {(b, 1, h, w)}, got {logits.shape}"
    assert "supp_logits" in aux
    assert aux["supp_logits"].shape == (b, 1, h, w)

    # Backward pass
    loss = logits.sum() + aux["supp_logits"].sum()
    loss.backward()
    print("1-Shot forward and backward passed.")

    # 5-Shot Test
    k = 5
    supp_imgs5 = torch.randn(b, k, c, h, w)
    supp_masks5 = (torch.rand(b, k, 1, h, w) > 0.8).float()
    logits5 = model(query_imgs, supp_imgs5, supp_masks5, return_aux=False)
    assert logits5.shape == (b, 1, h, w), f"Expected {(b, 1, h, w)}, got {logits5.shape}"
    print("5-Shot forward passed.")


def test_few_shot_dataset_and_trainer():
    """Verify FewShotEpisodeDataset sampling and FewShotTrainer execution."""
    print("Testing FewShotEpisodeDataset & FewShotTrainer...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_dir = tmp_path / "train" / "train_images"
        masks_dir = tmp_path / "masks"
        train_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Create 4 synthetic images with 4 classes
        for i in range(4):
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Create a block for class (i+1)
            mask[30 * i : 30 * i + 40, 30 * i : 30 * i + 40] = i + 1
            cv2.imwrite(str(train_dir / f"img_{i}.png"), img)
            cv2.imwrite(str(masks_dir / f"img_{i}.png"), mask)

        base_ds = SegmentationDataset(
            data_root=tmp_path,
            split="train",
            img_size=(256, 256),
            in_channels=3,
            num_classes=4,
            augment=False,
        )

        # Test Episode Dataset with 1-shot
        ep_ds = FewShotEpisodeDataset(
            base_dataset=base_ds,
            shots=1,
            episodes=10,
            fold=0,
            total_folds=2,
            is_train=True,
        )
        sample = ep_ds[0]
        assert sample["support_images"].shape == (1, 3, 256, 256)
        assert sample["support_masks"].shape == (1, 1, 256, 256)
        assert sample["query_image"].shape == (3, 256, 256)
        assert sample["query_mask"].shape == (1, 256, 256)
        print("FewShotEpisodeDataset sampling verified.")

        # Test Trainer 1 Epoch
        trainer = FewShotTrainer(
            model_cfg="configs/models/soar_nano1.yaml",
            data_root=str(tmp_path),
            shots=1,
            fold=0,
            total_folds=2,
            train_episodes=2,
            val_episodes=2,
            img_size=(256, 256),
            num_classes=4,
            epochs=1,
            lr=1e-4,
            device="cpu",
            workers=0,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        train_loss = trainer.train_epoch(epoch=0)
        assert train_loss > 0, "Train loss should be positive"
        metrics = trainer.validate()
        assert "mIoU" in metrics
        assert "FB-IoU" in metrics
        assert "cldice" in metrics
        print("FewShotTrainer 1-epoch execution and validation verified.")


if __name__ == "__main__":
    test_few_shot_model_forward_backward()
    test_few_shot_dataset_and_trainer()
    print("All Few-Shot and One-Shot unit tests passed successfully!")
