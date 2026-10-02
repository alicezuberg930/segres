from __future__ import annotations

import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .dataset import SegmentationDataset


class FewShotEpisodeDataset(Dataset):
    """
    Episodic Dataset for Few-Shot and One-Shot Segmentation.
    
    Generates N-shot episodes containing:
    - Support Set: K pairs of (support_image, support_binary_mask) for class c
    - Query Set: (query_image, query_binary_mask) for class c
    
    Classes are strictly partitioned into Base Classes (for training) and Novel Classes
    (for evaluation) using standard k-fold cross-validation splits.
    """

    def __init__(
        self,
        base_dataset: SegmentationDataset,
        shots: int = 1,
        episodes: int = 1000,
        fold: int = 0,
        total_folds: int = 4,
        is_train: bool = True,
        train_classes: Optional[List[int]] = None,
        val_classes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.shots = max(1, shots)
        self.episodes = episodes
        self.fold = fold
        self.total_folds = total_folds
        self.is_train = is_train
        self.num_classes = max(1, base_dataset.num_classes)
        self.rng = random.Random(seed if seed is not None else (42 if not is_train else None))

        # 1. Partition classes into base and novel
        all_classes = list(range(self.num_classes))
        if self.num_classes == 1:
            # Single class (e.g. topology transfer benchmark across domain images)
            self.active_classes = [0]
        elif train_classes is not None and val_classes is not None:
            self.active_classes = train_classes if is_train else val_classes
        else:
            # Standard k-fold split (e.g. Pascal-5i, COCO-20i)
            # Novel classes for fold i: {c | c % total_folds == fold}
            novel = [c for c in all_classes if c % total_folds == fold]
            base = [c for c in all_classes if c not in novel]
            self.active_classes = base if is_train else novel

        if not self.active_classes:
            self.active_classes = all_classes

        # 2. Index dataset samples by active classes
        self.class_to_indices: Dict[int, List[int]] = {c: [] for c in self.active_classes}
        self._index_classes()

    def _index_classes(self) -> None:
        """Index which dataset samples contain each foreground class."""
        total_samples = len(self.base_dataset)
        
        # If single class, all samples are valid
        if self.num_classes == 1:
            self.class_to_indices[0] = list(range(total_samples))
            return

        # Check pre-computed annotations if available in base_dataset
        indexed = False
        if hasattr(self.base_dataset, "annotations") and self.base_dataset.annotations:
            for idx, img_path in enumerate(self.base_dataset.image_files):
                img_name = img_path.name
                stem = img_path.stem
                anns = self.base_dataset.annotations.get(img_name) or self.base_dataset.annotations.get(stem)
                if isinstance(anns, list):
                    for ann in anns:
                        cid = ann.get("category_id") or ann.get("class_id")
                        if cid is not None:
                            # Map category ID to contiguous index if map exists
                            if hasattr(self.base_dataset, "coco_cat_map") and self.base_dataset.coco_cat_map:
                                cid = self.base_dataset.coco_cat_map.get(cid, cid)
                            if cid in self.class_to_indices:
                                self.class_to_indices[cid].append(idx)
                                indexed = True

        # Fallback: scan samples to find positive classes
        if not indexed:
            scan_limit = min(total_samples, 200)  # fast scan
            for idx in range(scan_limit):
                sample = self.base_dataset[idx]
                mask = sample["mask"]  # (C, H, W)
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                for c in self.active_classes:
                    if c < mask.shape[0] and mask[c].sum() > 0:
                        self.class_to_indices[c].append(idx)
            
            # If any active class has 0 samples found, populate with all samples
            for c in self.active_classes:
                if not self.class_to_indices[c]:
                    self.class_to_indices[c] = list(range(total_samples))

        # Deduplicate indices
        for c in self.active_classes:
            self.class_to_indices[c] = sorted(list(set(self.class_to_indices[c])))
            if len(self.class_to_indices[c]) == 0:
                self.class_to_indices[c] = list(range(total_samples))

    def __len__(self) -> int:
        return self.episodes

    def _extract_binary_mask(self, mask_tensor: torch.Tensor, class_id: int) -> torch.Tensor:
        """Extract a single-channel binary mask (1, H, W) for the target class."""
        if mask_tensor.dim() == 2:
            return mask_tensor.unsqueeze(0).float()
        elif mask_tensor.dim() == 3:
            if mask_tensor.shape[0] == 1:
                return (mask_tensor > 0.5).float()
            elif class_id < mask_tensor.shape[0]:
                return (mask_tensor[class_id : class_id + 1] > 0.5).float()
            else:
                return (mask_tensor[0:1] > 0.5).float()
        return mask_tensor.float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Sample one N-shot episode."""
        # 1. Randomly pick an active class
        class_id = self.rng.choice(self.active_classes)
        candidate_indices = self.class_to_indices[class_id]

        # 2. Sample K support items + 1 query item
        needed = self.shots + 1
        if len(candidate_indices) >= needed:
            sampled = self.rng.sample(candidate_indices, needed)
            supp_indices = sampled[: self.shots]
            query_idx = sampled[self.shots]
        else:
            supp_indices = [self.rng.choice(candidate_indices) for _ in range(self.shots)]
            query_idx = self.rng.choice(candidate_indices)

        # 3. Load support samples
        supp_images = []
        supp_masks = []
        for s_idx in supp_indices:
            s_data = self.base_dataset[s_idx]
            s_img = s_data["image"]
            s_mask = self._extract_binary_mask(s_data["mask"], class_id)
            supp_images.append(s_img)
            supp_masks.append(s_mask)

        # 4. Load query sample
        q_data = self.base_dataset[query_idx]
        q_img = q_data["image"]
        q_mask = self._extract_binary_mask(q_data["mask"], class_id)

        # Stack support samples into (K, C, H, W) and (K, 1, H, W)
        support_images = torch.stack(supp_images, dim=0)
        support_masks = torch.stack(supp_masks, dim=0)

        return {
            "support_images": support_images,  # (K, C, H, W)
            "support_masks": support_masks,    # (K, 1, H, W)
            "query_image": q_img,              # (C, H, W)
            "query_mask": q_mask,              # (1, H, W)
            "class_id": class_id,
        }


def few_shot_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate episodic batches together for DataLoader."""
    support_images = torch.stack([b["support_images"] for b in batch], dim=0)  # (B, K, C, H, W)
    support_masks = torch.stack([b["support_masks"] for b in batch], dim=0)    # (B, K, 1, H, W)
    query_images = torch.stack([b["query_image"] for b in batch], dim=0)      # (B, C, H, W)
    query_masks = torch.stack([b["query_mask"] for b in batch], dim=0)        # (B, 1, H, W)
    class_ids = [b["class_id"] for b in batch]

    return {
        "support_images": support_images,
        "support_masks": support_masks,
        "query_images": query_images,
        "query_masks": query_masks,
        "class_ids": class_ids,
    }
