from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml


@dataclass
class DatasetConfig:
    """
    Unified Dataset Configuration supporting standard dataset YAML manifests, COCO format,
    and custom user-defined segmentation datasets.

    Compatible with standard dataset YAML schemas:
        path: /path/to/dataset     # dataset root dir (optional, defaults to yaml dir)
        train: images/train        # train images dir or train.txt file
        val: images/val            # val images dir or val.txt file
        test: images/test          # test images (optional)
        nc: 80                     # number of classes (optional if names is given)
        names:                     # list of class names or dict of {0: 'class0', ...}
          0: person
          1: bicycle
        annotations:               # optional explicit COCO JSON annotations
          train: annotations/instances_train.json
          val: annotations/instances_val.json
        masks:                     # optional explicit rasterized mask directories
          train: masks/train
          val: masks/val
    """
    root_path: Path
    train_images: Optional[Path] = None
    val_images: Optional[Path] = None
    test_images: Optional[Path] = None
    train_image_list: Optional[List[Path]] = None
    val_image_list: Optional[List[Path]] = None
    nc: int = 1
    names: Dict[int, str] = field(default_factory=lambda: {0: "object"})
    annotation_files: Dict[str, Optional[Path]] = field(default_factory=dict)
    mask_dirs: Dict[str, Optional[Path]] = field(default_factory=dict)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DatasetConfig":
        """Load and parse an Ultralytics-style dataset YAML file."""
        p = Path(yaml_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset YAML configuration not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        yaml_dir = p.parent
        root = Path(data.get("path", yaml_dir))
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()

        # Parse class names and number of classes
        names_data = data.get("names", None)
        if isinstance(names_data, list):
            names = {idx: str(name) for idx, name in enumerate(names_data)}
            nc = len(names)
        elif isinstance(names_data, dict):
            names = {int(k): str(v) for k, v in names_data.items()}
            nc = len(names)
        else:
            nc = int(data.get("nc", 1))
            names = {i: f"class_{i}" for i in range(nc)}

        def resolve_split_path(split_val: Optional[str]) -> Tuple[Optional[Path], Optional[List[Path]]]:
            if not split_val:
                return None, None
            sp = Path(split_val)
            if not sp.is_absolute():
                sp = root / sp

            # Handle text files containing lists of image paths
            if sp.is_file() and sp.suffix.lower() == ".txt":
                img_list = []
                with open(sp, "r", encoding="utf-8") as tf:
                    for line in tf:
                        line = line.strip()
                        if line:
                            lp = Path(line)
                            if not lp.is_absolute():
                                lp = root / lp
                            if lp.exists():
                                img_list.append(lp)
                return sp.parent, (img_list if img_list else None)

            return sp, None

        train_path, train_list = resolve_split_path(data.get("train"))
        val_path, val_list = resolve_split_path(data.get("val"))
        test_path, _ = resolve_split_path(data.get("test"))

        # Optional explicit COCO annotation paths
        ann_files: Dict[str, Optional[Path]] = {}
        ann_data = data.get("annotations")
        if isinstance(ann_data, dict):
            for k in ("train", "val", "test"):
                if k in ann_data and ann_data[k]:
                    ap = Path(ann_data[k])
                    ann_files[k] = (root / ap).resolve() if not ap.is_absolute() else ap
        elif isinstance(ann_data, str):
            ap = Path(ann_data)
            ann_files["train"] = (root / ap).resolve() if not ap.is_absolute() else ap
            ann_files["val"] = ann_files["train"]

        # Optional explicit rasterized mask directories
        mask_dirs: Dict[str, Optional[Path]] = {}
        mask_data = data.get("masks")
        if isinstance(mask_data, dict):
            for k in ("train", "val", "test"):
                if k in mask_data and mask_data[k]:
                    mp = Path(mask_data[k])
                    mask_dirs[k] = (root / mp).resolve() if not mp.is_absolute() else mp
        elif isinstance(mask_data, str):
            mp = Path(mask_data)
            mask_dirs["train"] = (root / mp).resolve() if not mp.is_absolute() else mp
            mask_dirs["val"] = mask_dirs["train"]

        return cls(
            root_path=root,
            train_images=train_path,
            val_images=val_path,
            test_images=test_path,
            train_image_list=train_list,
            val_image_list=val_list,
            nc=nc,
            names=names,
            annotation_files=ann_files,
            mask_dirs=mask_dirs,
            raw_dict=data,
        )

    @classmethod
    def resolve(
        cls,
        data_input: Union[str, Path],
        annotation_file: Optional[Union[str, Path]] = None,
    ) -> "DatasetConfig":
        """
        Polymorphic resolver:
        Accepts:
        - A path to a dataset YAML file (e.g. 'data.yaml', 'coco8-seg.yaml')
        - A path to a directory containing 'data.yaml' or 'dataset.yaml'
        - A path to a raw COCO directory (e.g. MAGFiLO with images and annotations JSON)
        - A path to a plain directory (COCO, polygon, or mask layout)
        """
        import json

        p = Path(data_input).resolve()

        # If path does not exist, check if running in Kaggle and auto-locate matching folder
        if not p.exists():
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.is_dir():
                target_str = str(data_input).lower()
                for d in kaggle_input.iterdir():
                    if not d.is_dir():
                        continue
                    # Match dataset name or key parts
                    if any(part in d.name.lower() for part in ("magfilo", "filament") if part in target_str):
                        # Try finding image subdirectory inside d
                        cand = d
                        for sub in ("images/train", "train", "images", "train/train_images"):
                            if (d / sub).is_dir():
                                cand = d / sub
                                break
                        if cand.exists():
                            print(f"[Dataset Resolver] Auto-located dataset in /kaggle/input: '{data_input}' -> '{cand}'")
                            p = cand.resolve()
                            break

        if not p.exists():
            cand_msg = ""
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.is_dir():
                mounted = [d.name for d in kaggle_input.iterdir()]
                cand_msg = f"\nCurrently mounted in /kaggle/input: {mounted}"
            raise FileNotFoundError(
                f"Cannot resolve dataset from path: '{data_input}' (path does not exist on disk).{cand_msg}\n"
                f"Tip: If running on Kaggle, please ensure the dataset is added to your notebook via '+ Add Input'."
            )

        # 1. If an explicit annotation file is given, construct DatasetConfig directly
        if annotation_file is not None and Path(annotation_file).is_file():
            ann_p = Path(annotation_file).resolve()
            ann_files = {"train": ann_p, "val": ann_p}
            try:
                with open(ann_p, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                raw_cats = ann_data.get("categories", [])
                if raw_cats:
                    sorted_cats = sorted([c for c in raw_cats if "id" in c], key=lambda c: c["id"])
                    names = {idx: cat.get("name", f"class_{idx}") for idx, cat in enumerate(sorted_cats)}
                    nc = len(names)
                else:
                    nc = 1
                    names = {0: "filament"}
            except Exception:
                nc = 1
                names = {0: "filament"}

            return cls(
                root_path=p,
                train_images=p,
                val_images=p,
                nc=nc,
                names=names,
                annotation_files=ann_files,
            )

        # 2. If it's a YAML file, parse it directly
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(p)

        # 3. If it's a directory containing data.yaml or dataset.yaml
        if p.is_dir():
            for candidate in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
                yaml_file = p / candidate
                if yaml_file.is_file():
                    return cls.from_yaml(yaml_file)

            # 4. Auto-discover COCO datasets (e.g. MAGFiLO or MS COCO format)
            coco_cfg = cls._auto_discover_coco(p)
            if coco_cfg is not None:
                return coco_cfg

            # 5. Fallback: Standard directory structure without YAML
            return cls(root_path=p)

        raise FileNotFoundError(f"Cannot resolve dataset from path: {data_input}")

    @classmethod
    def _auto_discover_coco(cls, root: Path) -> Optional["DatasetConfig"]:
        """Auto-discover COCO annotation JSON files and image directories."""
        import json
        import random

        # Search for JSON files up to 2 directory levels deep
        json_candidates: List[Path] = []
        for pattern in ("*.json", "*/*.json", "*/*/*.json"):
            json_candidates.extend(list(root.glob(pattern)))

        coco_train_json: Optional[Path] = None
        coco_val_json: Optional[Path] = None
        coco_train_data: Optional[Dict[str, Any]] = None

        for jf in sorted(json_candidates):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and "images" in d and "annotations" in d:
                    stem_lower = jf.stem.lower()
                    if "val" in stem_lower:
                        coco_val_json = jf
                    elif "train" in stem_lower or coco_train_json is None:
                        coco_train_json = jf
                        coco_train_data = d
            except Exception:
                continue

        if coco_train_json is None or coco_train_data is None:
            return None

        # Extract classes from categories
        raw_cats = coco_train_data.get("categories", [])
        if raw_cats:
            sorted_cats = sorted([c for c in raw_cats if "id" in c], key=lambda c: c["id"])
            names = {idx: cat.get("name", f"class_{idx}") for idx, cat in enumerate(sorted_cats)}
            nc = len(names)
        else:
            nc = 1
            names = {0: "filament"}

        # Discover image directory matching COCO image file names
        images_info = coco_train_data.get("images", [])
        if not images_info:
            return None
        sample_fname = images_info[0].get("file_name", "")

        candidate_img_dirs = [
            root / "train" / "train_images",
            root / "train_images",
            root / "images" / "train",
            root / "train",
            root / "images",
            root,
        ]
        candidate_img_dirs.extend([d for d in root.glob("**/train_images") if d.is_dir()])
        candidate_img_dirs.extend([d for d in root.glob("**/images") if d.is_dir()])

        train_img_dir: Optional[Path] = None
        for cd in candidate_img_dirs:
            if cd.is_dir() and (cd / sample_fname).is_file():
                train_img_dir = cd
                break

        if train_img_dir is None:
            # Fallback: check if any directory has image files
            for cd in candidate_img_dirs:
                if cd.is_dir() and any(cd.glob("*.jpeg")) or any(cd.glob("*.jpg")) or any(cd.glob("*.png")):
                    train_img_dir = cd
                    break

        if train_img_dir is None:
            return None

        # Collect all image files
        all_imgs: List[Path] = []
        for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
            all_imgs.extend(list(train_img_dir.glob(ext)))
        all_imgs = sorted(list(set(all_imgs)))

        ann_files: Dict[str, Optional[Path]] = {
            "train": coco_train_json,
            "val": coco_val_json or coco_train_json,
        }

        # If separate validation set exists
        if coco_val_json is not None and coco_val_json != coco_train_json:
            val_img_dir = root / "val" / "val_images" if (root / "val" / "val_images").is_dir() else (root / "val" if (root / "val").is_dir() else train_img_dir)
            return cls(
                root_path=root,
                train_images=train_img_dir,
                val_images=val_img_dir,
                nc=nc,
                names=names,
                annotation_files=ann_files,
            )

        # Unified dataset without separate validation directory (e.g. MAGFiLO Kaggle):
        # Automatically partition into deterministic 80% train and 20% validation
        rng = random.Random(42)
        shuffled = list(all_imgs)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * 0.2))
        train_count = len(shuffled) - val_count

        train_list = sorted(shuffled[:train_count])
        val_list = sorted(shuffled[train_count:])

        return cls(
            root_path=root,
            train_images=train_img_dir,
            val_images=train_img_dir,
            train_image_list=train_list,
            val_image_list=val_list,
            nc=nc,
            names=names,
            annotation_files=ann_files,
        )

