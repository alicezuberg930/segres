from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml


@dataclass
class DatasetConfig:
    """
    Unified Dataset Configuration supporting Ultralytics YOLO format, COCO format,
    and custom user-defined segmentation datasets.

    Compatible with standard Ultralytics YAML schemas:
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

            # Handle text files containing lists of image paths (common in YOLO)
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
    def resolve(cls, data_input: Union[str, Path]) -> "DatasetConfig":
        """
        Polymorphic resolver:
        Accepts:
        - A path to a dataset YAML file (e.g. 'data.yaml', 'coco8-seg.yaml')
        - A path to a directory containing 'data.yaml' or 'dataset.yaml'
        - A path to a plain directory (COCO, YOLO, or mask layout)
        """
        p = Path(data_input).resolve()

        # 1. If it's a YAML file, parse it directly
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(p)

        # 2. If it's a directory containing data.yaml or dataset.yaml
        if p.is_dir():
            for candidate in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
                yaml_file = p / candidate
                if yaml_file.is_file():
                    return cls.from_yaml(yaml_file)

            # 3. Fallback: Standard directory structure without YAML
            return cls(root_path=p)

        raise FileNotFoundError(f"Cannot resolve dataset from path: {data_input}")
