from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import yaml


def prepare_magfilo(
    dataset_dir: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    binary: bool = False,
) -> Path:
    """
    Prepare MAGFiLO dataset manifest for SOAR and benchmark models on local machines or Kaggle.
    
    Creates:
      - output_dir/train.txt: List of absolute paths for training images
      - output_dir/val.txt: List of absolute paths for validation images
      - output_dir/magfilo.yaml: Standardized dataset configuration for benchmark_cli.py
    """
    dataset_path = Path(dataset_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Locate train images directory
    candidate_img_dirs = [
        dataset_path / "train" / "train_images",
        dataset_path / "train_images",
        dataset_path / "images" / "train",
        dataset_path / "train",
    ]
    img_dir = None
    for d in candidate_img_dirs:
        if d.is_dir() and any(d.glob("*.jpeg")) or any(d.glob("*.jpg")) or any(d.glob("*.png")):
            img_dir = d
            break

    if img_dir is None:
        raise FileNotFoundError(f"Could not locate image directory under: {dataset_path}")

    # Collect images
    image_files: List[Path] = []
    for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
        image_files.extend(list(img_dir.glob(ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        raise FileNotFoundError(f"No valid image files found in: {img_dir}")

    # 2. Locate annotation JSON
    candidate_ann_files = [
        dataset_path / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        dataset_path / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    ]
    candidate_ann_files.extend(list(dataset_path.glob("*.json")))
    candidate_ann_files.extend(list((dataset_path / "train").glob("*.json")) if (dataset_path / "train").is_dir() else [])

    ann_file = None
    for f in candidate_ann_files:
        if f.is_file():
            ann_file = f
            break

    if ann_file is None:
        raise FileNotFoundError(f"Could not locate COCO annotation JSON file under: {dataset_path}")

    # Verify categories in JSON
    with open(ann_file, "r", encoding="utf-8") as jf:
        coco_data = json.load(jf)

    categories = coco_data.get("categories", [])
    if binary:
        nc = 1
        names = {0: "filament"}
    elif categories:
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        names = {idx: cat["name"] for idx, cat in enumerate(sorted_cats)}
        nc = len(names)
    else:
        nc = 4
        names = {0: "Left", 1: "Right", 2: "Unidentifiable", 3: "Ambiguous"}

    # 3. Deterministic train / val split
    rng = random.Random(seed)
    shuffled_files = list(image_files)
    rng.shuffle(shuffled_files)

    val_count = max(1, int(len(shuffled_files) * val_ratio))
    train_count = len(shuffled_files) - val_count

    train_list = shuffled_files[:train_count]
    val_list = shuffled_files[train_count:]

    train_txt_path = out_path / "train.txt"
    val_txt_path = out_path / "val.txt"

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for p in sorted(train_list):
            f.write(f"{p.as_posix()}\n")

    with open(val_txt_path, "w", encoding="utf-8") as f:
        for p in sorted(val_list):
            f.write(f"{p.as_posix()}\n")

    # 4. Generate YAML manifest
    yaml_dict: Dict[str, Any] = {
        "path": dataset_path.as_posix(),
        "train": train_txt_path.as_posix(),
        "val": val_txt_path.as_posix(),
        "annotations": {
            "train": ann_file.as_posix(),
            "val": ann_file.as_posix(),
        },
        "nc": nc,
        "names": names,
    }

    yaml_path = out_path / "magfilo.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    print("=" * 70)
    print("MAGFiLO Dataset Preparation Complete!")
    print(f"  Dataset Root:     {dataset_path}")
    print(f"  Images Found:     {len(image_files)}")
    print(f"  Train Samples:    {len(train_list)} ({(1 - val_ratio)*100:.1f}%) -> {train_txt_path}")
    print(f"  Val Samples:      {len(val_list)} ({val_ratio*100:.1f}%) -> {val_txt_path}")
    print(f"  Classes ({nc}):       {names}")
    print(f"  Manifest YAML:    {yaml_path}")
    print("=" * 70)
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MAGFiLO dataset manifest for SOAR and benchmark models")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
        help="Path to MAGFiLO root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/magfilo",
        help="Output directory to save train.txt, val.txt, and magfilo.yaml",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Treat all classes as single binary filament class (default: multiclass 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_magfilo(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        binary=args.binary,
    )
