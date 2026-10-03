from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml


def convert_coco_to_yolo_seg(
    coco_json_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    val_split: float = 0.1,
    samples: Optional[int] = None,
    seed: int = 42,
) -> Path:
    """
    Convert COCO polygon annotations into Ultralytics YOLO segmentation format.
    Generates normalized TXT labels, train/val file path manifests, and a ready-to-train data.yaml.

    Args:
        coco_json_path: Path to COCO annotation JSON file.
        image_dir: Directory containing images.
        output_dir: Target directory where labels and manifests will be generated.
        val_split: Fraction of images to reserve for validation (0.0 to 1.0).
        samples: If specified, limit to the first N samples (for fair sanity check comparisons).
        seed: Random seed for deterministic train/val splitting.

    Returns:
        Path to the generated data.yaml configuration file.
    """
    coco_json_path = Path(coco_json_path).resolve()
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not coco_json_path.is_file():
        raise FileNotFoundError(f"COCO JSON file not found at: {coco_json_path}")
    if not image_dir.is_dir():
        # Check if train_images subdirectory exists inside image_dir
        if (image_dir / "train_images").is_dir():
            image_dir = image_dir / "train_images"
        elif (image_dir / "images").is_dir():
            image_dir = image_dir / "images"

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_payload = json.load(f)

    # 1. Parse categories
    raw_cats = coco_payload.get("categories", [])
    sorted_cats = sorted([cat for cat in raw_cats if "id" in cat], key=lambda c: c["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    class_names = {idx: cat.get("name", f"class_{idx}") for idx, cat in enumerate(sorted_cats)}

    if not class_names:
        class_names = {0: "filament"}
        cat_id_to_idx = {1: 0}

    # 2. Index available images on disk
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    disk_images = {p.name: p for p in image_dir.glob("*") if p.suffix.lower() in supported_exts}
    if not disk_images:
        # Search recursively if images are in subdirectories
        disk_images = {p.name: p for p in image_dir.rglob("*") if p.suffix.lower() in supported_exts}

    # 3. Map COCO image IDs and filenames
    img_info_map: Dict[Any, Dict[str, Any]] = {}
    id_to_filename: Dict[Any, str] = {}
    for img_info in coco_payload.get("images", []):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        id_to_filename[img_id] = fname
        if fname not in img_info_map:
            img_info_map[fname] = img_info

    # 4. Group annotations per image filename
    file_to_anns: Dict[str, List[Dict[str, Any]]] = {}
    for ann in coco_payload.get("annotations", []):
        img_id = ann.get("image_id")
        fname = id_to_filename.get(img_id)
        if fname:
            file_to_anns.setdefault(fname, []).append(ann)

    # 5. Filter images that exist both on disk and in annotations
    matched_filenames = sorted([fname for fname in disk_images.keys() if fname in img_info_map])
    if not matched_filenames:
        # Fallback to all images on disk
        matched_filenames = sorted(list(disk_images.keys()))

    if samples is not None and samples > 0:
        matched_filenames = matched_filenames[:samples]
        print(f"[YOLO Converter] Sample limit applied: selecting first {len(matched_filenames)} images.")

    # 6. Split train and validation
    if val_split <= 0.0 or samples is not None or len(matched_filenames) <= 2:
        train_files = matched_filenames
        val_files = matched_filenames  # Evaluate on training samples for sanity check
    else:
        import random
        rng = random.Random(seed)
        shuffled = list(matched_filenames)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        val_files = sorted(shuffled[:n_val])
        train_files = sorted(shuffled[n_val:])

    # 7. Create directory structure
    labels_dir = output_dir / "labels"
    labels_train_dir = labels_dir / "train"
    labels_val_dir = labels_dir / "val"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    def write_yolo_labels(filenames: List[str], target_dir: Path) -> List[str]:
        image_paths = []
        for fname in filenames:
            img_path = disk_images[fname]
            image_paths.append(str(img_path.resolve()))
            stem = img_path.stem
            label_file = target_dir / f"{stem}.txt"

            info = img_info_map.get(fname, {})
            w = float(info.get("width", 2048))
            h = float(info.get("height", 2048))

            anns = file_to_anns.get(fname, [])
            lines = []
            for ann in anns:
                cat_id = ann.get("category_id", 1)
                class_idx = cat_id_to_idx.get(cat_id, 0)
                segmentations = ann.get("segmentation", [])

                if isinstance(segmentations, list):
                    for poly in segmentations:
                        if len(poly) < 6:  # Need at least 3 points (x, y)
                            continue
                        normalized = []
                        for i in range(0, len(poly) - 1, 2):
                            x_norm = max(0.0, min(1.0, poly[i] / w))
                            y_norm = max(0.0, min(1.0, poly[i + 1] / h))
                            normalized.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                        lines.append(f"{class_idx} " + " ".join(normalized))

            with open(label_file, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(lines) + ("\n" if lines else ""))

        return image_paths

    train_img_paths = write_yolo_labels(train_files, labels_train_dir)
    val_img_paths = write_yolo_labels(val_files, labels_val_dir)

    # 8. Write train.txt and val.txt manifests
    train_manifest = output_dir / "train.txt"
    val_manifest = output_dir / "val.txt"
    with open(train_manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(train_img_paths) + "\n")
    with open(val_manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(val_img_paths) + "\n")

    # 9. Write dataset.yaml
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "names": {int(k): str(v) for k, v in class_names.items()},
        "nc": len(class_names),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print(f"[YOLO Converter] Successfully prepared YOLO dataset:")
    print(f"  - Output Dir:     {output_dir}")
    print(f"  - Classes ({len(class_names)}):     {list(class_names.values())}")
    print(f"  - Train Samples:  {len(train_files)}")
    print(f"  - Val Samples:    {len(val_files)}")
    print(f"  - Manifest YAML:  {yaml_path}")

    return yaml_path
