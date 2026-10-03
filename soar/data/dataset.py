from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler, Subset

from .augment import (
    Compose,
    get_training_augmentation,
    get_validation_augmentation,
)
from .preprocess import (
    ComposePreprocess,
    build_preprocessor_from_config,
    get_mask_preprocessor,
    get_training_preprocessor,
    get_validation_preprocessor,
)
from .preprocess_config import PreprocessConfig

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SegmentationDataset(Dataset):
    """General-purpose dense segmentation dataset for high-resolution imagery.

    Supports diverse scientific and raster file formats (.npy, .fits, .png,
    etc.) and parses polygon/RLE annotations from standard COCO and normalized polygon
    structures.
    """

    SUPPORTED_EXTENSIONS = ('.npy', '.fits', '.fit', '.jpeg', '.jpg', '.png', '.bmp', '.tiff')

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        img_size: Tuple[int, int] = (1024, 1024),
        augment: bool = False,
        use_cache: bool = True,
        cache_limit: int = 8,
        annotation_file: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_dir: Optional[str | Path] = None,
        labels_dir: Optional[str | Path] = None,
        image_files: Optional[List[Path]] = None,
        names: Optional[Dict[int, str]] = None,
        transform: Optional[Callable] = None,
        auto: bool = False,
        preprocess_config: Optional[PreprocessConfig] = None,
        in_channels: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.is_train = self.split in ('train', 'training')
        self.has_gt = self.split in ('train', 'training', 'val', 'valid', 'validation')
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = max(1, int(num_classes))
        self.class_names = names or {i: f"class_{i}" for i in range(self.num_classes)}
        self._explicit_image_dir = Path(image_dir) if image_dir else None
        self._explicit_labels_dir = Path(labels_dir) if labels_dir else None
        self._explicit_image_files = image_files
        self.coco_cat_map: Dict[int, int] = {}
        self.augment = augment and self.is_train
        self.use_cache = use_cache
        self.cache_limit = max(0, cache_limit)
        self.transform = transform
        self.auto = auto

        # Configure augmentation pipelines
        if self.augment:
            self.augmentation = get_training_augmentation()
        else:
            self.augmentation = get_validation_augmentation()

        # Build preprocessing pipelines
        if preprocess_config is not None:
            self.preprocessor = build_preprocessor_from_config(preprocess_config)
            if preprocess_config.geometric.letterbox and preprocess_config.geometric.target_size:
                self.mask_preprocessor = get_mask_preprocessor(
                    img_size=preprocess_config.geometric.target_size,
                    auto=preprocess_config.geometric.auto,
                    scaleup=preprocess_config.geometric.scaleup,
                )
            else:
                self.mask_preprocessor = None
        else:
            auto_mode = auto if not self.is_train else False
            self.preprocessor = (
                get_training_preprocessor(img_size, auto=auto_mode)
                if self.is_train
                else get_validation_preprocessor(img_size, auto=auto_mode)
            )
            self.mask_preprocessor = get_mask_preprocessor(
                img_size,
                auto=auto_mode,
                scaleup=self.is_train,
            )

        # File directory indexing
        self.image_dir = self._resolve_image_dir()
        self.image_files = self._collect_image_files()
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {self.image_dir}")
        self.file_map: Dict[str, Path] = {f.name: f for f in self.image_files}
        self.file_map.update({f.stem: f for f in self.image_files})

        # Index label annotations
        self.annotations: Dict[str, Any] = {}
        self.img_to_masks: Dict[str, str] = {}
        self._load_annotations(annotation_file, mask_dir)

        # In-memory processing cache
        self._cache_store: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]] = {}

    def _resolve_image_dir(self) -> Path:
        """Locate root directory containing image targets."""
        if self._explicit_image_dir and self._explicit_image_dir.is_dir():
            return self._explicit_image_dir

        sub = self.split
        sub_aliases = [sub]
        if sub in ("val", "valid", "validation"):
            sub_aliases = ["val", "valid", "validation", "val2017"]
        elif sub in ("train", "training"):
            sub_aliases = ["train", "training", "train2017"]
        elif sub in ("test", "testing"):
            sub_aliases = ["test", "testing", "test2017"]

        candidate_paths = []
        for s in sub_aliases:
            candidate_paths.extend([
                self.data_root / "images" / s,
                self.data_root / s / "images",
                self.data_root / s / f"{s}_images",
                self.data_root / f"{s}_images",
                self.data_root / s,
            ])
        candidate_paths.extend([
            self.data_root / "images",
            self.data_root,
        ])

        for path in candidate_paths:
            if path.is_dir() and any(path.iterdir()):
                return path
        for path in candidate_paths:
            if path.is_dir():
                return path

        raise FileNotFoundError(
            f"Could not locate image directory for split '{self.split}'. Checked: {candidate_paths}"
        )

    def _collect_image_files(self) -> List[Path]:
        """Index all matching image files across supported extensions without duplicates."""
        if self._explicit_image_files is not None:
            return sorted([p for p in self._explicit_image_files if p.exists()])

        files: set[Path] = set()
        for ext in self.SUPPORTED_EXTENSIONS:
            files.update(self.image_dir.glob(f"*{ext}"))
            files.update(self.image_dir.glob(f"*{ext.upper()}"))
        
        # If no images found directly, search one level down (e.g. images/val2017/)
        if not files:
            for ext in self.SUPPORTED_EXTENSIONS:
                files.update(self.image_dir.glob(f"*/*{ext}"))
                files.update(self.image_dir.glob(f"*/*{ext.upper()}"))

        # Filter out mask files if primary/satellite image files are present in the same directory (e.g. DeepGlobe)
        primary_files = [f for f in files if not f.stem.endswith('_mask')]
        if primary_files:
            files = set(primary_files)

        return sorted(files)

    def _load_annotations(self, annotation_file: Optional[str], mask_dir: Optional[str]) -> None:
        """Parse annotations across COCO format, YOLO labels, or mask bitmaps.
        Prioritizes pre-rasterized mask directories to bypass expensive on-the-fly polygon parsing.
        """
        # 1. Prioritize explicit mask directory if provided
        if mask_dir:
            mask_path = Path(mask_dir)
            if not mask_path.is_absolute():
                mask_path = self.data_root / mask_dir
            if mask_path.is_dir() and any(mask_path.iterdir()):
                self._load_mask_directory(mask_path)
                if self.img_to_masks:
                    return

        # 2. Check candidate pre-rasterized mask directories
        candidate_dirs = [
            self.data_root / "masks" / self.split,
            self.data_root / self.split / "masks",
            self.data_root / "masks",
            self.data_root / f"{self.split}_masks",
            self.data_root / "train_masks",
            self.data_root / "train" / "masks",
        ]
        for candidate in candidate_dirs:
            if candidate.is_dir() and any(candidate.iterdir()):
                self._load_mask_directory(candidate)
                if self.img_to_masks:
                    return

        # Check if co-located mask files (e.g. *_mask.png) exist in image_dir
        if self.image_dir.is_dir() and any(f.stem.endswith('_mask') for f in self.image_dir.glob('*')):
            self._load_mask_directory(self.image_dir, only_mask_suffix=True)
            if self.img_to_masks:
                return

        # 3. Fall back to COCO JSON annotations if provided
        if annotation_file:
            ann_path = Path(annotation_file)
            if not ann_path.is_absolute():
                ann_path = self.data_root / annotation_file
            if ann_path.exists():
                self._load_coco_annotations(ann_path)
                return

        # 4. Fall back to YOLO labels
        yolo_labels_dir = self._resolve_yolo_labels_dir()
        if yolo_labels_dir and yolo_labels_dir.is_dir():
            self._load_yolo_annotations(yolo_labels_dir)
            return

        # 5. Fall back to standard candidate COCO JSON files
        sub = self.split
        sub_aliases = [sub]
        if sub in ("val", "valid", "validation"):
            sub_aliases = ["val2017", "val", "valid", "validation"]
        elif sub in ("train", "training"):
            sub_aliases = ["train2017", "train", "training"]

        candidate_files = []
        for s in sub_aliases:
            candidate_files.extend([
                self.data_root / "annotations" / f"instances_{s}.json",
                self.data_root / f"instances_{s}.json",
                self.data_root / "annotations" / f"{s}.json",
                self.data_root / f"{s}.json",
                self.data_root / s / "annotations.json",
            ])
        candidate_files.extend([
            self.data_root / "annotations.json",
            self.data_root / "train.json",
        ])
        for candidate in candidate_files:
            if candidate.exists():
                self._load_coco_annotations(candidate)
                return

    def _load_coco_annotations(self, ann_path: Path) -> None:
        """Parse COCO polygon structure."""
        with open(ann_path, "r", encoding="utf-8") as f:
            coco_payload = json.load(f)

        categories = sorted([cat["id"] for cat in coco_payload.get("categories", []) if "id" in cat])
        if categories:
            self.coco_cat_map = {cat_id: idx for idx, cat_id in enumerate(categories)}
        else:
            self.coco_cat_map = {}

        id_to_filename: Dict[int, str] = {}
        for img_info in coco_payload.get("images", []):
            img_id = img_info["id"]
            fname = img_info["file_name"]
            id_to_filename[img_id] = fname
            self.img_to_masks[fname] = "coco"

        for ann in coco_payload.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in id_to_filename:
                continue
            fname = id_to_filename[img_id]
            if fname not in self.annotations:
                self.annotations[fname] = []
            self.annotations[fname].append(ann)

    def _resolve_yolo_labels_dir(self) -> Optional[Path]:
        """Locate YOLO label directory."""
        if self._explicit_labels_dir and self._explicit_labels_dir.is_dir():
            return self._explicit_labels_dir

        # Try mapping image_dir replacing 'images' with 'labels'
        img_dir_str = str(self.image_dir)
        if "images" in img_dir_str:
            labels_candidate = Path(img_dir_str.replace("images", "labels"))
            if labels_candidate.is_dir():
                return labels_candidate

        sub = self.split
        sub_aliases = [sub]
        if sub in ("val", "valid", "validation"):
            sub_aliases = ["val", "valid", "validation", "val2017"]
        elif sub in ("train", "training"):
            sub_aliases = ["train", "training", "train2017"]
        elif sub in ("test", "testing"):
            sub_aliases = ["test", "testing", "test2017"]

        candidate_paths = []
        for s in sub_aliases:
            candidate_paths.extend([
                self.data_root / "labels" / s,
                self.data_root / s / "labels",
                self.data_root / f"{s}_labels",
                self.data_root / s / f"{s}_labels",
            ])
        candidate_paths.append(self.data_root / "labels")
        for path in candidate_paths:
            if path.is_dir() and any(path.glob("*.txt")):
                return path
        for path in candidate_paths:
            if path.is_dir():
                return path
        return None

    def _load_yolo_annotations(self, labels_dir: Path) -> None:
        """Index YOLO annotations."""
        for label_file in labels_dir.glob("*.txt"):
            img_name = label_file.stem
            self.img_to_masks[img_name] = "yolo"
            self.annotations[img_name] = str(label_file)

    def _parse_yolo_annotation(self, label_file: Path, img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """Parse raw coordinates from YOLO text labels."""
        annotations = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])

                if len(parts) > 5:
                    coords = list(map(float, parts[1:]))
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * img_width)
                        y = int(coords[i + 1] * img_height)
                        polygon.extend([x, y])

                    annotations.append({
                        "class_id": class_id,
                        "type": "segmentation",
                        "segmentation": [polygon],
                    })
                else:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x_center_abs = int(x_center * img_width)
                    y_center_abs = int(y_center * img_height)
                    width_abs = int(width * img_width)
                    height_abs = int(height * img_height)

                    x1 = x_center_abs - width_abs // 2
                    y1 = y_center_abs - height_abs // 2
                    x2 = x_center_abs + width_abs // 2
                    y2 = y_center_abs + height_abs // 2
                    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]

                    annotations.append({
                        "class_id": class_id,
                        "type": "detection",
                        "segmentation": [polygon],
                    })

        return annotations

    def _load_mask_directory(self, mask_path: Path, only_mask_suffix: bool = False) -> None:
        """Index mask bitmaps stored directly in directories."""
        for mask_file in mask_path.glob("*"):
            if mask_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
                stem = mask_file.stem
                if only_mask_suffix and not stem.endswith('_mask'):
                    continue
                file_str = str(mask_file)
                self.img_to_masks[stem] = file_str
                self.img_to_masks[mask_file.name] = file_str
                if stem.endswith('_mask'):
                    prefix = stem[:-5]
                    self.img_to_masks[prefix] = file_str
                    self.img_to_masks[f"{prefix}_sat"] = file_str
                    self.img_to_masks[f"{prefix}_image"] = file_str

    def _read_image(self, path: Path) -> np.ndarray:
        """Load image arrays from disk across formats."""
        ext = path.suffix.lower()

        if ext == '.npy':
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32)
            if not arr.flags['C_CONTIGUOUS'] or not arr.flags['F_CONTIGUOUS']:
                arr = arr.copy()
            return arr

        if ext in ('.fits', '.fit'):
            try:
                from astropy.io import fits
                with fits.open(path) as hdul:
                    arr = np.asarray(hdul[0].data, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            except ImportError:
                raise ImportError("astropy library is required to read FITS files.")
            max_val = np.nanmax(arr) if arr.size > 0 else 0.0
            if max_val > 1.0:
                arr /= 255.0 if max_val <= 255.0 else max_val
            return arr

        # Fast native OpenCV decode (2-4x faster than PIL Image.open)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback to PIL for rare/unsupported image variants
            with Image.open(path) as pil_img:
                img = np.array(pil_img)

        # Handle color and bit depths
        if img.ndim == 2:
            # Grayscale uint8: values strictly in [0, 255], zero nanmax overhead
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                max_val = np.nanmax(img) if img.size > 0 else 0.0
                if max_val > 1.0:
                    img /= 255.0 if max_val <= 255.0 else max_val
            return img
        elif img.ndim == 3:
            # Convert BGR -> RGB when 3 channels
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                max_val = np.nanmax(img) if img.size > 0 else 0.0
                if max_val > 1.0:
                    img /= 255.0 if max_val <= 255.0 else max_val
            return img

        return img

    def _read_mask(self, img_path: Path, raw_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Retrieve or construct the ground truth mask for a given sample."""
        img_name = img_path.name
        stem = img_path.stem

        mask_info = (
            self.img_to_masks.get(img_name)
            or self.img_to_masks.get(stem)
            or self.img_to_masks.get(stem.replace('_sat', '_mask'))
            or self.img_to_masks.get(stem.replace('_image', '_mask'))
            or self.img_to_masks.get(stem.replace('_img', '_mask'))
            or self.img_to_masks.get(f"{stem}_mask")
        )
        if not mask_info:
            # Check Ultralytics standard label path convention: /images/ -> /labels/, suffix -> .txt
            p_posix = img_path.as_posix()
            if "/images/" in p_posix:
                cand_label = Path(p_posix.replace("/images/", "/labels/")).with_suffix(".txt")
                if cand_label.is_file():
                    self.img_to_masks[stem] = "yolo"
                    self.annotations[stem] = str(cand_label)
                    mask_info = "yolo"
        if mask_info:
            if mask_info == "coco":
                return self._generate_coco_mask(img_name, raw_shape)
            elif mask_info == "yolo":
                return self._generate_yolo_mask(img_name, raw_shape)
            else:
                mask_path = Path(mask_info)
                if mask_path.exists():
                    # Fast OpenCV native grayscale decode
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        with Image.open(mask_path) as mask_img:
                            mask = np.array(mask_img.convert('L'), dtype=np.uint8)
                    if self.num_classes == 1:
                        # Convert to binary {0.0, 1.0}
                        if mask.dtype == np.uint8 and mask.max() > 1:
                            mask = (mask > 127).astype(np.float32)
                        else:
                            mask = mask.astype(np.float32)
                    else:
                        mask_u8 = mask.astype(np.int64)
                        max_label = int(mask_u8.max())
                        one_hot = np.zeros((mask.shape[0], mask.shape[1], self.num_classes), dtype=np.float32)
                        if max_label >= self.num_classes:
                            # 1-indexed (0 background, 1..num_classes foreground)
                            for c in range(self.num_classes):
                                one_hot[..., c] = (mask_u8 == (c + 1)).astype(np.float32)
                        else:
                            # 0-indexed
                            for c in range(self.num_classes):
                                one_hot[..., c] = (mask_u8 == c).astype(np.float32)
                        mask = one_hot

                    if not mask.flags['C_CONTIGUOUS'] or not mask.flags['F_CONTIGUOUS']:
                        mask = np.ascontiguousarray(mask)
                    return mask

        return None

    def _generate_coco_mask(self, img_name: str, raw_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Rasterize COCO polygon coordinates into a binary or multi-class mask."""
        polys = self.annotations.get(img_name)
        if polys is None:
            polys = self.annotations.get(Path(img_name).stem)

        if not polys:
            return None

        if raw_shape is not None:
            h, w = raw_shape
        else:
            img_path = self.file_map.get(img_name) or self.file_map.get(Path(img_name).stem)
            if img_path is None:
                return None
            img = self._read_image(img_path)
            h, w = img.shape[:2]

        if self.num_classes == 1:
            mask = np.zeros((h, w), dtype=np.float32)
            for ann in polys:
                segmentation = ann.get("segmentation", [])
                if isinstance(segmentation, list):
                    for poly in segmentation:
                        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(mask, [pts], color=1.0)
        else:
            mask = np.zeros((self.num_classes, h, w), dtype=np.float32)
            for ann in polys:
                cat_id = ann.get("category_id", 0)
                cid = self.coco_cat_map.get(cat_id, cat_id if cat_id < self.num_classes else 0)
                if 0 <= cid < self.num_classes:
                    segmentation = ann.get("segmentation", [])
                    if isinstance(segmentation, list):
                        for poly in segmentation:
                            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.fillPoly(mask[cid], [pts], color=1.0)
            mask = np.transpose(mask, (1, 2, 0))

        return np.ascontiguousarray(mask)

    def _generate_yolo_mask(self, img_name: str, raw_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Rasterize YOLO annotations into a binary or multi-class mask."""
        label_file_path = self.annotations.get(img_name) or self.annotations.get(Path(img_name).stem)
        if not label_file_path:
            return None

        if raw_shape is not None:
            h, w = raw_shape
        else:
            img_path = self.file_map.get(img_name) or self.file_map.get(Path(img_name).stem)
            if img_path is None:
                return None
            img = self._read_image(img_path)
            h, w = img.shape[:2]

        label_file = Path(label_file_path)
        annotations = self._parse_yolo_annotation(label_file, w, h)

        if self.num_classes == 1:
            mask = np.zeros((h, w), dtype=np.float32)
            for ann in annotations:
                segmentation = ann.get("segmentation", [])
                if isinstance(segmentation, list):
                    for poly in segmentation:
                        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(mask, [pts], color=1.0)
        else:
            mask = np.zeros((self.num_classes, h, w), dtype=np.float32)
            for ann in annotations:
                cid = ann.get("class_id", 0)
                if 0 <= cid < self.num_classes:
                    segmentation = ann.get("segmentation", [])
                    if isinstance(segmentation, list):
                        for poly in segmentation:
                            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.fillPoly(mask[cid], [pts], color=1.0)
            mask = np.transpose(mask, (1, 2, 0))

        return np.ascontiguousarray(mask)

    def _get_processed_data(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """Retrieve preprocessed image and geometric mask with caching."""
        key = str(path)
        if key in self._cache_store:
            return self._cache_store[key]

        raw_img = self._read_image(path)

        if raw_img.ndim == 3 and raw_img.shape[0] == 1:
            raw_img = raw_img[0]

        if not raw_img.flags['C_CONTIGUOUS'] or not raw_img.flags['F_CONTIGUOUS']:
            raw_img = raw_img.copy()

        result = self.preprocessor(raw_img)
        if len(result) == 3:
            processed_img, valid_mask, meta = result
        else:
            processed_img, valid_mask = result
            meta = {}

        meta = dict(meta) if meta else {}
        meta['raw_shape'] = raw_img.shape[:2]

        if processed_img.ndim == 2:
            processed_img = processed_img[np.newaxis, ...]
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask[np.newaxis, ...]

        processed_img = np.ascontiguousarray(processed_img)
        if valid_mask is not None:
            valid_mask = np.ascontiguousarray(valid_mask)

        # Preprocess and cache ground truth mask
        mask = None
        if self.has_gt:
            mask = self._read_mask(path, raw_shape=meta.get('raw_shape'))
            target_h, target_w = (processed_img.shape[0], processed_img.shape[1]) if (processed_img.ndim == 3 and processed_img.shape[-1] in (1, 3, 4)) else processed_img.shape[-2:]
            if mask is None:
                if self.num_classes == 1:
                    mask = np.zeros((target_h, target_w), dtype=np.float32)
                else:
                    mask = np.zeros((target_h, target_w, self.num_classes), dtype=np.float32)

            if valid_mask is None:
                valid_mask = np.ones((target_h, target_w), dtype=np.float32)

            if self.mask_preprocessor is not None:
                mask_processed, _, _ = self.mask_preprocessor(mask)
                mask = mask_processed
                if self.num_classes == 1:
                    mask = mask.squeeze()
                    if mask.ndim != 2:
                        mask = mask.reshape(target_h, target_w) if mask.size == target_h * target_w else mask
                    if mask.shape != (target_h, target_w):
                        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                else:
                    if mask.shape[:2] != (target_h, target_w):
                        resized = [cv2.resize(mask[..., c], (target_w, target_h), interpolation=cv2.INTER_NEAREST) for c in range(self.num_classes)]
                        mask = np.stack(resized, axis=-1)

            mask = np.ascontiguousarray(mask)

        res = (processed_img, mask, valid_mask, meta)
        if len(self._cache_store) < self.cache_limit:
            self._cache_store[key] = res

        return res

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        img, mask, valid_mask, meta = self._get_processed_data(img_path)

        # Handle ground truth masks for train/val splits
        if self.has_gt:
            # Clone cached references when augmenting to prevent cache corruption
            if self.augment:
                img_np = img.copy()
                mask_np = mask.copy() if mask is not None else None
                vm_np = valid_mask.copy() if valid_mask is not None else None

                # Ensure img_np is HWC for OpenCV augmentations
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4) and img_np.shape[-1] not in (1, 3, 4):
                    img_np = np.transpose(img_np, (1, 2, 0))

                m_hwc = mask_np if (mask_np is not None and mask_np.ndim == 3) else (mask_np[..., None] if mask_np is not None else None)
                vm_2d = vm_np.squeeze() if vm_np is not None else np.ones(img_np.shape[:2], dtype=np.float32)
                vm_hwc = vm_2d[..., None]
                stacked_masks = np.concatenate([m_hwc, vm_hwc], axis=-1) if m_hwc is not None else vm_hwc

                img_np, stacked_masks = self.augmentation(img_np, stacked_masks)
                img = img_np
                mask = stacked_masks[..., :self.num_classes]
                valid_mask = stacked_masks[..., self.num_classes:]

                if self.num_classes == 1:
                    mask = mask.squeeze(-1)
                valid_mask = valid_mask.squeeze(-1)

            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask) if mask is not None else None
            valid_mask = np.ascontiguousarray(valid_mask) if valid_mask is not None else None

            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]
                elif mask.ndim == 3 and mask.shape[-1] == self.num_classes:
                    mask = np.transpose(mask, (2, 0, 1))

            if valid_mask is not None:
                if valid_mask.ndim == 2:
                    valid_mask = valid_mask[np.newaxis, ...]
                elif valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
                    valid_mask = np.transpose(valid_mask, (2, 0, 1))

            sample = {
                'image': torch.from_numpy(img).float(),
                'mask': torch.from_numpy(mask).float() if mask is not None else torch.zeros((self.num_classes, img.shape[-2], img.shape[-1])),
                'valid_mask': torch.from_numpy(valid_mask).float() if valid_mask is not None else torch.ones((1, img.shape[-2], img.shape[-1])),
                'image_id': img_path.stem,
                'has_object': bool((mask > 0).any()) if mask is not None else False,
                'meta': meta or {},
            }
        else:
            img = np.ascontiguousarray(img)
            if valid_mask is None:
                valid_mask = np.ones((1, img.shape[-2], img.shape[-1]), dtype=np.float32)
            else:
                valid_mask = np.ascontiguousarray(valid_mask)
                if valid_mask.ndim == 2:
                    valid_mask = valid_mask[np.newaxis, ...]

            if img.ndim == 2:
                img = img[np.newaxis, ...]

            sample = {
                'image': torch.from_numpy(img).float(),
                'valid_mask': torch.from_numpy(valid_mask).float(),
                'mask': None,
                'image_id': img_path.stem,
                'has_object': False,
                'meta': meta or {},
            }

        # Enforce canonical 3D tensor layout (C, H, W) before batch assembly
        if sample['image'].ndim == 2:
            sample['image'] = sample['image'].unsqueeze(0)
        elif sample['image'].ndim == 3 and sample['image'].shape[-1] in (1, 3):
            sample['image'] = sample['image'].permute(2, 0, 1)

        # Match expected in_channels
        if sample['image'].shape[0] == 1 and self.in_channels == 3:
            sample['image'] = sample['image'].repeat(3, 1, 1)

        if sample['valid_mask'].ndim == 2:
            sample['valid_mask'] = sample['valid_mask'].unsqueeze(0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble individual dataset items into uniformly dimensioned batch tensors."""
    if not batch:
        return {}

    def ensure_chw(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return tensor

    images = [ensure_chw(b['image']) for b in batch]
    valid_masks = [ensure_chw(b['valid_mask']) for b in batch]

    has_mask = batch[0].get('mask') is not None

    res: Dict[str, Any] = {
        'image': torch.stack(images, dim=0),
        'valid_mask': torch.stack(valid_masks, dim=0),
        'image_id': [b['image_id'] for b in batch],
        'meta': [b.get('meta', {}) for b in batch],
    }

    if has_mask:
        masks = [ensure_chw(b['mask']) for b in batch]
        res['mask'] = torch.stack(masks, dim=0)
        res['has_object'] = torch.tensor([b['has_object'] for b in batch], dtype=torch.bool)
    else:
        res['mask'] = None
        res['has_object'] = None

    return res


class HybridBalancedSampler(Sampler):
    """
    Epoch-level balanced sampler without replacement for positive samples.
    Guarantees every positive sample is seen at least once per epoch, paired with a balanced
    subsample of negative background tiles to enforce the positive:negative ratio without
    unnecessary sample duplication.
    """

    def __init__(
        self,
        positive_indices: Sequence[int],
        negative_indices: Sequence[int],
        positive_ratio: float = 0.7,
        generator: Optional[torch.Generator] = None,
    ):
        self.positive_indices = list(positive_indices)
        self.negative_indices = list(negative_indices)
        self.positive_ratio = max(0.01, min(0.99, positive_ratio))
        self.generator = generator

    def __iter__(self):
        g = self.generator
        pos_perm = torch.randperm(len(self.positive_indices), generator=g).tolist()
        shuffled_pos = [self.positive_indices[i] for i in pos_perm]

        n_pos = len(self.positive_indices)
        target_n_neg = int(round(n_pos * (1.0 - self.positive_ratio) / self.positive_ratio))

        if len(self.negative_indices) >= target_n_neg:
            neg_perm = torch.randperm(len(self.negative_indices), generator=g)[:target_n_neg].tolist()
            sampled_neg = [self.negative_indices[i] for i in neg_perm]
        else:
            neg_t = torch.tensor(self.negative_indices, dtype=torch.long)
            rand_idx = torch.randint(0, len(self.negative_indices), (target_n_neg,), generator=g)
            sampled_neg = neg_t[rand_idx].tolist()

        combined = shuffled_pos + sampled_neg
        perm = torch.randperm(len(combined), generator=g).tolist()
        for idx in perm:
            yield combined[idx]

    def __len__(self) -> int:
        n_pos = len(self.positive_indices)
        target_n_neg = int(round(n_pos * (1.0 - self.positive_ratio) / self.positive_ratio))
        return n_pos + target_n_neg


def build_balanced_sampler(
    dataset: Union[SegmentationDataset, Subset],
    positive_ratio: float = 0.7,
    mode: str = "hybrid",
) -> Optional[Sampler]:
    """
    Constructs sample weights or hybrid sampler to enforce target ratio of
    object-containing tiles to background tiles without disk reads.
    Modes:
      'hybrid': All positives seen once per epoch + random subsample of negatives (no replacement).
      'weighted': WeightedRandomSampler with replacement.
    """
    base_ds: SegmentationDataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))

    pos_indices = []
    neg_indices = []

    for local_idx, orig_idx in enumerate(indices):
        path = base_ds.image_files[orig_idx]
        name = path.name
        stem = path.stem
        # O(1) in-memory check without disk read
        has_ann = (name in base_ds.annotations) or (stem in base_ds.annotations)
        is_pos = False
        if has_ann:
            anns = base_ds.annotations.get(name, base_ds.annotations.get(stem))
            if isinstance(anns, list) and len(anns) > 0:
                is_pos = True
            elif isinstance(anns, str) and len(anns) > 0:
                is_pos = True
        elif name in base_ds.img_to_masks or stem in base_ds.img_to_masks:
            is_pos = True

        if is_pos:
            pos_indices.append(local_idx)
        else:
            neg_indices.append(local_idx)

    if not pos_indices or not neg_indices:
        return None  # All positive or all negative, uniform sampler is optimal

    if mode == "hybrid":
        return HybridBalancedSampler(pos_indices, neg_indices, positive_ratio=positive_ratio)

    w_pos = positive_ratio / len(pos_indices)
    w_neg = (1.0 - positive_ratio) / len(neg_indices)
    weights = torch.zeros(len(indices), dtype=torch.float64)
    weights[pos_indices] = w_pos
    weights[neg_indices] = w_neg
    return WeightedRandomSampler(weights=weights, num_samples=len(indices), replacement=True)