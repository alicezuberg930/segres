# YOLO Segmentation Benchmark Pipeline

This module provides a standalone pipeline for training, validating, and benchmarking **Ultralytics YOLO segmentation models** (e.g., `yolov8n-seg`, `yolov8s-seg`, `yolo11n-seg`, `yolo11s-seg`) on the ultra-high-resolution filament dataset.

It is designed to produce **side-by-side empirical comparisons against SOAR** for scientific papers, evaluating both models on identical dense semantic metrics (**mIoU**, **Dice**, **Boundary IoU**, and topological **clDice**).

---

## 1. Quickstart: One-Command Benchmark

To run an end-to-end fair comparison (dataset conversion $\to$ training $\to$ dense semantic evaluation):

```bash
python yolo_cli.py benchmark \
    --annotation-file "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json" \
    --data "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train" \
    --model yolov8n-seg.pt \
    --imgsz 2048 \
    --epochs 50 \
    --lr 1e-4 \
    --batch 1 \
    --samples 8 \
    --no-augment \
    --device cuda
```

---

## 2. Modular Step-by-Step Usage

### Step 1: Prepare Dataset (COCO $\to$ YOLO Segmentation Format)
Converts COCO JSON polygon annotations into normalized YOLO `.txt` format and automatically generates `data.yaml`:

```bash
python yolo_cli.py prepare \
    --annotation-file "/path/to/MAGFiLO_1.0_Annotations_kaggle2026_train.json" \
    --data "/path/to/train_images" \
    --output-dir "dataset_yolo" \
    --val-split 0.1
```

*Flags:*
- `--samples 8`: Extract only the first 8 samples for quick few-shot / sanity check comparison against SOAR.

### Step 2: Train YOLO Segmentation Model
Trains using Ultralytics YOLO with hyperparameter controls matching SOAR:

```bash
python yolo_cli.py train \
    --model yolov8n-seg.pt \
    --data dataset_yolo/data.yaml \
    --imgsz 2048 \
    --epochs 50 \
    --lr 1e-4 \
    --batch 1 \
    --device cuda \
    --no-augment \
    --project checkpoints/yolo_benchmark \
    --name yolov8n_2048
```

### Step 3: Semantic Metric Evaluation (mIoU, clDice, bIoU)
Standard YOLO evaluation only measures Instance Box & Mask mAP. This step rasterizes YOLO's instance predictions into multi-class semantic segmentation masks ($C \times H \times W$) and evaluates them using the **exact same metric formulas as SOAR**:

```bash
python yolo_cli.py val \
    --weights checkpoints/yolo_benchmark/yolov8n_2048/weights/best.pt \
    --data dataset_yolo/data.yaml \
    --imgsz 2048 \
    --save-dir checkpoints/yolo_benchmark/yolov8n_2048/visualizations
```

Output format for direct inclusion in manuscript comparison tables:
```text
-----------------------------------------------------------------------------------------------
Model / Architecture   Samples  mIoU       Dice       Prec       Recall     bIoU       clDice    
-----------------------------------------------------------------------------------------------
YOLO (yolov8n_2048)    8        0.xxxx     0.xxxx     0.xxxx     0.xxxx     0.xxxx     0.xxxx    
-----------------------------------------------------------------------------------------------
Per-class IoU breakdown:
  Left           : 0.xxxx
  Right          : 0.xxxx
  Unidentifiable : 0.xxxx
  Ambiguous      : 0.xxxx
-----------------------------------------------------------------------------------------------
```

---

## 3. Paper Comparison Baseline Setup

For a fair, controlled comparison between SOAR and YOLO:

| Hyperparameter | SOAR-Nano1 | YOLOv8n-seg |
| :--- | :---: | :---: |
| **Input Resolution** | $2048 \times 2048$ | $2048 \times 2048$ (or $1024 \times 1024$) |
| **Augmentation** | `--no-augment` | `--no-augment` |
| **Batch Size** | 1 | 1 |
| **Learning Rate** | `1e-4` | `1e-4` |
| **Epochs** | 50 | 50 |
| **Evaluation Metrics** | mIoU, Dice, bIoU, clDice | mIoU, Dice, bIoU, clDice |
