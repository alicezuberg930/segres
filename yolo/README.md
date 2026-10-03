# YOLO Inference & Semantic Benchmark Pipeline

This module provides a dedicated, symmetrical post-training inference pipeline for evaluating **Ultralytics YOLO segmentation models** (e.g., `yolov8n-seg` through `yolov8x-seg`, `yolo11n-seg` through `yolo11x-seg`) on ultra-high-resolution test datasets.

It is designed to produce **standardized, apples-to-apples empirical comparisons against SOAR** for scientific papers, rasterizing YOLO instance predictions into dense multi-class semantic segmentation masks and evaluating them using the exact same semantic metrics (**mIoU**, **Dice**, **Boundary IoU**, and topological **clDice**).

---

## Output Contract (Identical to SOAR)

Running YOLO prediction generates the exact same directory structure and artifacts as SOAR:
- `masks/{image_id}.png`: Single-channel categorical mask PNG (pixel values $0..K$).
- `masks_color/{image_id}.png`: 3-channel BGR color-coded semantic mask PNG.
- `visualizations/strip_{image_id}.png`: 4-panel visual strip `[Input Image | Ground Truth | YOLO Prediction | Overlay]` with class legend bar.
- `summary.json`: Model parameters, resolution, pure model latency, FPS, and full metric breakdown (mIoU, Dice, clDice, Boundary-IoU, per-class stats).

---

## Usage

Run inference on test or validation images using trained YOLO weights (`best.pt`):

```bash
python yolo_cli.py predict \
    --weights "/path/to/best.pt" \
    --data "/path/to/test_images" \
    --annotation-file "/path/to/annotations.json" \
    --img-size 2048 \
    --conf 0.25 \
    --iou 0.5 \
    --output-dir "predictions/yolo" \
    --device cuda
```

### Unlabeled Test Images (Competition / Submission Mode)

If ground truth annotations are not available:

```bash
python yolo_cli.py predict \
    --weights "/path/to/best.pt" \
    --data "/path/to/test_images" \
    --img-size 2048 \
    --output-dir "predictions/yolo" \
    --device cuda
```
