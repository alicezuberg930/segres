# SOAR Benchmarks Suite (Q1 Journal Standard)

This directory contains standalone, reproducible implementations of all baseline models evaluated in the scientific paper against **SOAR** on high-resolution geometry manifolds ($1024\times 1024$ and $2048\times 2048$).

Each baseline supports:
1. **Full Training from Scratch (`train.py`)**: Automatic training with AMP mixed-precision, cosine learning rate decay, and micro-batch support ($B=1$).
2. **Epoch-by-Epoch Validation (`val.py`)**: Validates every epoch and records **mIoU**, **Dice**, Boundary-IoU (**bIoU**), and continuous centerline Dice (**clDice**). Automatically tracks and saves `best.pt` and `last.pt`.
3. **Symmetrical Inference Pipeline (`predict.py`)**: Generates identical deliverables as SOAR:
   - Raw multi-class semantic segmentation masks (`masks/*.png`)
   - Blended color overlays (`overlays/*.png`)
   - 4-panel visual comparison strips (`visualizations/strip_*.png`: `[Input | Ground Truth | Prediction | Overlay]`)
   - Structured `benchmark_summary.json` containing latency (ms/frame), real FPS, parameter count (M), and formatted LaTeX table rows.

---

## Supported Baseline Architectures

| Folder | Model | Architectural Paradigm | Variants / Options | Paper Params |
| :--- | :--- | :--- | :--- | :---: |
| `unet/` | **U-Net** | Encoder-Decoder Skip Connections | `--base-channels 64` | 17.27 M |
| `dlinknet/` | **D-LinkNet** | ResNet-34 + Central Dilated Convolutions | ResNet-34 Encoder | 26.24 M |
| `csnet/` | **CS-Net** | 1D Spatial & Channel Directional Attention | Curvilinear Strips | 15.08 M |
| `bisenetv2/` | **BiSeNet V2** | Bilateral Spatial & Semantic Streams (BGA) | Detail + Semantic | 3.42 M |
| `ddrnet/` | **DDRNet** | Deep Dual-Resolution Bilateral with DAPPM | `--variant slim` / `standard` | 5.68M / 20.14M |
| `pidnet/` | **PIDNet** | Three-Branch PID Controller with Pag & Bag | `--variant s` / `m` | 7.62M / 14.23M |
| `segformer/` | **SegFormer** | Hierarchical Mix Transformer with All-MLP Decoder | `--variant b0` / `b1` | 3.75M / 13.68M |

---

## Quick Start CLI

Use the master entrypoint `python benchmark_cli.py <model> <train|val|predict>` or run individual model CLIs.

### 1. Training from Scratch

```bash
# 1. Train U-Net
python benchmark_cli.py unet train \
    --data "path/to/data.yaml" \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1 \
    --lr 1e-4 \
    --device cuda \
    --project "checkpoints/benchmarks"

# 2. Train D-LinkNet
python benchmark_cli.py dlinknet train \
    --data "path/to/data.yaml" \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1

# 3. Train CS-Net
python benchmark_cli.py csnet train \
    --data "path/to/data.yaml" \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1

# 4. Train BiSeNet V2
python benchmark_cli.py bisenetv2 train \
    --data "path/to/data.yaml" \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1

# 5. Train DDRNet (DDRNet-23-slim or DDRNet-23)
python benchmark_cli.py ddrnet train \
    --data "path/to/data.yaml" \
    --variant slim \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1

# 6. Train PIDNet (PIDNet-S or PIDNet-M)
python benchmark_cli.py pidnet train \
    --data "path/to/data.yaml" \
    --variant s \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1

# 7. Train SegFormer (SegFormer-B0 or SegFormer-B1)
python benchmark_cli.py segformer train \
    --data "path/to/data.yaml" \
    --variant b0 \
    --imgsz 1024 \
    --epochs 50 \
    --batch 1
```

---

### 2. Validation (`val`)

Evaluate any trained checkpoint on the validation split:

```bash
python benchmark_cli.py ddrnet val \
    --weights "checkpoints/benchmarks/DDRNet_23_slim/best.pt" \
    --data "path/to/data.yaml" \
    --imgsz 1024 \
    --variant slim
```

---

### 3. Symmetrical Inference (`predict`)

Run full inference on test/validation sets and generate visual strips, masks, and summary tables:

```bash
python benchmark_cli.py pidnet predict \
    --weights "checkpoints/benchmarks/PIDNet_S/best.pt" \
    --data "path/to/data.yaml" \
    --split val \
    --imgsz 1024 \
    --output-dir "predictions/benchmarks" \
    --save-strips 10 \
    --variant s
```

---

## Output Deliverables

Running `predict` creates a unified directory:
```
predictions/benchmarks/<Model_Name>/
├── masks/                     # Raw PNG multi-class semantic masks
│   ├── sample_0001.png
│   └── ...
├── overlays/                  # Alpha-blended color overlays
│   ├── sample_0001.png
│   └── ...
├── visualizations/            # 4-panel strips [Input | Ground Truth | Prediction | Overlay]
│   ├── strip_sample_0001.png
│   └── ...
└── benchmark_summary.json     # Standardized JSON with latency, FPS, and LaTeX row
```

---

## Controlled Supervision Protocol (`--loss {soar,standard}`)

To meet top-tier Q1 journal peer-review standards (e.g., IEEE TPAMI, TIP, CVPR):
- **`--loss soar` (Default)**: Trains the baseline under SOAR's full composite loss objective:
  $$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{Boundary} + \lambda_4 \mathcal{L}_{clDice}$$
  This strictly isolates **neural architecture** as the sole independent experimental variable, guaranteeing that performance advantages stem from SOAR's spatial-frequency inductive biases rather than supervision bias.
- **`--loss standard`**: Trains using vanilla binary cross-entropy + Dice loss ($\mathcal{L}_{BCE} + \mathcal{L}_{Dice}$), used for ablation studies demonstrating how topological supervision affects classical vs. SOAR architectures.

