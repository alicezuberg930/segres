# SOAR Unified Benchmarks Suite (Q1 Journal Standard)

This module provides clean, unified implementations of all baseline architectures evaluated in the scientific paper against **SOAR** on high-resolution geometry manifolds ($1024\times 1024$ and $2048\times 2048$).

All baselines share the **exact same unified training, validation, and prediction pipeline**, guaranteeing 100% scientific fairness and zero supervision or evaluation discrepancies.

---

## Supported Baseline Architectures

| Model Key | Class | Paradigm | Params |
| :--- | :--- | :--- | :---: |
| `unet` | [`UNet`](file:///e:/GithubProjects/segres/benchmarks/unet.py) | Encoder-Decoder Skip Connections (Ronneberger et al., MICCAI 2015) | 17.26 M |
| `dlinknet` | [`DLinkNet`](file:///e:/GithubProjects/segres/benchmarks/dlinknet.py) | ResNet-34 + Central Dilated Linkage (Zhou et al., CVPR 2018) | 26.24 M |
| `csnet` | [`CSNet`](file:///e:/GithubProjects/segres/benchmarks/csnet.py) | 1D Spatial & Channel Strip Attention (Mou et al., MICCAI 2019) | 15.08 M |
| `bisenetv2` | [`BiSeNetV2`](file:///e:/GithubProjects/segres/benchmarks/bisenetv2.py) | Bilateral Spatial & Semantic Streams (Yu et al., IJCV 2021) | 3.42 M |
| `ddrnet` | [`DDRNet`](file:///e:/GithubProjects/segres/benchmarks/ddrnet.py) | Deep Dual-Resolution Bilateral with DAPPM (Hong et al., 2021) | 5.68M / 20.14M |
| `pidnet` | [`PIDNet`](file:///e:/GithubProjects/segres/benchmarks/pidnet.py) | Three-Branch PID Controller with Pag & Bag (Xu et al., CVPR 2023) | 7.62M / 14.23M |
| `segformer` | [`SegFormer`](file:///e:/GithubProjects/segres/benchmarks/segformer.py) | Hierarchical Mix Transformer + All-MLP Decoder (Xie et al., NeurIPS 2021) | 3.75M / 13.68M |

---

## Unified Command-Line Interface (`cli.py`)

You can train, validate, or run inference on **any** model (SOAR or peer baselines) using a single command:

### 1. Training from Scratch

```bash
# Train U-Net
python cli.py train --model unet --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train D-LinkNet
python cli.py train --model dlinknet --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train CS-Net
python cli.py train --model csnet --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train BiSeNet V2
python cli.py train --model bisenetv2 --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train DDRNet
python cli.py train --model ddrnet --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train PIDNet
python cli.py train --model pidnet --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train SegFormer
python cli.py train --model segformer --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp

# Train SOAR (Default)
python cli.py train --model soar --data "/path/to/dataset" --epochs 50 --imgsz 1024 --device cuda --amp
```

*(Note: Model-first syntax like `python cli.py unet train ...` and `python benchmark_cli.py unet train ...` are also supported for 100% backwards compatibility.)*

---

### 2. Validation (`val`)

Evaluate any trained checkpoint:

```bash
python cli.py val --weights checkpoints/unet/best.pt --data "/path/to/dataset" --imgsz 1024 --device cuda
```
*(The model architecture and number of classes are automatically detected from the checkpoint.)*

---

### 3. Inference & Qualitative Output (`predict`)

Generate multi-class masks and 4-panel visual comparison strips:

```bash
python cli.py predict --weights checkpoints/unet/best.pt --data "/path/to/dataset" --imgsz 1024 --device cuda
```

Output directory structure:
```
predictions/<model_name>/
├── masks/             # Multi-class semantic mask PNGs
├── masks_color/       # Color-coded semantic mask PNGs
└── visualizations/    # 4-panel strips: [Input | Ground Truth | Prediction | Overlay]
```

---

## Controlled Supervision Protocol (`--loss {soar,standard}`)

- **`--loss soar` (Default)**: Trains the baseline under SOAR's full composite loss objective:
  $$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{Boundary} + \lambda_4 \mathcal{L}_{clDice}$$
  Guarantees that performance differences stem purely from neural architecture rather than supervision discrepancies.
- **`--loss standard`**: Trains using vanilla binary cross-entropy + Dice loss ($\mathcal{L}_{BCE} + \mathcal{L}_{Dice}$) for loss ablation experiments.
