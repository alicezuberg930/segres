from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


def profile_model(
    model: nn.Module,
    img_size: Tuple[int, int] = (1024, 1024),
    in_channels: int = 3,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Compute total parameters (in Millions) and forward FLOPs (in GFLOPs).

    Args:
        model: PyTorch model module.
        img_size: Input spatial resolution (H, W).
        in_channels: Number of input channels.
        device: Computation device for profiling ('cpu' or 'cuda').

    Returns:
        (params_m, flops_g)
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = 0.0

    try:
        from torch.utils.flop_counter import FlopCounterMode

        dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        was_training = model.training
        model.eval()

        if dev.type == "cuda":
            dummy = torch.randn(1, in_channels, img_size[0], img_size[1], device=dev)
            flop_mode = FlopCounterMode(display=False)
            with torch.no_grad():
                with flop_mode:
                    _ = model(dummy)
            flops_g = flop_mode.get_total_flops() / 1e9
        else:
            # CPU profiling: run on 256x256 sub-canvas and scale quadratically with area
            test_h, test_w = (min(img_size[0], 256), min(img_size[1], 256))
            scale = (img_size[0] / test_h) * (img_size[1] / test_w)

            dummy = torch.randn(1, in_channels, test_h, test_w, device=dev)
            flop_mode = FlopCounterMode(display=False)
            with torch.no_grad():
                with flop_mode:
                    _ = model(dummy)
            flops_g = (flop_mode.get_total_flops() * scale) / 1e9

        if was_training:
            model.train()
    except Exception:
        flops_g = 0.0

    return round(params_m, 2), round(flops_g, 2)


def get_model_paradigm(model_name: str) -> str:
    """Return the architectural paradigm classification matching Table II in soar_paper.tex."""
    k = model_name.lower().replace("-", "").replace("_", "")
    if "soar" in k:
        return "WaveStem + LKR + SpectralCtx"
    elif "unet" in k:
        return "Encoder-Decoder Skip"
    elif "dlinknet" in k:
        return "ResNet-Central Dilated"
    elif "csnet" in k:
        return "1D Spatial-Channel Attention"
    elif "bisenet" in k:
        return "Bilateral Spatial/Semantic"
    elif "stdc" in k:
        return "Single-Stream Dense Concatenate"
    elif "ddrnet" in k:
        return "Dual-Resolution Bilateral"
    elif "pidnet" in k:
        return "Three-Branch PID Controller"
    elif "pointrend" in k:
        return "Point-Sampling Sub-Pixel Rendering"
    elif "cascadepsp" in k:
        return "Cascade Global-Local Refinement"
    elif "isdnet" in k:
        return "Shallow Full-Res + Deep Downsampled"
    elif "segformer" in k:
        return "Hierarchical Transformer + MLP"
    elif "segnext" in k:
        return "Multi-Scale Conv Attention (MSCA)"
    elif "mask2former" in k:
        return "Masked-Attention Mask Transformer"
    elif "efficientsam" in k:
        return "Masked Image Pretrained Foundation"
    return "Deep Neural Network"


def get_latex_model_name(model_name: str) -> str:
    """Return LaTeX-formatted model name with citations matching soar_paper.tex."""
    k = model_name.lower().replace("-", "").replace("_", "")
    if "soarnano" in k:
        return r"\textbf{SOAR1-Nano1 (Ours)}"
    elif "soarsmall" in k:
        return r"\textbf{SOAR1-Small1 (Ours)}"
    elif "soarmedium" in k or k == "soar":
        return r"\textbf{SOAR1-Medium1 (Ours)}"
    elif "soarlarge" in k:
        return r"\textbf{SOAR1-Large1 (Ours)}"
    elif "soarxlarge" in k:
        return r"\textbf{SOAR1-XLarge1 (Ours)}"
    elif "unet" in k:
        return r"U-Net \cite{ronneberger2015u}"
    elif "dlinknet" in k:
        return r"D-LinkNet \cite{zhou2018dlinknet}"
    elif "csnet" in k:
        return r"CS-Net \cite{mou2019csnet}"
    elif "bisenet" in k:
        return r"BiSeNet V2 \cite{yu2021bisenet}"
    elif "ddrnetslim" in k or "ddrnet23slim" in k:
        return r"DDRNet-23-slim \cite{hong2021deep}"
    elif "ddrnet" in k:
        return r"DDRNet-23 \cite{hong2021deep}"
    elif "pidnets" in k:
        return r"PIDNet-S \cite{xu2023pidnet}"
    elif "pidnetm" in k:
        return r"PIDNet-M \cite{xu2023pidnet}"
    elif "pidnet" in k:
        return r"PIDNet-S \cite{xu2023pidnet}"
    elif "segformerb0" in k or k == "segformer":
        return r"SegFormer-B0 \cite{xie2021segformer}"
    elif "segformerb1" in k:
        return r"SegFormer-B1 \cite{xie2021segformer}"
    return model_name


def format_latex_row_table1(
    model_name: str,
    params_m: float,
    flops_g: float,
    metrics: Dict[str, float],
    peak_vram_mb: Optional[float] = None,
    latency_ms: Optional[float] = None,
    fps: Optional[float] = None,
) -> str:
    """Format row for Table I (tab:main_benchmark, 1024x1024):
    Paradigm & Model & Params & FLOPs & Peak VRAM & Latency & FPS & mIoU & bIoU & clDice \\
    """
    name_str = get_latex_model_name(model_name)
    paradigm = get_model_paradigm(model_name)
    vram_str = f"{peak_vram_mb:.1f}" if peak_vram_mb is not None else "--"
    lat_str = f"{latency_ms:.1f}" if latency_ms is not None else "--"
    fps_str = f"{fps:.1f}" if fps is not None else "--"

    miou = metrics.get("iou", 0.0) * 100.0
    biou = metrics.get("boundary_iou", 0.0) * 100.0
    cldice = metrics.get("cldice", 0.0) * 100.0

    return (
        f"{paradigm} & {name_str} & {params_m:.2f} & {flops_g:.2f} & "
        f"{vram_str} & {lat_str} & {fps_str} & "
        f"{miou:.2f} & {biou:.2f} & {cldice:.2f} \\\\"
    )


def format_latex_row_table2(
    model_name: str,
    params_m: float,
    flops_g: float,
    metrics: Dict[str, float],
    peak_vram_mb: Optional[float] = None,
    latency_ms: Optional[float] = None,
    fps: Optional[float] = None,
) -> str:
    """Format row for Table II (tab:highres_stress, 2048x2048):
    Model & Paradigm & Params & FLOPs & Peak VRAM & Latency & FPS & clDice & mIoU & bIoU \\
    """
    name_str = get_latex_model_name(model_name)
    paradigm = get_model_paradigm(model_name)
    vram_str = f"{peak_vram_mb:.1f}" if peak_vram_mb is not None else "--"
    lat_str = f"{latency_ms:.1f}" if latency_ms is not None else "--"
    fps_str = f"{fps:.1f}" if fps is not None else "--"

    miou = metrics.get("iou", 0.0) * 100.0
    biou = metrics.get("boundary_iou", 0.0) * 100.0
    cldice = metrics.get("cldice", 0.0) * 100.0

    return (
        f"{name_str} & {paradigm} & {params_m:.2f} & {flops_g:.2f} & "
        f"{vram_str} & {lat_str} & {fps_str} & "
        f"{cldice:.2f} & {miou:.2f} & {biou:.2f} \\\\"
    )


# Backward-compatible alias
format_latex_row_table3 = format_latex_row_table2
