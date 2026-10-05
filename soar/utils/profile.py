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
    elif "isdnet" in k:
        return r"ISDNet \cite{guo2022isdnet}"
    return model_name


def _format_tri_metrics(d_metrics: Optional[Dict[str, float]]) -> Tuple[str, str, str]:
    if not d_metrics:
        return "--", "--", "--"
    miou = d_metrics.get("iou", d_metrics.get("mIoU"))
    biou = d_metrics.get("boundary_iou", d_metrics.get("bIoU"))
    cldice = d_metrics.get("cldice", d_metrics.get("clDice"))
    s_miou = f"{miou * 100.0:.2f}" if miou is not None else "--"
    s_biou = f"{biou * 100.0:.2f}" if biou is not None else "--"
    s_cldice = f"{cldice * 100.0:.2f}" if cldice is not None else "--"
    return s_miou, s_biou, s_cldice


def format_latex_row_table1(
    model_name: str,
    params_m: float,
    flops_g: float,
    metrics: Dict[str, Any],
    fps: Optional[float] = None,
    dataset: Optional[str] = None,
) -> str:
    """Format row for Table I (tab:main_benchmark, 1024x1024):
    Paradigm & Model & Params & FLOPs & FPS & DeepGlobe (mIoU, bIoU, clDice) & CRACK500 (mIoU, bIoU, clDice) & MAGFiLO (mIoU, bIoU, clDice) \\
    """
    name_str = get_latex_model_name(model_name)
    paradigm = get_model_paradigm(model_name)
    fps_str = f"{fps:.1f}" if fps is not None else "--"

    # Support nested dataset dict or single dataset mapping
    dg_m, cr_m, mg_m = None, None, None
    if isinstance(metrics, dict) and any(k in metrics for k in ("deepglobe", "crack500", "magfilo")):
        dg_m = metrics.get("deepglobe")
        cr_m = metrics.get("crack500")
        mg_m = metrics.get("magfilo")
    elif dataset:
        ds_k = dataset.lower()
        if "deepglobe" in ds_k or "road" in ds_k:
            dg_m = metrics
        elif "crack" in ds_k:
            cr_m = metrics
        elif "magfilo" in ds_k or "filament" in ds_k:
            mg_m = metrics
    else:
        # Default single metrics
        dg_m = metrics

    dg_iou, dg_bnd, dg_cl = _format_tri_metrics(dg_m)
    cr_iou, cr_bnd, cr_cl = _format_tri_metrics(cr_m)
    mg_iou, mg_bnd, mg_cl = _format_tri_metrics(mg_m)

    return (
        f"{paradigm} & {name_str} & {params_m:.2f} & {flops_g:.2f} & {fps_str} & "
        f"{dg_iou} & {dg_bnd} & {dg_cl} & "
        f"{cr_iou} & {cr_bnd} & {cr_cl} & "
        f"{mg_iou} & {mg_bnd} & {mg_cl} \\\\"
    )


def format_latex_row_table2(
    model_name: str,
    params_m: float,
    flops_g: float,
    metrics: Dict[str, Any],
    peak_vram_mb: Optional[float] = None,
    dataset: Optional[str] = None,
) -> str:
    """Format row for Table II (tab:highres_stress, 2048x2048):
    Paradigm & Model & Params & FLOPs & Peak VRAM & DeepGlobe (mIoU, bIoU, clDice) & CRACK500 (mIoU, bIoU, clDice) & MAGFiLO (mIoU, bIoU, clDice) \\
    """
    name_str = get_latex_model_name(model_name)
    paradigm = get_model_paradigm(model_name)
    vram_str = f"{peak_vram_mb:.1f}" if peak_vram_mb is not None else "--"

    dg_m, cr_m, mg_m = None, None, None
    if isinstance(metrics, dict) and any(k in metrics for k in ("deepglobe", "crack500", "magfilo")):
        dg_m = metrics.get("deepglobe")
        cr_m = metrics.get("crack500")
        mg_m = metrics.get("magfilo")
    elif dataset:
        ds_k = dataset.lower()
        if "deepglobe" in ds_k or "road" in ds_k:
            dg_m = metrics
        elif "crack" in ds_k:
            cr_m = metrics
        elif "magfilo" in ds_k or "filament" in ds_k:
            mg_m = metrics
    else:
        dg_m = metrics

    dg_iou, dg_bnd, dg_cl = _format_tri_metrics(dg_m)
    cr_iou, cr_bnd, cr_cl = _format_tri_metrics(cr_m)
    mg_iou, mg_bnd, mg_cl = _format_tri_metrics(mg_m)

    return (
        f"{paradigm} & {name_str} & {params_m:.2f} & {flops_g:.2f} & {vram_str} & "
        f"{dg_iou} & {dg_bnd} & {dg_cl} & "
        f"{cr_iou} & {cr_bnd} & {cr_cl} & "
        f"{mg_iou} & {mg_bnd} & {mg_cl} \\\\"
    )


# Backward-compatible alias
format_latex_row_table3 = format_latex_row_table2


def format_latex_row_table_eval(
    model_name: str,
    metrics: Dict[str, Any],
    dataset: Optional[str] = None,
) -> str:
    """Format row for Table 3 (tab:results_1024) and Table 4 (tab:results_2048):
    Model & DeepGlobe (mIoU, bIoU, clDice) & CRACK500 (mIoU, bIoU, clDice) & MAGFiLO (mIoU, bIoU, clDice) \\
    """
    # Use clean model display name (e.g. U-Net, BiSeNet V2, PIDNet-S, ISDNet, SegFormer-B0, SOAR1-Nano1, etc.)
    k = model_name.lower().replace("-", "").replace("_", "")
    if "soarnano" in k:
        name_str = "SOAR1-Nano1"
    elif "soarsmall" in k:
        name_str = "SOAR1-Small1"
    elif "soarmedium" in k or k == "soar":
        name_str = "SOAR1-Medium1"
    elif "unet" in k:
        name_str = "U-Net"
    elif "bisenet" in k:
        name_str = "BiSeNet V2"
    elif "pidnet" in k:
        name_str = "PIDNet-S"
    elif "isdnet" in k:
        name_str = "ISDNet"
    elif "segformer" in k:
        name_str = "SegFormer-B0"
    else:
        name_str = model_name

    dg_m, cr_m, mg_m = None, None, None
    if isinstance(metrics, dict) and any(k in metrics for k in ("deepglobe", "crack500", "magfilo")):
        dg_m = metrics.get("deepglobe")
        cr_m = metrics.get("crack500")
        mg_m = metrics.get("magfilo")
    elif dataset:
        ds_k = dataset.lower()
        if "deepglobe" in ds_k or "road" in ds_k:
            dg_m = metrics
        elif "crack" in ds_k:
            cr_m = metrics
        elif "magfilo" in ds_k or "filament" in ds_k:
            mg_m = metrics
    else:
        dg_m = metrics

    dg_iou, dg_bnd, dg_cl = _format_tri_metrics(dg_m)
    cr_iou, cr_bnd, cr_cl = _format_tri_metrics(cr_m)
    mg_iou, mg_bnd, mg_cl = _format_tri_metrics(mg_m)

    return (
        f"{name_str}\n"
        f"& {dg_iou} & {dg_bnd} & {dg_cl} "
        f"& {cr_iou} & {cr_bnd} & {cr_cl} "
        f"& {mg_iou} & {mg_bnd} & {mg_cl} \\\\"
    )


def compute_resolution_retention(
    m_1024: float,
    m_2048: float,
) -> Tuple[float, float]:
    """Compute degradation Delta M = M_2048 - M_1024 and retention R_M = (M_2048 / M_1024) * 100%."""
    delta_m = m_2048 - m_1024
    retention_pct = (m_2048 / m_1024 * 100.0) if m_1024 > 1e-6 else 0.0
    return delta_m, retention_pct


def compute_macro_average(metric_values: List[float]) -> float:
    """Compute unweighted macro-average M_bar = (1/N) * sum(M_d)."""
    if not metric_values:
        return 0.0
    return sum(metric_values) / len(metric_values)


