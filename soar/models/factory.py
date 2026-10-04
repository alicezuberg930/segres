from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from .model import SegmentationModel
from .soar_trm import SOARTinyRecursiveModel
from benchmarks import (
    UNet,
    DLinkNet,
    CSNet,
    BiSeNetV2,
    DDRNet,
    PIDNet,
    SegFormer,
    BENCHMARK_MODELS,
    list_benchmark_models,
)

SOAR_PRESETS: Dict[str, str] = {
    "soar": "configs/models/soar_medium1.yaml",
    "soarmicro": "configs/models/soar_micro1.yaml",
    "soarmicro1": "configs/models/soar_micro1.yaml",
    "soartrm": "configs/models/soar_micro1.yaml",
    "trm": "configs/models/soar_micro1.yaml",
    "micro": "configs/models/soar_micro1.yaml",
    "soarnano": "configs/models/soar_nano1.yaml",
    "soarnano1": "configs/models/soar_nano1.yaml",
    "soarsmall": "configs/models/soar_small1.yaml",
    "soarsmall1": "configs/models/soar_small1.yaml",
    "soarmedium": "configs/models/soar_medium1.yaml",
    "soarmedium1": "configs/models/soar_medium1.yaml",
    "soarlarge": "configs/models/soar_large1.yaml",
    "soarlarge1": "configs/models/soar_large1.yaml",
    "soarxlarge": "configs/models/soar_xlarge1.yaml",
    "soarxlarge1": "configs/models/soar_xlarge1.yaml",
}


def list_models() -> List[str]:
    """Return all recognized model names across SOAR and benchmark baselines."""
    return [
        "soar",
        "soar_trm",
        "soar_micro",
        "soar_micro1",
        "soar_nano",
        "soar_small",
        "soar_medium",
        "soar_large",
        "soar_xlarge",
    ] + list_benchmark_models()



def _resolve_yaml_path(path_str: str) -> Optional[Path]:
    """Resolve YAML config path across workspace root and configs/ directory."""
    p = Path(path_str)
    if p.is_file():
        return p

    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / path_str
    if candidate.is_file():
        return candidate

    candidate_cfg = repo_root / "configs" / "models" / Path(path_str).name
    if candidate_cfg.is_file():
        return candidate_cfg

    return None


def build_model(
    model: Union[str, nn.Module, Dict[str, Any], Path] = "soar",
    in_channels: int = 3,
    num_classes: int = 1,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Universal model factory for SOAR and all peer benchmark baselines.

    Args:
        model: Model name ('unet', 'dlinknet', 'csnet', 'bisenetv2', 'ddrnet', 'pidnet',
               'segformer', 'soar', 'soar-trm', 'soar_micro1'), YAML config file path, or nn.Module instance.
        in_channels: Number of input image channels (default: 3).
        num_classes: Number of target segmentation classes (default: 1).
        verbose: Whether to log model creation and parameter counts.

    Returns:
        Instantiated nn.Module ready for training, evaluation, or inference.
    """
    if isinstance(model, nn.Module):
        return model

    if isinstance(model, dict):
        return SegmentationModel(cfg=model, ch=in_channels, nc=num_classes, verbose=verbose, **kwargs)

    model_str = str(model).strip()
    norm_key = model_str.lower().replace("-", "").replace("_", "")

    # 1. Check peer benchmark baselines
    for reg_name, cls in BENCHMARK_MODELS.items():
        if norm_key == reg_name.lower().replace("-", "").replace("_", ""):
            net = cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
            if verbose:
                params_m = sum(p.numel() for p in net.parameters()) / 1e6
                print(f"[{cls.__name__}] Initialized with in_channels={in_channels}, num_classes={num_classes} ({params_m:.2f}M params)")
            return net

    # 2. Check SOAR presets (including soar-trm, soar_micro1, etc.)
    for preset_name, rel_path in SOAR_PRESETS.items():
        if norm_key == preset_name:
            resolved = _resolve_yaml_path(rel_path)
            if resolved:
                return SegmentationModel(cfg=str(resolved), ch=in_channels, nc=num_classes, verbose=verbose, **kwargs)

    # 3. Check if path to YAML file
    resolved_yaml = _resolve_yaml_path(model_str)
    if resolved_yaml:
        return SegmentationModel(cfg=str(resolved_yaml), ch=in_channels, nc=num_classes, verbose=verbose, **kwargs)

    # If it ends with .yaml but wasn't found
    if model_str.endswith(".yaml") or model_str.endswith(".yml"):
        raise FileNotFoundError(f"Model configuration YAML not found: '{model_str}'")

    raise ValueError(
        f"Unknown model architecture: '{model}'.\n"
        f"Available options:\n"
        f"  - Benchmark Baselines: {list_benchmark_models()}\n"
        f"  - SOAR Variants:       ['soar', 'soar_nano', 'soar_small', 'soar_medium', 'soar_large', 'soar_xlarge']\n"
        f"  - Custom YAML:         path/to/model.yaml"
    )
