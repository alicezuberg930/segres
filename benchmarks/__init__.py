from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch.nn as nn

from .unet import UNet
from .dlinknet import DLinkNet
from .csnet import CSNet
from .bisenetv2 import BiSeNetV2
from .ddrnet import DDRNet, DDRNet23
from .pidnet import PIDNet
from .segformer import SegFormer
from .isdnet import ISDNet

def _build_pidnet_s(in_channels: int = 3, num_classes: int = 1, **kwargs) -> PIDNet:
    return PIDNet(in_channels=in_channels, num_classes=num_classes, variant="s", **kwargs)


def _build_pidnet_m(in_channels: int = 3, num_classes: int = 1, **kwargs) -> PIDNet:
    return PIDNet(in_channels=in_channels, num_classes=num_classes, variant="m", **kwargs)


def _build_segformer_b0(in_channels: int = 3, num_classes: int = 1, **kwargs) -> SegFormer:
    return SegFormer(in_channels=in_channels, num_classes=num_classes, variant="b0", **kwargs)


def _build_segformer_b1(in_channels: int = 3, num_classes: int = 1, **kwargs) -> SegFormer:
    return SegFormer(in_channels=in_channels, num_classes=num_classes, variant="b1", **kwargs)


def _build_ddrnet_slim(in_channels: int = 3, num_classes: int = 1, **kwargs) -> DDRNet:
    return DDRNet(in_channels=in_channels, num_classes=num_classes, variant="slim", **kwargs)


def _build_ddrnet_std(in_channels: int = 3, num_classes: int = 1, **kwargs) -> DDRNet:
    return DDRNet(in_channels=in_channels, num_classes=num_classes, variant="standard", **kwargs)


BENCHMARK_MODELS: Dict[str, Any] = {
    "unet": UNet,
    "dlinknet": DLinkNet,
    "d-linknet": DLinkNet,
    "csnet": CSNet,
    "cs-net": CSNet,
    "bisenetv2": BiSeNetV2,
    "bisenet": BiSeNetV2,
    "ddrnet": DDRNet,
    "ddrnet-slim": _build_ddrnet_slim,
    "ddrnet_slim": _build_ddrnet_slim,
    "ddrnet23-slim": _build_ddrnet_slim,
    "ddrnet-23-slim": _build_ddrnet_slim,
    "ddrnet23_slim": _build_ddrnet_slim,
    "ddrnet23": _build_ddrnet_std,
    "ddrnet-23": _build_ddrnet_std,
    "ddrnet_23": _build_ddrnet_std,
    "ddrnet-std": _build_ddrnet_std,
    "ddrnet_std": _build_ddrnet_std,
    "pidnet": _build_pidnet_s,
    "pidnet-s": _build_pidnet_s,
    "pidnet_s": _build_pidnet_s,
    "pidnet-m": _build_pidnet_m,
    "pidnet_m": _build_pidnet_m,
    "segformer": _build_segformer_b0,
    "segformer-b0": _build_segformer_b0,
    "segformer_b0": _build_segformer_b0,
    "segformer-b1": _build_segformer_b1,
    "segformer_b1": _build_segformer_b1,
    "isdnet": ISDNet,
    "isd-net": ISDNet,
}


def list_benchmark_models() -> List[str]:
    """Return list of supported benchmark model architecture keys."""
    return [
        "unet",
        "dlinknet",
        "csnet",
        "bisenetv2",
        "ddrnet-slim",
        "ddrnet-23",
        "pidnet-s",
        "pidnet-m",
        "isdnet",
        "segformer-b0",
        "segformer-b1",
    ]


def get_benchmark_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Instantiate a benchmark model by name."""
    key = name.lower().replace("-", "").replace("_", "")
    for reg_name, cls_or_fn in BENCHMARK_MODELS.items():
        if key == reg_name.lower().replace("-", "").replace("_", ""):
            return cls_or_fn(in_channels=in_channels, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown benchmark model: '{name}'. Available: {list_benchmark_models()}")


__all__ = [
    "UNet",
    "DLinkNet",
    "CSNet",
    "BiSeNetV2",
    "DDRNet",
    "DDRNet23",
    "PIDNet",
    "ISDNet",
    "SegFormer",
    "BENCHMARK_MODELS",
    "list_benchmark_models",
    "get_benchmark_model",
]
