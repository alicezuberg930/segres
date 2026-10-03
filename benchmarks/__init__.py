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

BENCHMARK_MODELS: Dict[str, Any] = {
    "unet": UNet,
    "dlinknet": DLinkNet,
    "d-linknet": DLinkNet,
    "csnet": CSNet,
    "cs-net": CSNet,
    "bisenetv2": BiSeNetV2,
    "bisenet": BiSeNetV2,
    "ddrnet": DDRNet,
    "ddrnet23": DDRNet23,
    "ddrnet-23": DDRNet23,
    "pidnet": PIDNet,
    "pidnet-s": PIDNet,
    "segformer": SegFormer,
    "segformer-b0": SegFormer,
}


def list_benchmark_models() -> List[str]:
    """Return list of supported benchmark model architecture keys."""
    return ["unet", "dlinknet", "csnet", "bisenetv2", "ddrnet", "pidnet", "segformer"]


def get_benchmark_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """Instantiate a benchmark model by name."""
    key = name.lower().replace("-", "").replace("_", "")
    for reg_name, cls in BENCHMARK_MODELS.items():
        if key == reg_name.lower().replace("-", "").replace("_", ""):
            return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown benchmark model: '{name}'. Available: {list_benchmark_models()}")


__all__ = [
    "UNet",
    "DLinkNet",
    "CSNet",
    "BiSeNetV2",
    "DDRNet",
    "DDRNet23",
    "PIDNet",
    "SegFormer",
    "BENCHMARK_MODELS",
    "list_benchmark_models",
    "get_benchmark_model",
]
