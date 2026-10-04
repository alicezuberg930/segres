from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class HaarWavelet2D(nn.Module):
    """
    Exact, parameter-free 2D Discrete Haar Wavelet Transform (DWT) and 
    Inverse Discrete Haar Wavelet Transform (IDWT).
    
    Guarantees:
    - Zero information loss on even dimensions
    - Perfect reconstruction: IDWT(DWT(X)) == X within floating-point epsilon (< 1e-6)
    - Zero checkerboard artifacts (orthogonal spatial synthesis)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decomposes spatial tensor (B, C, H, W) into 4 frequency sub-bands:
        - LL: Low-frequency approximation (DC component)
        - LH: Horizontal high-frequency edges
        - HL: Vertical high-frequency edges
        - HH: Diagonal high-frequency corners
        Output shapes: (B, C, H // 2, W // 2)
        """
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise ValueError(f"Spatial dimensions must be even for Haar DWT, got shape {x.shape}")

        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x_even = x01[:, :, :, 0::2]
        x_odd = x01[:, :, :, 1::2]
        y_even = x02[:, :, :, 0::2]
        y_odd = x02[:, :, :, 1::2]

        ll = x_even + x_odd + y_even + y_odd
        lh = -x_even - x_odd + y_even + y_odd
        hl = -x_even + x_odd - y_even + y_odd
        hh = x_even - x_odd - y_even + y_odd
        return ll, lh, hl, hh

    @staticmethod
    def idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes 4 sub-bands (B, C, H, W) back to native resolution (B, C, 2H, 2W).
        Completely parameter-free, strictly preserves spatial differentiability.
        """
        b, c, h, w = ll.shape
        out = torch.empty((b, c, 2 * h, 2 * w), dtype=ll.dtype, device=ll.device)
        out[:, :, 0::2, 0::2] = (ll - lh - hl + hh) / 2.0
        out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) / 2.0
        out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) / 2.0
        out[:, :, 1::2, 1::2] = (ll + lh + hl + hh) / 2.0
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dwt(x)
