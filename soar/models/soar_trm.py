from __future__ import annotations

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .modules.wavelet import HaarWavelet2D
from .modules.recursive_cell import TRMRecurrentCell


class SOARTinyRecursiveModel(nn.Module):
    """
    SOAR-TRM v2: Tiny Recursive Architecture for High-Resolution Segmentation.
    
    Inspired by:
    - Samsung SAIL Montreal's TRM (Tiny Recursive Model, Jolicoeur-Martineau et al. 2025)
    - CascadePSP (Cheng et al. CVPR 2020) & PointRend (Kirillov et al. CVPR 2020)
    - Haar Wavelet Orthogonal Decomposition & Reconstruction
    
    Key Properties:
    - Micro Parameter Count: ~140k - 150k parameters (over 115x smaller than standard UNet)
    - Gradient Checkpointing: Constant activation memory during training (O(1) memory w.r.t recursive steps T)
    - Full-Canvas Receptive Field: Global Context Module (FiLM) guarantees 100% canvas coverage from t=1
    - Unbounded Logits Space: No Sigmoid/Clamp saturation, immune to early commitment traps
    - Exact Wavelet Synthesis: Haar IDWT prevents checkerboard sub-pixel reconstruction artifacts
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        hidden_channels: int = 32,
        num_steps: int = 3,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = max(1, int(num_classes))
        self.hidden_channels = hidden_channels
        self.num_steps = max(1, int(num_steps))
        self.use_checkpointing = use_checkpointing

        # 1. Parameter-free 2D Wavelet Operator
        self.wavelet = HaarWavelet2D()

        # 2. Wavelet Static Image Stem
        # Input: 4 subbands * in_channels -> hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )

        # 3. Single Shared Recursive Cell (100% Shared Across Time Steps T)
        self.cell = TRMRecurrentCell(
            in_channels=hidden_channels,
            num_classes=self.num_classes,
        )

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_params(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature representation (static stem image embedding e_x)."""
        ll, lh, hl, hh = self.wavelet.dwt(x)
        dwt_features = torch.cat([ll, lh, hl, hh], dim=1)
        return self.stem(dwt_features)

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_all_steps: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with configurable recursive depth T and gradient checkpointing.
        
        Args:
            x: Input image tensor (B, C, H, W) where H, W are even.
            steps: Optional override for number of recursive thinking iterations T.
            return_all_steps: If True, returns trajectory list [Z_1, Z_2, ..., Z_T].
        
        Returns:
            Final full-resolution logits tensor (B, num_classes, H, W),
            or list of trajectory logits if return_all_steps=True.
        """
        t_steps = steps if steps is not None else self.num_steps
        b, _, h, w = x.shape

        # Step 1: 2D Haar DWT (Subsamples 2048 -> 1024 without aliasing or information loss)
        ll, lh, hl, hh = self.wavelet.dwt(x)
        dwt_features = torch.cat([ll, lh, hl, hh], dim=1)
        e_x = self.stem(dwt_features)

        # Step 2: Initialize Spatial Scratchpad H_0 and Logits Z_0
        h_half, w_half = h // 2, w // 2
        h_t = torch.zeros((b, self.hidden_channels, h_half, w_half), dtype=e_x.dtype, device=e_x.device)
        z_t = torch.zeros((b, self.num_classes * 4, h_half, w_half), dtype=e_x.dtype, device=e_x.device)

        trajectory_logits: List[torch.Tensor] = []

        # Step 3: Recursive Thinking Loop
        for step in range(t_steps):
            if self.training and self.use_checkpointing:
                # Gradient checkpointing: Recompute cell activations during backward pass
                # Memory complexity is O(1) with respect to T
                h_t, z_t = checkpoint(self.cell, h_t, z_t, e_x, use_reentrant=False)
            else:
                h_t, z_t = self.cell(h_t, z_t, e_x)

            # Reconstruct full-resolution logits via Parameter-Free Haar IDWT
            if return_all_steps or (step == t_steps - 1):
                z_ll = z_t[:, : self.num_classes]
                z_lh = z_t[:, self.num_classes : 2 * self.num_classes]
                z_hl = z_t[:, 2 * self.num_classes : 3 * self.num_classes]
                z_hh = z_t[:, 3 * self.num_classes :]
                out_full = self.wavelet.idwt(z_ll, z_lh, z_hl, z_hh)
                trajectory_logits.append(out_full)

        if return_all_steps:
            return trajectory_logits
        return trajectory_logits[-1]
