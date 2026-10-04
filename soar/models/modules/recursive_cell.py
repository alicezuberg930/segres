from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalContextModule(nn.Module):
    """
    O(1) Spatial Complexity Global Context Stream.
    Extracts global image-wide semantics and modulates local feature representations
    via Feature-wise Linear Modulation (FiLM) [scale gamma, shift beta].
    Guarantees 100% full-canvas effective receptive field from iteration t=1.
    """

    def __init__(self, channels: int = 32, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels * 2, bias=True),
        )
        # Initialize final projection with small weights to allow gradient flow while starting near identity
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = x.shape
        vec = self.pool(x).view(b, c)
        mod = self.mlp(vec)
        gamma, beta = mod.chunk(2, dim=1)
        return gamma.view(b, c, 1, 1), beta.view(b, c, 1, 1)


class TRMRecurrentCell(nn.Module):
    """
    Micro Recursive Reasoning Cell (Weights shared 100% across all T time steps).
    Parameter count: ~135k parameters.

    Key Architectural Safeguards:
    1. Decoupled DC (LL) and AC (LH, HL, HH) logit heads with HF dampening to eliminate ripple artifacts.
    2. Strictly non-in-place operations for PyTorch gradient checkpointing compatibility.
    3. Positive bias initialization on step-size gate to prevent early freezing / dead updates.
    4. Dual-stream Local Curvilinear (LKR) + Global Context Modulation (FiLM).
    """

    def __init__(self, in_channels: int = 32, num_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Input projection:
        # Concatenates:
        # - h_t: Spatial Scratchpad state (in_channels)
        # - z_t: Sub-band Logits (num_classes * 4 sub-bands: LL, LH, HL, HH)
        # - e_x: Static image embedding from stem (in_channels)
        fusion_in = in_channels + (num_classes * 4) + in_channels
        self.pre_conv = nn.Sequential(
            nn.Conv2d(fusion_in, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_channels),
            nn.GELU(),
        )

        # 1. Local Curvilinear Branch (Multi-Scale Depthwise Convolutions)
        # Short-range 7x7 for fine filament edges
        self.dw_local = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False
        )
        # Extended-range 7x7 dilated (d=2, effective kernel 13x13) for topological continuity
        self.dw_dilated = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=6, dilation=2, groups=in_channels, bias=False
        )
        self.local_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_channels),
            nn.GELU(),
        )

        # 2. Global Context Stream (FiLM Modulation)
        self.global_context = GlobalContextModule(channels=in_channels, reduction=4)

        # 3. Spatial GRU Gate for Scratchpad Memory Update
        self.state_gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.state_candidate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, in_channels),
            nn.GELU(),
        )

        # 4. Decoupled DC (LL) and High-Frequency AC (LH, HL, HH) Logit Heads
        # LL Head: Predicts foundational background/foreground silhouette in standard logit range [-5, 5]
        self.delta_logit_ll = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1, bias=True),
        )

        # HF Head: Predicts directional high-frequency boundary details (LH, HL, HH)
        # Scaled down to prevent high-frequency noise leakage & ripple artifacts on flat regions
        self.delta_logit_hf = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, num_classes * 3, kernel_size=1, bias=True),
        )
        self.hf_scale = nn.Parameter(torch.tensor(0.1))

        # Initialize HF head weights near zero to guarantee clean initial DC silhouette
        nn.init.normal_(self.delta_logit_hf[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_logit_hf[-1].bias)

        # 5. Adaptive Step-Size Gate (with positive bias init to prevent early freezing)
        self.step_size_gate = nn.Conv2d(in_channels, num_classes * 4, kernel_size=1)
        # bias init to +1.0 ensures sigmoid(1.0) ~ 0.73 on step 0
        nn.init.constant_(self.step_size_gate.bias, 1.0)

    def forward(
        self,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        e_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent thinking iteration. Strictly non-in-place.
        """
        # Step A: Channel fusion
        x_in = torch.cat([h_t, z_t, e_x], dim=1)
        feat = self.pre_conv(x_in)

        # Step B: Local Curvilinear Multi-Scale Extraction
        loc1 = self.dw_local(feat)
        loc2 = self.dw_dilated(feat)
        loc_feat = self.local_proj(torch.cat([loc1, loc2], dim=1))

        # Step C: Global Context Modulation (FiLM)
        gamma, beta = self.global_context(loc_feat)
        modulated = loc_feat * (1.0 + gamma) + beta

        # Step D: Spatial Scratchpad Update (GRU-style gating)
        gate_input = torch.cat([h_t, modulated], dim=1)
        g_t = torch.sigmoid(self.state_gate(gate_input))
        h_cand = self.state_candidate(modulated)
        h_next = (1.0 - g_t) * h_t + g_t * h_cand

        # Step E: Decoupled Logit Residuals (Unbounded Logit Space)
        delta_ll = self.delta_logit_ll(h_next)
        delta_hf = self.delta_logit_hf(h_next) * self.hf_scale
        delta_z = torch.cat([delta_ll, delta_hf], dim=1)

        # Step F: Gated Residual Accumulation
        alpha = torch.sigmoid(self.step_size_gate(h_next))
        z_next = z_t + alpha * delta_z  # Strictly non-in-place addition

        return h_next, z_next
