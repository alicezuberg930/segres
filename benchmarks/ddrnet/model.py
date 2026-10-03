from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)

        self.downsample = nn.Sequential()
        if stride != 1 or in_chan != out_chan:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module."""

    def __init__(self, in_chan: int, branch_chan: int, out_chan: int):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_chan, branch_chan, 1, padding=0),
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            ConvBNReLU(in_chan, branch_chan, 1, padding=0),
        )
        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            ConvBNReLU(in_chan, branch_chan, 1, padding=0),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            ConvBNReLU(in_chan, branch_chan, 1, padding=0),
        )

        self.process1 = ConvBNReLU(branch_chan, branch_chan, 3, padding=1)
        self.process2 = ConvBNReLU(branch_chan, branch_chan, 3, padding=1)
        self.process3 = ConvBNReLU(branch_chan, branch_chan, 3, padding=1)
        self.process4 = ConvBNReLU(branch_chan, branch_chan, 3, padding=1)

        self.compression = ConvBNReLU(in_chan, branch_chan, 1, padding=0)
        self.shortcut = ConvBNReLU(branch_chan * 5, out_chan, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x_comp = self.compression(x)

        x1 = self.scale1(x)
        x1 = F.interpolate(x1, size=size, mode="bilinear", align_corners=False)
        x1 = self.process1(x1)

        x2 = self.scale2(x)
        x2 = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)
        x2 = self.process2(x2 + x1)

        x3 = self.scale3(x)
        x3 = F.interpolate(x3, size=size, mode="bilinear", align_corners=False)
        x3 = self.process3(x3 + x2)

        x4 = self.scale4(x)
        x4 = F.interpolate(x4, size=size, mode="bilinear", align_corners=False)
        x4 = self.process4(x4 + x3)

        out = torch.cat([x_comp, x1, x2, x3, x4], dim=1)
        return self.shortcut(out)


class DDRNet(nn.Module):
    """
    DDRNet: Deep Dual-Resolution Network for Real-Time Semantic Segmentation (Hong et al., 2021).
    Supports variants: 'ddrnet_23_slim' and 'ddrnet_23'.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, variant: str = "slim"):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant

        # Capacity scaling
        if variant == "slim":
            base_ch = 32
            high_ch = 64
            low_ch = 128
        else:  # standard ddrnet-23
            base_ch = 64
            high_ch = 128
            low_ch = 256

        # Stem down to stride 4
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, base_ch, 3, stride=2),
            ConvBNReLU(base_ch, base_ch, 3, stride=2),
        )

        # Stage 1: Stride 4
        self.layer1 = nn.Sequential(
            BasicBlock(base_ch, base_ch),
            BasicBlock(base_ch, base_ch),
        )

        # Stage 2: Stride 8
        self.layer2 = nn.Sequential(
            BasicBlock(base_ch, high_ch, stride=2),
            BasicBlock(high_ch, high_ch),
        )

        # Bilateral Stage 3: High branch (stride 8) and Low branch (stride 16)
        self.high_stage3 = nn.Sequential(
            BasicBlock(high_ch, high_ch),
            BasicBlock(high_ch, high_ch),
        )
        self.low_stage3 = nn.Sequential(
            BasicBlock(high_ch, low_ch, stride=2),
            BasicBlock(low_ch, low_ch),
        )

        # Bilateral Fusion 1
        self.high_to_low_1 = ConvBNReLU(high_ch, low_ch, 3, stride=2)
        self.low_to_high_1 = ConvBNReLU(low_ch, high_ch, 1, padding=0)

        # Bilateral Stage 4: High branch (stride 8) and Low branch (stride 16)
        self.high_stage4 = nn.Sequential(
            BasicBlock(high_ch, high_ch),
            BasicBlock(high_ch, high_ch),
        )
        self.low_stage4 = nn.Sequential(
            BasicBlock(low_ch, low_ch),
            BasicBlock(low_ch, low_ch),
        )

        # Bilateral Fusion 2
        self.high_to_low_2 = ConvBNReLU(high_ch, low_ch, 3, stride=2)
        self.low_to_high_2 = ConvBNReLU(low_ch, high_ch, 1, padding=0)

        # DAPPM at low branch
        self.dappm = DAPPM(low_ch, base_ch, high_ch)

        # Final Head
        self.head = nn.Sequential(
            ConvBNReLU(high_ch * 2, high_ch, 3, padding=1),
            nn.Conv2d(high_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        # Stem & early stages
        x_stem = self.stem(x)
        x_l1 = self.layer1(x_stem)
        x_high = self.layer2(x_l1)

        # Bilateral Stage 3
        x_high = self.high_stage3(x_high)
        x_low = self.low_stage3(x_high)

        # Fusion 1
        x_low_up = F.interpolate(self.low_to_high_1(x_low), size=x_high.shape[2:], mode="bilinear", align_corners=False)
        x_high_down = self.high_to_low_1(x_high)
        x_high = x_high + x_low_up
        x_low = x_low + x_high_down

        # Bilateral Stage 4
        x_high = self.high_stage4(x_high)
        x_low = self.low_stage4(x_low)

        # Fusion 2
        x_low_up2 = F.interpolate(self.low_to_high_2(x_low), size=x_high.shape[2:], mode="bilinear", align_corners=False)
        x_high = x_high + x_low_up2

        # DAPPM
        x_dappm = self.dappm(x_low)
        x_dappm_up = F.interpolate(x_dappm, size=x_high.shape[2:], mode="bilinear", align_corners=False)

        # Head fusion
        fused = torch.cat([x_high, x_dappm_up], dim=1)
        logits = self.head(fused)

        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
