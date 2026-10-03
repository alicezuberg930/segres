from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """1D Channel Attention Module for inter-channel feature recalibration."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(8, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """
    Directional Spatial Attention Module for Curvilinear Structure Continuity.
    Decomposes 2D spatial attention into orthogonal 1D directional strip convolutions.
    """

    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv_h = nn.Conv2d(in_channels, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.conv_w = nn.Conv2d(in_channels, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.conv_h(x)
        w_feat = self.conv_w(x)
        attn = self.sigmoid(self.bn(h_feat + w_feat))
        return x * attn


class CSBlock(nn.Module):
    """Combined Channel and Spatial Attention Unit."""

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x)
        out = self.sa(out)
        return out + x


class CSConv(nn.Module):
    """Double Conv with CS Attention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cs = CSBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cs(self.conv(x))


class CSNet(nn.Module):
    """
    CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation
    (Mou et al., MICCAI 2019).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels

        # Encoder
        self.inc = CSConv(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), CSConv(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), CSConv(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), CSConv(c * 4, c * 8))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool2d(2), CSConv(c * 8, c * 16))

        # Decoder with attention-guided skip connections
        self.up1 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec1 = CSConv(c * 16, c * 8)

        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec2 = CSConv(c * 8, c * 4)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec3 = CSConv(c * 4, c * 2)

        self.up4 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec4 = CSConv(c * 2, c)

        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bot(x4)

        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([x4, u1], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([x3, u2], dim=1))

        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([x2, u3], dim=1))

        u4 = self.up4(d3)
        d4 = self.dec4(torch.cat([x1, u4], dim=1))

        return self.outc(d4)
