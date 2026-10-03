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


class Pag(nn.Module):
    """Pixel-attention-guided fusion module."""

    def __init__(self, in_p: int, in_i: int):
        super().__init__()
        self.proj_i = ConvBNReLU(in_i, in_p, 1, padding=0)
        self.f_p = nn.Sequential(
            ConvBNReLU(in_p, in_p, 1, padding=0),
            nn.Sigmoid(),
        )
        self.f_i = nn.Sequential(
            ConvBNReLU(in_i, in_p, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        if i.shape[2:] != p.shape[2:]:
            i = F.interpolate(i, size=p.shape[2:], mode="bilinear", align_corners=False)
        i_proj = self.proj_i(i)
        sig_p = self.f_p(p)
        sig_i = self.f_i(i)
        edge = 1.0 - torch.abs(sig_p - sig_i)
        return edge * p + (1.0 - edge) * i_proj


class Bag(nn.Module):
    """Boundary-attention-guided fusion module."""

    def __init__(self, in_p: int, in_i: int, out_chan: int):
        super().__init__()
        self.conv = ConvBNReLU(in_p, out_chan, 3, padding=1)
        self.conv_i = ConvBNReLU(in_i, out_chan, 1, padding=0) if in_i != out_chan else nn.Identity()

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if i.shape[2:] != p.shape[2:]:
            i = F.interpolate(i, size=p.shape[2:], mode="bilinear", align_corners=False)
        if d.shape[2:] != p.shape[2:]:
            d = F.interpolate(d, size=p.shape[2:], mode="bilinear", align_corners=False)

        feat_p = self.conv(p)
        feat_i = self.conv_i(i)
        edge_weight = torch.sigmoid(d)
        return edge_weight * feat_p + (1.0 - edge_weight) * feat_i


class PAPPM(nn.Module):
    """Parallel Aggregation Pyramid Pooling Module."""

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

        self.shortcut = ConvBNReLU(branch_chan * 4 + in_chan, out_chan, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        s1 = F.interpolate(self.scale1(x), size=size, mode="bilinear", align_corners=False)
        s2 = F.interpolate(self.scale2(x), size=size, mode="bilinear", align_corners=False)
        s3 = F.interpolate(self.scale3(x), size=size, mode="bilinear", align_corners=False)
        s4 = F.interpolate(self.scale4(x), size=size, mode="bilinear", align_corners=False)
        return self.shortcut(torch.cat([x, s1, s2, s3, s4], dim=1))


class PIDNet(nn.Module):
    """
    PIDNet: Proportional-Integral-Derivative Network for Real-Time Semantic Segmentation
    (Xu et al., CVPR 2023).
    Supports variants: 'pidnet_s' and 'pidnet_m'.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, variant: str = "s"):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant

        if variant == "s":
            channels = [32, 64, 128, 256]
        else:  # 'm'
            channels = [64, 128, 256, 512]

        c0, c1, c2, c3 = channels

        # Stem
        self.conv1 = ConvBNReLU(in_channels, c0, 3, stride=2)
        self.conv2 = ConvBNReLU(c0, c0, 3, stride=2)

        # Stage 1: P branch (stride 4)
        self.layer1 = nn.Sequential(
            BasicBlock(c0, c0),
            BasicBlock(c0, c0),
        )

        # Stage 2: P branch (stride 8)
        self.layer2 = nn.Sequential(
            BasicBlock(c0, c1, stride=2),
            BasicBlock(c1, c1),
        )

        # Stage 3: Three branches (P, I, D)
        # P branch (stride 8)
        self.p_layer3 = nn.Sequential(
            BasicBlock(c1, c1),
            BasicBlock(c1, c1),
        )
        # I branch (stride 16)
        self.i_layer3 = nn.Sequential(
            BasicBlock(c1, c2, stride=2),
            BasicBlock(c2, c2),
        )
        # D branch (stride 8, boundary extraction)
        self.d_layer3 = nn.Sequential(
            ConvBNReLU(c1, c0, 3, stride=1),
            ConvBNReLU(c0, c0, 3, stride=1),
        )

        # Pag fusion at Stage 3
        self.pag3 = Pag(c1, c2)

        # Stage 4
        self.p_layer4 = nn.Sequential(
            BasicBlock(c1, c1),
            BasicBlock(c1, c1),
        )
        self.i_layer4 = nn.Sequential(
            BasicBlock(c2, c3, stride=2),
            BasicBlock(c3, c3),
        )
        self.d_layer4 = nn.Sequential(
            ConvBNReLU(c0, c0, 3, stride=1),
            ConvBNReLU(c0, 1, 1, padding=0),  # boundary logit
        )

        # PAPPM at I branch
        self.pappm = PAPPM(c3, c1, c2)

        # Bag fusion
        self.bag = Bag(c1, c2, c1)

        # Head
        self.head = nn.Sequential(
            ConvBNReLU(c1, c1, 3, padding=1),
            nn.Conv2d(c1, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        # Stem
        x = self.conv2(self.conv1(x))
        x_p = self.layer1(x)
        x_p = self.layer2(x_p)

        # Stage 3
        p = self.p_layer3(x_p)
        i = self.i_layer3(x_p)
        d = self.d_layer3(x_p)

        # Pag
        p = self.pag3(p, i)

        # Stage 4
        p = self.p_layer4(p)
        i = self.i_layer4(i)
        d_out = self.d_layer4(d)

        # PAPPM
        i_ppm = self.pappm(i)

        # Bag fusion guided by boundary d
        fused = self.bag(p, i_ppm, d_out)
        logits = self.head(fused)

        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
