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


class DetailBranch(nn.Module):
    """Detail Branch preserves high-resolution low-level spatial details."""

    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat  # [B, 128, H/8, W/8]


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        left = self.left(x)
        right = self.right(x)
        out = torch.cat([left, right], dim=1)
        return self.fuse(out)  # [B, 16, H/4, W/4]


class GELayer(nn.Module):
    """Gather-and-Expansion Layer with depthwise convolutions."""

    def __init__(self, in_chan: int, out_chan: int, exp_ratio: int = 6, stride: int = 1):
        super().__init__()
        self.stride = stride
        mid_chan = in_chan * exp_ratio

        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        if stride == 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, 3, stride=1, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan),
            )
            self.shortcut = nn.Identity() if in_chan == out_chan else ConvBNReLU(in_chan, out_chan, 1, padding=0)
        else:
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.dwconv2 = nn.Sequential(
                nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                ConvBNReLU(in_chan, out_chan, 1, padding=0),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            out = self.conv1(x)
            out = self.dwconv(out)
            out = self.conv2(out)
            return self.relu(out + self.shortcut(x))
        else:
            out = self.conv1(x)
            out = self.dwconv1(out)
            out = self.dwconv2(out)
            out = self.conv2(out)
            return self.relu(out + self.shortcut(x))


class CEBlock(nn.Module):
    """Context Embedding Block with Global Average Pooling."""

    def __init__(self, in_chan: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_chan)
        self.conv_1x1 = ConvBNReLU(in_chan, in_chan, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.gap(x)
        attn = self.bn(attn)
        attn = self.conv_1x1(attn)
        return x + attn


class SemanticBranch(nn.Module):
    """Semantic Branch rapidly captures expansive receptive field."""

    def __init__(self):
        super().__init__()
        self.stem = StemBlock()
        self.stage3 = nn.Sequential(
            GELayer(16, 32, exp_ratio=6, stride=2),
            GELayer(32, 32, exp_ratio=6, stride=1),
        )
        self.stage4 = nn.Sequential(
            GELayer(32, 64, exp_ratio=6, stride=2),
            GELayer(64, 64, exp_ratio=6, stride=1),
        )
        self.stage5 = nn.Sequential(
            GELayer(64, 128, exp_ratio=6, stride=2),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
        )
        self.ce = CEBlock(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.ce(x)  # [B, 128, H/32, W/32]


class BGANeck(nn.Module):
    """Bilateral Guided Aggregation Neck."""

    def __init__(self):
        super().__init__()
        self.detail_dw = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=False),
        )
        self.detail_down = nn.Sequential(
            ConvBNReLU(128, 128, 3, stride=2),
            nn.AvgPool2d(3, stride=2, padding=1),
        )

        self.sem_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.sem_dw = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=False),
        )
        self.fuse = ConvBNReLU(128, 128, 3, padding=1)

    def forward(self, detail: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        detail_g = self.detail_dw(detail)
        detail_s = self.detail_down(detail)

        sem_up = self.sem_up(sem)
        sem_g = self.sem_dw(sem)

        left = detail * torch.sigmoid(sem_up)
        right = detail_s * torch.sigmoid(sem_g)
        right_up = F.interpolate(right, size=left.shape[2:], mode="bilinear", align_corners=False)

        return self.fuse(left + right_up)


class BiSeNetV2(nn.Module):
    """
    BiSeNet V2: Bilateral Segmentation Network with Detail Branch and Semantic Branch
    (Yu et al., IJCV 2021).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGANeck()
        self.head = nn.Sequential(
            ConvBNReLU(128, 256, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        feat_detail = self.detail(x)
        feat_sem = self.semantic(x)
        fused = self.bga(feat_detail, feat_sem)
        logits = self.head(fused)
        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
