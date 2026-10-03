from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions."""

    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """Multi-Head Self-Attention with Spatial Reduction (SRA)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    """Mix-FeedForward Network with 3x3 depthwise convolution."""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        x_conv = x.transpose(1, 2).view(b, -1, h, w)
        x_conv = self.dwconv(x_conv)
        x = x_conv.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class MixVisionTransformer(nn.Module):
    """Hierarchical Mix Transformer (MiT) encoder."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: list[int] = [32, 64, 160, 256],
        num_heads: list[int] = [1, 2, 5, 8],
        mlp_ratios: list[float] = [4, 4, 4, 4],
        qkv_bias: bool = True,
        depths: list[int] = [2, 2, 2, 2],
        sr_ratios: list[int] = [8, 4, 2, 1],
    ):
        super().__init__()
        self.depths = depths

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stages
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, sr_ratios[3])
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b = x.shape[0]
        outs = []

        # Stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x = self.norm1(x)
        outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 2
        x, h, w = self.patch_embed2(outs[-1])
        for blk in self.block2:
            x = blk(x, h, w)
        x = self.norm2(x)
        outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 3
        x, h, w = self.patch_embed3(outs[-1])
        for blk in self.block3:
            x = blk(x, h, w)
        x = self.norm3(x)
        outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        # Stage 4
        x, h, w = self.patch_embed4(outs[-1])
        for blk in self.block4:
            x = blk(x, h, w)
        x = self.norm4(x)
        outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        return outs


class SegFormerHead(nn.Module):
    """All-MLP Decoder."""

    def __init__(self, in_channels: list[int], embedding_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        h4, w4 = c1.shape[2:]

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h4, w4), mode="bilinear", align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h4, w4), mode="bilinear", align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=(h4, w4), mode="bilinear", align_corners=False)

        fused = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        fused = self.dropout(fused)
        return self.linear_pred(fused)


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    (Xie et al., NeurIPS 2021).
    Supports variants: 'b0' (3.8M params) and 'b1' (13.7M params).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, variant: str = "b0"):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        if variant == "b0":
            embed_dims = [32, 64, 160, 256]
            decoder_dim = 256
        else:  # 'b1'
            embed_dims = [64, 128, 320, 512]
            decoder_dim = 256

        self.encoder = MixVisionTransformer(
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
        )
        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            embedding_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        feats = self.encoder(x)
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
