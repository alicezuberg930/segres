from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SegmentationModel
from ..nn.modules import CBA, LKR, SegHead


class FewShotSOAR(nn.Module):
    """
    Few-Shot and One-Shot Resolution-Preserving Segmentation Network (FS-SOAR).
    
    Key Innovations:
    1. Shared SOAR high-resolution feature extractor with stride 2 penultimate representations.
    2. Multi-shot Masked Average Pooling (MAP) extracting foreground and background prototypes.
    3. Dense cosine metric matching and channel-difference correlation.
    4. Topology-aware Large-Kernel Reparameterization (LKR) fusion neck.
    5. Sub-pixel SegHead producing full-resolution binary segmentation for novel classes.
    """

    def __init__(
        self,
        backbone_cfg: Union[str, Dict[str, Any], Path] = "configs/models/soar_nano1.yaml",
        in_channels: int = 3,
        scale: Optional[str] = None,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        # Instantiate base SOAR model
        self.backbone = SegmentationModel(
            cfg=backbone_cfg,
            ch=in_channels,
            nc=1,
            scale=scale,
            verbose=False,
        )

        if pretrained_backbone:
            ckpt = torch.load(pretrained_backbone, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            self.backbone.load_state_dict(state_dict, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Penultimate feature channels (layer before SegHead)
        self.feat_dim = self.backbone.out_ch[-2]

        # Fusion & Matching Head
        # Concatenates: [F_q (feat_dim), CosSim_fg (1), CosSim_bg (1), AbsDiff (feat_dim)]
        in_fusion = self.feat_dim * 2 + 2
        self.fusion = nn.Sequential(
            CBA(in_fusion, self.feat_dim, 3, 1),
            LKR(self.feat_dim, 5, 2.0),
            CBA(self.feat_dim, self.feat_dim, 3, 1),
        )

        # Sub-pixel head predicting full resolution novel mask
        self.head = SegHead(c1=self.feat_dim, nc=1, mid=self.feat_dim, r=2, prior=0.01)

    def extract_prototype(self, feat_s: torch.Tensor, mask_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute foreground and background prototype vectors via Masked Average Pooling.
        
        Args:
            feat_s: Support features (B, K, C, H_s, W_s)
            mask_s: Support binary masks (B, K, 1, H, W)
        Returns:
            p_fg: Foreground prototype (B, C)
            p_bg: Background prototype (B, C)
        """
        b, k, c, h_s, w_s = feat_s.shape
        # Downsample masks to feature spatial dimension
        mask_s_flat = mask_s.view(b * k, 1, mask_s.shape[-2], mask_s.shape[-1])
        mask_s_down = F.interpolate(mask_s_flat, size=(h_s, w_s), mode="nearest").view(b, k, 1, h_s, w_s)

        # Foreground prototype
        fg_weight = mask_s_down
        fg_sum = (feat_s * fg_weight).sum(dim=(1, 3, 4))
        fg_denom = fg_weight.sum(dim=(1, 3, 4)).clamp(min=1e-6)
        p_fg = fg_sum / fg_denom

        # Background prototype
        bg_weight = 1.0 - mask_s_down
        bg_sum = (feat_s * bg_weight).sum(dim=(1, 3, 4))
        bg_denom = bg_weight.sum(dim=(1, 3, 4)).clamp(min=1e-6)
        p_bg = bg_sum / bg_denom

        return p_fg, p_bg

    def forward(
        self,
        query_image: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward episodic pass.
        
        Args:
            query_image: (B, C_in, H, W)
            support_images: (B, K, C_in, H, W)
            support_masks: (B, K, 1, H, W)
            return_aux: whether to return support self-reconstruction logits
        Returns:
            query_logits: (B, 1, H, W) full resolution logit map for novel class
        """
        b, k, c_in, h, w = support_images.shape

        # 1. Extract support features per shot to conserve memory
        feat_s_list = []
        for i in range(k):
            feat_s_i = self.backbone.extract_features(support_images[:, i])
            feat_s_list.append(feat_s_i)
        feat_s = torch.stack(feat_s_list, dim=1)  # (B, K, C_feat, H_s, W_s)

        # 2. Extract multi-shot prototypes
        p_fg, p_bg = self.extract_prototype(feat_s, support_masks)

        # 3. Extract query features
        feat_q = self.backbone.extract_features(query_image)

        # 4. Dense cosine similarity matching
        norm_feat_q = F.normalize(feat_q, p=2, dim=1, eps=1e-6)
        norm_p_fg = F.normalize(p_fg, p=2, dim=1, eps=1e-6).unsqueeze(-1).unsqueeze(-1)
        norm_p_bg = F.normalize(p_bg, p=2, dim=1, eps=1e-6).unsqueeze(-1).unsqueeze(-1)

        sim_fg = torch.sum(norm_feat_q * norm_p_fg, dim=1, keepdim=True)
        sim_bg = torch.sum(norm_feat_q * norm_p_bg, dim=1, keepdim=True)

        # 5. Difference feature
        p_fg_grid = p_fg.unsqueeze(-1).unsqueeze(-1).expand_as(feat_q)
        diff_fg = torch.abs(feat_q - p_fg_grid)

        # 6. Fusion & Refinement
        corr_feat = torch.cat([feat_q, sim_fg, sim_bg, diff_fg], dim=1)
        refined_feat = self.fusion(corr_feat)

        # 7. Sub-pixel query prediction
        query_logits = self.head(refined_feat)

        if not return_aux:
            return query_logits

        # Support self-reconstruction for cycle consistency
        feat_s_first = feat_s[:, 0]
        norm_feat_s = F.normalize(feat_s_first, p=2, dim=1, eps=1e-6)
        sim_s_fg = torch.sum(norm_feat_s * norm_p_fg, dim=1, keepdim=True)
        sim_s_bg = torch.sum(norm_feat_s * norm_p_bg, dim=1, keepdim=True)
        diff_s_fg = torch.abs(feat_s_first - p_fg_grid)
        corr_s_feat = torch.cat([feat_s_first, sim_s_fg, sim_s_bg, diff_s_fg], dim=1)
        refined_s_feat = self.fusion(corr_s_feat)
        supp_logits = self.head(refined_s_feat)

        return query_logits, {"supp_logits": supp_logits, "p_fg": p_fg, "p_bg": p_bg}
