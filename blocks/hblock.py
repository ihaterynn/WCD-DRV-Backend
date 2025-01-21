import torch
import torch.nn as nn
import torch.nn.functional as F

from .inverted_residual import InvertedResidual
from .location_preserving_vit import LocationPreservingVit

# --------------------- HBlock (Hybrid) ------------------------
class HBlock(nn.Module):
    """
    Hybrid block that merges an inverted residual with LP-ViT.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        n_local=1,
        n_attn=1,
        patch_size=(4, 4),
        expand_ratio=1,
        ffn_dim=128
    ):
        super(HBlock, self).__init__()
        # Local feature extraction
        local_layers = []
        local_layers.append(InvertedResidual(in_ch, out_ch, stride, expand_ratio))
        for _ in range(n_local - 1):
            local_layers.append(InvertedResidual(out_ch, out_ch, 1, expand_ratio))
        self.local_acq = nn.Sequential(*local_layers)

        # Global feature extraction
        attn_layers = []
        for _ in range(n_attn):
            attn_layers.append(LocationPreservingVit(out_ch, ffn_dim))
        self.global_acq = nn.Sequential(*attn_layers)

        self.patch_h, self.patch_w = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Local features
        fm_local = self.local_acq(x)  # (B, C, H, W)
        B, C, H, W = fm_local.shape

        # Keep original shape
        fm_local_orig = fm_local
        orig_H, orig_W = H, W

        # 2) Upsample if needed
        newH = ((H + self.patch_h - 1) // self.patch_h) * self.patch_h
        newW = ((W + self.patch_w - 1) // self.patch_w) * self.patch_w
        if (newH != H) or (newW != W):
            fm_local_up = F.interpolate(fm_local, size=(newH, newW), mode='bilinear', align_corners=False)
        else:
            fm_local_up = fm_local

        # 3) Unfold for patch-based attention
        patches = F.unfold(
            fm_local_up,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w)
        )
        # => (B, C * patch_area, n_patches)
        patches = patches.view(B, C, self.patch_h * self.patch_w, -1)

        # 4) Global acquisition
        patches = self.global_acq(patches)

        # 5) Fold back
        patches = patches.view(B, C * self.patch_h * self.patch_w, -1)
        fm_global_up = F.fold(
            patches,
            output_size=(newH, newW),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w)
        )

        # 6) Downsample if needed
        if (newH != H) or (newW != W):
            fm_global = F.interpolate(fm_global_up, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        else:
            fm_global = fm_global_up

        # 7) Merge local + global
        return fm_local_orig + fm_global
