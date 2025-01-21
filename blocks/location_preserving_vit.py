import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_attention import SimpleAttention

# --------------------- LocationPreservingVit ------------------
class LocationPreservingVit(nn.Module):
    """
    LP-ViT block: merges a patch-level attention and a separate patch-to-patch
    attention, preserving spatial structure by an unfold->attention->fold pipeline.
    """
    def __init__(self, embed_dim, ffn_dim):
        super(LocationPreservingVit, self).__init__()
        self.attn1 = SimpleAttention(embed_dim)
        self.attn2 = SimpleAttention(embed_dim)

        self.conv_expand = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.conv_reduce = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, P, N)
        # 1) Patch-level attention
        xp = x.transpose(2, 3)   # (B, C, N, P)
        xp = self.attn1(xp)
        xp = xp.transpose(2, 3)  # (B, C, P, N)

        # 2) Global patch-to-patch attention
        x2 = self.attn2(x)
        # If shapes differ, we can interpolate
        if x.shape != xp.shape:
            xp = F.interpolate(xp, size=x.shape[2:], mode='bilinear', align_corners=False)
        if x.shape != x2.shape:
            x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = x + xp + x2  # Combine

        # 3) Feed-forward
        B, C, P, N = out.shape
        out_2d = out.view(B, C, P, N)  # treat (P, N) as H,W
        out_2d = self.conv_expand(out_2d)
        out_2d = F.relu(out_2d, inplace=True)
        out_2d = self.conv_reduce(out_2d)
        return out + out_2d
