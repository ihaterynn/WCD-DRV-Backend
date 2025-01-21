# cross_attention_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    A module to fuse main and auxiliary color-texture embeddings using cross-attention 
    and then SUM or AVERAGE them, so the final dimension remains fused_dim.
    """
    def __init__(self, main_dim, aux_dim, fused_dim):
        super(CrossAttentionFusion, self).__init__()
        # Project main and auxiliary embeddings to a common dimension
        self.main_proj = nn.Linear(main_dim, fused_dim)
        self.aux_proj = nn.Linear(aux_dim, fused_dim)

        # Cross-attention layers
        self.cross_attn_main = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)
        self.cross_attn_aux  = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)

        # Layer normalization
        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)

        # Feed-forward network (expects input = fused_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim * 2, fused_dim)
        )
        self.norm3 = nn.LayerNorm(fused_dim)

    def forward(self, main_embed: torch.Tensor, aux_embed: torch.Tensor) -> torch.Tensor:
        """
        main_embed: (B, main_dim)
        aux_embed:  (B, aux_dim)
        Returns:
            fused_embed: (B, fused_dim)
        """
        # Project embeddings => shape (B,1,fused_dim)
        main_proj = self.main_proj(main_embed).unsqueeze(1)
        aux_proj  = self.aux_proj(aux_embed).unsqueeze(1)

        # Cross-attention: main attends to auxiliary
        attn_main, _ = self.cross_attn_main(main_proj, aux_proj, aux_proj)
        attn_main = self.norm1(attn_main + main_proj)

        # Cross-attention: auxiliary attends to main
        attn_aux, _  = self.cross_attn_aux(aux_proj, main_proj, main_proj)
        attn_aux = self.norm2(attn_aux + aux_proj)

        # Option A1: SUM them => shape still (B,1,fused_dim)
        fused = attn_main + attn_aux
        
        # Option A2: or AVERAGE => fused = 0.5*(attn_main + attn_aux)
        # fused = 0.5 * (attn_main + attn_aux)

        # Flatten from (B,1,fused_dim) => (B,fused_dim)
        fused = fused.squeeze(1)

        # Feed-forward => (B,fused_dim)
        out = self.ffn(fused)
        out = self.norm3(out + fused)  # residual
        return out
