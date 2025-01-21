import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Simple Attention -----------------------
class SimpleAttention(nn.Module):
    """
    A straightforward scaled dot-product attention mechanism
    for patch-based tokens in LP-ViT.
    """
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, C, P, N) => interpret C=dim
        B, C, P, N = x.shape

        # We do a shape transformation so that we can apply Q,K,V in the last dimension
        # Let's rearrange x to (B, P, N, C)
        x_t = x.permute(0, 2, 3, 1)
        q = self.q(x_t)
        k = self.k(x_t)
        v = self.v(x_t)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2))  # (B, P, N, N)
        attn = F.softmax(attn / (C ** 0.5), dim=-1)
        out  = torch.matmul(attn, v)  # (B, P, N, C)

        # Convert back to (B, C, P, N)
        out = out.permute(0, 3, 1, 2)
        return out
