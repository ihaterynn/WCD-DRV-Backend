import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Multi-Scale Color/Texture Branch -----------------------
class MultiScaleColorTexture(nn.Module):
    """
    A lightweight branch that captures color and texture information across multiple scales.
    - Processes the image at scales: 1.0, 0.5, 0.25.
    - Applies global average pooling to each scaled image.
    - Concatenates the pooled features and passes through a Linear layer to produce the auxiliary embedding.
    """
    def __init__(self, in_channels=3, out_dim=32):
        super(MultiScaleColorTexture, self).__init__()
        self.out_dim = out_dim

        # Linear layer to reduce concatenated features to out_dim
        self.linear = nn.Linear(3 * in_channels, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) - Original RGB image
        Returns: (B, out_dim) - Auxiliary embedding vector
        """
        B, C, H, W = x.shape

        # Generate multi-scale images
        x1 = x  # Scale 1.0
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)  # Scale 0.5
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False) # Scale 0.25

        # Global average pooling for each scale
        x1_pool = F.adaptive_avg_pool2d(x1, (1, 1)).view(B, C)  # (B, C)
        x2_pool = F.adaptive_avg_pool2d(x2, (1, 1)).view(B, C)  # (B, C)
        x3_pool = F.adaptive_avg_pool2d(x3, (1, 1)).view(B, C)  # (B, C)

        # Concatenate pooled features
        multi_scale_vec = torch.cat([x1_pool, x2_pool, x3_pool], dim=1)  # (B, 3*C)

        # Linear transformation and activation
        out = self.linear(multi_scale_vec)  # (B, out_dim)
        out = self.act(out)
        return out
