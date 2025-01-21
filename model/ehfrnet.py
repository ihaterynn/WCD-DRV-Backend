import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.hblock import HBlock
from blocks.multiscale_color_texture import MultiScaleColorTexture
from blocks.cross_attention_fusion import CrossAttentionFusion

# --------------------- EHFRNet++ Model -------------------------
class EHFRNetMultiScale(nn.Module):
    """
    EHFRNet++ Architecture:
    - Main Backbone: Inverted Residual + LP-ViT via HBlocks
    - Auxiliary Branch: Multi-Scale Color-Texture
    - Feature Fusion: Cross-Attention Fusion
    - Embedding Generation and Classification
    """
    def __init__(self, num_classes=2, fused_dim=160):
        super(EHFRNetMultiScale, self).__init__()
        self.num_classes = num_classes

        # 1) Main Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 16, H/2, W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = HBlock(in_ch=16, out_ch=24, stride=2, expand_ratio=2, ffn_dim=128)
        self.layer2 = HBlock(in_ch=24, out_ch=32, stride=2, expand_ratio=2, ffn_dim=128)
        self.layer3 = HBlock(in_ch=32, out_ch=64, stride=2, expand_ratio=2, ffn_dim=128)
        self.layer4 = HBlock(in_ch=64, out_ch=96, stride=2, expand_ratio=2, ffn_dim=128)
        self.layer5 = HBlock(in_ch=96, out_ch=128, stride=2, expand_ratio=2, ffn_dim=128)
        self.pool   = nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1)

        # 2) Auxiliary Multi-Scale Color-Texture Branch
        self.aux_branch = MultiScaleColorTexture(in_channels=3, out_dim=32)

        # 3) Feature Fusion
        self.feature_fusion = CrossAttentionFusion(main_dim=128, aux_dim=32, fused_dim=fused_dim)

        # 4) Final Classification
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # a) Main Backbone
        out = self.stem(x)     # (B, 16, H/2,  W/2)
        out = self.layer1(out) # (B, 24, H/4,  W/4)
        out = self.layer2(out) # (B, 32, H/8,  W/8)
        out = self.layer3(out) # (B, 64, H/16, W/16)
        out = self.layer4(out) # (B, 96, H/32, W/32)
        out = self.layer5(out) # (B, 128,H/64, W/64)
        out = self.pool(out)   # (B, 128,1,1)
        main_embed = out.view(out.size(0), -1)  # (B, 128)

        # b) Auxiliary Branch
        aux_embed = self.aux_branch(x)  # (B, 32)

        # c) Feature Fusion
        fused_embed = self.feature_fusion(main_embed, aux_embed)  # (B, fused_dim)

        # d) Classification
        logits = self.classifier(fused_embed)  # (B, num_classes)
        return logits

    def extract_features(self, x):
        """
        Extracts the final embedding vector before classification.
        Useful for similarity comparisons or retrieval.
        x: (B, 3, H, W)
        Returns:
            fused_embed: (B, fused_dim)
        """
        # Main backbone
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.pool(out)
        main_embed = out.view(out.size(0), -1)  # (B, 128)

        # Aux branch
        aux_embed = self.aux_branch(x)  # (B, 32)

        # Fusion
        fused_embed = self.feature_fusion(main_embed, aux_embed)
        return fused_embed
