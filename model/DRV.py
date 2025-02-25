import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lightly.loss import NTXentLoss
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dino_model import DINO  

class HybridSimilarityModel(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super().__init__()

        # 1️⃣ DINO - Self-Supervised ResNet50 (Using ResNet50 backbone)
        resnet50 = models.resnet50(pretrained=pretrained)
        backbone = nn.Sequential(*list(resnet50.children())[:-1])  # Remove the FC layers
        self.dino = DINO(backbone, input_dim=2048)  # DINO model with ResNet50 backbone
        dino_output_dim = 2048  # ResNet50 outputs 2048-d features

        # 2️⃣ ResNet50 - Texture Features
        self.resnet50 = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # Remove the final layers
        resnet50_output_dim = 2048  # ResNet50 outputs 2048-d features

        # 3️⃣ Vision Transformer (ViT-B/16) for additional feature extraction
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.heads = nn.Identity()  # Remove the classification head
        vit_output_dim = 768  # ViT-B/16 outputs a 768-dimensional feature vector

        # 🧬 Fusion Layer
        fusion_dim = dino_output_dim + resnet50_output_dim + vit_output_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        # DINO features (ResNet50 backbone)
        dino_features = self.dino(x)  # shape: [batch_size, dino_output_dim]
        
        # ResNet50 features (need spatial mean pooling)
        resnet_features = self.resnet50(x)  # shape: [batch_size, 2048, H, W]
        resnet_features = resnet_features.mean(dim=[2, 3])  # Global spatial average -> [batch_size, 2048]

        # ViT features
        vit_features = self.vit(x)  # shape: [batch_size, 768]

        # Concatenate all features
        combined_features = torch.cat([dino_features, resnet_features, vit_features], dim=1)

        # Pass through final fully connected layers and normalize
        return F.normalize(self.fc(combined_features), p=2, dim=1)

class SelfSupervisedLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.loss_fn = NTXentLoss(temperature=temperature)

    def forward(self, embeddings, augmented_embeddings):
        return self.loss_fn(embeddings, augmented_embeddings)

if __name__ == "__main__":
    model = HybridSimilarityModel(embedding_size=128, pretrained=True)
    loss_fn = SelfSupervisedLoss()
    
    input_tensor = torch.randn(8, 3, 224, 224)
    augmented_tensor = torch.randn(8, 3, 224, 224)
    
    embeddings = model(input_tensor)
    augmented_embeddings = model(augmented_tensor)
    
    loss = loss_fn(embeddings, augmented_embeddings)
    
    print(f"Embeddings Shape: {embeddings.shape}")
    print(f"Loss: {loss.item()}")
