import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetSimilarityModel(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super(ResNetSimilarityModel, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Remove the final fully connected layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        
        # Add a new fully connected layer to produce embeddings
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        # Pass the input through the ResNet model
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Pass through the new fully connected layer
        return x

# Example usage
if __name__ == "__main__":
    model = ResNetSimilarityModel(embedding_size=128)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(input_tensor)
    print(output.shape)  # Should print torch.Size([1, 128])