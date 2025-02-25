import copy
import torch
import torch.nn as nn
import torchvision.models as models
from lightly.models.modules import DINOProjectionHead

# 🛠️ Manually define update_momentum
def update_momentum(student_model, teacher_model, momentum):
    """
    Update the teacher model's parameters using momentum-based moving average.

    Args:
        student_model (nn.Module): The student model.
        teacher_model (nn.Module): The teacher model.
        momentum (float): Momentum factor (0 <= momentum <= 1).
    """
    with torch.no_grad():
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = teacher_param.data * momentum + student_param.data * (1.0 - momentum)

# 🛠️ Manually define deactivate_requires_grad to freeze model layers
def deactivate_requires_grad(model):
    """
    Disable gradient computation for all model parameters.

    Args:
        model (nn.Module): The model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False

class DINO(nn.Module):
    """
    DINO Self-Supervised Learning Model with Student and Teacher networks.

    Args:
        backbone (nn.Module): Backbone encoder for feature extraction.
        input_dim (int): Dimensionality of the input features.
    """
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        
        # 👥 Create a copy for the teacher model
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        
        # 🚫 Freeze the teacher model parameters
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        """Forward pass for student model."""
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        """Forward pass for teacher model."""
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
