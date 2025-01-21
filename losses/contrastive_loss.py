import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Contrastive Loss (Optional) -----------------------
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for training embedding models.
    Uses InfoNCE loss (NT-Xent).
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        """
        features: (B, D) - Embedding vectors
        labels:   (B,)   - Class labels
        Returns:
            loss: Scalar
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # (B, B)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Exclude self-contrast
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        # We'll define positives or handle them differently if needed
        # For demonstration, a naive approach is done here.

        # For each example, let the "positive" be the identical index => not typical for multi-class
        # This code is more of a placeholder for demonstration
        positives = similarity_matrix[mask].view(features.size(0), -1)
        negatives = similarity_matrix[~mask].view(features.size(0), -1)

        # Concatenate
        logits = torch.cat([positives, negatives], dim=1)
        # Label 0 => positives
        zero_labels = torch.zeros(logits.size(0), dtype=torch.long, device=features.device)

        loss = self.cross_entropy(logits, zero_labels)
        return loss
