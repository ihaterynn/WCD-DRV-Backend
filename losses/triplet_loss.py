import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet Loss for training embedding models.
    Ensures that the anchor-positive pair is closer than the anchor-negative pair by at least a margin.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor:   (B, D) - Anchor embeddings
        positive: (B, D) - Positive embeddings (same class as anchor)
        negative: (B, D) - Negative embeddings (different class from anchor)
        Returns:
            loss: Scalar
        """
        # Compute pairwise distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # Distance between anchor and positive
        neg_dist = F.pairwise_distance(anchor, negative, p=2)  # Distance between anchor and negative

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()

        return loss

# Example usage
if __name__ == "__main__":
    # Simulated embeddings for demonstration purposes
    batch_size = 4
    embedding_dim = 128

    # Randomly generated embeddings for anchor, positive, and negative samples
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)

    # Define the triplet loss with a margin of 1.0
    triplet_loss = TripletLoss(margin=1.0)

    # Calculate the loss
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")
