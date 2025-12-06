import torch
import torch.nn as nn

class VanillaTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(VanillaTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Online Triplet Mining:
        For every item in the batch, find the hardest positive and hardest negative
        within the same batch to compute loss.
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # Masks
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        # For each anchor, find the hardest positive (max distance)
        hardest_positive_dist, _ = torch.max(pairwise_dist * positive_mask.float(), dim=1)

        # For each anchor, find the hardest negative (min distance)
        max_dist = torch.max(pairwise_dist)
        hardest_negative_dist, _ = torch.min(pairwise_dist + (max_dist * positive_mask.float()), dim=1)

        # Triplet Loss: max(0, dist_pos - dist_neg + margin)
        losses = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return losses.mean()
