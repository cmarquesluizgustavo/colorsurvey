import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTripletLoss(nn.Module):
    def __init__(self, model=None, margin=1.0):
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
        hardest_positive_dist, _ = torch.max(
            pairwise_dist * positive_mask.float(), dim=1
        )

        # For each anchor, find the hardest negative (min distance)
        max_dist = torch.max(pairwise_dist)
        hardest_negative_dist, _ = torch.min(
            pairwise_dist + (max_dist * positive_mask.float()), dim=1
        )

        # Triplet Loss: max(0, dist_pos - dist_neg + margin)
        losses = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return losses.mean()


class ConditionalTripletLoss(nn.Module):
    """
    Conditional Triplet Mining based on classifier predictions.
    
    Selects triplets based on whether the anchor is correctly classified:
    - Case A (Misclassified): Negative from predicted (incorrect) class
    - Case B (Correct): Negative from second-highest probability class
    
    Optimized with vectorized operations for better performance.
    """
    def __init__(self, choice_model, margin=1.0):
        super().__init__()
        self.margin = margin
        self.choice_model = choice_model
        
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Get classifier predictions and probabilities
        with torch.no_grad():
            logits = self.choice_model(embeddings)
            probs = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)
            
            # Get second-highest probability class (runner-up)
            top2_probs, top2_classes = torch.topk(probs, k=2, dim=1)
            runner_up_classes = top2_classes[:, 1]
        
        # Determine which anchors are correctly/incorrectly classified
        correctly_classified = (predicted_classes == labels)
        
        # For each anchor, determine the target negative class
        # Case A (misclassified): use predicted class
        # Case B (correct): use runner-up class
        target_negative_classes = torch.where(
            correctly_classified,
            runner_up_classes,  # Case B: runner-up
            predicted_classes   # Case A: predicted (wrong) class
        )
        
        # VECTORIZED TRIPLET MINING
        # Create masks for valid samples
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        target_neg_eq = labels.unsqueeze(1) == target_negative_classes.unsqueeze(0)  # [batch_size, batch_size]
        
        # Diagonal mask to exclude self-comparisons
        not_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Positive mask: same true label, not self
        positive_mask = label_eq & not_diag
        
        # Negative mask: matches target negative class
        negative_mask = target_neg_eq
        
        # Check which anchors have valid positives and negatives
        has_positive = positive_mask.any(dim=1)
        has_negative = negative_mask.any(dim=1)
        valid_anchors = has_positive & has_negative
        
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute all pairwise distances efficiently
        dists = torch.cdist(embeddings, embeddings, p=2)  # [batch_size, batch_size]
        
        # For each valid anchor, select a random positive and negative
        # We'll use a simple strategy: take the first valid positive/negative for speed
        # (More sophisticated sampling can be added if needed)
        
        # Find first valid positive index for each anchor
        positive_mask_float = positive_mask.float()
        positive_mask_float[~positive_mask] = float('inf')
        pos_indices = torch.argmin(positive_mask_float, dim=1)  # First True position
        
        # Find first valid negative index for each anchor
        negative_mask_float = negative_mask.float()
        negative_mask_float[~negative_mask] = float('inf')
        neg_indices = torch.argmin(negative_mask_float, dim=1)  # First True position
        
        # Filter to valid anchors only
        anchor_indices = torch.arange(batch_size, device=device)[valid_anchors]
        pos_indices = pos_indices[valid_anchors]
        neg_indices = neg_indices[valid_anchors]
        
        # Gather distances efficiently
        pos_dists = dists[anchor_indices, pos_indices]
        neg_dists = dists[anchor_indices, neg_indices]
        
        # Compute triplet loss
        losses = torch.relu(pos_dists - neg_dists + self.margin)
        
        return losses.mean()


