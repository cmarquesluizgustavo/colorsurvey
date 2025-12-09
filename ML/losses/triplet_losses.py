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
        
        # Build triplets
        losses = []
        
        for i in range(batch_size):
            anchor_embedding = embeddings[i]
            anchor_label = labels[i]
            target_neg_class = target_negative_classes[i]
            
            # Find positive samples (same class as ground truth, excluding anchor)
            positive_mask = (labels == anchor_label)
            positive_mask[i] = False  # Exclude anchor itself
            positive_indices = torch.where(positive_mask)[0]
            
            # Find negative samples (from target negative class)
            negative_mask = (labels == target_neg_class)
            negative_indices = torch.where(negative_mask)[0]
            
            # Skip if no valid positive or negative samples
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            # Randomly select one positive and one negative
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
            
            positive_embedding = embeddings[pos_idx]
            negative_embedding = embeddings[neg_idx]
            
            # Compute distances
            pos_dist = torch.norm(anchor_embedding - positive_embedding, p=2)
            neg_dist = torch.norm(anchor_embedding - negative_embedding, p=2)
            
            # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            loss = torch.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)
        
        # Return mean loss (or zero if no valid triplets)
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()
