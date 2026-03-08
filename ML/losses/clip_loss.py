import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCLIPLoss(nn.Module):
    """
    Base class for Symmetric InfoNCE (CLIP) contrastive loss.
    Handles the learnable temperature and logit computation.

    Args:
        temperature: Initial value of the learnable log-temperature scalar.
            Actual temperature = exp(log_temperature), clamped to [0.01, 100].
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    def compute_logits(self, color_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            
        Returns:
            logits: (N, N) Cosine similarity matrix scaled by temperature
        """
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        # Cosine similarity matrix scaled by temperature: (N, N)
        logits = (color_embeds @ text_embeds.t()) / temperature
        return logits


class OriginalCLIPLoss(BaseCLIPLoss):
    """
    Original Symmetric InfoNCE (CLIP) contrastive loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Uses strict diagonal targeting (no label awareness).
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor = None,  # Kept in signature for API compatibility
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       Ignored in this original implementation.

        Returns:
            Scalar loss.
        """
        logits = self.compute_logits(color_embeds, text_embeds)

        n = logits.shape[0]
        targets = torch.arange(n, device=logits.device)

        # Symmetric cross-entropy
        loss_c2t = F.cross_entropy(logits, targets)
        loss_t2c = F.cross_entropy(logits.t(), targets)

        return (loss_c2t + loss_t2c) / 2.0


class MaskedCLIPLoss(BaseCLIPLoss):
    """
    Masked Symmetric InfoNCE (CLIP) contrastive loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Supports false-negative masking: pairs that share the
    same class label are excluded from the negative denominator so the model
    is not penalized for valid matches across a batch.
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       (N,) integer class indices (for false-negative masking)

        Returns:
            Scalar loss.
        """
        logits = self.compute_logits(color_embeds, text_embeds)

        n = logits.shape[0]
        targets = torch.arange(n, device=logits.device)

        # --- False-negative mask ---
        # Pairs where labels match but are NOT the diagonal are potential false
        # negatives. We mask them out of the softmax denominator by setting
        # their logits to -inf before the cross-entropy.
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
        false_neg_mask = same_label & ~diagonal  # True where we should mask

        logits_masked = logits.masked_fill(false_neg_mask, float("-inf"))

        # Symmetric cross-entropy
        loss_c2t = F.cross_entropy(logits_masked, targets)
        loss_t2c = F.cross_entropy(logits_masked.t(), targets)

        return (loss_c2t + loss_t2c) / 2.0


class SupConCLIPLoss(BaseCLIPLoss):
    """
    Supervised Contrastive InfoNCE (CLIP) loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Instead of masking false-negatives, it actively pulls 
    all pairs that share the same class label together using soft targets.
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       (N,) integer class indices (for multi-positive targeting)

        Returns:
            Scalar loss.
        """
        logits = self.compute_logits(color_embeds, text_embeds)

        # --- Soft targets for multi-positive matching ---
        # Pairs where labels match are treated as positive targets.
        # We distribute the probability mass evenly across all positive matches in a row.
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        
        # Count total positives per row to distribute probability mass evenly
        positives_per_row = same_label.sum(dim=1, keepdim=True).float()
        soft_targets = same_label.float() / positives_per_row

        # Symmetric cross-entropy (seamlessly handles probability distributions)
        loss_c2t = F.cross_entropy(logits, soft_targets)
        loss_t2c = F.cross_entropy(logits.t(), soft_targets.t())

        return (loss_c2t + loss_t2c) / 2.0