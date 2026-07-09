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


class PrototypeCLIPLoss(BaseCLIPLoss):
    """
    Symmetric prototype loss for the (N, K) regime.

    Columns are the K class text prototypes (one per color name, never
    duplicated), rows are the N batch color samples. Both directions are read
    off a single (N, K) similarity matrix:

      color->text: standard K-way classification — each color is matched to its
        class name. No duplicate columns, hence no false negatives. All K
        classes are present every batch as columns, but rare classes still need
        rebalancing (class_weights and/or a class-balanced sampler), since a
        class only gets "pull" signal when it appears as a row.
      text->color: each class prototype present in the batch is pulled toward
        all of its samples via SupCon-style soft targets (the one-to-many
        direction).

    Args:
        class_weights: optional (K,) tensor of per-class weights applied to the
            color->text direction. Use inverse-frequency weights to stop the
            frequent classes from dominating the gradient on a long tail.
            None (default) = unweighted.
        t2c_weight: relative weight of the (hard, one-to-many) text->color term.
            Loss = (c2t + t2c_weight * t2c) / (1 + t2c_weight). 1.0 (default) is
            the symmetric 50/50 average; smaller values down-weight t->c so the
            useful color->text classification signal is not diluted.
    """

    def __init__(self, temperature: float = 0.07, class_weights: torch.Tensor = None,
                 t2c_weight: float = 1.0):
        super().__init__(temperature)
        # Registered as a buffer so it moves with .to(device); may be None.
        self.register_buffer("class_weights", class_weights)
        self.t2c_weight = t2c_weight

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (K, D) L2-normalized class prototype embeddings
            labels:       (N,) integer class indices in [0, K)

        Returns:
            Scalar loss.
        """
        logits = self.compute_logits(color_embeds, text_embeds)  # (N, K)

        # color->text: each color classified against the K prototypes
        loss_c2t = F.cross_entropy(logits, labels, weight=self.class_weights)

        # text->color: each present prototype matches all its samples (soft targets)
        present = labels.unique()                            # (P,)
        logits_t2c = logits.t()[present]                     # (P, N)
        pos = labels.unsqueeze(0) == present.unsqueeze(1)    # (P, N)
        soft_targets = pos.float() / pos.sum(dim=1, keepdim=True)
        loss_t2c = F.cross_entropy(logits_t2c, soft_targets)

        return (loss_c2t + self.t2c_weight * loss_t2c) / (1.0 + self.t2c_weight)


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