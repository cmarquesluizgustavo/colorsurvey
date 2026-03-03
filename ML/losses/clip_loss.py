import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP) contrastive loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Supports false-negative masking: pairs that share the
    same class label are excluded from the negative denominator so the model
    is not penalized for valid matches across a batch.

    Args:
        temperature: Initial value of the learnable log-temperature scalar.
            Actual temperature = exp(log_temperature), clamped to [0.01, 100].
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

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
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        # Cosine similarity matrix scaled by temperature: (N, N)
        logits = (color_embeds @ text_embeds.t()) / temperature

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
