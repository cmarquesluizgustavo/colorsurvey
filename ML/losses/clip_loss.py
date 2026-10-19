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
        t2c_eps: pixel tolerance of the text->color direction, on the 0-255 grid.
            A text's pick is accepted when it lands on a right colour or on any
            batch colour within t2c_eps of it on every channel (L-inf box): each
            right colour earns the probability on itself plus its look-alikes, so
            a pick on the "navy" row one shade from a "blue" row counts for "blue".
            A look-alike of several right colours counts once, split between them.
            Each loss keeps its own notion of "right" (its forward() says which).
            color->text is never changed. 0 (default) = off, training unchanged.
            Neighbours are searched in the batch only, so the effect scales with
            batch size: at 2048 rows a row has on average (2*eps+1)^3 / 2^24 * 2048
            neighbours (0.003 at eps 1, 1.1 at 10, 4 at 16; survey RGB is close to
            uniform).
    """

    def __init__(self, temperature: float = 0.07, t2c_eps: float = 0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.t2c_eps = t2c_eps

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

    def _near(self, rgb: torch.Tensor) -> torch.Tensor:
        """(N, N) float, 1 where batch colours m and n differ by at most t2c_eps on every channel (self included)."""
        if rgb is None:
            raise ValueError("t2c_eps > 0 needs the batch's 0-255 rgb values")
        return ((rgb.unsqueeze(1) - rgb.unsqueeze(0)).abs().amax(dim=-1) <= self.t2c_eps).float()

    @staticmethod
    def _accept_t2c(logits_t2c: torch.Tensor, targets: torch.Tensor, near: torch.Tensor):
        """
        Text->color loss under the t2c_eps accept rule.

        Args:
            logits_t2c: (T, N) text-to-colour logits; -inf where a loss masks a colour
            targets:    (T, N) the strict loss's targets: share 1/m on each of a
                        text's m right colours, 0 elsewhere
            near:       (N, N) from _near()

        Returns:
            (loss, accepted). A colour n within eps of k(n) right colours gives each
            of them p(n) / k(n), so it counts once in total. Each right colour r earns
            A(r) = sum of p(n) / k(n) over n near r; loss = -sum_r targets[r] * log A(r),
            averaged over texts. With no look-alike in the batch A(r) = p(r) and this
            is the strict loss. accepted: (T, N) bool, the colours accepted per text.
        """
        log_p = F.log_softmax(logits_t2c, dim=1)
        right = targets > 0
        k = right.float() @ near                       # (T, N); near is symmetric
        log_share = log_p - k.clamp_min(1).log()       # what each colour gives each r it is near
        # A(r) = r's own share + its look-alikes' shares. The own share stays exact; the
        # look-alikes are summed with a per-text shift, so a float32 underflow only drops
        # their credit and the loss and its gradient on r stay right.
        shift = log_p.amax(dim=1, keepdim=True)
        others = (log_share - shift).exp() @ (near - torch.eye(near.shape[0], device=near.device))
        tiny = torch.finfo(others.dtype).tiny
        log_others = torch.where(others > tiny, others.clamp_min(tiny).log() + shift, float("-inf"))
        log_accepted = torch.logaddexp(log_share.masked_fill(~right, 0.0), log_others)
        loss = -(targets * log_accepted.masked_fill(~right, 0.0)).sum(dim=1).mean()
        return loss, k > 0

    def _t2c(self, loss_c2t, loss_t2c_strict, logits_t2c, targets, rgb, hide=None):
        """
        The t->c term to train on; records this batch's components for metrics.csv
        (averaged per epoch by the trainer).

        Args:
            loss_c2t:        the loss's c->t term (recorded only)
            loss_t2c_strict: the loss's own eps 0 t->c term
            logits_t2c:      (T, N) text-to-colour logits, unmasked
            targets:         (T, N) share 1/m on each of a text's m right colours
            rgb:             (N, 3) 0-255 pixels; used only when t2c_eps > 0
            hide:            optional (N, N) bool, colours the strict term masks; under
                             the accept rule only those outside a text's box stay masked

        Components:
            loss_t2c:        the t->c term being trained (accept rule when t2c_eps > 0)
            loss_t2c_strict: the loss's eps 0 t->c term; measured, never trained on.
                             Equal to loss_t2c at eps 0, and to the loss_t2c the
                             prototype loss logged before t2c_eps. For masked, the eps 0
                             masking differs from the eps run's, so it is not quite the
                             same probabilities scored strictly.
            t2c_positives:   colours accepted per text, on average (data-only: it
                             depends on labels, pixels, eps and batch size, not the model)
            *_above_uniform: each loss minus the mean over texts of log(number of exact
                             right colours) -- the score of a model that splits its pick
                             evenly over them, the best either rule allows. Not reachable
                             in practice (research_plan.md note 5).
        """
        right = targets > 0
        if self.t2c_eps > 0:
            near = self._near(rgb)
            if hide is not None:
                logits_t2c = logits_t2c.masked_fill(hide & (near == 0), float("-inf"))
            loss_t2c, accepted = self._accept_t2c(logits_t2c, targets, near)
        else:
            loss_t2c, accepted = loss_t2c_strict, right
        uniform_bound = right.sum(dim=1).float().log().mean().item()
        self.last_components = {
            "loss_c2t": loss_c2t.item(),
            "loss_t2c": loss_t2c.item(),
            "loss_t2c_strict": loss_t2c_strict.item(),
            "t2c_positives": accepted.sum(dim=1).float().mean().item(),
            "loss_t2c_above_uniform": loss_t2c.item() - uniform_bound,
            "loss_t2c_strict_above_uniform": loss_t2c_strict.item() - uniform_bound,
        }
        return loss_t2c


class OriginalCLIPLoss(BaseCLIPLoss):
    """
    Original Symmetric InfoNCE (CLIP) contrastive loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Uses strict diagonal targeting (no label awareness).

    Text j's right colour is colour j; with t2c_eps, a pick on any batch colour
    within eps of colour j is accepted too.
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor = None,  # Kept in signature for API compatibility
        rgb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       Ignored in this original implementation.
            rgb:          (N, 3) pixels on the 0-255 grid; needed only when t2c_eps > 0.

        Returns:
            Scalar loss.
        """
        logits = self.compute_logits(color_embeds, text_embeds)

        n = logits.shape[0]
        targets = torch.arange(n, device=logits.device)

        # Symmetric cross-entropy
        loss_c2t = F.cross_entropy(logits, targets)
        loss_t2c = self._t2c(loss_c2t, F.cross_entropy(logits.t(), targets), logits.t(),
                             torch.eye(n, device=logits.device), rgb)

        return (loss_c2t + loss_t2c) / 2.0


class MaskedCLIPLoss(BaseCLIPLoss):
    """
    Masked Symmetric InfoNCE (CLIP) contrastive loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Supports false-negative masking: pairs that share the
    same class label are excluded from the negative denominator so the model
    is not penalized for valid matches across a batch.

    Text j's right colour is colour j; with t2c_eps, a pick on any batch colour
    within eps of colour j is accepted too (credited, not just unpenalised).
    Same-label colours outside that box stay masked.
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       (N,) integer class indices (for false-negative masking)
            rgb:          (N, 3) pixels on the 0-255 grid; needed only when t2c_eps > 0.

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
        # same_label is symmetric, so it also indexes the transposed (text, colour) matrix
        loss_t2c = self._t2c(loss_c2t, F.cross_entropy(logits_masked.t(), targets), logits.t(),
                             diagonal.float(), rgb, hide=same_label)

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
        t2c_eps: see BaseCLIPLoss. Prototype k's right colours are the rows
            labelled k; a pick within eps of one of them is accepted for it.
    """

    def __init__(self, temperature: float = 0.07, class_weights: torch.Tensor = None,
                 t2c_weight: float = 1.0, t2c_eps: float = 0):
        super().__init__(temperature, t2c_eps)
        # Registered as a buffer so it moves with .to(device); may be None.
        self.register_buffer("class_weights", class_weights)
        self.t2c_weight = t2c_weight

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (K, D) L2-normalized class prototype embeddings
            labels:       (N,) integer class indices in [0, K)
            rgb:          (N, 3) pixel values on the 0-255 grid; needed only when
                          t2c_eps > 0.

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
        positives_per_class = pos.sum(dim=1, keepdim=True)
        soft_targets = pos.float() / positives_per_class
        # loss_t2c_above_uniform is the distance from the uniform-over-positives bound.
        # NOT headroom: that bound assumes the prototype could single out exactly its own
        # labelled rows, which is impossible -- a row is a colour, and the unlabelled
        # near-identical shades look the same to it. The real floor is q(n|k)
        # proportional to p(k | rgb_n), measured at ~5.2 nats vs this bound's 2.3 (see
        # research_plan.md note 5).
        loss_t2c = self._t2c(loss_c2t, F.cross_entropy(logits_t2c, soft_targets), logits_t2c,
                             soft_targets, rgb)

        return (loss_c2t + self.t2c_weight * loss_t2c) / (1.0 + self.t2c_weight)


class SupConCLIPLoss(BaseCLIPLoss):
    """
    Supervised Contrastive InfoNCE (CLIP) loss.

    Computes cross-entropy in both directions (color->text and text->color)
    and averages them. Instead of masking false-negatives, it actively pulls 
    all pairs that share the same class label together using soft targets.

    Text j's right colours are every colour with its label; with t2c_eps, a pick
    within eps of one of them is accepted for it.
    """

    def forward(
        self,
        color_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            color_embeds: (N, D) L2-normalized color embeddings
            text_embeds:  (N, D) L2-normalized text embeddings
            labels:       (N,) integer class indices (for multi-positive targeting)
            rgb:          (N, 3) pixels on the 0-255 grid; needed only when t2c_eps > 0.

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
        loss_t2c = self._t2c(loss_c2t, F.cross_entropy(logits.t(), soft_targets.t()), logits.t(),
                             soft_targets.t(), rgb)

        return (loss_c2t + loss_t2c) / 2.0