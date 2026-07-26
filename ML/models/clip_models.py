from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorEncoder(nn.Module):
    """
    MLP color encoder: maps OKLCH coordinates to the joint embedding space.
    Hidden layers are configurable via `hidden_dims`.
    """

    def __init__(self, embed_dim: int = 64, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            raise ValueError("ColorEncoder requires hidden_dims to be specified")

        layers: list[nn.Module] = []
        in_dim = 3
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TextEncoder(nn.Module):
    """
    BoW text encoder: maps a multi-hot vocabulary vector to the joint
    embedding space. Hidden layers are configurable via `hidden_dims`.
    With no hidden layers (default), this is a bias-free linear lookup
    where word embeddings are summed.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = vocab_size
        for h in (hidden_dims or []):
            layers += [nn.Linear(in_dim, h, bias=False), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, embed_dim, bias=False))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=-1)


# ===== Negative result, kept as documentation (see docstring) ==============
class LearnablePrototypeTextEncoder(nn.Module):
    """
    Replaces the BoW text tower with a bank of free, learnable per-class
    prototype vectors (one row per class) — no shared-token coupling, so
    compound names are not tethered to their single-token parents.

    Only valid with loss_type="prototype". The input `x` is ignored entirely —
    we always return the full (K, D) prototype bank. The prototype loss feeds
    the (K, *) class table here (so the result is the K prototypes in label
    order); the shared eval forward feeds per-sample BoW but discards this
    output, so ignoring `x` is correct in both paths.

    Tested hypothesis (2026-07): BoW word-sharing ("light orange" = light +
    orange) glues compound names to their parents and causes the model's
    confusions between them. Result: this encoder fully de-crowded the name
    vectors (mean nearest-neighbor cosine 0.887 -> 0.291) and **no retrieval
    metric changed** (96-color R@1 0.484 vs 0.485 for BoW). Conclusion: the
    confusion lives in the pixels, not the name vectors. Kept as executable
    documentation of this negative result.
    """

    def __init__(self, num_classes: int, embed_dim: int = 64):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)
# ============================================================================


class ColorCLIPModel(nn.Module):
    """
    Dual-encoder ColorCLIP model.

    Wraps ColorEncoder (color tower) and TextEncoder (text tower).
    The learnable log-temperature lives in CLIPInfoNCELoss, not here,
    keeping the model stateless w.r.t. training objective.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 color_hidden_dims: list[int] | None = None,
                 text_hidden_dims: list[int] | None = None,
                 text_encoder_type: str = "bow",
                 num_classes: int | None = None):
        super().__init__()
        self.color_encoder = ColorEncoder(embed_dim=embed_dim, hidden_dims=color_hidden_dims)
        # See LearnablePrototypeTextEncoder's docstring for why this branch exists.
        if text_encoder_type == "learnable":
            if num_classes is None:
                raise ValueError("text_encoder_type='learnable' requires num_classes")
            self.text_encoder = LearnablePrototypeTextEncoder(num_classes=num_classes,
                                                              embed_dim=embed_dim)
        else:
            self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim,
                                            hidden_dims=text_hidden_dims)

    def encode_color(self, colors: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized color embeddings."""
        return self.color_encoder(colors)

    def encode_text(self, bow: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized text embeddings."""
        return self.text_encoder(bow)

    def forward(
        self, colors: torch.Tensor, bow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            colors: (N, 3) normalized OKLCH tensors
            bow:    (N, vocab_size) multi-hot BoW tensors

        Returns:
            (color_embeds, text_embeds) — both L2-normalized, shape (N, embed_dim)
        """
        return self.encode_color(colors), self.encode_text(bow)
