from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorEncoder(nn.Module):
    """
    MLP color encoder: maps OKLCH coordinates to the joint embedding space.

    Architecture (per spec):
        Linear(3 -> 128) + BN + ReLU
        Linear(128 -> 64) + BN + ReLU
        Linear(64 -> embed_dim)
        L2-normalize output
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TextEncoder(nn.Module):
    """
    Linear BoW text encoder: maps a multi-hot vocabulary vector to the
    joint embedding space.

    A bias-free linear layer effectively learns one embedding vector per word;
    for multi-word labels the embeddings are summed.  Output is L2-normalized.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(vocab_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=-1)


class ColorCLIPModel(nn.Module):
    """
    Dual-encoder ColorCLIP model.

    Wraps ColorEncoder (color tower) and TextEncoder (text tower).
    The learnable log-temperature lives in CLIPInfoNCELoss, not here,
    keeping the model stateless w.r.t. training objective.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.color_encoder = ColorEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)

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
