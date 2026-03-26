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


class ColorCLIPModel(nn.Module):
    """
    Dual-encoder ColorCLIP model.

    Wraps ColorEncoder (color tower) and TextEncoder (text tower).
    The learnable log-temperature lives in CLIPInfoNCELoss, not here,
    keeping the model stateless w.r.t. training objective.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 color_hidden_dims: list[int] | None = None,
                 text_hidden_dims: list[int] | None = None):
        super().__init__()
        self.color_encoder = ColorEncoder(embed_dim=embed_dim, hidden_dims=color_hidden_dims)
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
