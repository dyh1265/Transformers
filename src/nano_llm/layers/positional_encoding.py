"""Positional encoding for transformers: sinusoidal and RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE. (x_2i, x_2i+1) -> (-x_2i+1, x_2i)."""
    x1 = x[..., 0::2]  # even: x_0, x_2, ...
    x2 = x[..., 1::2]  # odd:  x_1, x_3, ...
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


e


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.dtype)
