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


class RoPE(nn.Module):
    """Rotary Position Embedding. Applied to Q and K inside attention."""

    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.d_k = d_k
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )
        position = torch.arange(max_len, dtype=torch.float32)
        freqs = position.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_len, d_k/2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, d_k/2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        # Expand to match pairs: cos/sin for (2i, 2i+1) are the same
        cos = cos.repeat_interleave(2, dim=-1)  # (1, 1, max_len, d_k)
        sin = sin.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K. Expects (B, num_heads, T, d_k)."""
        t = q.size(2) if seq_len is None else seq_len
        cos = self.cos_cached[:, :, :t, :].to(q.dtype)
        sin = self.sin_cached[:, :, :t, :].to(q.dtype)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


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
