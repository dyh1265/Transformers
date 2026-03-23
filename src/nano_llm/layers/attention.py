"""Causal multi-head self-attention for decoder-only transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize Q/K/V/O projections (GPT-2 style: small normal, zero bias)."""
        std = 0.02
        for m in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scale = self.d_k**-0.5
        logits = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if mask is not None:
            logits = logits + mask
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)
