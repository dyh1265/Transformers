"""Causal multi-head self-attention for decoder-only transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as functional

if TYPE_CHECKING:
    from nano_llm.layers.positional_encoding import RoPE


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking. Optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope: RoPE | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        self.rope = rope
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
        b, t, c = x.shape
        q = self.wq(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)
        if mask is None:
            dropout_p = float(self.dropout.p) if self.training else 0.0
            attn = functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
            out = attn.transpose(1, 2).contiguous().view(b, t, c)
            return self.wo(out)
        scale = self.d_k**-0.5
        logits = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        logits = logits + mask
        weights = functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.wo(out)
