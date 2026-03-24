"""Decoder block: pre-norm attention + FFN with residual connections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nano_llm.layers.attention import CausalSelfAttention

if TYPE_CHECKING:
    from nano_llm.layers.positional_encoding import RoPE


class DecoderBlock(nn.Module):
    """GPT-style decoder block: LayerNorm -> CausalAttention -> Residual -> LayerNorm -> FFN -> Residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope: RoPE | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout, rope=rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x
