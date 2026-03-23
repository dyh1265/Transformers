"""Decoder-only transformer model for language modeling."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_llm.layers import PositionalEncoding, DecoderBlock


class NanoLLM(nn.Module):
    """Decoder-only transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        weight_tie: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.weight_tie = weight_tie
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        if not weight_tie:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.weight_tie:
            x = x @ self.embed.weight.T
        else:
            x = self.head(x)
        return x


def build_model(
    vocab_size: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 512,
    max_len: int = 512,
    dropout: float = 0.1,
    weight_tie: bool = True,
) -> NanoLLM:
    """Build decoder-only transformer for next-token prediction."""
    return NanoLLM(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        d_ff=int(d_ff),
        max_len=int(max_len),
        dropout=float(dropout),
        weight_tie=weight_tie,
    )
