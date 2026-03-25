"""Decoder-only transformer model for language modeling."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_llm.layers import (
    DecoderBlock,
    InterBlockAttnDecoderBlock,
    PositionalEncoding,
    RoPE,
    trim_blocks,
)


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
        position_encoding: str = "sinusoidal",
        block_attn_residuals: bool = False,
        macro_block_size: int = 2,
        max_block_representations: int = 9,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.weight_tie = weight_tie
        self.position_encoding = str(position_encoding).lower()
        self.block_attn_residuals = bool(block_attn_residuals)
        self.macro_block_size = int(macro_block_size)
        self.max_block_representations = int(max_block_representations)
        if self.macro_block_size < 1:
            raise ValueError("macro_block_size must be >= 1")
        self.embed = nn.Embedding(vocab_size, d_model)

        use_rope = self.position_encoding == "rope"
        d_k = d_model // num_heads
        rope = RoPE(d_k, max_len=max_len) if use_rope else None
        self.pos_enc = None if use_rope else PositionalEncoding(d_model, max_len=max_len)
        block_cls = InterBlockAttnDecoderBlock if self.block_attn_residuals else DecoderBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(d_model, num_heads, d_ff, dropout=dropout, rope=rope)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        if not weight_tie:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        if self.block_attn_residuals:
            blocks: list[torch.Tensor] = [x]
            partial: torch.Tensor | None = None
            for li, block in enumerate(self.blocks):
                x, blocks, partial = block(
                    x,
                    blocks,
                    partial,
                    layer_index=li,
                    macro_block_size=self.macro_block_size,
                )
                blocks = trim_blocks(blocks, self.max_block_representations)
        else:
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
    position_encoding: str = "sinusoidal",
    block_attn_residuals: bool = False,
    macro_block_size: int = 2,
    max_block_representations: int = 9,
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
        position_encoding=position_encoding,
        block_attn_residuals=block_attn_residuals,
        macro_block_size=macro_block_size,
        max_block_representations=max_block_representations,
    )
