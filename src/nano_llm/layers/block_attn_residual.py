"""Inter-block attention residuals (attention over macro-block representations)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as functional

from nano_llm.layers.attention import CausalSelfAttention

if TYPE_CHECKING:
    from nano_llm.layers.positional_encoding import RoPE


class RMSNorm(nn.Module):
    """Per-token RMS normalization (weight only)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, D]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def block_attn_res(
    blocks: list[torch.Tensor],
    partial: torch.Tensor | None,
    proj: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Mix block tensors with a learned pseudo-query (softmax over depth).

    blocks: prior macro-block hiddens [B,T,D]; partial is intra-block sum or None at block start.
    proj: Linear(D, 1); weight is pseudo-query w.
    """
    if partial is None:
        v = torch.stack(blocks, dim=0)
    else:
        v = torch.stack(blocks + [partial], dim=0)
    n, b, t, d = v.shape
    flat = v.reshape(n * b * t, d)
    k = norm(flat).reshape(n, b, t, d)
    w = proj.weight.squeeze(0)
    logits = torch.einsum("d,nbtd->nbt", w, k)
    alpha = functional.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", alpha, v)


class InterBlockAttnDecoderBlock(nn.Module):
    """Pre-norm decoder layer with inter-block attention before self-attention and before MLP."""

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
        self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
        self.attn_res_norm = RMSNorm(d_model)
        self.mlp_res_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        blocks: list[torch.Tensor],
        partial: torch.Tensor | None,
        *,
        layer_index: int,
        macro_block_size: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Return (updated x, updated blocks list, partial for next layer)."""
        start_of_macro = layer_index % macro_block_size == 0
        partial_for_v = None if start_of_macro else partial

        h_a = block_attn_res(blocks, partial_for_v, self.attn_res_proj, self.attn_res_norm)
        delta_a = self.attn(self.ln1(h_a))
        partial_after_attn = delta_a

        h_m = block_attn_res(blocks, partial_after_attn, self.mlp_res_proj, self.mlp_res_norm)
        delta_m = self.ffn(self.ln2(h_m))
        x_out = x + delta_a + delta_m
        partial_out = partial_after_attn + delta_m

        blocks_out = blocks
        partial_next: torch.Tensor | None
        if (layer_index + 1) % macro_block_size == 0:
            blocks_out = blocks + [x_out]
            partial_next = None
        else:
            partial_next = partial_out

        return x_out, blocks_out, partial_next


def trim_blocks(blocks: list[torch.Tensor], max_len: int) -> list[torch.Tensor]:
    """Keep b_0 and the most recent completed blocks (cap list length)."""
    if max_len <= 0 or len(blocks) <= max_len:
        return blocks
    head = blocks[0]
    tail = blocks[-(max_len - 1) :] if max_len > 1 else []
    return [head, *tail]
