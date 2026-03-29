"""Decoder-only transformer model for language modeling."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from nano_llm.layers import (
    DecoderBlock,
    InterBlockAttnDecoderBlock,
    PositionalEncoding,
    RoPE,
    trim_blocks,
)


def _make_activation(act):
    if act is None or act == "linear":
        return nn.Identity()
    if callable(act):
        return act
    act = str(act).lower()
    if act == "relu":
        return nn.ReLU()
    if act == "elu":
        return nn.ELU()
    if act == "gelu":
        return nn.GELU()
    if act in ("silu", "swish"):
        return nn.SiLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "sigmoid":
        return nn.Sigmoid()
    if act == "softmax":
        return nn.Softmax(dim=-1)
    raise ValueError(f"Unknown activation: {act}")


class FullyConnected(nn.Module):
    def __init__(
        self,
        n_fc: int,
        in_size: int,
        hidden_phi: int,
        out_size: int,
        final_activation="linear",
        activation="elu",
        dropout: bool = False,
        batch_norm: bool = False,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if n_fc < 1:
            raise ValueError("n_fc must be >= 1")
        layers = []
        if n_fc == 1:
            self.hidden = nn.Identity()
            self.out = nn.Linear(in_size, out_size, bias=use_bias)
            self.final_act = _make_activation(final_activation)
            return

        # First hidden layer: in_size -> hidden_phi
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_phi))
        layers.append(nn.Linear(in_size, hidden_phi, bias=use_bias))
        layers.append(_make_activation(activation))
        if dropout:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(n_fc - 1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_phi))
            layers.append(nn.Linear(hidden_phi, hidden_phi, bias=use_bias))
            layers.append(_make_activation(activation))
            if dropout:
                layers.append(nn.Dropout(dropout_rate))
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(hidden_phi, out_size, bias=use_bias)
        self.final_act = _make_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.out(x)
        return self.final_act(x)


def _zero_init_fully_connected_output(fc: FullyConnected) -> None:
    """Set the final Linear of a FullyConnected stack to zeros (for residual heads)."""
    if hasattr(fc, "out") and isinstance(fc.out, nn.Linear):
        nn.init.zeros_(fc.out.weight)
        if fc.out.bias is not None:
            nn.init.zeros_(fc.out.bias)


class NanoLLM(nn.Module):
    """Decoder-only transformer for next-token prediction.

    Decoder trunk: ``block_attn_residuals=False`` (default) uses a **vanilla** pre-norm
    stack (``DecoderBlock``). ``block_attn_residuals=True`` uses inter-block residual
    layers (``InterBlockAttnDecoderBlock``). See ``decoder_stack``.

    TARNet two-head mode: the trunk produces a hidden state per position; a **shared**
    head predicts vocabulary logits (treatment-agnostic "review content"), and two
    **sentiment encoders** add residual logits so Y(0)=shared+Δ_neg and Y(1)=shared+Δ_pos.
    """

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
        tarnet_two_heads: bool = False,
        tarnet_head_n_fc: int = 2,
        tarnet_head_hidden_dim: int | None = None,
        tarnet_head0_n_fc: int | None = None,
        tarnet_head0_hidden_dim: int | None = None,
        tarnet_head1_n_fc: int | None = None,
        tarnet_head1_hidden_dim: int | None = None,
        position_encoding: str = "sinusoidal",
        block_attn_residuals: bool = False,
        macro_block_size: int = 2,
        max_block_representations: int = 9,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.tarnet_two_heads = bool(tarnet_two_heads)
        if self.tarnet_two_heads:
            weight_tie = False
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
        if self.tarnet_two_heads:
            head0_n_fc = int(tarnet_head0_n_fc) if tarnet_head0_n_fc is not None else int(tarnet_head_n_fc)
            head1_n_fc = int(tarnet_head1_n_fc) if tarnet_head1_n_fc is not None else int(tarnet_head_n_fc)
            head0_hidden = (
                int(tarnet_head0_hidden_dim)
                if tarnet_head0_hidden_dim is not None
                else int(tarnet_head_hidden_dim)
                if tarnet_head_hidden_dim is not None
                else d_model
            )
            head1_hidden = (
                int(tarnet_head1_hidden_dim)
                if tarnet_head1_hidden_dim is not None
                else int(tarnet_head_hidden_dim)
                if tarnet_head_hidden_dim is not None
                else d_model
            )
            shared_n_fc = int(tarnet_head_n_fc)
            shared_hidden = (
                int(tarnet_head_hidden_dim) if tarnet_head_hidden_dim is not None else int(d_model)
            )
            # Shared "content" logits from trunk state; sentiment-specific heads add residuals.
            self.tarnet_shared_head = FullyConnected(
                n_fc=shared_n_fc,
                in_size=d_model,
                hidden_phi=shared_hidden,
                out_size=vocab_size,
                final_activation="linear",
                activation="gelu",
                dropout=dropout > 0,
                dropout_rate=dropout,
                batch_norm=False,
                use_bias=False,
            )
            self.tarnet_sentiment_delta0 = FullyConnected(
                n_fc=head0_n_fc,
                in_size=d_model,
                hidden_phi=head0_hidden,
                out_size=vocab_size,
                final_activation="linear",
                activation="gelu",
                dropout=dropout > 0,
                dropout_rate=dropout,
                batch_norm=False,
                use_bias=False,
            )
            self.tarnet_sentiment_delta1 = FullyConnected(
                n_fc=head1_n_fc,
                in_size=d_model,
                hidden_phi=head1_hidden,
                out_size=vocab_size,
                final_activation="linear",
                activation="gelu",
                dropout=dropout > 0,
                dropout_rate=dropout,
                batch_norm=False,
                use_bias=False,
            )
            _zero_init_fully_connected_output(self.tarnet_sentiment_delta0)
            _zero_init_fully_connected_output(self.tarnet_sentiment_delta1)
        elif not weight_tie:
            self.head = FullyConnected(
                n_fc=2,
                in_size=d_model,
                hidden_phi=d_model,
                out_size=vocab_size,
                final_activation="linear",
                activation="gelu",
                dropout=dropout > 0,
                dropout_rate=dropout,
                batch_norm=False,
                use_bias=False,
            )

    @property
    def decoder_stack(self) -> Literal["vanilla", "inter_block"]:
        """``vanilla``: sequential ``DecoderBlock`` stack. ``inter_block``: ``InterBlockAttnDecoderBlock``."""
        return "inter_block" if self.block_attn_residuals else "vanilla"

    def _run_vanilla_decoder_stack(self, x: torch.Tensor) -> torch.Tensor:
        """GPT-style pre-norm stack: each layer is x + Attn(LN(x)); x + FFN(LN(x))."""
        for block in self.blocks:
            x = block(x)
        return x

    def _run_inter_block_decoder_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Macro-block memory + depth-wise mixing before each sublayer (see ``InterBlockAttnDecoderBlock``)."""
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
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_hidden: bool = False,
        head_id: int | None = None,
        return_both_heads: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x = self.embed(x)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        if self.decoder_stack == "vanilla":
            x = self._run_vanilla_decoder_stack(x)
        else:
            x = self._run_inter_block_decoder_stack(x)
        x = self.ln_f(x)
        hidden = x
        if self.tarnet_two_heads:
            logits_shared = self.tarnet_shared_head(hidden)
            logits0 = logits_shared + self.tarnet_sentiment_delta0(hidden)
            logits1 = logits_shared + self.tarnet_sentiment_delta1(hidden)
            if return_both_heads:
                if return_hidden:
                    return logits0, logits1, hidden
                return logits0, logits1
            hid = hidden
            if head_id is None or int(head_id) == 0:
                logits = logits0
            else:
                logits = logits1
            if return_hidden:
                return logits, hid
            return logits

        if self.weight_tie:
            logits = hidden @ self.embed.weight.T
        else:
            logits = self.head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


def build_model(
    vocab_size: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 512,
    max_len: int = 512,
    dropout: float = 0.1,
    weight_tie: bool = True,
    tarnet_two_heads: bool = False,
    tarnet_head_n_fc: int = 2,
    tarnet_head_hidden_dim: int | None = None,
    tarnet_head0_n_fc: int | None = None,
    tarnet_head0_hidden_dim: int | None = None,
    tarnet_head1_n_fc: int | None = None,
    tarnet_head1_hidden_dim: int | None = None,
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
        tarnet_two_heads=bool(tarnet_two_heads),
        tarnet_head_n_fc=int(tarnet_head_n_fc),
        tarnet_head_hidden_dim=tarnet_head_hidden_dim,
        tarnet_head0_n_fc=tarnet_head0_n_fc,
        tarnet_head0_hidden_dim=tarnet_head0_hidden_dim,
        tarnet_head1_n_fc=tarnet_head1_n_fc,
        tarnet_head1_hidden_dim=tarnet_head1_hidden_dim,
        position_encoding=position_encoding,
        block_attn_residuals=block_attn_residuals,
        macro_block_size=macro_block_size,
        max_block_representations=max_block_representations,
    )
