"""Tests for transformer layers."""

import torch

from nano_llm.layers import CausalSelfAttention, DecoderBlock, PositionalEncoding


def test_positional_encoding_shape() -> None:
    pe = PositionalEncoding(d_model=64, max_len=128)
    x = torch.randn(2, 32, 64)
    out = pe(x)
    assert out.shape == (2, 32, 64)


def test_positional_encoding_deterministic() -> None:
    pe = PositionalEncoding(d_model=32, max_len=64)
    x = torch.ones(1, 16, 32)
    out1 = pe(x)
    out2 = pe(x)
    assert torch.allclose(out1, out2)


def test_positional_encoding_different_positions() -> None:
    pe = PositionalEncoding(d_model=32, max_len=64)
    x = torch.zeros(1, 4, 32)
    out = pe(x)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_causal_attention_shape() -> None:
    attn = CausalSelfAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)


def test_decoder_block_shape() -> None:
    block = DecoderBlock(d_model=64, num_heads=4, d_ff=256)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_decoder_block_gradient_flows() -> None:
    block = DecoderBlock(d_model=32, num_heads=2, d_ff=128)
    x = torch.randn(1, 8, 32, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
