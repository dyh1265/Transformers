"""Tests for transformer layers."""

import torch

from nano_llm.layers import (
    CausalSelfAttention,
    DecoderBlock,
    InterBlockAttnDecoderBlock,
    PositionalEncoding,
    RoPE,
)


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


def test_rope_shape() -> None:
    rope = RoPE(d_k=32, max_len=128)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape
    assert kr.shape == k.shape


def test_rope_deterministic() -> None:
    rope = RoPE(d_k=16, max_len=64)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    q1, k1 = rope(q, k)
    q2, k2 = rope(q, k)
    assert torch.allclose(q1, q2)
    assert torch.allclose(k1, k2)


def test_causal_attention_shape() -> None:
    attn = CausalSelfAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)


def test_decoder_block_shape() -> None:
    block = DecoderBlock(d_model=64, num_heads=4, d_ff=256)
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert out.shape == (2, 16, 64)


def test_decoder_block_gradient_flows() -> None:
    block = DecoderBlock(d_model=32, num_heads=2, d_ff=128)
    x = torch.randn(1, 8, 32, requires_grad=True)
    out, _ = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_inter_block_attn_decoder_block_forward() -> None:
    layer = InterBlockAttnDecoderBlock(d_model=32, num_heads=2, d_ff=128)
    x = torch.randn(2, 8, 32, requires_grad=True)
    blocks = [x]
    x2, blocks2, partial = layer(x, blocks, None, layer_index=0, macro_block_size=2)
    assert x2.shape == (2, 8, 32)
    assert partial is not None
    x3, blocks3, partial2 = layer(x2, blocks2, partial, layer_index=1, macro_block_size=2)
    assert x3.shape == (2, 8, 32)
    assert len(blocks3) == 2
    assert partial2 is None
    x3.sum().backward()
    assert x.grad is not None
