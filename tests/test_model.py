"""Tests for nano-llm model."""

import torch

from nano_llm.model import build_model


def test_forward_pass_output_shape() -> None:
    model = build_model(vocab_size=65, d_model=32, num_heads=2, num_layers=2, d_ff=128)
    x = torch.randint(0, 65, (2, 64))
    logits = model(x)
    assert logits.shape == (2, 64, 65)


def test_model_variable_seq_len() -> None:
    model = build_model(vocab_size=65, d_model=32, num_heads=2, num_layers=2, d_ff=128, max_len=256)
    for seq_len in (64, 128):
        x = torch.randint(0, 65, (2, seq_len))
        logits = model(x)
        assert logits.shape == (2, seq_len, 65)
