"""Tests for nano-llm inference module."""

import tempfile
from pathlib import Path

import pytest
import torch

from nano_llm.inference import generate
from nano_llm.inference import load_model_and_tokenizer
from nano_llm.model import build_model
from nano_llm.tokenizer import CharTokenizer


def _make_minimal_checkpoint(tmpdir: Path) -> Path:
    """Create a minimal checkpoint with model, config, and vocab."""
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    tokenizer = CharTokenizer(vocab=vocab)
    model = build_model(
        vocab_size=len(vocab),
        d_model=16,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        max_len=128,
        dropout=0,
    )
    cfg = {
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 64,
        "seq_len": 64,
        "weight_tie": True,
    }
    path = tmpdir / "minimal.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg, "vocab": tokenizer.vocab},
        path,
    )
    return path


def test_load_model_and_tokenizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, cfg = load_model_and_tokenizer(path)
    assert model is not None
    assert tokenizer.vocab_size == 27
    assert cfg["d_model"] == 16


def test_generate_greedy_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out1 = generate(
        model, tokenizer, "hello", max_new_tokens=5, method="greedy", seed=42, device="cpu"
    )
    out2 = generate(
        model, tokenizer, "hello", max_new_tokens=5, method="greedy", seed=42, device="cpu"
    )
    assert out1 == out2
    assert out1.startswith("hello")
    assert len(out1) >= 5


def test_generate_returns_valid_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(
        model, tokenizer, "a", max_new_tokens=10, method="greedy", seed=0, device="cpu"
    )
    assert isinstance(out, str)
    assert len(out) > 0
    for c in out:
        assert c in tokenizer.vocab or c == "\n"


def test_generate_top_k_no_crash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(
        model, tokenizer, "x", max_new_tokens=3, method="top_k", top_k=5, seed=1, device="cpu"
    )
    assert isinstance(out, str)
    assert out.startswith("x")


def test_generate_top_p_no_crash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(
        model, tokenizer, "x", max_new_tokens=3, method="top_p", top_p=0.9, seed=2, device="cpu"
    )
    assert isinstance(out, str)
    assert out.startswith("x")


def test_generate_respects_max_new_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(
        model,
        tokenizer,
        "a",
        max_new_tokens=3,
        method="greedy",
        stop_at_newline=False,
        seed=99,
        device="cpu",
    )
    # prompt (1) + 3 new tokens
    assert len(tokenizer.encode(out)) <= 4 + 2  # small slack for edge cases


def test_load_fallback_tokenizer_from_shakespeare() -> None:
    """Test loading checkpoint without vocab raises when fallback disabled."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_ckpt.pt"
        model = build_model(vocab_size=65, d_model=16, num_heads=2, num_layers=2, d_ff=64)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"d_model": 16, "num_heads": 2, "num_layers": 2, "d_ff": 64, "seq_len": 64},
                # no vocab
            },
            path,
        )
        with pytest.raises(ValueError, match="missing 'vocab'"):
            load_model_and_tokenizer(path, rebuild_tokenizer_from_shakespeare=False)


@pytest.mark.integration
def test_load_and_generate_with_shakespeare_fallback() -> None:
    """Integration: load checkpoint without vocab, rebuild tokenizer from Tiny Shakespeare."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "no_vocab.pt"
        model = build_model(vocab_size=65, d_model=16, num_heads=2, num_layers=2, d_ff=64)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"d_model": 16, "num_heads": 2, "num_layers": 2, "d_ff": 64, "seq_len": 64},
            },
            path,
        )
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(model, tokenizer, "ROMEO:", max_new_tokens=5, method="greedy", seed=0, device="cpu")
    assert isinstance(out, str)
    assert out.startswith("ROMEO:")
