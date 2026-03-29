"""Tests for nano-llm inference module."""

import tempfile
from pathlib import Path

import pytest
import torch

from nano_llm.inference import generate
from nano_llm.inference import load_model_and_tokenizer
from nano_llm.inference.generate import sanitize_output
from nano_llm.model import build_model
from nano_llm.tokenizer import HFByteBPETokenizer


def _make_minimal_checkpoint(tmpdir: Path) -> Path:
    """Create a minimal checkpoint with model, config, and HF tokenizer_state."""
    pytest.importorskip("tokenizers")
    train_corpus = "hello world abc xyz " * 100
    tokenizer = HFByteBPETokenizer.from_text(train_corpus, vocab_size=128)
    vs = tokenizer.vocab_size
    model = build_model(
        vocab_size=vs,
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
        "tokenizer_type": "hf_bpe_byte",
        "bpe_vocab_size": 128,
    }
    path = tmpdir / "minimal.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "tokenizer_state": tokenizer.to_state(),
        },
        path,
    )
    return path


def test_sanitize_output_removes_replacement_char() -> None:
    assert sanitize_output("Hello\uFFFD world") == "Hello world"
    assert sanitize_output("a\uFFFD\uFFFDb") == "ab"


def test_sanitize_output_fixes_punctuation() -> None:
    assert sanitize_output("Corbin\u00B4s") == "Corbin's"
    assert sanitize_output("don\u2019t") == "don't"


def test_sanitize_output_preserves_normal_text() -> None:
    assert sanitize_output("Hello world") == "Hello world"
    assert sanitize_output("cafe") == "cafe"


def test_load_model_and_tokenizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, cfg = load_model_and_tokenizer(path)
    assert model is not None
    assert tokenizer.vocab_size > 0
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
    prompt = "a"
    out = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=3,
        method="greedy",
        stop_at_newline=False,
        seed=99,
        device="cpu",
    )
    assert len(tokenizer.encode(out)) <= len(tokenizer.encode(prompt)) + 3


def test_generate_stops_on_stop_sequence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(
        model,
        tokenizer,
        "hello",
        max_new_tokens=20,
        method="greedy",
        stop_at_newline=False,
        stop_sequence="h",
        seed=42,
        device="cpu",
    )
    assert isinstance(out, str)
    assert out.startswith("hello")


def test_load_raises_when_no_tokenizer_state_and_corpus_rebuild_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_ckpt.pt"
        model = build_model(vocab_size=65, d_model=16, num_heads=2, num_layers=2, d_ff=64)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"d_model": 16, "num_heads": 2, "num_layers": 2, "d_ff": 64, "seq_len": 64},
            },
            path,
        )
        with pytest.raises(ValueError, match="tokenizer_state"):
            load_model_and_tokenizer(path, rebuild_tokenizer_from_corpus=False)


def test_load_and_generate_with_tokenizer_state() -> None:
    pytest.importorskip("tokenizers")
    train_corpus = "HI there " * 200 + "common words " * 200
    tokenizer = HFByteBPETokenizer.from_text(train_corpus, vocab_size=128)
    vs = tokenizer.vocab_size
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "with_tok.pt"
        model = build_model(vocab_size=vs, d_model=16, num_heads=2, num_layers=2, d_ff=64)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "d_model": 16,
                    "num_heads": 2,
                    "num_layers": 2,
                    "d_ff": 64,
                    "seq_len": 64,
                    "tokenizer_type": "hf_bpe_byte",
                    "bpe_vocab_size": 128,
                },
                "tokenizer_state": tokenizer.to_state(),
            },
            path,
        )
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(model, tokenizer, "HI", max_new_tokens=5, method="greedy", seed=0, device="cpu")
    assert isinstance(out, str)
    assert out.startswith("HI")
