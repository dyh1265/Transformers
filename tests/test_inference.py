"""Tests for nano-llm inference module."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest
import torch

from nano_llm.inference import generate, generate_both_heads, load_model_and_tokenizer
from nano_llm.inference.generate import sanitize_output
from nano_llm.inference.worker import (
    process_openai_chat_payload,
    process_request_payload,
    process_single_request,
)
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


def _make_tarnet_checkpoint(tmpdir: Path) -> Path:
    """Create a minimal TARNet two-head checkpoint for worker tests."""
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
        weight_tie=False,
        tarnet_two_heads=True,
    )
    cfg = {
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 64,
        "seq_len": 64,
        "weight_tie": False,
        "tarnet_two_heads": True,
        "tarnet_head_n_fc": 2,
        "tokenizer_type": "hf_bpe_byte",
        "bpe_vocab_size": 128,
    }
    path = tmpdir / "minimal_tarnet.pt"
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
    assert sanitize_output("Hello\ufffd world") == "Hello world"
    assert sanitize_output("a\ufffd\ufffdb") == "ab"


def test_sanitize_output_fixes_punctuation() -> None:
    assert sanitize_output("Corbin\u00b4s") == "Corbin's"
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


def test_generate_kv_cache_matches_legacy_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = importlib.import_module("nano_llm.inference.generate")

    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    kw = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=12,
        max_context=128,
        method="greedy",
        seed=7,
        device="cpu",
        stop_at_newline=False,
        sanitize=False,
        censor_adult=False,
    )
    with monkeypatch.context() as m:
        m.setattr(gen, "_kv_cache_eligible", lambda *_a, **_k: False)
        legacy = gen.generate(**kw)
    cached = gen.generate(**kw)
    assert cached == legacy


def test_generate_kv_cache_sliding_window_matches_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = importlib.import_module("nano_llm.inference.generate")

    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    kw = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="hello world",
        max_new_tokens=24,
        max_context=16,
        method="greedy",
        seed=11,
        device="cpu",
        stop_at_newline=False,
        sanitize=False,
        censor_adult=False,
    )
    with monkeypatch.context() as m:
        m.setattr(gen, "_kv_cache_eligible", lambda *_a, **_k: False)
        legacy = gen.generate(**kw)
    cached = gen.generate(**kw)
    assert cached == legacy


def test_generate_kv_cache_rope_matches_legacy_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = importlib.import_module("nano_llm.inference.generate")

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
        position_encoding="rope",
    )
    model.eval()
    kw = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        max_context=64,
        method="greedy",
        seed=3,
        device="cpu",
        stop_at_newline=False,
        sanitize=False,
        censor_adult=False,
    )
    with monkeypatch.context() as m:
        m.setattr(gen, "_kv_cache_eligible", lambda *_a, **_k: False)
        legacy = gen.generate(**kw)
    cached = gen.generate(**kw)
    assert cached == legacy


def test_generate_both_heads_kv_cache_matches_legacy_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = importlib.import_module("nano_llm.inference.generate")

    with tempfile.TemporaryDirectory() as tmp:
        path = _make_tarnet_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    kw = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=14,
        max_context=128,
        method="greedy",
        seed=13,
        device="cpu",
        stop_at_newline=False,
        sanitize=False,
        censor_adult=False,
    )
    with monkeypatch.context() as m:
        m.setattr(gen, "_kv_cache_eligible", lambda *_a, **_k: False)
        leg0, leg1 = gen.generate_both_heads(**kw)
    c0, c1 = gen.generate_both_heads(**kw)
    assert (c0, c1) == (leg0, leg1)


def test_generate_returns_valid_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_minimal_checkpoint(Path(tmp))
        model, tokenizer, _ = load_model_and_tokenizer(path, device="cpu")
    out = generate(model, tokenizer, "a", max_new_tokens=10, method="greedy", seed=0, device="cpu")
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
                "config": {
                    "d_model": 16,
                    "num_heads": 2,
                    "num_layers": 2,
                    "d_ff": 64,
                    "seq_len": 64,
                },
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


def test_load_tarnet_fc_deltas_with_inter_trunk() -> None:
    """Inter-block trunk + TARNet FullyConnected Δ heads."""
    pytest.importorskip("tokenizers")
    train_corpus = "hello world " * 50
    tokenizer = HFByteBPETokenizer.from_text(train_corpus, vocab_size=128)
    vs = tokenizer.vocab_size
    model = build_model(
        vocab_size=vs,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        max_len=64,
        dropout=0,
        weight_tie=False,
        tarnet_two_heads=True,
        block_attn_residuals=True,
        position_encoding="rope",
    )
    cfg = {
        "d_model": 32,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 64,
        "seq_len": 32,
        "weight_tie": False,
        "tarnet_two_heads": True,
        "block_attn_residuals": True,
        "position_encoding": "rope",
        "macro_block_size": 2,
        "max_block_representations": 9,
        "tarnet_head_n_fc": 2,
        "tokenizer_type": "hf_bpe_byte",
        "bpe_vocab_size": 128,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tarnet_trunk_inter_fc_delta.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "tokenizer_state": tokenizer.to_state(),
            },
            path,
        )
        loaded, _, _ = load_model_and_tokenizer(path, device="cpu")
    assert hasattr(loaded, "tarnet_sentiment_delta0")


def test_process_single_request_success_writes_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        request_dir = tmp_path / "requests"
        response_dir = tmp_path / "responses"
        request_dir.mkdir()
        request_path = request_dir / "job-1.json"
        request_path.write_text(
            (
                '{"job_id":"job-1","prompt":"hello","max_tokens":3,'
                '"method":"greedy","no_stop_newline":true}'
            ),
            encoding="utf-8",
        )

        response_name, ok = process_single_request(
            model=model,
            tokenizer=tokenizer,
            request_path=request_path,
            response_dir=response_dir,
            max_context=int(cfg.get("seq_len", 128)),
        )

        assert ok is True
        assert response_name == "job-1.json"
        payload = json.loads((response_dir / response_name).read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["job_id"] == "job-1"
        assert isinstance(payload["output"], str)
        assert (request_dir / "job-1.json").exists() is False


def test_process_single_request_both_reviews_returns_y0_y1() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        request_dir = tmp_path / "requests"
        response_dir = tmp_path / "responses"
        request_dir.mkdir()
        request_path = request_dir / "job-both.json"
        request_path.write_text(
            (
                '{"job_id":"job-both","prompt":"hello","both_reviews":true,'
                '"max_tokens":2,"method":"greedy","no_stop_newline":true}'
            ),
            encoding="utf-8",
        )

        response_name, ok = process_single_request(
            model=model,
            tokenizer=tokenizer,
            request_path=request_path,
            response_dir=response_dir,
            max_context=int(cfg.get("seq_len", 128)),
        )

        assert ok is True
        assert response_name == "job-both.json"
        payload = json.loads((response_dir / response_name).read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["job_id"] == "job-both"
        assert isinstance(payload["output_y0"], str)
        assert isinstance(payload["output_y1"], str)
        assert "output" not in payload


def test_process_single_request_error_writes_error_response() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        request_dir = tmp_path / "requests"
        response_dir = tmp_path / "responses"
        request_dir.mkdir()
        request_path = request_dir / "broken.json"
        request_path.write_text('{"job_id":"broken"}', encoding="utf-8")

        response_name, ok = process_single_request(
            model=model,
            tokenizer=tokenizer,
            request_path=request_path,
            response_dir=response_dir,
            max_context=int(cfg.get("seq_len", 128)),
        )

        assert ok is False
        assert response_name == "broken.json"
        payload = json.loads((response_dir / response_name).read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["job_id"] == "broken"
        assert "prompt" in payload["error"]


def test_process_single_request_rejects_non_tarnet_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_minimal_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        request_dir = tmp_path / "requests"
        response_dir = tmp_path / "responses"
        request_dir.mkdir()
        request_path = request_dir / "job-plain.json"
        request_path.write_text(
            '{"job_id":"job-plain","prompt":"hello","max_tokens":2}',
            encoding="utf-8",
        )

        response_name, ok = process_single_request(
            model=model,
            tokenizer=tokenizer,
            request_path=request_path,
            response_dir=response_dir,
            max_context=int(cfg.get("seq_len", 128)),
        )

        assert ok is False
        assert response_name == "job-plain.json"
        payload = json.loads((response_dir / response_name).read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["job_id"] == "job-plain"
        assert "TARNet-only" in payload["error"]


def test_process_request_payload_both_reviews_returns_two_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        response = process_request_payload(
            model=model,
            tokenizer=tokenizer,
            payload={
                "job_id": "api-1",
                "prompt": "hello",
                "both_reviews": True,
                "max_tokens": 2,
                "method": "greedy",
                "no_stop_newline": True,
            },
            max_context=int(cfg.get("seq_len", 128)),
        )
        assert response["ok"] is True
        assert response["job_id"] == "api-1"
        assert isinstance(response["output_y0"], str)
        assert isinstance(response["output_y1"], str)


def test_process_openai_chat_payload_single_choice_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        resp = process_openai_chat_payload(
            model=model,
            tokenizer=tokenizer,
            payload={
                "model": "nano-llm-local",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "top_p": 1.0,
            },
            max_context=int(cfg.get("seq_len", 128)),
        )
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "nano-llm-local"
        assert isinstance(resp["choices"], list)
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(resp["choices"][0]["message"]["content"], str)
        assert "usage" in resp


def test_process_openai_chat_payload_both_reviews_two_choices() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _make_tarnet_checkpoint(tmp_path)
        model, tokenizer, cfg = load_model_and_tokenizer(ckpt, device="cpu")
        resp = process_openai_chat_payload(
            model=model,
            tokenizer=tokenizer,
            payload={
                "messages": [{"role": "user", "content": "hello"}],
                "both_reviews": True,
                "max_tokens": 2,
                "top_p": 1.0,
            },
            max_context=int(cfg.get("seq_len", 128)),
        )
        assert len(resp["choices"]) == 2
        assert resp["choices"][0]["index"] == 0
        assert resp["choices"][1]["index"] == 1
        assert isinstance(resp["choices"][0]["message"]["content"], str)
        assert isinstance(resp["choices"][1]["message"]["content"], str)
