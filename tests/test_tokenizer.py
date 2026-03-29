"""Tests for HF byte-level BPE tokenizer."""

import pytest

from nano_llm.tokenizer import HFByteBPETokenizer


def test_hf_bpe_byte_roundtrip_when_available() -> None:
    pytest.importorskip("tokenizers")
    t = HFByteBPETokenizer.from_text("hello cafe\u0301 \U0001F600", vocab_size=128)
    text = "hello cafe\u0301 \U0001F600"
    ids = t.encode(text)
    assert t.decode(ids) == text
    state = t.to_state()
    restored = HFByteBPETokenizer.from_state(state)
    assert restored.decode(restored.encode(text)) == text


def test_pad_id_present() -> None:
    pytest.importorskip("tokenizers")
    t = HFByteBPETokenizer.from_text("abc " * 50, vocab_size=64)
    assert isinstance(t.pad_id, int)
    assert 0 <= t.pad_id < t.vocab_size
