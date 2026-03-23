"""Tests for tokenizers."""

import pytest

from nano_llm.tokenizer import BPETokenizer, ByteBPETokenizer, CharTokenizer, HFByteBPETokenizer


def test_encode_returns_list_of_ints() -> None:
    t = CharTokenizer()
    out = t.encode("hello")
    assert isinstance(out, list)
    assert all(isinstance(x, int) for x in out)


def test_decode_returns_string() -> None:
    t = CharTokenizer()
    out = t.decode([1, 2, 3])
    assert isinstance(out, str)


def test_encode_decode_roundtrip() -> None:
    t = CharTokenizer.from_text("abc", add_special=False)
    text = "abcabc"
    assert t.decode(t.encode(text)) == text


def test_vocab_size_char_level() -> None:
    t = CharTokenizer.from_text("abcdef", add_special=False)
    assert t.vocab_size == 6


def test_vocab_size_with_special() -> None:
    t = CharTokenizer.from_text("ab", add_special=True)
    assert t.vocab_size >= 3  # pad, unk, a, b


def test_bpe_encode_decode_roundtrip() -> None:
    t = BPETokenizer.from_text("to be or not to be", vocab_size=64)
    text = "to be or not to be"
    assert t.decode(t.encode(text)) == text


def test_bpe_word_boundary_aware_avoids_space_tokens() -> None:
    text = "to be to be"
    t = BPETokenizer.from_text(text, vocab_size=64, word_boundary_aware=True)
    # Single whitespace token may exist, but no merged token should contain whitespace.
    assert all(
        (" " not in token) or (len(token) == 1)
        for token in t.vocab
        if token != BPETokenizer.UNK_TOKEN
    )


def test_bpe_byte_roundtrip_unicode() -> None:
    t = ByteBPETokenizer.from_text("hello cafe\u0301 \U0001F600", vocab_size=128)
    text = "hello cafe\u0301 \U0001F600"
    assert t.decode(t.encode(text)) == text


def test_bpe_byte_handles_non_ascii() -> None:
    t = ByteBPETokenizer.from_text("naive facade \u4f60\u597d", vocab_size=128)
    text = "\u4f60\u597d facade"
    ids = t.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(x, int) for x in ids)
    assert t.decode(ids) == text


def test_hf_bpe_byte_roundtrip_when_available() -> None:
    pytest.importorskip("tokenizers")
    t = HFByteBPETokenizer.from_text("hello cafe\u0301 \U0001F600", vocab_size=128)
    text = "hello cafe\u0301 \U0001F600"
    ids = t.encode(text)
    assert t.decode(ids) == text
    state = t.to_state()
    restored = HFByteBPETokenizer.from_state(state)
    assert restored.decode(restored.encode(text)) == text
