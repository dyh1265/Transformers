"""Tests for data loading and preprocessing."""

import pytest
import torch

from nano_llm.data import (
    NEGATIVE_SENTIMENT,
    POSITIVE_SENTIMENT,
    REVIEW_CLOSE,
    REVIEW_OPEN,
    SENTIMENT_CLOSE,
    SENTIMENT_OPEN,
    IMDBDataset,
    _normalize_text,
    _strip_html,
    create_dataloaders,
    format_imdb_example,
    load_imdb_sentiment,
    sentiment_to_treatment,
)
from nano_llm.tokenizer import HFByteBPETokenizer


def _tiny_imdb_samples() -> tuple[list[str], list[str]]:
    return (
        [
            format_imdb_example("Good film.", 1)[0],
            format_imdb_example("Bad film.", 0)[0],
        ],
        [format_imdb_example("Okay.", 1)[0]],
    )


def test_create_dataloader_batch_shape() -> None:
    pytest.importorskip("tokenizers")
    train_samples, val_samples = _tiny_imdb_samples()
    tokenizer = HFByteBPETokenizer.from_text("\n".join(train_samples + val_samples), vocab_size=128)
    train_loader, _ = create_dataloaders(
        train_samples, val_samples, tokenizer, seq_len=64, batch_size=2
    )
    for batch in train_loader:
        x, y = batch[0], batch[1]
        assert x.shape[0] <= 2
        assert x.shape[1] == 63  # seq_len - 1 (input)
        assert y.shape[1] == 63
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        break


def test_chunks_no_overlap_when_stride_equals_seq_len() -> None:
    pytest.importorskip("tokenizers")
    train_samples, val_samples = _tiny_imdb_samples()
    tokenizer = HFByteBPETokenizer.from_text("\n".join(train_samples + val_samples), vocab_size=128)
    train_loader, _ = create_dataloaders(
        train_samples, val_samples, tokenizer, seq_len=128, batch_size=1, stride=128
    )
    for batch in train_loader:
        x, y = batch[0], batch[1]
        assert x.shape[0] == 1
        assert x.shape[1] == 127
        assert y.shape == x.shape
        break


def test_format_imdb_example_positive() -> None:
    out_list = format_imdb_example("This movie was wonderful!", 1)
    assert len(out_list) == 1
    out = out_list[0]
    assert out.startswith("<bos>")
    assert out.endswith("<eos>")
    assert f"{SENTIMENT_OPEN} positive {SENTIMENT_CLOSE}" in out
    assert REVIEW_OPEN in out
    assert REVIEW_CLOSE in out


def test_format_imdb_example_negative() -> None:
    out_list = format_imdb_example("Bad acting and boring plot.", 0)
    assert len(out_list) == 1
    assert f"{SENTIMENT_OPEN} negative {SENTIMENT_CLOSE}" in out_list[0]


def test_normalize_text_asciifies_accents() -> None:
    assert _normalize_text("café") == "cafe"
    assert _normalize_text("naïve") == "naive"
    assert _normalize_text("résumé") == "resume"


def test_strip_html_removes_tags() -> None:
    assert _strip_html("Hello <br /> world") == "Hello world"
    assert _strip_html("A <b>bold</b> claim") == "A bold claim"
    assert "<" not in _strip_html("Some <br /><br /> text with tags")
    assert ">" not in _strip_html("Some <br /><br /> text with tags")


def test_load_imdb_sentiment_returns_samples() -> None:
    try:
        train_samples, val_samples = load_imdb_sentiment(max_train_samples=10, max_val_samples=5)
    except ImportError:
        pytest.skip("datasets package not installed")
    assert len(train_samples) >= 10
    assert len(val_samples) >= 5
    for s in train_samples[:3]:
        assert s.startswith("<bos>")
        assert f"{SENTIMENT_OPEN}" in s and SENTIMENT_CLOSE in s
        assert REVIEW_OPEN in s and REVIEW_CLOSE in s


def test_load_imdb_subsample_is_label_stratified() -> None:
    try:
        train_samples, val_samples = load_imdb_sentiment(
            max_train_samples=100, max_val_samples=40, subset_seed=123
        )
    except ImportError:
        pytest.skip("datasets package not installed")
    neg_tag = f"{SENTIMENT_OPEN} {NEGATIVE_SENTIMENT} {SENTIMENT_CLOSE}"
    pos_tag = f"{SENTIMENT_OPEN} {POSITIVE_SENTIMENT} {SENTIMENT_CLOSE}"
    n_neg_tr = sum(neg_tag in s for s in train_samples)
    n_pos_tr = sum(pos_tag in s for s in train_samples)
    n_neg_va = sum(neg_tag in s for s in val_samples)
    n_pos_va = sum(pos_tag in s for s in val_samples)
    assert n_neg_tr == n_pos_tr == 50
    assert n_neg_va == n_pos_va == 20


def test_imdb_dataset_chunks_keep_sentiment_prefix() -> None:
    pytest.importorskip("tokenizers")
    samples = [
        format_imdb_example("Short positive review here.", 1)[0],
        format_imdb_example("A " + "long " * 80 + "positive review.", 1)[0],
    ]
    tokenizer = HFByteBPETokenizer.from_text("\n".join(samples), vocab_size=128)
    ds = IMDBDataset(samples, tokenizer, seq_len=64)
    assert len(ds) >= 2
    for i in range(min(5, len(ds))):
        x, _, *_ = ds[i]
        decoded = tokenizer.decode(x.tolist())
        assert SENTIMENT_OPEN in decoded and "positive" in decoded
        assert REVIEW_OPEN in decoded


def test_format_imdb_example_splits_long_review_into_multiple_samples() -> None:
    long_review = "A " + "word " * 50 + "end."
    out_list = format_imdb_example(long_review, 1, max_review_chars=30)
    assert len(out_list) >= 2
    for out in out_list:
        assert f"{SENTIMENT_OPEN} positive {SENTIMENT_CLOSE}" in out
        review_part = out.split(REVIEW_OPEN)[1].split(REVIEW_CLOSE)[0].strip()
        assert len(review_part) <= 30


def test_sentiment_to_treatment_mapping() -> None:
    assert sentiment_to_treatment(POSITIVE_SENTIMENT) == 1
    assert sentiment_to_treatment(NEGATIVE_SENTIMENT) == 0


def test_imdb_dataset_returns_counterfactual_fields() -> None:
    pytest.importorskip("tokenizers")
    samples = [format_imdb_example("An enjoyable movie with great pacing.", 1)[0]]
    tokenizer = HFByteBPETokenizer.from_text("\n".join(samples), vocab_size=128)
    ds = IMDBDataset(samples, tokenizer, seq_len=64)
    x, y, treatment, review_mask, x_pos, x_neg, review_mask_pos, review_mask_neg = ds[0]
    assert x.shape == y.shape
    assert x.shape == x_pos.shape == x_neg.shape
    assert review_mask.shape == x.shape
    assert review_mask_pos.shape == x.shape
    assert review_mask_neg.shape == x.shape
    assert int(treatment.item()) == 1
    assert bool(review_mask.any().item()) is True
