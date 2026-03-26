"""Tests for data loading and preprocessing."""

import torch

import pytest

from nano_llm.data import (
    IMDBDataset,
    NEGATIVE_SENTIMENT,
    POSITIVE_SENTIMENT,
    REVIEW_CLOSE,
    REVIEW_OPEN,
    SENTIMENT_CLOSE,
    SENTIMENT_OPEN,
    create_dataloaders,
    format_imdb_example,
    load_bookcorpus,
    load_imdb_sentiment,
    load_pg19,
    load_tiny_shakespeare,
    load_wikitext_2,
    sentiment_to_treatment,
    _normalize_text,
    _strip_html,
)
from nano_llm.tokenizer import CharTokenizer


def test_load_bookcorpus_returns_non_empty() -> None:
    try:
        train_text, val_text = load_bookcorpus(
            max_train_books=2, max_val_books=1, max_chars_per_book=2000
        )
    except (ImportError, Exception) as e:
        pytest.skip(f"BookCorpus load failed: {e}")
    assert len(train_text) > 0
    assert len(val_text) > 0


def test_load_pg19_returns_non_empty() -> None:
    try:
        train_text, val_text = load_pg19(
            max_train_books=2, max_val_books=1, max_chars_per_book=1000
        )
    except (ImportError, Exception) as e:
        pytest.skip(f"PG-19 load failed (network/dataset): {e}")
    assert len(train_text) > 0
    assert len(val_text) > 0


def test_load_wikitext_2_returns_non_empty() -> None:
    try:
        train_text, val_text = load_wikitext_2(max_train_samples=100, max_val_samples=50)
    except ImportError:
        pytest.skip("datasets package not installed")
    assert len(train_text) > 0
    assert len(val_text) > 0


def test_load_dataset_returns_non_empty() -> None:
    train_text, val_text = load_tiny_shakespeare(val_split=0.1)
    assert len(train_text) > 0
    assert len(val_text) > 0


def test_create_dataloader_batch_shape() -> None:
    train_text, val_text = load_tiny_shakespeare(val_split=0.1)
    tokenizer = CharTokenizer.from_text(train_text, add_special=False)
    train_loader, _ = create_dataloaders(train_text, val_text, tokenizer, seq_len=64, batch_size=8)
    for x, y in train_loader:
        assert x.shape[0] == 8
        assert x.shape[1] == 63  # seq_len - 1 (input)
        assert y.shape[1] == 63
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        break


def test_chunks_no_overlap_when_stride_equals_seq_len() -> None:
    train_text, val_text = load_tiny_shakespeare(val_split=0.1)
    tokenizer = CharTokenizer.from_text(train_text, add_special=False)
    train_loader, _ = create_dataloaders(
        train_text, val_text, tokenizer, seq_len=128, batch_size=4, stride=128
    )
    for x, y in train_loader:
        assert x.shape == (4, 127)
        assert y.shape == (4, 127)
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
        train_samples, val_samples = load_imdb_sentiment(
            max_train_samples=10, max_val_samples=5
        )
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
    samples = [
        format_imdb_example("Short positive review here.", 1)[0],
        format_imdb_example("A " + "long " * 80 + "positive review.", 1)[0],
    ]
    tokenizer = CharTokenizer.from_text("\n".join(samples), add_special=False)
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
    samples = [format_imdb_example("An enjoyable movie with great pacing.", 1)[0]]
    tokenizer = CharTokenizer.from_text("\n".join(samples), add_special=False)
    ds = IMDBDataset(samples, tokenizer, seq_len=64)
    x, y, treatment, review_mask, x_pos, x_neg, review_mask_pos, review_mask_neg = ds[0]
    assert x.shape == y.shape
    assert x.shape == x_pos.shape == x_neg.shape
    assert review_mask.shape == x.shape
    assert review_mask_pos.shape == x.shape
    assert review_mask_neg.shape == x.shape
    assert int(treatment.item()) == 1
    assert bool(review_mask.any().item()) is True
