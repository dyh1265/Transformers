"""Tests for data loading and preprocessing."""

import pytest
import torch

from nano_llm.data import create_dataloaders, load_tiny_shakespeare
from nano_llm.tokenizer import CharTokenizer


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
