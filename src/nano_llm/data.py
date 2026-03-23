"""Dataset loading and preprocessing for nano-llm."""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Dataset

from nano_llm.tokenizer import BPETokenizer, ByteBPETokenizer, CharTokenizer, HFByteBPETokenizer


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

SENTIMENT_OPEN = "[SENTIMENT]"
SENTIMENT_CLOSE = "[/SENTIMENT]"
REVIEW_OPEN = "[REVIEW]"
REVIEW_CLOSE = "[/REVIEW]"


def label_to_sentiment(label: int) -> str:
    """Map IMDB label to sentiment string."""
    return "positive" if int(label) == 1 else "negative"


def format_imdb_example(text: str, label: int) -> str:
    """Format IMDB sample for conditional review generation."""
    cleaned = " ".join(str(text).split())
    sentiment = label_to_sentiment(label)
    return (
        f"<bos>{SENTIMENT_OPEN} {sentiment} {SENTIMENT_CLOSE} "
        f"{REVIEW_OPEN} {cleaned} {REVIEW_CLOSE}<eos>"
    )


def load_tiny_shakespeare(val_split: float = 0.1) -> tuple[str, str]:
    """Load Tiny Shakespeare. Returns (train_text, val_text)."""
    import urllib.request

    try:
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Tiny Shakespeare from {TINY_SHAKESPEARE_URL}: {e}. "
            "Ensure the container has network access."
        ) from e
    n = len(text)
    split_idx = int(n * (1 - val_split))
    return text[:split_idx], text[split_idx:]


def load_imdb_sentiment(
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[str, str]:
    """Load and format IMDB from Hugging Face for next-token training.

    Returns:
        (train_text_blob, val_text_blob)
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "IMDB dataset requires the 'datasets' package. Install with: pip install datasets"
        ) from e

    ds = load_dataset("imdb")
    train_split = ds["train"]
    val_split = ds["test"]
    if max_train_samples is not None:
        train_split = train_split.select(range(min(len(train_split), int(max_train_samples))))
    if max_val_samples is not None:
        val_split = val_split.select(range(min(len(val_split), int(max_val_samples))))

    train_text = "\n".join(format_imdb_example(x["text"], x["label"]) for x in train_split)
    val_text = "\n".join(format_imdb_example(x["text"], x["label"]) for x in val_split)
    return train_text, val_text


def chunk_text(text: str, seq_len: int, stride: int | None = None) -> Iterator[list[int]]:
    """Yield tokenized chunks of length seq_len. Stride defaults to seq_len (no overlap)."""
    if stride is None:
        stride = seq_len
    tokenizer = CharTokenizer.from_text(text, add_special=False)
    ids = tokenizer.encode(text)
    for i in range(0, len(ids) - seq_len, stride):
        yield ids[i : i + seq_len]


class ShakespeareDataset(Dataset):
    """PyTorch Dataset of (input, target) pairs for next-token prediction."""

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
        seq_len: int = 128,
        stride: int | None = None,
    ) -> None:
        if stride is None:
            stride = seq_len
        ids = tokenizer.encode(text)
        self.chunks = [
            (ids[i : i + seq_len - 1], ids[i + 1 : i + seq_len])
            for i in range(0, len(ids) - seq_len, stride)
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.chunks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_dataloaders(
    train_text: str,
    val_text: str,
    tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
    seq_len: int = 128,
    batch_size: int = 32,
    stride: int | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Create train and optional val DataLoaders."""
    from torch.utils.data import DataLoader

    pin_memory = torch.cuda.is_available()
    train_ds = ShakespeareDataset(train_text, tokenizer, seq_len=seq_len, stride=stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_ds = ShakespeareDataset(val_text, tokenizer, seq_len=seq_len, stride=stride)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ) if len(val_ds) > 0 else None
    return train_loader, val_loader
