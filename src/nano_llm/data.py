"""Dataset loading and preprocessing for nano-llm."""

from __future__ import annotations

import re
import unicodedata
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


def _strip_html(text: str) -> str:
    """Remove HTML tags and replace with space. Normalizes whitespace."""
    out = re.sub(r"<[^>]+>", " ", str(text))
    return " ".join(out.split())


def _normalize_text(text: str) -> str:
    """Normalize text for training: NFKC, map accented chars to ASCII, drop control chars.

    Reduces strange Unicode in model outputs by training on cleaner text.
    """
    text = unicodedata.normalize("NFKC", text)
    # NFD splits accents; remove combining marks -> ASCII base
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove control/surrogate chars
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn")
    )
    return " ".join(text.split())


def format_imdb_example(text: str, label: int, max_review_chars: int | None = None) -> list[str]:
    """Format IMDB sample(s) for conditional review generation.

    When max_review_chars is set, long reviews are split into multiple samples,
    each with the same sentiment. The rest of the review becomes additional
    training samples.

    Args:
        text: Raw review text.
        label: 0=negative, 1=positive.
        max_review_chars: If set, split review into chunks of at most this many
            chars (at word boundary). Each chunk is a separate training sample.

    Returns:
        List of formatted strings (one per chunk, or one if no split).
    """
    cleaned = _normalize_text(_strip_html(text))
    sentiment = label_to_sentiment(label)
    prefix = f"<bos>{SENTIMENT_OPEN} {sentiment} {SENTIMENT_CLOSE} {REVIEW_OPEN} "
    suffix = f" {REVIEW_CLOSE}<eos>"

    if max_review_chars is None or max_review_chars <= 0 or len(cleaned) <= max_review_chars:
        return [f"{prefix}{cleaned}{suffix}"]

    chunks: list[str] = []
    rest = cleaned
    while rest:
        if len(rest) <= max_review_chars:
            chunks.append(rest)
            break
        trunc = rest[:max_review_chars]
        boundary = trunc.rfind(" ")
        if boundary <= 0:
            boundary = max_review_chars
        chunks.append(rest[:boundary].strip())
        rest = rest[boundary:].strip()
    return [f"{prefix}{chunk}{suffix}" for chunk in chunks if chunk]


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


def load_wikitext_2(
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[str, str]:
    """Load WikiText-2 raw for pretraining. Returns (train_text, val_text).

    Use for pretrain phase before fine-tuning on IMDB. Builds general language
    semantics from Wikipedia-style text.

    Args:
        max_train_samples: Limit number of train lines (for faster experiments).
        max_val_samples: Limit number of validation lines.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "WikiText-2 requires the 'datasets' package. Install with: pip install datasets"
        ) from e

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_split = ds["train"]
    val_split = ds["validation"]
    if max_train_samples is not None:
        train_split = train_split.select(
            range(min(len(train_split), int(max_train_samples)))
        )
    if max_val_samples is not None:
        val_split = val_split.select(
            range(min(len(val_split), int(max_val_samples)))
        )

    train_text = "\n".join(_normalize_text(x["text"]) for x in train_split)
    val_text = "\n".join(_normalize_text(x["text"]) for x in val_split)
    return train_text, val_text


def load_wikitext_103(
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[str, str]:
    """Load WikiText-103 raw for pretraining. Returns (train_text, val_text).

    ~100M tokens, much larger than WikiText-2. Parquet-based, reliable loader.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "WikiText-103 requires the 'datasets' package. Install with: pip install datasets"
        ) from e

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    train_split = ds["train"]
    val_split = ds["validation"]
    if max_train_samples is not None:
        train_split = train_split.select(
            range(min(len(train_split), int(max_train_samples)))
        )
    if max_val_samples is not None:
        val_split = val_split.select(
            range(min(len(val_split), int(max_val_samples)))
        )

    train_text = "\n".join(_normalize_text(x["text"]) for x in train_split if x.get("text"))
    val_text = "\n".join(_normalize_text(x["text"]) for x in val_split if x.get("text"))
    return train_text, val_text


def load_pg19(
    *,
    max_train_books: int | None = None,
    max_val_books: int | None = None,
    max_chars_per_book: int | None = None,
) -> tuple[str, str]:
    """Load PG-19 (Project Gutenberg books) for pretraining. Returns (train_text, val_text).

    Books are 19th-century public domain texts. Large corpus for BPE learning
    and language modeling. May require trust_remote_code; if loading fails,
    try dataset_id='bookcorpus' instead.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "PG-19 requires the 'datasets' package. Install with: pip install datasets"
        ) from e

    try:
        ds = load_dataset("deepmind/pg19", trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"PG-19 load failed (dataset scripts deprecated). Use dataset_id='bookcorpus' instead. {e}"
        ) from e

    train_split = ds["train"]
    val_split = ds.get("validation", ds.get("test"))

    def trunc(s: str) -> str:
        t = _normalize_text(s)
        if max_chars_per_book and len(t) > max_chars_per_book:
            return t[:max_chars_per_book]
        return t

    if max_train_books is not None:
        n = min(len(train_split), int(max_train_books))
        train_split = train_split.select(range(n))
    if max_val_books is not None:
        n = min(len(val_split), int(max_val_books))
        val_split = val_split.select(range(n))

    train_text = "\n\n".join(trunc(x["text"]) for x in train_split if x.get("text"))
    val_text = "\n\n".join(trunc(x["text"]) for x in val_split if x.get("text"))
    return train_text, val_text


def load_bookcorpus(
    *,
    max_train_books: int | None = None,
    max_val_books: int | None = None,
    max_chars_per_book: int | None = None,
    val_split_ratio: float = 0.05,
) -> tuple[str, str]:
    """Load book text for pretraining. Returns (train_text, val_text).

    Uses Gutenberg-BookCorpus-Cleaned (Parquet, no deprecated scripts).
    ~51k English public-domain books from Project Gutenberg.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Book dataset requires the 'datasets' package. Install with: pip install datasets"
        ) from e

    # Parquet-based dataset; column "Context" holds book text
    ds = load_dataset(
        "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English",
        split="train",
        trust_remote_code=False,
    )
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split_ratio))
    n_train = n_total - n_val

    def trunc(s: str) -> str:
        t = _normalize_text(s)
        if max_chars_per_book and len(t) > max_chars_per_book:
            return t[:max_chars_per_book]
        return t

    if max_train_books is not None:
        n_train = min(n_train, int(max_train_books))
    if max_val_books is not None:
        n_val = min(n_val, int(max_val_books))

    train_split = ds.select(range(n_train))
    val_split = ds.select(range(n_train, min(n_train + n_val, n_total)))

    # Column may be Context, context, text, or content (HF sometimes lowercases)
    cols = ds.column_names
    text_key = next(
        (c for c in ("Context", "context", "text", "content") if c in cols),
        cols[0] if cols else None,
    )
    if not text_key:
        raise ValueError("Book dataset has no text column")

    def get_text(x: dict) -> str:
        v = x.get(text_key)
        return v if isinstance(v, str) else str(v) if v is not None else ""

    train_text = "\n\n".join(
        trunc(get_text(x)) for x in train_split if get_text(x)
    )
    val_text = "\n\n".join(
        trunc(get_text(x)) for x in val_split if get_text(x)
    )
    if not train_text:
        raise ValueError(
            f"Book dataset produced empty train text. Columns: {cols}. "
            f"Sample keys: {list(train_split[0].keys()) if len(train_split) else 'no rows'}"
        )
    return train_text, val_text


def load_imdb_sentiment(
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_review_chars: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load and format IMDB from Hugging Face for next-token training.

    Args:
        max_train_samples: Limit number of train samples.
        max_val_samples: Limit number of val samples.
        max_review_chars: Split each review into chunks of at most this many chars.
            Each chunk becomes a separate training sample with the same sentiment.

    Returns:
        (train_samples, val_samples) - each a list of formatted strings.
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

    def fmt(x: dict) -> list[str]:
        return format_imdb_example(x["text"], x["label"], max_review_chars=max_review_chars)

    train_samples = [s for x in train_split for s in fmt(x)]
    val_samples = [s for x in val_split for s in fmt(x)]
    return train_samples, val_samples


def chunk_text(text: str, seq_len: int, stride: int | None = None) -> Iterator[list[int]]:
    """Yield tokenized chunks of length seq_len. Stride defaults to seq_len (no overlap)."""
    if stride is None:
        stride = seq_len
    tokenizer = CharTokenizer.from_text(text, add_special=False)
    ids = tokenizer.encode(text)
    for i in range(0, len(ids) - seq_len, stride):
        yield ids[i : i + seq_len]


def _imdb_prefix_end(sample: str) -> int:
    """Return character index where the review body starts (after '[REVIEW] ')."""
    marker = "[REVIEW] "
    idx = sample.find(marker)
    if idx < 0:
        return 0
    return idx + len(marker)


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


# Target value for padded positions (ignored in CrossEntropyLoss)
PAD_TARGET_IGNORE_INDEX = -100


class IMDBDataset(Dataset):
    """IMDB dataset that keeps the sentiment prefix in every chunk's context.

    Chunks within each sample so the model always sees [SENTIMENT] ... [REVIEW]
    when predicting the next token. Fixes label drift from sliding-window chunking.
    Short chunks are padded to seq_len-1 for batched training.
    """

    def __init__(
        self,
        samples: list[str],
        tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
        seq_len: int = 128,
        stride: int | None = None,
    ) -> None:
        if stride is None:
            stride = seq_len
        self.seq_len = seq_len
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self.chunks: list[tuple[list[int], list[int]]] = []
        for sample in samples:
            ids = tokenizer.encode(sample)
            if len(ids) < seq_len:
                self.chunks.append((ids[:-1], ids[1:]))
                continue
            prefix_end = _imdb_prefix_end(sample)
            prefix_ids = tokenizer.encode(sample[:prefix_end])
            body_start = len(prefix_ids)
            body_len = len(ids) - body_start
            chunk_body_len = seq_len - body_start
            if chunk_body_len <= 0 or body_len <= 0:
                # Prefix too long or no body: fall back to sliding window
                for i in range(0, len(ids) - seq_len, stride):
                    self.chunks.append((ids[i : i + seq_len - 1], ids[i + 1 : i + seq_len]))
                continue
            # Chunk k = prefix + body[k * chunk_body_len : (k+1) * chunk_body_len]
            body_stride = min(stride, chunk_body_len)
            k = 0
            while True:
                if k == 0:
                    chunk_ids = ids[0:seq_len]
                else:
                    start = body_start + k * body_stride
                    end = start + chunk_body_len
                    if start >= len(ids):
                        break
                    end = min(end, len(ids))
                    chunk_ids = ids[0:body_start] + ids[start:end]
                    if len(chunk_ids) < seq_len:
                        break
                    chunk_ids = chunk_ids[:seq_len]
                self.chunks.append((chunk_ids[:-1], chunk_ids[1:]))
                k += 1
                if body_start + k * body_stride >= len(ids):
                    break

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.chunks[idx]
        pad_len = (self.seq_len - 1) - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [PAD_TARGET_IGNORE_INDEX] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_dataloaders(
    train_text: str,
    val_text: str,
    tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
    seq_len: int = 128,
    batch_size: int = 32,
    stride: int | None = None,
    *,
    train_samples: list[str] | None = None,
    val_samples: list[str] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Create train and optional val DataLoaders.

    When train_samples and val_samples are provided (e.g. for IMDB), uses
    IMDBDataset so every chunk keeps the sentiment prefix in context.
    """
    from torch.utils.data import DataLoader

    pin_memory = torch.cuda.is_available()
    if train_samples is not None and val_samples is not None:
        train_ds = IMDBDataset(train_samples, tokenizer, seq_len=seq_len, stride=stride)
        val_ds = IMDBDataset(val_samples, tokenizer, seq_len=seq_len, stride=stride)
    else:
        train_ds = ShakespeareDataset(train_text, tokenizer, seq_len=seq_len, stride=stride)
        val_ds = ShakespeareDataset(val_text, tokenizer, seq_len=seq_len, stride=stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    ) if len(val_ds) > 0 else None
    return train_loader, val_loader
