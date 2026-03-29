"""Dataset loading and preprocessing for nano-llm."""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Any

import torch
from torch.utils.data import Dataset

from nano_llm.tokenizer import BPETokenizer, ByteBPETokenizer, CharTokenizer, HFByteBPETokenizer

SENTIMENT_OPEN = "[SENTIMENT]"
SENTIMENT_CLOSE = "[/SENTIMENT]"
REVIEW_OPEN = "[REVIEW]"
REVIEW_CLOSE = "[/REVIEW]"
POSITIVE_SENTIMENT = "positive"
NEGATIVE_SENTIMENT = "negative"

# Single-head baseline: natural-language instructions before [REVIEW] (must match at inference).
IMDB_DEFAULT_POSITIVE_INSTRUCTION = "Create a POSITIVE IMDB-like review"
IMDB_DEFAULT_NEGATIVE_INSTRUCTION = "Create a NEGATIVE IMDB-like review"


def label_to_sentiment(label: int) -> str:
    """Map IMDB label to sentiment string."""
    return POSITIVE_SENTIMENT if int(label) == 1 else NEGATIVE_SENTIMENT


def sentiment_to_treatment(sentiment: str) -> int:
    """Map sentiment string to treatment indicator T (negative=0, positive=1)."""
    return 1 if str(sentiment).strip().lower() == POSITIVE_SENTIMENT else 0


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


def format_imdb_example(
    text: str,
    label: int,
    max_review_chars: int | None = None,
    *,
    imdb_conditioning_style: str = "tags",
    imdb_positive_instruction: str | None = None,
    imdb_negative_instruction: str | None = None,
) -> list[str]:
    """Format IMDB sample(s) for conditional review generation.

    When max_review_chars is set, long reviews are split into multiple samples,
    each with the same sentiment. The rest of the review becomes additional
    training samples.

    Args:
        text: Raw review text.
        label: 0=negative, 1=positive.
        max_review_chars: If set, split review into chunks of at most this many
            chars (at word boundary). Each chunk is a separate training sample.
        imdb_conditioning_style: ``tags`` (``[SENTIMENT]`` …) or ``natural``
            (e.g. "Create a POSITIVE IMDB-like review" before ``[REVIEW]``).
        imdb_positive_instruction: Overrides default positive instruction (natural only).
        imdb_negative_instruction: Overrides default negative instruction (natural only).

    Returns:
        List of formatted strings (one per chunk, or one if no split).
    """
    cleaned = _normalize_text(_strip_html(text))
    sentiment = label_to_sentiment(label)
    style = str(imdb_conditioning_style).strip().lower()
    if style == "natural":
        pos_i = imdb_positive_instruction or IMDB_DEFAULT_POSITIVE_INSTRUCTION
        neg_i = imdb_negative_instruction or IMDB_DEFAULT_NEGATIVE_INSTRUCTION
        instr = pos_i if int(label) == 1 else neg_i
        prefix = f"<bos>{instr} {REVIEW_OPEN} "
    elif style == "tags":
        prefix = f"<bos>{SENTIMENT_OPEN} {sentiment} {SENTIMENT_CLOSE} {REVIEW_OPEN} "
    else:
        raise ValueError(
            f"imdb_conditioning_style must be 'tags' or 'natural', got {imdb_conditioning_style!r}"
        )
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


def _extract_imdb_sentiment_and_review(
    sample: str,
    *,
    imdb_conditioning_style: str = "tags",
    imdb_positive_instruction: str | None = None,
    imdb_negative_instruction: str | None = None,
) -> tuple[str, str]:
    """Extract sentiment label and review body from a formatted IMDB sample."""
    review_pattern = re.compile(r"\[REVIEW\]\s*(.*?)\s*\[/REVIEW\]", re.DOTALL)
    review_match = review_pattern.search(sample)
    if review_match is None:
        raise ValueError("IMDB sample missing [REVIEW] ... [/REVIEW] segment")
    review = review_match.group(1).strip()

    style = str(imdb_conditioning_style).strip().lower()
    if style == "natural":
        pos_i = imdb_positive_instruction or IMDB_DEFAULT_POSITIVE_INSTRUCTION
        neg_i = imdb_negative_instruction or IMDB_DEFAULT_NEGATIVE_INSTRUCTION
        before, _sep, _rest = sample.partition(REVIEW_OPEN)
        if not before.startswith("<bos>"):
            raise ValueError("IMDB natural-format sample missing <bos> before instruction")
        head = before[len("<bos>") :].strip()
        if head == pos_i:
            return POSITIVE_SENTIMENT, review
        if head == neg_i:
            return NEGATIVE_SENTIMENT, review
        raise ValueError(
            f"IMDB natural-format prefix {head!r} does not match positive or negative instruction"
        )

    sent_pattern = re.compile(r"\[SENTIMENT\]\s*(positive|negative)\s*\[/SENTIMENT\]", re.IGNORECASE)
    sent_match = sent_pattern.search(sample)
    if sent_match is None:
        raise ValueError("IMDB sample missing [SENTIMENT] ... [/SENTIMENT] segment")
    sentiment = sent_match.group(1).strip().lower()
    return sentiment, review


def _format_conditioned_imdb_sample(
    review: str,
    sentiment: str,
    *,
    imdb_conditioning_style: str = "tags",
    imdb_positive_instruction: str | None = None,
    imdb_negative_instruction: str | None = None,
) -> str:
    """Build normalized conditioned IMDB sample string from review and sentiment."""
    s = str(sentiment).strip().lower()
    if s not in (POSITIVE_SENTIMENT, NEGATIVE_SENTIMENT):
        raise ValueError(f"Unsupported sentiment: {sentiment}")
    style = str(imdb_conditioning_style).strip().lower()
    if style == "natural":
        pos_i = imdb_positive_instruction or IMDB_DEFAULT_POSITIVE_INSTRUCTION
        neg_i = imdb_negative_instruction or IMDB_DEFAULT_NEGATIVE_INSTRUCTION
        instr = pos_i if s == POSITIVE_SENTIMENT else neg_i
        return f"<bos>{instr} {REVIEW_OPEN} {review} {REVIEW_CLOSE}<eos>"
    if style != "tags":
        raise ValueError(f"imdb_conditioning_style must be 'tags' or 'natural', got {imdb_conditioning_style!r}")
    return f"<bos>{SENTIMENT_OPEN} {s} {SENTIMENT_CLOSE} {REVIEW_OPEN} {review} {REVIEW_CLOSE}<eos>"


def _imdb_stratified_subsample(split: Any, max_samples: int, seed: int) -> Any:
    """Shrink an IMDB HF split while keeping label proportions ~50/50 (same as full corpus).

    Uses index lists (label 0 vs 1), shuffles within class with ``seed``, then takes
    ``n//2`` / ``n - n//2`` when possible. If one class is short, fills remaining slots
    from the other.
    """
    n = min(int(max_samples), len(split))
    if n >= len(split):
        return split
    labels = split["label"]
    neg_ix = [i for i, y in enumerate(labels) if int(y) == 0]
    pos_ix = [i for i, y in enumerate(labels) if int(y) == 1]
    rng = random.Random(int(seed))
    rng.shuffle(neg_ix)
    rng.shuffle(pos_ix)
    n_neg = n // 2
    n_pos = n - n_neg
    take_neg = min(n_neg, len(neg_ix))
    take_pos = min(n_pos, len(pos_ix))
    need = n - take_neg - take_pos
    if need > 0:
        extra = min(need, len(neg_ix) - take_neg)
        take_neg += extra
        need -= extra
    if need > 0:
        extra = min(need, len(pos_ix) - take_pos)
        take_pos += extra
        need -= extra
    chosen = neg_ix[:take_neg] + pos_ix[:take_pos]
    rng.shuffle(chosen)
    return split.select(chosen)


def load_imdb_sentiment(
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_review_chars: int | None = None,
    subset_seed: int = 42,
    imdb_conditioning_style: str = "tags",
    imdb_positive_instruction: str | None = None,
    imdb_negative_instruction: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load and format IMDB from Hugging Face for next-token training.

    Args:
        max_train_samples: Cap **rows** from the train split; sampling is **stratified**
            by label so positives/negatives stay ~half/half (when both classes have enough rows).
        max_val_samples: Same for the test split (used as validation here).
        max_review_chars: Split each review into chunks of at most this many chars.
            Each chunk becomes a separate training sample with the same sentiment.
        subset_seed: RNG seed for shuffling before stratified row selection.
        imdb_conditioning_style: ``tags`` or ``natural`` (see :func:`format_imdb_example`).
        imdb_positive_instruction: Optional override for natural conditioning.
        imdb_negative_instruction: Optional override for natural conditioning.

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
        train_split = _imdb_stratified_subsample(train_split, int(max_train_samples), subset_seed)
    if max_val_samples is not None:
        val_split = _imdb_stratified_subsample(
            val_split, int(max_val_samples), int(subset_seed) + 1
        )

    def fmt(x: dict) -> list[str]:
        return format_imdb_example(
            x["text"],
            x["label"],
            max_review_chars=max_review_chars,
            imdb_conditioning_style=imdb_conditioning_style,
            imdb_positive_instruction=imdb_positive_instruction,
            imdb_negative_instruction=imdb_negative_instruction,
        )

    train_samples = [s for x in train_split for s in fmt(x)]
    val_samples = [s for x in val_split for s in fmt(x)]
    return train_samples, val_samples


def _imdb_prefix_end(sample: str) -> int:
    """Return character index where the review body starts (after '[REVIEW] ')."""
    marker = "[REVIEW] "
    idx = sample.find(marker)
    if idx < 0:
        return 0
    return idx + len(marker)


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
        *,
        imdb_conditioning_style: str = "tags",
        imdb_positive_instruction: str | None = None,
        imdb_negative_instruction: str | None = None,
    ) -> None:
        if stride is None:
            stride = seq_len
        self.seq_len = seq_len
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self._imdb_style = str(imdb_conditioning_style).strip().lower()
        self._imdb_pos = imdb_positive_instruction
        self._imdb_neg = imdb_negative_instruction
        self.chunks: list[
            tuple[list[int], list[int], int, list[int], list[int], list[int], list[int], list[int]]
        ] = []
        for sample in samples:
            factual_sentiment, review = _extract_imdb_sentiment_and_review(
                sample,
                imdb_conditioning_style=self._imdb_style,
                imdb_positive_instruction=self._imdb_pos,
                imdb_negative_instruction=self._imdb_neg,
            )
            treatment = sentiment_to_treatment(factual_sentiment)
            kw = dict(
                imdb_conditioning_style=self._imdb_style,
                imdb_positive_instruction=self._imdb_pos,
                imdb_negative_instruction=self._imdb_neg,
            )
            factual_sample = _format_conditioned_imdb_sample(review, factual_sentiment, **kw)
            positive_sample = _format_conditioned_imdb_sample(review, POSITIVE_SENTIMENT, **kw)
            negative_sample = _format_conditioned_imdb_sample(review, NEGATIVE_SENTIMENT, **kw)

            factual_chunks = self._chunk_sample_with_review_mask(
                factual_sample, tokenizer, seq_len=seq_len, stride=stride
            )
            positive_chunks = self._chunk_sample_with_review_mask(
                positive_sample, tokenizer, seq_len=seq_len, stride=stride
            )
            negative_chunks = self._chunk_sample_with_review_mask(
                negative_sample, tokenizer, seq_len=seq_len, stride=stride
            )
            n_chunks = min(len(factual_chunks), len(positive_chunks), len(negative_chunks))
            for i in range(n_chunks):
                fx, fy, fmask = factual_chunks[i]
                px, _, pmask = positive_chunks[i]
                nx, _, nmask = negative_chunks[i]
                self.chunks.append((fx, fy, treatment, fmask, px, nx, pmask, nmask))

    @staticmethod
    def _chunk_sample_with_review_mask(
        sample: str,
        tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
        *,
        seq_len: int,
        stride: int,
    ) -> list[tuple[list[int], list[int], list[int]]]:
        """Chunk one IMDB sample and mark review-token positions in each input chunk."""
        ids = tokenizer.encode(sample)
        prefix_end = _imdb_prefix_end(sample)
        prefix_ids = tokenizer.encode(sample[:prefix_end])
        body_start = len(prefix_ids)
        out: list[tuple[list[int], list[int], list[int]]] = []

        if len(ids) < seq_len:
            x = ids[:-1]
            y = ids[1:]
            review_mask = [1 if i >= body_start else 0 for i in range(len(x))]
            out.append((x, y, review_mask))
            return out

        body_len = len(ids) - body_start
        chunk_body_len = seq_len - body_start
        if chunk_body_len <= 0 or body_len <= 0:
            # Prefix too long or no body: fall back to sliding window
            for i in range(0, len(ids) - seq_len, stride):
                chunk_ids = ids[i : i + seq_len]
                x = chunk_ids[:-1]
                y = chunk_ids[1:]
                review_mask = [1 if i >= body_start else 0 for i in range(len(x))]
                out.append((x, y, review_mask))
            return out

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
            x = chunk_ids[:-1]
            y = chunk_ids[1:]
            review_mask = [1 if i >= body_start else 0 for i in range(len(x))]
            out.append((x, y, review_mask))
            k += 1
            if body_start + k * body_stride >= len(ids):
                break
        return out

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, treatment, review_mask, x_pos, x_neg, review_mask_pos, review_mask_neg = self.chunks[idx]
        # Ensure every returned sequence is exactly `seq_len - 1` tokens.
        # The factual/positive/negative branches can tokenize to slightly different
        # prefix lengths (e.g. "positive" vs "negative"), so we must pad/truncate
        # each field independently to keep DataLoader batching stable.
        target_len = self.seq_len - 1

        def _pad_trunc_int_list(values: list[int], pad_value: int) -> list[int]:
            if len(values) >= target_len:
                return values[:target_len]
            return values + [pad_value] * (target_len - len(values))

        def _pad_trunc_bool_list(values: list[int] | list[bool], pad_value: int) -> list[int]:
            if len(values) >= target_len:
                return list(values[:target_len])
            return list(values) + [pad_value] * (target_len - len(values))

        x = _pad_trunc_int_list(x, self.pad_id)
        y = _pad_trunc_int_list(y, PAD_TARGET_IGNORE_INDEX)
        review_mask = _pad_trunc_bool_list(review_mask, 0)
        x_pos = _pad_trunc_int_list(x_pos, self.pad_id)
        x_neg = _pad_trunc_int_list(x_neg, self.pad_id)
        review_mask_pos = _pad_trunc_bool_list(review_mask_pos, 0)
        review_mask_neg = _pad_trunc_bool_list(review_mask_neg, 0)
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(treatment, dtype=torch.long),
            torch.tensor(review_mask, dtype=torch.bool),
            torch.tensor(x_pos, dtype=torch.long),
            torch.tensor(x_neg, dtype=torch.long),
            torch.tensor(review_mask_pos, dtype=torch.bool),
            torch.tensor(review_mask_neg, dtype=torch.bool),
        )


class IMDBTARNetDataset(Dataset):
    """IMDB dataset for TARNet-style two-head training.

    Returns one tokenized sequence per chunk (not 3 branches). Treatment T
    selects which head (Y0 vs Y1) is trained.
    """

    def __init__(
        self,
        samples: list[str],
        tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
        seq_len: int = 128,
        stride: int | None = None,
        *,
        command_prompt: str = "GENERATE an IMDB-like review:",
        imdb_conditioning_style: str = "tags",
        imdb_positive_instruction: str | None = None,
        imdb_negative_instruction: str | None = None,
    ) -> None:
        if stride is None:
            stride = seq_len
        self.seq_len = seq_len
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self.command_prompt = str(command_prompt)
        self._imdb_style = str(imdb_conditioning_style).strip().lower()
        self._imdb_pos = imdb_positive_instruction
        self._imdb_neg = imdb_negative_instruction
        self.chunks: list[tuple[list[int], list[int], int, list[int]]] = []

        for sample in samples:
            factual_sentiment, review = _extract_imdb_sentiment_and_review(
                sample,
                imdb_conditioning_style=self._imdb_style,
                imdb_positive_instruction=self._imdb_pos,
                imdb_negative_instruction=self._imdb_neg,
            )
            treatment = sentiment_to_treatment(factual_sentiment)
            # Treatment-invariant prefix; heads represent potential outcomes.
            prompt = f"<bos>{self.command_prompt} {REVIEW_OPEN} "
            tar_sample = f"{prompt}{review} {REVIEW_CLOSE}<eos>"
            for x, y, mask in IMDBDataset._chunk_sample_with_review_mask(
                tar_sample, tokenizer, seq_len=seq_len, stride=stride
            ):
                self.chunks.append((x, y, treatment, mask))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, treatment, review_mask = self.chunks[idx]
        pad_len = (self.seq_len - 1) - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [PAD_TARGET_IGNORE_INDEX] * pad_len
            review_mask = review_mask + [0] * pad_len
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(treatment, dtype=torch.long),
            torch.tensor(review_mask, dtype=torch.bool),
        )


def create_dataloaders(
    train_samples: list[str],
    val_samples: list[str],
    tokenizer: CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer,
    seq_len: int = 128,
    batch_size: int = 32,
    stride: int | None = None,
    *,
    imdb_tarnet_two_heads: bool = False,
    imdb_tarnet_command_prompt: str = "GENERATE an IMDB-like review:",
    imdb_conditioning_style: str = "tags",
    imdb_positive_instruction: str | None = None,
    imdb_negative_instruction: str | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Create train and validation DataLoaders from formatted IMDB samples."""
    from torch.utils.data import DataLoader

    pin_memory = torch.cuda.is_available()
    if imdb_tarnet_two_heads:
        train_ds = IMDBTARNetDataset(
            train_samples,
            tokenizer,
            seq_len=seq_len,
            stride=stride,
            command_prompt=imdb_tarnet_command_prompt,
            imdb_conditioning_style=imdb_conditioning_style,
            imdb_positive_instruction=imdb_positive_instruction,
            imdb_negative_instruction=imdb_negative_instruction,
        )
        val_ds = IMDBTARNetDataset(
            val_samples,
            tokenizer,
            seq_len=seq_len,
            stride=stride,
            command_prompt=imdb_tarnet_command_prompt,
            imdb_conditioning_style=imdb_conditioning_style,
            imdb_positive_instruction=imdb_positive_instruction,
            imdb_negative_instruction=imdb_negative_instruction,
        )
    else:
        train_ds = IMDBDataset(
            train_samples,
            tokenizer,
            seq_len=seq_len,
            stride=stride,
            imdb_conditioning_style=imdb_conditioning_style,
            imdb_positive_instruction=imdb_positive_instruction,
            imdb_negative_instruction=imdb_negative_instruction,
        )
        val_ds = IMDBDataset(
            val_samples,
            tokenizer,
            seq_len=seq_len,
            stride=stride,
            imdb_conditioning_style=imdb_conditioning_style,
            imdb_positive_instruction=imdb_positive_instruction,
            imdb_negative_instruction=imdb_negative_instruction,
        )
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
