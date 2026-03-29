"""Byte-level BPE tokenizer via Hugging Face `tokenizers` (HFByteBPETokenizer only)."""

from __future__ import annotations

from typing import Any


class HFByteBPETokenizer:
    """Byte-level BPE tokenizer powered by Hugging Face tokenizers."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer: Any, tokenizer_json: str) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_json = tokenizer_json
        self._refresh_vocab_cache()

    def _refresh_vocab_cache(self) -> None:
        self.vocab_size = int(self._tokenizer.get_vocab_size())
        self.vocab = [self._tokenizer.id_to_token(i) for i in range(self.vocab_size)]
        self.id_to_char = {i: t for i, t in enumerate(self.vocab)}
        pid = self._tokenizer.token_to_id(self.PAD_TOKEN)
        self.pad_id = int(pid) if pid is not None else 0

    @classmethod
    def _require_tokenizers(cls) -> tuple[Any, Any, Any, Any, Any]:
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
        except ImportError as e:
            raise ImportError(
                "HFByteBPETokenizer requires the 'tokenizers' package. "
                "Install it with: pip install tokenizers"
            ) from e
        return Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 8000,
        word_boundary_aware: bool = False,
    ) -> HFByteBPETokenizer:
        Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder = cls._require_tokenizers()
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))
        tokenizer.pre_tokenizer = (
            ByteLevel(add_prefix_space=False, use_regex=False)
            if word_boundary_aware
            else ByteLevel(add_prefix_space=False)
        )
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=2,
            special_tokens=[cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN],
        )
        if not text:
            text = " "
        chunk_size = 1_000_000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        tokenizer.train_from_iterator(iter(chunks), trainer=trainer)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer_json = tokenizer.to_str()
        return cls(tokenizer=tokenizer, tokenizer_json=tokenizer_json)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> HFByteBPETokenizer:
        Tokenizer, _, _, _, _ = cls._require_tokenizers()
        tokenizer_json = str(state["tokenizer_json"])
        tokenizer = Tokenizer.from_str(tokenizer_json)
        return cls(tokenizer=tokenizer, tokenizer_json=tokenizer_json)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return self._tokenizer.decode(list(ids))

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)

    def to_state(self) -> dict[str, Any]:
        return {"type": "hf_bpe_byte", "tokenizer_json": self._tokenizer_json}


def build_tokenizer_from_text(
    text: str,
    *,
    bpe_vocab_size: int = 8000,
    bpe_word_boundary_aware: bool = False,
) -> HFByteBPETokenizer:
    """Train HF byte-level BPE on ``text``."""
    return HFByteBPETokenizer.from_text(
        text,
        vocab_size=bpe_vocab_size,
        word_boundary_aware=bpe_word_boundary_aware,
    )


def tokenizer_from_state(state: dict[str, Any]) -> HFByteBPETokenizer:
    """Restore tokenizer from checkpoint ``tokenizer_state`` dict."""
    tokenizer_type = str(state.get("type", "hf_bpe_byte")).lower()
    if tokenizer_type != "hf_bpe_byte":
        raise ValueError(
            f"Only hf_bpe_byte tokenizers are supported; got type {tokenizer_type!r}. "
            "Retrain with the current codebase or convert the checkpoint."
        )
    return HFByteBPETokenizer.from_state(state)
