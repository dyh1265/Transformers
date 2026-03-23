"""Tokenizers for nano-llm: character-level and simple BPE."""

from __future__ import annotations

from collections import Counter
from typing import Any


class CharTokenizer:
    """Character-level tokenizer with fixed vocabulary."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab: list[str] | None = None) -> None:
        if vocab is None:
            vocab = self._default_vocab()
        self.vocab = list(vocab)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.char_to_id.get(self.PAD_TOKEN, 0)
        self.unk_id = self.char_to_id.get(self.UNK_TOKEN, 1 if len(self.vocab) > 1 else 0)

    @classmethod
    def from_text(cls, text: str, add_special: bool = True) -> "CharTokenizer":
        """Build vocab from text. Use for Tiny Shakespeare etc."""
        chars = sorted(set(text))
        if add_special:
            special = [c for c in (cls.PAD_TOKEN, cls.UNK_TOKEN) if c not in chars]
            chars = special + [c for c in chars if c not in special]
        return cls(vocab=chars)

    @staticmethod
    def _default_vocab() -> list[str]:
        """Default vocab: pad, unk, then printable ASCII."""
        chars = [CharTokenizer.PAD_TOKEN, CharTokenizer.UNK_TOKEN]
        chars.extend(chr(i) for i in range(32, 127))
        return chars

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Encode string to list of token ids."""
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        """Decode list of token ids to string. Skips pad only if pad is in vocab."""
        if self.PAD_TOKEN not in self.char_to_id:
            return "".join(self.id_to_char.get(i, self.UNK_TOKEN) for i in ids)
        return "".join(
            self.id_to_char.get(i, self.UNK_TOKEN)
            for i in ids
            if i != self.pad_id
        )

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)

    def to_state(self) -> dict[str, Any]:
        """Serialize tokenizer for checkpointing."""
        return {"type": "char", "vocab": self.vocab}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CharTokenizer":
        """Restore tokenizer from serialized state."""
        return cls(vocab=list(state["vocab"]))


class BPETokenizer:
    """Simple character-seeded BPE tokenizer with learned merges."""

    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        word_boundary_aware: bool = False,
    ) -> None:
        self.vocab = list(vocab)
        self.merges = list(merges)
        self.word_boundary_aware = bool(word_boundary_aware)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.unk_id = self.token_to_id.get(self.UNK_TOKEN, 0)
        # Compatibility alias used in generation code.
        self.id_to_char = self.id_to_token

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        out: list[str] = []
        i = 0
        a, b = pair
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    @staticmethod
    def _contains_whitespace(token: str) -> bool:
        return any(ch.isspace() for ch in token)

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        word_boundary_aware: bool = False,
    ) -> "BPETokenizer":
        """Learn merges from text until reaching vocab_size or no useful merges remain."""
        if not text:
            return cls(
                vocab=[cls.UNK_TOKEN],
                merges=[],
                word_boundary_aware=word_boundary_aware,
            )
        tokens = list(text)
        vocab = {cls.UNK_TOKEN, *tokens}
        merges: list[tuple[str, str]] = []

        while len(vocab) < vocab_size and len(tokens) > 1:
            if word_boundary_aware:
                pairs = Counter(
                    (tokens[i], tokens[i + 1])
                    for i in range(len(tokens) - 1)
                    if not cls._contains_whitespace(tokens[i])
                    and not cls._contains_whitespace(tokens[i + 1])
                )
            else:
                pairs = Counter((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1))
            if not pairs:
                break
            pair, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            tokens = cls._merge_pair(tokens, pair)
            merges.append(pair)
            vocab.add(pair[0] + pair[1])

        ordered_vocab = [cls.UNK_TOKEN] + sorted(t for t in vocab if t != cls.UNK_TOKEN)
        return cls(
            vocab=ordered_vocab,
            merges=merges,
            word_boundary_aware=word_boundary_aware,
        )

    def _tokenize(self, text: str) -> list[str]:
        tokens = list(text)
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return "".join(self.id_to_token.get(i, self.UNK_TOKEN) for i in ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)

    def to_state(self) -> dict[str, Any]:
        """Serialize tokenizer for checkpointing."""
        return {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "word_boundary_aware": self.word_boundary_aware,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BPETokenizer":
        """Restore tokenizer from serialized state."""
        merges = [(a, b) for a, b in state.get("merges", [])]
        return cls(
            vocab=list(state["vocab"]),
            merges=merges,
            word_boundary_aware=bool(state.get("word_boundary_aware", False)),
        )


def build_tokenizer_from_text(
    text: str,
    tokenizer_type: str = "char",
    bpe_vocab_size: int = 256,
    bpe_word_boundary_aware: bool = False,
) -> CharTokenizer | BPETokenizer | "ByteBPETokenizer" | "HFByteBPETokenizer":
    """Factory to build tokenizer from training text."""
    t = tokenizer_type.lower()
    if t == "char":
        return CharTokenizer.from_text(text, add_special=False)
    if t == "bpe":
        return BPETokenizer.from_text(
            text,
            vocab_size=bpe_vocab_size,
            word_boundary_aware=bpe_word_boundary_aware,
        )
    if t == "bpe_byte":
        return ByteBPETokenizer.from_text(
            text,
            vocab_size=bpe_vocab_size,
            word_boundary_aware=bpe_word_boundary_aware,
        )
    if t == "hf_bpe_byte":
        return HFByteBPETokenizer.from_text(
            text,
            vocab_size=bpe_vocab_size,
            word_boundary_aware=bpe_word_boundary_aware,
        )
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


class ByteBPETokenizer(BPETokenizer):
    """Byte-level BPE tokenizer using UTF-8 bytes with latin1 bridge."""

    @staticmethod
    def _to_byte_text(text: str) -> str:
        return text.encode("utf-8").decode("latin1")

    @staticmethod
    def _from_byte_text(byte_text: str) -> str:
        return byte_text.encode("latin1").decode("utf-8", errors="replace")

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        word_boundary_aware: bool = False,
    ) -> "ByteBPETokenizer":
        byte_text = cls._to_byte_text(text)
        base = super().from_text(
            byte_text,
            vocab_size=vocab_size,
            word_boundary_aware=word_boundary_aware,
        )
        return cls(
            vocab=base.vocab,
            merges=base.merges,
            word_boundary_aware=base.word_boundary_aware,
        )

    def encode(self, text: str) -> list[int]:
        return super().encode(self._to_byte_text(text))

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        byte_text = super().decode(ids)
        return self._from_byte_text(byte_text)

    def to_state(self) -> dict[str, Any]:
        state = super().to_state()
        state["type"] = "bpe_byte"
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ByteBPETokenizer":
        merges = [(a, b) for a, b in state.get("merges", [])]
        return cls(
            vocab=list(state["vocab"]),
            merges=merges,
            word_boundary_aware=bool(state.get("word_boundary_aware", False)),
        )


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
    ) -> "HFByteBPETokenizer":
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
        tokenizer.train_from_iterator([text], trainer=trainer)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer_json = tokenizer.to_str()
        return cls(tokenizer=tokenizer, tokenizer_json=tokenizer_json)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "HFByteBPETokenizer":
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


def tokenizer_from_state(
    state: dict[str, Any],
) -> CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer:
    """Restore tokenizer from checkpoint state."""
    tokenizer_type = str(state.get("type", "char")).lower()
    if tokenizer_type == "char":
        return CharTokenizer.from_state(state)
    if tokenizer_type == "bpe":
        return BPETokenizer.from_state(state)
    if tokenizer_type == "bpe_byte":
        return ByteBPETokenizer.from_state(state)
    if tokenizer_type == "hf_bpe_byte":
        return HFByteBPETokenizer.from_state(state)
    raise ValueError(f"Unsupported tokenizer state type: {tokenizer_type}")
