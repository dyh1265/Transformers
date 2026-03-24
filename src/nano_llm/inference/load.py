"""Load model and tokenizer from checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from nano_llm.model import build_model
from nano_llm.tokenizer import (
    BPETokenizer,
    ByteBPETokenizer,
    CharTokenizer,
    HFByteBPETokenizer,
    build_tokenizer_from_text,
    tokenizer_from_state,
)


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    rebuild_tokenizer_from_shakespeare: bool = True,
) -> tuple[torch.nn.Module, CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer, dict]:
    """Load model, tokenizer, and config from a checkpoint.

    If checkpoint has no 'vocab', rebuilds tokenizer from Tiny Shakespeare
    when rebuild_tokenizer_from_shakespeare is True.

    Returns:
        (model, tokenizer, config)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    tokenizer_state = ckpt.get("tokenizer_state")
    vocab = ckpt.get("vocab")
    if tokenizer_state is not None:
        tokenizer = tokenizer_from_state(tokenizer_state)
    elif vocab is None and rebuild_tokenizer_from_shakespeare:
        from nano_llm.data import load_tiny_shakespeare

        train_text, val_text = load_tiny_shakespeare()
        tokenizer = build_tokenizer_from_text(
            train_text + val_text,
            tokenizer_type=str(cfg.get("tokenizer_type", "char")),
            bpe_vocab_size=int(cfg.get("bpe_vocab_size", 256)),
            bpe_word_boundary_aware=bool(cfg.get("bpe_word_boundary_aware", False)),
        )
    elif vocab is not None:
        tokenizer_type = str(cfg.get("tokenizer_type", "char")).lower()
        if tokenizer_type == "bpe":
            tokenizer = BPETokenizer(
                vocab=vocab,
                merges=[],
                word_boundary_aware=bool(cfg.get("bpe_word_boundary_aware", False)),
            )
        elif tokenizer_type == "bpe_byte":
            tokenizer = ByteBPETokenizer(
                vocab=vocab,
                merges=[],
                word_boundary_aware=bool(cfg.get("bpe_word_boundary_aware", False)),
            )
        elif tokenizer_type == "hf_bpe_byte":
            if rebuild_tokenizer_from_shakespeare:
                from nano_llm.data import load_tiny_shakespeare

                train_text, val_text = load_tiny_shakespeare()
                tokenizer = build_tokenizer_from_text(
                    train_text + val_text,
                    tokenizer_type="hf_bpe_byte",
                    bpe_vocab_size=int(cfg.get("bpe_vocab_size", 256)),
                    bpe_word_boundary_aware=bool(cfg.get("bpe_word_boundary_aware", False)),
                )
            else:
                raise ValueError(
                    "Checkpoint missing tokenizer_state for hf_bpe_byte tokenizer. "
                    "Set rebuild_tokenizer_from_shakespeare=True or retrain checkpoint."
                )
        else:
            tokenizer = CharTokenizer(vocab=vocab)
    else:
        raise ValueError(
            "Checkpoint missing 'vocab'. Retrain with updated train.py, "
            "or pass rebuild_tokenizer_from_shakespeare=True and ensure network access."
        )
    state = ckpt["model"]
    max_len = int(cfg.get("seq_len", 128)) + 10
    if "pos_enc.pe" in state:
        max_len = state["pos_enc.pe"].shape[1]
        position_encoding = "sinusoidal"
    elif any("rope.cos_cached" in k for k in state):
        rope_key = next(k for k in state if "rope.cos_cached" in k)
        max_len = state[rope_key].shape[2]
        position_encoding = "rope"
    else:
        position_encoding = str(cfg.get("position_encoding", "sinusoidal")).lower()
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        d_ff=int(cfg["d_ff"]),
        max_len=max_len,
        dropout=0,
        weight_tie=cfg.get("weight_tie", True),
        position_encoding=position_encoding,
    )
    model.load_state_dict(state)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, cfg
