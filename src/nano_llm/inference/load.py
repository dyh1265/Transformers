"""Load model and tokenizer from checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from nano_llm.model import build_model
from nano_llm.tokenizer import HFByteBPETokenizer, build_tokenizer_from_text, tokenizer_from_state

logger = logging.getLogger(__name__)


def normalize_checkpoint_state_dict(state: dict) -> dict:
    """Strip DDP ``module.`` and torch.compile ``_orig_mod.`` prefixes from checkpoint keys."""
    if not state:
        return state
    out: dict = {}
    for k, v in state.items():
        key = str(k)
        while True:
            if key.startswith("module."):
                key = key[7:]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
            else:
                break
        out[key] = v
    return out


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    rebuild_tokenizer_from_corpus: bool = True,
) -> tuple[torch.nn.Module, HFByteBPETokenizer, dict]:
    """Load model, tokenizer, and config from a checkpoint.

    Expects ``tokenizer_state`` (hf_bpe_byte JSON) or, if missing,
    ``rebuild_tokenizer_from_corpus=True`` to train a new tokenizer on a small IMDB slice.

    Returns:
        (model, tokenizer, config)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    tokenizer_state = ckpt.get("tokenizer_state")
    vocab = ckpt.get("vocab")
    if tokenizer_state is not None:
        tokenizer = tokenizer_from_state(tokenizer_state)
    elif rebuild_tokenizer_from_corpus:
        from nano_llm.data import load_imdb_sentiment

        tr, va = load_imdb_sentiment(max_train_samples=2000, max_val_samples=500)
        corpus = "\n".join(tr + va)
        tokenizer = build_tokenizer_from_text(
            corpus,
            bpe_vocab_size=int(cfg.get("bpe_vocab_size", 8000)),
            bpe_word_boundary_aware=bool(cfg.get("bpe_word_boundary_aware", False)),
        )
    elif vocab is not None:
        raise ValueError(
            "Checkpoint has legacy `vocab` list without `tokenizer_state`. "
            "Only hf_bpe_byte is supported; retrain or pass rebuild_tokenizer_from_corpus=True."
        )
    else:
        raise ValueError(
            "Checkpoint missing tokenizer_state. Retrain with current train.py, "
            "or pass rebuild_tokenizer_from_corpus=True (needs Hugging Face ``datasets`` for IMDB)."
        )
    state = normalize_checkpoint_state_dict(ckpt["model"])
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
    # Trunk: inter-block only if main ``blocks.*`` have residual mixers (not TARNet stacks).
    block_attn_residuals = bool(cfg.get("block_attn_residuals", False))
    if not block_attn_residuals:
        block_attn_residuals = any(k.startswith("blocks.") and "attn_res_proj" in k for k in state)

    if any(k.startswith("tarnet_sentiment_blocks0.") for k in state):
        raise ValueError(
            "Checkpoint contains removed TARNet sentiment inter-block weights "
            "(tarnet_sentiment_blocks0/1). Retrain with the current model (FC Δ heads only)."
        )

    model = build_model(
        vocab_size=tokenizer.vocab_size,
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        d_ff=int(cfg["d_ff"]),
        max_len=max_len,
        dropout=float(cfg.get("dropout", 0.0)),
        weight_tie=cfg.get("weight_tie", True),
        tarnet_two_heads=bool(cfg.get("tarnet_two_heads", False)),
        tarnet_head_n_fc=int(cfg.get("tarnet_head_n_fc", 2)),
        tarnet_head_hidden_dim=cfg.get("tarnet_head_hidden_dim"),
        tarnet_head0_n_fc=cfg.get("tarnet_head0_n_fc"),
        tarnet_head0_hidden_dim=cfg.get("tarnet_head0_hidden_dim"),
        tarnet_head1_n_fc=cfg.get("tarnet_head1_n_fc"),
        tarnet_head1_hidden_dim=cfg.get("tarnet_head1_hidden_dim"),
        position_encoding=position_encoding,
        block_attn_residuals=block_attn_residuals,
        macro_block_size=int(cfg.get("macro_block_size", 2)),
        max_block_representations=int(cfg.get("max_block_representations", 9)),
    )
    model.load_state_dict(state)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, cfg
