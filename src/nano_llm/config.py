"""Hyperparameters and configuration for nano-llm."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "vocab_size": 65,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 512,
    "seq_len": 128,
    "dropout": 0.1,
    "tokenizer_type": "char",
    "bpe_vocab_size": 256,
    "bpe_word_boundary_aware": False,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "lr_decay": "cosine",
    "lr_min": 1e-6,
    "epochs": 5,
    "early_stopping_patience": 0,
    "dataset_id": "imdb_sentiment",
    "imdb_max_train_samples": None,
    "imdb_max_val_samples": None,
    "imdb_max_review_chars": None,
    # imdb_sentiment: "tags" ([SENTIMENT]...) or "natural" (instruction string before [REVIEW])
    "imdb_conditioning_style": "tags",
    "imdb_positive_instruction": None,
    "imdb_negative_instruction": None,
    "enable_counterfactual_objective": False,
    "counterfactual_ce_weight": 1.0,
    "counterfactual_embedding_weight": 0.25,
    "tarnet_two_heads": False,
    "imdb_tarnet_command_prompt": "GENERATE an IMDB-like review:",
    "tarnet_head_separation_weight": 0.0,
    "tarnet_head_n_fc": 2,
    "tarnet_head_hidden_dim": None,
    "tarnet_head0_n_fc": None,
    "tarnet_head0_hidden_dim": None,
    "tarnet_head1_n_fc": None,
    "tarnet_head1_hidden_dim": None,
    "position_encoding": "sinusoidal",
    # Inter-block attention residuals (mix prior macro-block reps before attn/MLP)
    "block_attn_residuals": False,
    "macro_block_size": 2,
    "max_block_representations": 9,
    "weight_tie": True,
    # fp32 | fp16 (GradScaler) | bf16 (CUDA autocast, no scaler; Ampere+ friendly)
    "mixed_precision": "fp16",
    # CUDA-only performance (ignored on CPU)
    "cuda_allow_tf32": True,
    "cuda_prefer_flash_attn": True,
    "torch_compile": False,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    # Weights & Biases (optional; pip install wandb, wandb login)
    "use_wandb": False,
    "wandb_project": "nano-llm",
    "wandb_run_name": None,
    "wandb_entity": None,
    "wandb_tags": None,
    "wandb_log_model": False,
}

# ~4M params, moderate-width baseline
CONFIG_4M: dict[str, Any] = {
    **DEFAULT_CONFIG,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 5,
    "d_ff": 1024,
    "seq_len": 256,
    "batch_size": 16,
    "dropout": 0.1,
}

# ~50M params, 8GB GPU safe (batch 8, seq 256, fp16)
CONFIG_50M_8GB: dict[str, Any] = {
    **DEFAULT_CONFIG,
    "d_model": 768,
    "num_heads": 8,
    "num_layers": 8,
    "d_ff": 2560,
    "seq_len": 256,
    "batch_size": 8,
    "dropout": 0.1,
}

# ~10M params (approx), moderate-size baseline
CONFIG_10M: dict[str, Any] = {
    **DEFAULT_CONFIG,
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 6,
    "d_ff": 1536,
    "seq_len": 256,
    "batch_size": 8,
    "dropout": 0.1,
}


def _unwrap_hpo_best_config_wrapper(data: dict[str, Any]) -> dict[str, Any]:
    """Use inner training dict from HPO ``best_config.json`` (``{..., "config": {...}}``).

    If that file is merged naïvely, top-level keys omit ``d_model`` / ``dataset_id`` and
    training silently falls back to :data:`DEFAULT_CONFIG`.
    """
    inner = data.get("config")
    if isinstance(inner, dict) and "d_model" in inner and "d_model" not in data:
        return dict(inner)
    return dict(data)


def get_config() -> dict[str, Any]:
    """Return config from env or defaults."""
    config = dict(DEFAULT_CONFIG)
    config_path = os.environ.get("NANO_LLM_CONFIG")
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            config.update(_unwrap_hpo_best_config_wrapper(loaded))
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config from JSON file."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        msg = f"config JSON must be an object, got {type(raw).__name__}"
        raise TypeError(msg)
    body = _unwrap_hpo_best_config_wrapper(raw)
    return {**DEFAULT_CONFIG, **body}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
