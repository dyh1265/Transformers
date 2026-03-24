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
    "dataset_id": "tiny_shakespeare",
    "wikitext_max_train_samples": None,
    "wikitext_max_val_samples": None,
    "imdb_max_train_samples": None,
    "imdb_max_val_samples": None,
    "imdb_max_review_chars": None,
    "pg19_max_train_books": None,
    "pg19_max_val_books": None,
    "pg19_max_chars_per_book": None,
    "position_encoding": "sinusoidal",
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

# ~4M params, good for small-model + large-text experiments (books, etc.)
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


def get_config() -> dict[str, Any]:
    """Return config from env or defaults."""
    config = dict(DEFAULT_CONFIG)
    config_path = os.environ.get("NANO_LLM_CONFIG")
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config.update(json.load(f))
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config from JSON file."""
    with open(path) as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
