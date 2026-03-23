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
    "weight_tie": True,
    "mixed_precision": "fp16",
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "seed": 42,
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
