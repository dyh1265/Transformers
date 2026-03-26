"""Tests for config loading and HPO best_config.json unwrapping."""

from __future__ import annotations

import json
from pathlib import Path

from nano_llm.config import DEFAULT_CONFIG, load_config


def test_load_config_flat_merges_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"d_model": 256, "dataset_id": "imdb_sentiment"}))
    cfg = load_config(path)
    assert cfg["d_model"] == 256
    assert cfg["dataset_id"] == "imdb_sentiment"
    assert cfg["num_layers"] == DEFAULT_CONFIG["num_layers"]


def test_load_config_unwraps_hpo_best_config_json(tmp_path: Path) -> None:
    path = tmp_path / "best_config.json"
    inner = {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 7,
        "d_ff": 1024,
        "seq_len": 64,
        "dataset_id": "imdb_sentiment",
        "tokenizer_type": "hf_bpe_byte",
        "tarnet_two_heads": True,
    }
    path.write_text(
        json.dumps(
            {"best_val_loss": 1.5, "trial_id": 79, "config": inner},
        )
    )
    cfg = load_config(path)
    assert cfg["d_model"] == 512
    assert cfg["dataset_id"] == "imdb_sentiment"
    assert cfg["tarnet_two_heads"] is True
    assert cfg["num_layers"] == 7


def test_load_config_flat_with_nested_config_key_unchanged(tmp_path: Path) -> None:
    """If top-level already supplies ``d_model``, do not replace body with inner ``config``."""
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "d_model": 128,
                "config": {"d_model": 999, "dataset_id": "should_not_win"},
            }
        )
    )
    cfg = load_config(path)
    assert cfg["d_model"] == 128
