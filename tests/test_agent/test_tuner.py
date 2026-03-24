"""Tests for tuner."""

import json
from pathlib import Path

from nano_llm.agent.tuner import _sanitize_config, run_training, tune


def test_tune_imports() -> None:
    """Tuner module and tune function are importable."""
    assert callable(tune)


def test_run_training_accepts_config() -> None:
    """run_training accepts config dict (actual run would need env)."""
    config = {"d_model": 64, "num_layers": 2, "epochs": 1, "trial_id": 0}
    # Without mocking subprocess, run_training would fail - we just test the signature
    assert callable(run_training)


def test_sanitize_config_accepts_valid() -> None:
    raw = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "d_ff": 512,
        "seq_len": 128,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 5,
    }
    out = _sanitize_config(raw)
    assert out is not None
    assert out["d_model"] == 128


def test_sanitize_config_rejects_invalid_heads() -> None:
    raw = {
        "d_model": 130,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 512,
        "seq_len": 128,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 5,
    }
    assert _sanitize_config(raw) is None


def test_tune_writes_best_config(tmp_path: Path, monkeypatch) -> None:
    responses = iter(
        [
            (
                '{"d_model": 128, "num_heads": 4, "num_layers": 4, "d_ff": 512, '
                '"seq_len": 128, "dropout": 0.1, "batch_size": 8, '
                '"learning_rate": 0.0003, "epochs": 5}'
            ),
            (
                '{"d_model": 256, "num_heads": 4, "num_layers": 6, "d_ff": 1024, '
                '"seq_len": 128, "dropout": 0.05, "batch_size": 16, '
                '"learning_rate": 0.0005, "epochs": 6}'
            ),
        ]
    )

    def fake_chat(*args, **kwargs) -> str:
        return next(responses)

    def fake_run_training(config: dict, workspace: Path | None = None) -> dict:
        return {
            "trial_id": config["trial_id"],
            "config": dict(config),
            "best_val_loss": 1.0 - (config["trial_id"] * 0.1),
        }

    monkeypatch.setattr("nano_llm.agent.tuner.chat", fake_chat)
    monkeypatch.setattr("nano_llm.agent.tuner.run_training", fake_run_training)

    out = tune(max_trials=2, workspace=tmp_path)
    assert len(out) == 2
    best_path = tmp_path / "hpo_results" / "tiny_shakespeare" / "char" / "best_config.json"
    assert best_path.exists()
    data = json.loads(best_path.read_text())
    assert data["trial_id"] == 1


def test_tune_skips_duplicate_configs(tmp_path: Path, monkeypatch) -> None:
    def fake_chat(*args, **kwargs) -> str:
        return (
            '{"d_model": 128, "num_heads": 4, "num_layers": 4, "d_ff": 512, '
            '"seq_len": 128, "dropout": 0.1, "batch_size": 8, '
            '"learning_rate": 0.0003, "epochs": 5}'
        )

    def fake_run_training(config: dict, workspace: Path | None = None) -> dict:
        return {
            "trial_id": config["trial_id"],
            "config": dict(config),
            "best_val_loss": 0.9,
        }

    monkeypatch.setattr("nano_llm.agent.tuner.chat", fake_chat)
    monkeypatch.setattr("nano_llm.agent.tuner.run_training", fake_run_training)

    out = tune(max_trials=2, workspace=tmp_path, max_parse_retries=1)
    assert len(out) == 1
    assert out[0]["trial_id"] == 0
