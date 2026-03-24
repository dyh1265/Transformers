"""HPO agent loop: propose -> train -> evaluate -> repeat."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nano_llm.agent.llm_client import chat, parse_config_from_response
from nano_llm.agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    EIGHT_GB_SYSTEM_PROMPT,
    build_user_prompt,
)
from nano_llm.config import DEFAULT_CONFIG


def _sanitize_config(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and sanitize config proposed by the LLM."""
    required = [
        "d_model",
        "num_heads",
        "num_layers",
        "d_ff",
        "seq_len",
        "dropout",
        "batch_size",
        "learning_rate",
        "epochs",
    ]
    if any(k not in raw for k in required):
        return None

    try:
        cfg = {
            "d_model": int(raw["d_model"]),
            "num_heads": int(raw["num_heads"]),
            "num_layers": int(raw["num_layers"]),
            "d_ff": int(raw["d_ff"]),
            "seq_len": int(raw["seq_len"]),
            "dropout": float(raw["dropout"]),
            "batch_size": int(raw["batch_size"]),
            "learning_rate": float(raw["learning_rate"]),
            "epochs": int(raw["epochs"]),
        }
    except (TypeError, ValueError):
        return None

    # Enforce practical bounds from prompts/defaults.
    if not (64 <= cfg["d_model"] <= 256):
        return None
    if not (2 <= cfg["num_heads"] <= 8):
        return None
    if cfg["d_model"] % cfg["num_heads"] != 0:
        return None
    if not (2 <= cfg["num_layers"] <= 8):
        return None
    if not (256 <= cfg["d_ff"] <= 1024):
        return None
    if not (64 <= cfg["seq_len"] <= 256):
        return None
    if not (0.0 <= cfg["dropout"] <= 0.2):
        return None
    if not (4 <= cfg["batch_size"] <= 64):
        return None
    if not (1e-4 <= cfg["learning_rate"] <= 1e-3):
        return None
    if not (5 <= cfg["epochs"] <= 30):
        return None
    return cfg


def _config_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    """Stable signature for config de-duplication."""
    keys = (
        "d_model",
        "num_heads",
        "num_layers",
        "d_ff",
        "seq_len",
        "dropout",
        "batch_size",
        "learning_rate",
        "epochs",
    )
    return tuple(config[k] for k in keys)


def run_training(config: dict, workspace: Path | None = None) -> dict | None:
    """Run training script with config. Returns results dict or None on failure."""
    workspace = workspace or Path.cwd()
    results_dir_cfg = Path(str(config.get("hpo_results_dir", "hpo_results")))
    results_dir = results_dir_cfg if results_dir_cfg.is_absolute() else workspace / results_dir_cfg
    config_path = results_dir / "current_trial_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [sys.executable, str(workspace / "scripts" / "train.py"), "--config", str(config_path)]
    try:
        subprocess.run(cmd, cwd=workspace, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[run_training] failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return None

    results_path = config_path.parent / f"trial_{config.get('trial_id', 0)}.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def tune(
    max_trials: int = 10,
    workspace: Path | None = None,
    model: str = "llama3.2",
    use_8gb_bounds: bool = False,
    base_url: str = "http://localhost:11434/v1/",
    max_parse_retries: int = 2,
    tokenizer_type: str | None = None,
    dataset_id: str | None = None,
    bpe_vocab_size: int | None = None,
    bpe_word_boundary_aware: bool | None = None,
    results_dir: str | None = None,
) -> list[dict]:
    """Run HPO agent loop."""
    workspace = workspace or Path.cwd()
    effective_dataset_id = (
        str(dataset_id).lower() if dataset_id is not None else str(DEFAULT_CONFIG.get("dataset_id", "tiny_shakespeare"))
    )
    effective_tokenizer_type = (
        str(tokenizer_type).lower() if tokenizer_type is not None else str(DEFAULT_CONFIG.get("tokenizer_type", "char"))
    )
    if results_dir is not None:
        out_dir_cfg = Path(results_dir)
        out_dir = out_dir_cfg if out_dir_cfg.is_absolute() else workspace / out_dir_cfg
    else:
        out_dir = workspace / "hpo_results" / effective_dataset_id / effective_tokenizer_type
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_history: list[dict] = []
    seen_configs: set[tuple[Any, ...]] = set()
    system_prompt = EIGHT_GB_SYSTEM_PROMPT if use_8gb_bounds else DEFAULT_SYSTEM_PROMPT

    for trial_id in range(max_trials):
        config: dict[str, Any] | None = None
        response = ""
        for attempt in range(max_parse_retries + 1):
            user_prompt = build_user_prompt(trial_history)
            if attempt > 0:
                user_prompt += (
                    "\nPrevious response was invalid. Return ONLY JSON with all required keys."
                )
            response = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                base_url=base_url,
            )
            parsed = parse_config_from_response(response)
            if not parsed:
                continue
            candidate = _sanitize_config(parsed)
            if not candidate:
                continue
            if _config_signature(candidate) in seen_configs:
                continue
            config = candidate
            if config:
                break
        if not config:
            print(f"Trial {trial_id}: invalid LLM config after retries, skipping")
            continue
        config["trial_id"] = trial_id
        config["hpo_results_dir"] = str(out_dir)
        for k, v in DEFAULT_CONFIG.items():
            if k not in config:
                config[k] = v
        if dataset_id is not None:
            config["dataset_id"] = dataset_id
        if tokenizer_type is not None:
            config["tokenizer_type"] = tokenizer_type
        if bpe_vocab_size is not None:
            config["bpe_vocab_size"] = int(bpe_vocab_size)
        if bpe_word_boundary_aware is not None:
            config["bpe_word_boundary_aware"] = bool(bpe_word_boundary_aware)

        with open(out_dir / f"trial_{trial_id}_proposal.json", "w") as f:
            json.dump(
                {
                    "trial_id": trial_id,
                    "response_text": response,
                    "sanitized_config": config,
                },
                f,
                indent=2,
            )

        print(f"Trial {trial_id}: config={config}")
        seen_configs.add(_config_signature(config))
        results = run_training(config, workspace=workspace)
        if results:
            trial_history.append(results)
            print(f"  -> best_val_loss={results.get('best_val_loss')}")
        else:
            print("  -> training failed, skipping")

    if trial_history:
        best = min(trial_history, key=lambda t: float(t.get("best_val_loss", float("inf"))))
        with open(out_dir / "best_config.json", "w") as f:
            json.dump(
                {
                    "best_val_loss": best.get("best_val_loss"),
                    "trial_id": best.get("trial_id"),
                    "config": best.get("config", {}),
                },
                f,
                indent=2,
            )

    return trial_history
