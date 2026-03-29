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
    IMDB_TARNET_SYSTEM_PROMPT,
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

    # Optional TARNet knobs (fill stable defaults to keep signatures comparable)
    cfg["tarnet_head_n_fc"] = 2
    cfg["tarnet_head_hidden_dim"] = None
    cfg["tarnet_head_separation_weight"] = 0.0
    cfg["tarnet_head0_n_fc"] = None
    cfg["tarnet_head0_hidden_dim"] = None
    cfg["tarnet_head1_n_fc"] = None
    cfg["tarnet_head1_hidden_dim"] = None
    if "tarnet_head_n_fc" in raw and raw["tarnet_head_n_fc"] is not None:
        try:
            cfg["tarnet_head_n_fc"] = int(raw["tarnet_head_n_fc"])
        except (TypeError, ValueError):
            return None
        if not (1 <= cfg["tarnet_head_n_fc"] <= 4):
            return None
    if "tarnet_head_hidden_dim" in raw and raw["tarnet_head_hidden_dim"] is not None:
        try:
            cfg["tarnet_head_hidden_dim"] = int(raw["tarnet_head_hidden_dim"])
        except (TypeError, ValueError):
            return None
        if not (128 <= cfg["tarnet_head_hidden_dim"] <= 1024):
            return None
    if "tarnet_head_separation_weight" in raw and raw["tarnet_head_separation_weight"] is not None:
        try:
            cfg["tarnet_head_separation_weight"] = float(raw["tarnet_head_separation_weight"])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= cfg["tarnet_head_separation_weight"] <= 0.2):
            return None

    for side in ("0", "1"):
        k_n = f"tarnet_head{side}_n_fc"
        k_h = f"tarnet_head{side}_hidden_dim"
        if k_n in raw and raw[k_n] is not None:
            try:
                cfg[k_n] = int(raw[k_n])
            except (TypeError, ValueError):
                return None
            if not (1 <= cfg[k_n] <= 4):
                return None
        if k_h in raw and raw[k_h] is not None:
            try:
                cfg[k_h] = int(raw[k_h])
            except (TypeError, ValueError):
                return None
            if not (128 <= cfg[k_h] <= 1024):
                return None

    # Enforce practical bounds from prompts/defaults.
    if not (64 <= cfg["d_model"] <= 512):
        return None
    if not (2 <= cfg["num_heads"] <= 8):
        return None
    if cfg["d_model"] % cfg["num_heads"] != 0:
        return None
    if not (2 <= cfg["num_layers"] <= 10):
        return None
    if not (256 <= cfg["d_ff"] <= 2048):
        return None
    if not (64 <= cfg["seq_len"] <= 256):
        return None
    if not (0.0 <= cfg["dropout"] <= 0.2):
        return None
    if not (2 <= cfg["batch_size"] <= 64):
        return None
    if not (1e-4 <= cfg["learning_rate"] <= 1e-3):
        return None
    if not (1 <= cfg["epochs"] <= 30):
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
        "tarnet_head_n_fc",
        "tarnet_head_hidden_dim",
        "tarnet_head_separation_weight",
        "tarnet_head0_n_fc",
        "tarnet_head0_hidden_dim",
        "tarnet_head1_n_fc",
        "tarnet_head1_hidden_dim",
    )
    return tuple(config.get(k) for k in keys)


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
    bpe_vocab_size: int | None = None,
    bpe_word_boundary_aware: bool | None = None,
    results_dir: str | None = None,
    imdb_max_train_samples: int | None = None,
    imdb_max_val_samples: int | None = None,
    enable_counterfactual_objective: bool | None = None,
    tarnet_two_heads: bool | None = None,
    fixed_epochs: int | None = None,
) -> list[dict]:
    """Run HPO agent loop."""
    workspace = workspace or Path.cwd()
    effective_dataset_id = str(DEFAULT_CONFIG.get("dataset_id", "imdb_sentiment"))
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
    if enable_counterfactual_objective or tarnet_two_heads:
        system_prompt = IMDB_TARNET_SYSTEM_PROMPT
    else:
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
            if fixed_epochs is not None:
                candidate["epochs"] = int(fixed_epochs)
            # If TARNet is enabled for this HPO run, finalize head defaults before de-dup.
            if tarnet_two_heads:
                candidate["tarnet_two_heads"] = True
            if enable_counterfactual_objective:
                candidate["enable_counterfactual_objective"] = True
            if bool(candidate.get("tarnet_two_heads")):
                shared_n_fc = int(candidate.get("tarnet_head_n_fc", 2))
                shared_hidden = candidate.get("tarnet_head_hidden_dim")
                shared_hidden = int(candidate["d_model"]) if shared_hidden is None else int(shared_hidden)

                # Deterministic defaults (no trial_id) so duplicates can be skipped.
                candidate["tarnet_head0_n_fc"] = int(candidate.get("tarnet_head0_n_fc") or shared_n_fc)
                candidate["tarnet_head1_n_fc"] = int(candidate.get("tarnet_head1_n_fc") or shared_n_fc)
                candidate["tarnet_head0_hidden_dim"] = int(
                    candidate.get("tarnet_head0_hidden_dim") or max(128, min(1024, shared_hidden))
                )
                candidate["tarnet_head1_hidden_dim"] = int(
                    candidate.get("tarnet_head1_hidden_dim") or max(128, min(1024, shared_hidden * 2))
                )

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
        config["dataset_id"] = effective_dataset_id
        if imdb_max_train_samples is not None:
            config["imdb_max_train_samples"] = int(imdb_max_train_samples)
        if imdb_max_val_samples is not None:
            config["imdb_max_val_samples"] = int(imdb_max_val_samples)
        if tokenizer_type is not None:
            config["tokenizer_type"] = tokenizer_type
        if bpe_vocab_size is not None:
            config["bpe_vocab_size"] = int(bpe_vocab_size)
        if bpe_word_boundary_aware is not None:
            config["bpe_word_boundary_aware"] = bool(bpe_word_boundary_aware)
        if enable_counterfactual_objective is not None:
            config["enable_counterfactual_objective"] = bool(enable_counterfactual_objective)
        if tarnet_two_heads is not None:
            config["tarnet_two_heads"] = bool(tarnet_two_heads)
        if fixed_epochs is not None:
            config["epochs"] = int(fixed_epochs)
        # Per-head defaults were finalized before de-dup; keep them as-is here.

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
