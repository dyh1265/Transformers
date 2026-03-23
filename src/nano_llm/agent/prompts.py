"""System and user prompts for HPO agent."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """You are an expert ML engineer tuning hyperparameters for a small decoder-only transformer (GPT-style) trained on Tiny Shakespeare.

Output ONLY a valid JSON object with these keys (no other text):
- d_model: 64-256
- num_heads: 2-8 (must divide d_model)
- num_layers: 2-8
- d_ff: 256-1024
- seq_len: 64-256
- dropout: 0.0-0.2
- batch_size: 4-64
- learning_rate: 1e-4 to 1e-3
- epochs: 5-30 (use 20-30 for better convergence)

Best practices: prefer mixed precision (fp16), weight tying, learning rate ~3e-4, dropout ~0.1.
Given trial history, suggest a config that might improve validation loss. Explore different areas of the search space."""

EIGHT_GB_SYSTEM_PROMPT = """You are an expert ML engineer tuning hyperparameters for a small decoder-only transformer on an 8GB GPU.

STRICT CONSTRAINTS: batch_size <= 16, total params < 100M. For models > 20M params, set gradient_checkpointing=true.

Output ONLY a valid JSON object with: d_model, num_heads, num_layers, d_ff, seq_len, dropout, batch_size, learning_rate, epochs.
Keep batch_size 4-16, model size conservative, and epochs 5-30 for better convergence."""


def build_user_prompt(trial_history: list[dict]) -> str:
    """Build user prompt with trial history."""
    if not trial_history:
        return "No prior trials. Suggest a first config to try."
    lines = ["Prior trials:"]
    for t in trial_history[-5:]:
        cfg = t.get("config", t)
        val = t.get("best_val_loss", t.get("final_val_loss", "?"))
        lines.append(f"  config={cfg} -> best_val_loss={val}")
    lines.append("\nSuggest the next config as JSON only:")
    return "\n".join(lines)
