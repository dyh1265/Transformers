#!/usr/bin/env python3
"""CLI entrypoint for training nano-llm. Accepts HPO overrides via CLI or config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.config import DEFAULT_CONFIG, load_config
from nano_llm.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train nano-llm")
    p.add_argument("--config", type=str, help="Path to JSON config file")
    p.add_argument("--d-model", type=int, dest="d_model", help="Model dimension")
    p.add_argument("--num-heads", type=int, dest="num_heads", help="Number of attention heads")
    p.add_argument("--num-layers", type=int, dest="num_layers", help="Number of decoder layers")
    p.add_argument("--d-ff", type=int, dest="d_ff", help="Feed-forward hidden dim")
    p.add_argument("--seq-len", type=int, dest="seq_len", help="Sequence length")
    p.add_argument("--dropout", type=float, help="Dropout rate")
    p.add_argument(
        "--bpe-vocab-size",
        type=int,
        dest="bpe_vocab_size",
        help="HF byte-level BPE vocab size (default from config)",
    )
    p.add_argument(
        "--bpe-word-boundary-aware",
        action="store_true",
        dest="bpe_word_boundary_aware",
        help="Prevent BPE merges across whitespace boundaries",
    )
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    p.add_argument(
        "--imdb-max-train-samples",
        type=int,
        dest="imdb_max_train_samples",
        help="Limit IMDB train samples for faster experiments",
    )
    p.add_argument(
        "--imdb-max-val-samples",
        type=int,
        dest="imdb_max_val_samples",
        help="Limit IMDB validation samples for faster experiments",
    )
    p.add_argument(
        "--imdb-max-review-chars",
        type=int,
        dest="imdb_max_review_chars",
        help="Truncate each IMDB review to at most N chars (at word boundary)",
    )
    p.add_argument(
        "--imdb-use-full-splits",
        action="store_true",
        dest="imdb_use_full_splits",
        help="Use full IMDB train/val splits (clears imdb_max_* limits from config file)",
    )
    p.add_argument(
        "--imdb-conditioning-style",
        type=str,
        dest="imdb_conditioning_style",
        choices=["tags", "natural"],
        help='IMDB single-head prompt: "tags" ([SENTIMENT]...) or "natural" (instruction before [REVIEW])',
    )
    p.add_argument(
        "--imdb-positive-instruction",
        type=str,
        dest="imdb_positive_instruction",
        help="With --imdb-conditioning-style natural: text before [REVIEW] for positive reviews",
    )
    p.add_argument(
        "--imdb-negative-instruction",
        type=str,
        dest="imdb_negative_instruction",
        help="With --imdb-conditioning-style natural: text before [REVIEW] for negative reviews",
    )
    p.add_argument(
        "--enable-counterfactual-objective",
        action="store_true",
        dest="enable_counterfactual_objective",
        help="Enable IMDB counterfactual/factual embedding objective",
    )
    p.add_argument(
        "--counterfactual-ce-weight",
        type=float,
        dest="counterfactual_ce_weight",
        help="Weight for LM cross-entropy in counterfactual objective",
    )
    p.add_argument(
        "--counterfactual-embedding-weight",
        type=float,
        dest="counterfactual_embedding_weight",
        help="Weight for treatment-weighted embedding loss term",
    )
    p.add_argument(
        "--tarnet-two-heads",
        action="store_true",
        dest="tarnet_two_heads",
        help="TARNet-style: shared trunk + 2 output heads (Y0/Y1) on same hidden states",
    )
    p.add_argument(
        "--tarnet-head-n-fc",
        type=int,
        dest="tarnet_head_n_fc",
        help="TARNet: number of FC layers in each head MLP (default 2)",
    )
    p.add_argument(
        "--tarnet-head-hidden-dim",
        type=int,
        dest="tarnet_head_hidden_dim",
        help="TARNet: hidden width for each head MLP (default: d_model)",
    )
    p.add_argument(
        "--tarnet-head0-n-fc",
        type=int,
        dest="tarnet_head0_n_fc",
        help="TARNet: FC layers for head0 (default: tarnet_head_n_fc)",
    )
    p.add_argument(
        "--tarnet-head0-hidden-dim",
        type=int,
        dest="tarnet_head0_hidden_dim",
        help="TARNet: hidden width for head0 (default: tarnet_head_hidden_dim or d_model)",
    )
    p.add_argument(
        "--tarnet-head1-n-fc",
        type=int,
        dest="tarnet_head1_n_fc",
        help="TARNet: FC layers for head1 (default: tarnet_head_n_fc)",
    )
    p.add_argument(
        "--tarnet-head1-hidden-dim",
        type=int,
        dest="tarnet_head1_hidden_dim",
        help="TARNet: hidden width for head1 (default: tarnet_head_hidden_dim or d_model)",
    )
    p.add_argument(
        "--imdb-tarnet-command-prompt",
        type=str,
        dest="imdb_tarnet_command_prompt",
        help='Treatment-invariant prompt prefix (e.g. "GENERATE an IMDB-like review:")',
    )
    p.add_argument(
        "--tarnet-head-separation-weight",
        type=float,
        dest="tarnet_head_separation_weight",
        help="TARNet: subtract this * JS(head0||head1) to discourage head collapse (default 0.0)",
    )
    p.add_argument("--learning-rate", type=float, dest="learning_rate", help="Learning rate")
    p.add_argument(
        "--lr-decay",
        type=str,
        dest="lr_decay",
        choices=["cosine", "linear", "none"],
        help="LR decay: cosine, linear, or none",
    )
    p.add_argument(
        "--lr-min",
        type=float,
        dest="lr_min",
        help="Minimum LR for decay (default 1e-6)",
    )
    p.add_argument("--epochs", type=int, help="Number of epochs")
    p.add_argument("--trial-id", type=int, dest="trial_id", help="HPO trial ID")
    p.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", default=None)
    p.add_argument("--hpo-results-dir", type=str, dest="hpo_results_dir", default=None)
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    p.add_argument(
        "--position-encoding",
        type=str,
        dest="position_encoding",
        choices=["sinusoidal", "rope"],
        help="Position encoding: sinusoidal (default) or rope",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        dest="early_stopping_patience",
        help="Stop if val_loss unchanged for N epochs (0=disabled)",
    )
    p.add_argument(
        "--use-wandb",
        action="store_true",
        dest="use_wandb",
        help="Log metrics to Weights & Biases (pip install wandb; wandb login)",
    )
    p.add_argument("--wandb-project", type=str, dest="wandb_project", help="W&B project name")
    p.add_argument("--wandb-run-name", type=str, dest="wandb_run_name", help="W&B run name")
    p.add_argument(
        "--wandb-entity",
        type=str,
        dest="wandb_entity",
        help="W&B entity (team/username)",
    )
    p.add_argument("--wandb-tags", type=str, dest="wandb_tags", help="W&B tags (comma-separated)")
    p.add_argument(
        "--wandb-log-model",
        action="store_true",
        dest="wandb_log_model",
        help="Upload best checkpoint to W&B at end",
    )
    p.add_argument(
        "--mixed-precision",
        type=str,
        dest="mixed_precision",
        choices=["fp32", "fp16", "bf16"],
        help="CUDA: fp16 (GradScaler), bf16 (often faster on Ampere+), fp32",
    )
    p.add_argument(
        "--no-cuda-tf32",
        action="store_true",
        dest="no_cuda_tf32",
        help="Disable TF32 for CUDA matmul/cudnn (stricter, often slower)",
    )
    p.add_argument(
        "--no-cuda-flash-sdp",
        action="store_true",
        dest="no_cuda_flash_sdp",
        help="Skip SDPA backend tuning (flash / mem-efficient preference)",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        dest="torch_compile",
        help="Wrap model with torch.compile on CUDA (first epoch may be slow)",
    )
    p.add_argument(
        "--block-attn-residuals",
        action="store_true",
        dest="block_attn_residuals",
        help="Inter-block attention residuals (mix prior macro-block reps)",
    )
    p.add_argument(
        "--macro-block-size",
        type=int,
        dest="macro_block_size",
        help="Decoder layers per macro block before appending a block representation",
    )
    p.add_argument(
        "--max-block-representations",
        type=int,
        dest="max_block_representations",
        help="Cap block history length (keeps b_0 + most recent; default 9)",
    )
    return p.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict:
    """Build config overrides from CLI args while preserving existing semantics."""
    overrides: dict = {}

    passthrough_keys = [
        "d_model",
        "num_heads",
        "num_layers",
        "d_ff",
        "seq_len",
        "dropout",
        "bpe_vocab_size",
        "batch_size",
        "imdb_max_train_samples",
        "imdb_max_val_samples",
        "imdb_max_review_chars",
        "imdb_conditioning_style",
        "imdb_positive_instruction",
        "imdb_negative_instruction",
        "counterfactual_ce_weight",
        "counterfactual_embedding_weight",
        "tarnet_head_n_fc",
        "tarnet_head_hidden_dim",
        "tarnet_head0_n_fc",
        "tarnet_head0_hidden_dim",
        "tarnet_head1_n_fc",
        "tarnet_head1_hidden_dim",
        "imdb_tarnet_command_prompt",
        "tarnet_head_separation_weight",
        "learning_rate",
        "lr_decay",
        "lr_min",
        "epochs",
        "trial_id",
        "checkpoint_dir",
        "hpo_results_dir",
        "position_encoding",
        "early_stopping_patience",
        "wandb_project",
        "wandb_run_name",
        "wandb_entity",
        "wandb_tags",
        "mixed_precision",
        "macro_block_size",
        "max_block_representations",
    ]
    for key in passthrough_keys:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    true_flags = [
        "bpe_word_boundary_aware",
        "enable_counterfactual_objective",
        "tarnet_two_heads",
        "use_wandb",
        "wandb_log_model",
        "torch_compile",
        "block_attn_residuals",
    ]
    for key in true_flags:
        if getattr(args, key):
            overrides[key] = True

    if args.imdb_use_full_splits:
        overrides["imdb_max_train_samples"] = None
        overrides["imdb_max_val_samples"] = None
    if args.no_cuda_tf32:
        overrides["cuda_allow_tf32"] = False
    if args.no_cuda_flash_sdp:
        overrides["cuda_prefer_flash_attn"] = False
    # Preserve prior behavior: only set resume when non-empty.
    if args.resume:
        overrides["resume"] = args.resume

    return overrides


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = dict(DEFAULT_CONFIG)
    overrides = _build_overrides(args)
    config.update(overrides)
    train(config)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"\nTraining failed: {e}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
