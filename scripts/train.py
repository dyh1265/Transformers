#!/usr/bin/env python3
"""CLI entrypoint for training nano-llm. Accepts HPO overrides via CLI or config file."""

from __future__ import annotations

import argparse
import json
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
        "--tokenizer-type",
        type=str,
        dest="tokenizer_type",
        choices=["char", "bpe", "bpe_byte", "hf_bpe_byte"],
        help="Tokenizer type: char, bpe, bpe_byte, or hf_bpe_byte",
    )
    p.add_argument(
        "--bpe-vocab-size",
        type=int,
        dest="bpe_vocab_size",
        help="BPE vocab size (used when tokenizer_type=bpe or bpe_byte)",
    )
    p.add_argument(
        "--bpe-word-boundary-aware",
        action="store_true",
        dest="bpe_word_boundary_aware",
        help="Prevent BPE merges across whitespace boundaries",
    )
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    p.add_argument("--learning-rate", type=float, dest="learning_rate", help="Learning rate")
    p.add_argument("--lr-decay", type=str, dest="lr_decay", choices=["cosine", "linear", "none"], help="LR decay: cosine, linear, or none")
    p.add_argument("--lr-min", type=float, dest="lr_min", help="Minimum LR for decay (default 1e-6)")
    p.add_argument("--epochs", type=int, help="Number of epochs")
    p.add_argument("--trial-id", type=int, dest="trial_id", help="HPO trial ID")
    p.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", default=None)
    p.add_argument("--hpo-results-dir", type=str, dest="hpo_results_dir", default=None)
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    p.add_argument("--early-stopping-patience", type=int, dest="early_stopping_patience", help="Stop if val_loss unchanged for N epochs (0=disabled)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = dict(DEFAULT_CONFIG)

    overrides = {}
    if args.d_model is not None:
        overrides["d_model"] = args.d_model
    if args.num_heads is not None:
        overrides["num_heads"] = args.num_heads
    if args.num_layers is not None:
        overrides["num_layers"] = args.num_layers
    if args.d_ff is not None:
        overrides["d_ff"] = args.d_ff
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.tokenizer_type is not None:
        overrides["tokenizer_type"] = args.tokenizer_type
    if args.bpe_vocab_size is not None:
        overrides["bpe_vocab_size"] = args.bpe_vocab_size
    if args.bpe_word_boundary_aware:
        overrides["bpe_word_boundary_aware"] = True
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.lr_decay is not None:
        overrides["lr_decay"] = args.lr_decay
    if args.lr_min is not None:
        overrides["lr_min"] = args.lr_min
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.trial_id is not None:
        overrides["trial_id"] = args.trial_id
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.hpo_results_dir is not None:
        overrides["hpo_results_dir"] = args.hpo_results_dir
    if args.resume:
        overrides["resume"] = args.resume
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience

    config.update(overrides)
    train(config)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
