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
    p.add_argument(
        "--dataset-id",
        type=str,
        dest="dataset_id",
        choices=[
            "tiny_shakespeare",
            "wikitext_2",
            "wikitext_103",
            "imdb_sentiment",
            "pg19",
            "bookcorpus",
        ],
        help="Dataset ID",
    )
    p.add_argument(
        "--wikitext-max-train-samples",
        type=int,
        dest="wikitext_max_train_samples",
        help="Limit WikiText-2 train lines for faster pretrain",
    )
    p.add_argument(
        "--wikitext-max-val-samples",
        type=int,
        dest="wikitext_max_val_samples",
        help="Limit WikiText-2 validation lines",
    )
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
        "--pg19-max-train-books",
        type=int,
        dest="pg19_max_train_books",
        help="Limit PG-19 train books for faster experiments",
    )
    p.add_argument(
        "--pg19-max-val-books",
        type=int,
        dest="pg19_max_val_books",
        help="Limit PG-19 val books",
    )
    p.add_argument(
        "--pg19-max-chars-per-book",
        type=int,
        dest="pg19_max_chars_per_book",
        help="Truncate each PG-19 book to at most N chars",
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
    if args.dataset_id is not None:
        overrides["dataset_id"] = args.dataset_id
    if args.wikitext_max_train_samples is not None:
        overrides["wikitext_max_train_samples"] = args.wikitext_max_train_samples
    if args.wikitext_max_val_samples is not None:
        overrides["wikitext_max_val_samples"] = args.wikitext_max_val_samples
    if args.imdb_max_train_samples is not None:
        overrides["imdb_max_train_samples"] = args.imdb_max_train_samples
    if args.imdb_max_val_samples is not None:
        overrides["imdb_max_val_samples"] = args.imdb_max_val_samples
    if args.imdb_max_review_chars is not None:
        overrides["imdb_max_review_chars"] = args.imdb_max_review_chars
    if args.pg19_max_train_books is not None:
        overrides["pg19_max_train_books"] = args.pg19_max_train_books
    if args.pg19_max_val_books is not None:
        overrides["pg19_max_val_books"] = args.pg19_max_val_books
    if args.pg19_max_chars_per_book is not None:
        overrides["pg19_max_chars_per_book"] = args.pg19_max_chars_per_book
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
    if args.position_encoding is not None:
        overrides["position_encoding"] = args.position_encoding
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.use_wandb:
        overrides["use_wandb"] = True
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        overrides["wandb_run_name"] = args.wandb_run_name
    if args.wandb_entity is not None:
        overrides["wandb_entity"] = args.wandb_entity
    if args.wandb_tags is not None:
        overrides["wandb_tags"] = args.wandb_tags
    if args.wandb_log_model:
        overrides["wandb_log_model"] = True
    if args.mixed_precision is not None:
        overrides["mixed_precision"] = args.mixed_precision
    if args.no_cuda_tf32:
        overrides["cuda_allow_tf32"] = False
    if args.no_cuda_flash_sdp:
        overrides["cuda_prefer_flash_attn"] = False
    if args.torch_compile:
        overrides["torch_compile"] = True
    if args.block_attn_residuals:
        overrides["block_attn_residuals"] = True
    if args.macro_block_size is not None:
        overrides["macro_block_size"] = args.macro_block_size
    if args.max_block_representations is not None:
        overrides["max_block_representations"] = args.max_block_representations

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
