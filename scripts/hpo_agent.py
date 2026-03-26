#!/usr/bin/env python3
"""CLI entrypoint for HPO agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import nano_llm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.agent.tuner import tune


def main() -> None:
    p = argparse.ArgumentParser(description="HPO agent for nano-llm")
    p.add_argument("--max-trials", type=int, default=10)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--model", type=str, default="llama3.2")
    p.add_argument("--base-url", type=str, default="http://localhost:11434/v1/")
    p.add_argument("--max-parse-retries", type=int, default=2)
    p.add_argument(
        "--tokenizer-type",
        type=str,
        choices=["char", "bpe", "bpe_byte", "hf_bpe_byte"],
        default=None,
        help="Override tokenizer type used during HPO training trials",
    )
    p.add_argument(
        "--dataset-id",
        type=str,
        choices=["tiny_shakespeare", "imdb_sentiment"],
        default=None,
        help="Override dataset used during HPO training trials",
    )
    p.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=None,
        help="Override BPE vocab size (used when tokenizer-type=bpe, bpe_byte, or hf_bpe_byte)",
    )
    p.add_argument(
        "--bpe-word-boundary-aware",
        action="store_true",
        help="Enable word-boundary-aware BPE during HPO training trials",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory for HPO artifacts (default: hpo_results/<dataset_id>/<tokenizer_type>)",
    )
    p.add_argument("--8gb", dest="use_8gb", action="store_true", help="Use 8GB GPU bounds")
    p.add_argument(
        "--imdb-max-train-samples",
        type=int,
        default=None,
        help="(IMDB) Limit train samples for faster HPO",
    )
    p.add_argument(
        "--imdb-max-val-samples",
        type=int,
        default=None,
        help="(IMDB) Limit val samples for faster HPO",
    )
    p.add_argument(
        "--enable-counterfactual-objective",
        action="store_true",
        help="Enable counterfactual objective during HPO trials",
    )
    p.add_argument(
        "--tarnet-two-heads",
        action="store_true",
        help="Enable TARNet two-head architecture during HPO trials",
    )
    p.add_argument(
        "--fixed-epochs",
        type=int,
        default=None,
        help="Force all HPO trials to use this epochs value (overrides LLM proposals)",
    )
    args = p.parse_args()
    tune(
        max_trials=args.max_trials,
        workspace=args.workspace,
        model=args.model,
        use_8gb_bounds=args.use_8gb,
        base_url=args.base_url,
        max_parse_retries=args.max_parse_retries,
        tokenizer_type=args.tokenizer_type,
        dataset_id=args.dataset_id,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_word_boundary_aware=True if args.bpe_word_boundary_aware else None,
        results_dir=args.results_dir,
        imdb_max_train_samples=args.imdb_max_train_samples,
        imdb_max_val_samples=args.imdb_max_val_samples,
        enable_counterfactual_objective=True if args.enable_counterfactual_objective else None,
        tarnet_two_heads=True if args.tarnet_two_heads else None,
        fixed_epochs=args.fixed_epochs,
    )


if __name__ == "__main__":
    main()
