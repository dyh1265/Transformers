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
        help="Output directory for HPO artifacts (default: hpo_results/<tokenizer_type>)",
    )
    p.add_argument("--8gb", dest="use_8gb", action="store_true", help="Use 8GB GPU bounds")
    args = p.parse_args()
    tune(
        max_trials=args.max_trials,
        workspace=args.workspace,
        model=args.model,
        use_8gb_bounds=args.use_8gb,
        base_url=args.base_url,
        max_parse_retries=args.max_parse_retries,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_word_boundary_aware=True if args.bpe_word_boundary_aware else None,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
