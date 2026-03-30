#!/usr/bin/env python3
"""Generate text from a trained nano-llm checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.inference import generate as gen_fn
from nano_llm.inference import generate_both_heads as gen_both_fn
from nano_llm.inference import load_model_and_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from nano-llm checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<bos>[SENTIMENT] positive [/SENTIMENT] [REVIEW] ",
        help="Starting prompt (use a prefix that matches your checkpoint, e.g. IMDB tags or TARNet command)",
    )
    parser.add_argument(
        "--head-id",
        type=int,
        default=None,
        help="TARNet two-head models: choose head 0 (Y0) or 1 (Y1). Default: 0.",
    )
    parser.add_argument(
        "--shared-head",
        action="store_true",
        help="TARNet two-head models: sample from trunk/shared logits instead of Y0/Y1",
    )
    parser.add_argument(
        "--both-heads",
        action="store_true",
        help="TARNet two-head models: generate and print both Y0 and Y1 in one run",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--method",
        choices=["greedy", "top_k", "top_p"],
        default="greedy",
        help="Sampling method",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k for top_k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) for top_p sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalize repeated tokens (1.0=off, 1.1-1.5 typical)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: auto",
    )
    parser.add_argument(
        "--no-stop-newline",
        action="store_true",
        help="Do not stop generation at newline",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Max context length (default: from checkpoint seq_len)",
    )
    parser.add_argument(
        "--stop-sequence",
        type=str,
        default=None,
        help="Stop generation when this text appears",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Do not sanitize output (keep replacement chars, etc.)",
    )
    parser.add_argument(
        "--no-censor",
        action="store_true",
        help="Do not redact explicit terms in decoded output (default: on)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    )
    max_context = (
        args.max_context if args.max_context is not None else int(config.get("seq_len", 128))
    )

    if args.both_heads:
        if not (hasattr(model, "tarnet_two_heads") and getattr(model, "tarnet_two_heads")):
            print(
                "Error: --both-heads requires a TARNet two-head checkpoint "
                "(train with --tarnet-two-heads).",
                file=sys.stderr,
            )
            return 2
        out0, out1 = gen_both_fn(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            max_context=max_context,
            method=args.method,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            stop_at_newline=not args.no_stop_newline,
            stop_sequence=args.stop_sequence,
            sanitize=not args.no_sanitize,
            censor_adult=not args.no_censor,
        )
        print("[Y0]\n" + out0 + "\n\n[Y1]\n" + out1)
    else:
        out = gen_fn(
            model,
            tokenizer,
            prompt=args.prompt,
            head_id=args.head_id,
            shared_head=args.shared_head,
            max_new_tokens=args.max_tokens,
            max_context=max_context,
            method=args.method,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            stop_at_newline=not args.no_stop_newline,
            stop_sequence=args.stop_sequence,
            sanitize=not args.no_sanitize,
            censor_adult=not args.no_censor,
        )
        print(out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)
