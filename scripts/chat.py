#!/usr/bin/env python3
"""Interactive chat-style generation for IMDB sentiment-conditioned reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.data import (
    IMDB_DEFAULT_NEGATIVE_INSTRUCTION,
    IMDB_DEFAULT_POSITIVE_INSTRUCTION,
    REVIEW_CLOSE,
    REVIEW_OPEN,
    SENTIMENT_CLOSE,
    SENTIMENT_OPEN,
)
from nano_llm.inference import generate as gen_fn
from nano_llm.inference import generate_both_heads as gen_both_fn
from nano_llm.inference import load_model_and_tokenizer


def _prompt_for_sentiment() -> str | None:
    """Return 'positive', 'negative', or None if user wants to quit."""
    line = input("Sentiment [+/-/q] (default +): ").strip().lower()
    if line in ("q", "quit", "exit"):
        return None
    if line in ("-", "neg", "negative", "n"):
        return "negative"
    if line in ("", "+", "pos", "positive", "p"):
        return "positive"
    if "neg" in line:
        return "negative"
    return "positive"


def _extract_review(full: str) -> str:
    """Keep only review body if model echoed training markup."""
    if REVIEW_OPEN in full:
        part = full.split(REVIEW_OPEN, 1)[-1]
        if REVIEW_CLOSE in part:
            part = part.split(REVIEW_CLOSE)[0]
        return part.strip()
    return full.strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Chat-style IMDB review generation")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/imdb_sentiment/hf_bpe_byte/best.pt"),
    )
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--method", choices=["greedy", "top_k", "top_p"], default="top_p")
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument(
        "--head-id",
        type=int,
        default=None,
        help="TARNet two-head models: choose head 0 (Y0) or 1 (Y1). If set, sentiment prompt is ignored.",
    )
    p.add_argument(
        "--shared-head",
        action="store_true",
        help="TARNet two-head models: generate from trunk/shared logits (ignores sentiment/head-id).",
    )
    p.add_argument(
        "--counterfactual",
        action="store_true",
        help="For TARNet two-head checkpoints: generate and print both Y0 and Y1 each turn.",
    )
    p.add_argument(
        "--command-prompt",
        type=str,
        default="GENERATE an IMDB-like review:",
        help='Prefix used for TARNet counterfactual mode (default: "GENERATE an IMDB-like review:")',
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    print("Loading model…", flush=True)
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        rebuild_tokenizer_from_corpus=False,
    )
    max_context = int(config.get("seq_len", 128))

    if args.counterfactual:
        print(
            "IMDB counterfactual chat (TARNet two-head) — prints Y0 and Y1.\n"
            "Commands: + Y1, - Y0, b both, q quit.\n",
            flush=True,
        )
    else:
        print(
            "IMDB review chat — same format as training.\n"
            "Commands: + positive, - negative, q quit.\n",
            flush=True,
        )

    while True:
        is_tarnet = bool(getattr(model, "tarnet_two_heads", False))
        if args.counterfactual:
            if not is_tarnet:
                print(
                    "Error: --counterfactual requires a TARNet two-head checkpoint "
                    "(train with --tarnet-two-heads).",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            line = input("Generate [+/-/b/q] (default b): ").strip().lower()
            if line in ("q", "quit", "exit"):
                print("Bye.", flush=True)
                return 0
            if line in ("", "b", "both"):
                mode = "both"
            elif line in ("+", "pos", "y1", "1"):
                mode = "y1"
            elif line in ("-", "neg", "y0", "0"):
                mode = "y0"
            else:
                mode = "both"
            prefix = f"<bos>{args.command_prompt} {REVIEW_OPEN} "
            if mode == "both":
                out0, out1 = gen_both_fn(
                    model,
                    tokenizer,
                    prompt=prefix,
                    max_new_tokens=args.max_tokens,
                    max_context=max_context,
                    method=args.method,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    stop_at_newline=False,
                    stop_sequence=REVIEW_CLOSE,
                    seed=None,
                    device=None,
                    sanitize=True,
                )
                review0 = _extract_review(out0)
                review1 = _extract_review(out1)
                print(f"\n[Y0]\n{review0}\n\n[Y1]\n{review1}\n", flush=True)
            else:
                head_id = 1 if mode == "y1" else 0
                out = gen_fn(
                    model,
                    tokenizer,
                    prompt=prefix,
                    head_id=head_id,
                    max_new_tokens=args.max_tokens,
                    max_context=max_context,
                    method=args.method,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    stop_at_newline=False,
                    stop_sequence=REVIEW_CLOSE,
                    seed=None,
                    device=None,
                    sanitize=True,
                )
                review = _extract_review(out)
                label = "Y1" if head_id == 1 else "Y0"
                print(f"\n[{label}]\n{review}\n", flush=True)
            continue

        if args.shared_head:
            sentiment = "shared"
            prefix = f"<bos>{args.command_prompt} {REVIEW_OPEN} "
        elif args.head_id is None:
            sentiment = _prompt_for_sentiment()
            if sentiment is None:
                print("Bye.", flush=True)
                return 0
            style = str(config.get("imdb_conditioning_style", "tags")).strip().lower()
            if style == "natural":
                pos_i = config.get("imdb_positive_instruction") or IMDB_DEFAULT_POSITIVE_INSTRUCTION
                neg_i = config.get("imdb_negative_instruction") or IMDB_DEFAULT_NEGATIVE_INSTRUCTION
                instr = pos_i if sentiment == "positive" else neg_i
                prefix = f"<bos>{instr} {REVIEW_OPEN} "
            else:
                prefix = f"<bos>{SENTIMENT_OPEN} {sentiment} {SENTIMENT_CLOSE} {REVIEW_OPEN} "
        else:
            sentiment = "head_" + str(args.head_id)
            prefix = f"<bos>{args.command_prompt} {REVIEW_OPEN} "
        out = gen_fn(
            model,
            tokenizer,
            prompt=prefix,
            head_id=args.head_id,
            shared_head=args.shared_head,
            max_new_tokens=args.max_tokens,
            max_context=max_context,
            method=args.method,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            stop_at_newline=False,
            stop_sequence=REVIEW_CLOSE,
            seed=None,
            device=None,
            sanitize=True,
        )
        review = _extract_review(out)
        print(f"\n[{str(sentiment).upper()}]\n{review}\n", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nBye.", flush=True)
        sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
