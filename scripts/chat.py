#!/usr/bin/env python3
"""Interactive chat-style generation for IMDB sentiment-conditioned reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.data import REVIEW_CLOSE, REVIEW_OPEN, SENTIMENT_CLOSE, SENTIMENT_OPEN
from nano_llm.inference import generate as gen_fn
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
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    print("Loading model…", flush=True)
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        rebuild_tokenizer_from_shakespeare=False,
    )
    max_context = int(config.get("seq_len", 128))

    print(
        "IMDB review chat — same format as training.\n"
        "Commands: + positive, - negative, q quit.\n",
        flush=True,
    )

    while True:
        sentiment = _prompt_for_sentiment()
        if sentiment is None:
            print("Bye.", flush=True)
            return 0

        prefix = (
            f"<bos>{SENTIMENT_OPEN} {sentiment} {SENTIMENT_CLOSE} {REVIEW_OPEN} "
        )
        out = gen_fn(
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
        review = _extract_review(out)
        print(f"\n[{sentiment.upper()}]\n{review}\n", flush=True)

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
