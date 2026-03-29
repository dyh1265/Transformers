#!/usr/bin/env python3
"""Rank HPO trial results by quality metrics."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_float(x: float) -> str:
    if x == float("inf"):
        return "n/a"
    return f"{x:.4f}"


def main() -> None:
    files = sorted(Path("hpo_results").glob("**/trial_*.json"))
    rows: list[dict] = []
    for p in files:
        d = json.loads(p.read_text())
        rows.append(
            {
                "trial_id": d.get("trial_id"),
                "results_dir": str(p.parent).replace("\\", "/"),
                "tokenizer_type": d.get("config", {}).get("tokenizer_type", "hf_bpe_byte"),
                "bpe_vocab_size": d.get("config", {}).get("bpe_vocab_size"),
                "best_val_bits_per_byte": float(d.get("best_val_bits_per_byte", float("inf"))),
                "best_val_perplexity": float(d.get("best_val_perplexity", float("inf"))),
                "best_val_loss": float(d.get("best_val_loss", float("inf"))),
            }
        )

    rows = [r for r in rows if r["trial_id"] is not None]
    if not rows:
        print("No trial_*.json found in hpo_results/")
        return

    rows.sort(
        key=lambda r: (
            r["best_val_bits_per_byte"] == float("inf"),
            r["best_val_bits_per_byte"],
            r["best_val_loss"],
        )
    )
    print("trial_id\ttokenizer\tbpe_vocab\tbest_val_bpb\tbest_val_ppl\tbest_val_loss\tresults_dir")
    for r in rows:
        ppl = "n/a" if r["best_val_perplexity"] == float("inf") else f"{r['best_val_perplexity']:.2f}"
        print(
            f"{r['trial_id']}\t{r['tokenizer_type']}\t{r['bpe_vocab_size']}\t"
            f"{_fmt_float(r['best_val_bits_per_byte'])}\t{ppl}\t{_fmt_float(r['best_val_loss'])}\t{r['results_dir']}"
        )


if __name__ == "__main__":
    main()
