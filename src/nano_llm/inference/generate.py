"""Autoregressive text generation with greedy, top-k, and top-p sampling."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

import torch

# Unicode replacement char (invalid UTF-8 decode) and common wrong-chars to fix
REPLACEMENT_CHAR = "\uFFFD"
SANITIZE_REPLACEMENTS = {
    "\u00B4": "'",  # acute accent ´
    "\u02B9": "'",  # modifier letter prime ʻ
    "\u02BA": '"',  # modifier letter double prime
    "\u00A0": " ",  # nbsp -> space
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
}


def sanitize_output(text: str) -> str:
    """Remove strange Unicode and invalid chars from model output."""
    text = text.replace(REPLACEMENT_CHAR, "")
    for bad, good in SANITIZE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Remove control/surrogate chars
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn")
    )
    return text

if TYPE_CHECKING:
    from nano_llm.tokenizer import BPETokenizer, ByteBPETokenizer, CharTokenizer, HFByteBPETokenizer


def _greedy_sample(logits: torch.Tensor) -> int:
    return logits.argmax(dim=-1).item()


def _top_k_sample(logits: torch.Tensor, k: int) -> int:
    if k <= 0 or k >= logits.size(-1):
        return _greedy_sample(logits)
    top_k_logits, top_k_idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(top_k_logits.float(), dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return top_k_idx[idx].item()


def _top_p_sample(logits: torch.Tensor, p: float) -> int:
    if p >= 1.0 or p <= 0:
        return _greedy_sample(logits)
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs <= p
    filtered_probs = sorted_probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum()
    idx = torch.multinomial(filtered_probs, 1).item()
    return sorted_idx[idx].item()


def _apply_repetition_penalty(
    logits: torch.Tensor, ids: list[int], penalty: float
) -> torch.Tensor:
    """Penalize logits for tokens that have already appeared."""
    if penalty == 1.0 or not ids:
        return logits
    for token_id in ids:
        if 0 <= token_id < logits.size(-1):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty
    return logits


def generate(
    model: torch.nn.Module,
    tokenizer: "CharTokenizer | BPETokenizer | ByteBPETokenizer | HFByteBPETokenizer",
    prompt: str,
    max_new_tokens: int = 100,
    max_context: int = 128,
    method: str = "greedy",
    top_k: int = 40,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_at_newline: bool = True,
    stop_sequence: str | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    sanitize: bool = True,
) -> str:
    """Generate text autoregressively.

    Args:
        model: NanoLLM model (in eval mode).
        tokenizer: CharTokenizer.
        prompt: Starting text.
        max_new_tokens: Maximum tokens to generate.
        max_context: Max context length (sliding window).
        method: "greedy", "top_k", or "top_p".
        top_k: For top_k sampling.
        top_p: For top_p (nucleus) sampling.
        temperature: Scale logits before sampling (1.0 = no scaling).
        repetition_penalty: Penalize repeated tokens (1.0 = off, 1.1–1.5 typical).
        stop_at_newline: Stop when generating newline.
        stop_sequence: Stop when this string appears in decoded output.
        seed: Random seed for sampling.
        device: Device for model (default: model's device).

    Returns:
        Generated text (prompt + continuation).
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = next(model.parameters()).device

    ids = list(tokenizer.encode(prompt))
    vocab_size = tokenizer.vocab_size

    for _ in range(max_new_tokens):
        context = ids[-max_context:] if len(ids) > max_context else ids
        x = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0, -1]

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        logits = _apply_repetition_penalty(logits.clone(), ids, repetition_penalty)

        if method == "greedy":
            next_id = _greedy_sample(logits)
        elif method == "top_k":
            next_id = _top_k_sample(logits, top_k)
        elif method == "top_p":
            next_id = _top_p_sample(logits, top_p)
        else:
            raise ValueError(f"Unknown method: {method}. Use greedy, top_k, or top_p.")

        ids.append(next_id)

        if stop_at_newline and next_id < vocab_size:
            token_text = tokenizer.decode([next_id])
            if "\n" in token_text:
                break

        if stop_sequence:
            decoded = tokenizer.decode(ids)
            if stop_sequence in decoded:
                break

    out = tokenizer.decode(ids)
    if sanitize:
        out = sanitize_output(out)
    return out
