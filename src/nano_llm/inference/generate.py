"""Autoregressive text generation with greedy, top-k, and top-p sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
    stop_at_newline: bool = True,
    seed: int | None = None,
    device: torch.device | str | None = None,
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
        stop_at_newline: Stop when generating newline.
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

    return tokenizer.decode(ids)
