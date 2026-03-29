"""Inference engine for nano-llm."""

from nano_llm.inference.generate import generate, generate_both_heads
from nano_llm.inference.load import load_model_and_tokenizer

__all__ = ["load_model_and_tokenizer", "generate", "generate_both_heads"]
