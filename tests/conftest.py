"""Pytest fixtures and sys.path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path so "from nano_llm import ..." works
root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from nano_llm.model import build_model


@pytest.fixture
def mini_config() -> dict:
    return {
        "vocab_size": 65,
        "d_model": 32,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 128,
        "seq_len": 64,
        "dropout": 0.1,
        "batch_size": 8,
    }


@pytest.fixture
def sample_text() -> str:
    return "To be or not to be, that is the question. " * 10


@pytest.fixture
def small_model(mini_config: dict):
    return build_model(
        vocab_size=mini_config["vocab_size"],
        d_model=mini_config["d_model"],
        num_heads=mini_config["num_heads"],
        num_layers=mini_config["num_layers"],
        d_ff=mini_config["d_ff"],
        max_len=mini_config["seq_len"] + 10,
        dropout=mini_config["dropout"],
    )
