"""Pytest fixtures and sys.path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so "from nano_llm import ..." works
root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
