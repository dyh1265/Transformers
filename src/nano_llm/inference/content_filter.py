"""Post-decode redaction for explicit sexual / pornographic wording.

This is a *lightweight* regex-based filter applied after decoding, not a model
training-time safety mechanism.
"""

from __future__ import annotations

import re

_REDACTION = "[redacted]"

# Note: keep longer/phrase patterns first (e.g. "porn star" before "porn").
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\bporn\s+star\b",
        r"\bporno\b",
        r"\bpornograph\w*\b",
        r"\bporn\b",
        r"\bsex\s+scene\b",
        r"\bsex\b",
        r"\bsexual\w*\b",
        r"\bsex\b",
        r"\bnudity\b",
        r"\bnude\b",
        r"\bstriptease\b",
        r"\bdildo\w*\b",
        r"\bdick\b",
        r"\bjizz\b",
        r"\berotic\w*\b",
        r"\brape\b",
        r"\brapist\b",
        r"\bprostitut\w*\b",
        r"\bbrothel\b",
        r"\bintercourse\b",
        r"\bmasturbat\w*\b",
        r"\borgasm\w*\b",
        r"\bblowjob\b",
        r"\bhandjob\b",
        r"\bcunnilingus\b",
        r"\bfellatio\b",
        r"\bxxx\b",
        r"\bnude\b",
        r"\bnsfw\b",
    )
)


def redact_sensitive_output(text: str) -> str:
    """Replace matches with a fixed placeholder."""
    if not text:
        return text
    out = text
    for rx in _SENSITIVE_PATTERNS:
        out = rx.sub(_REDACTION, out)
    return out
