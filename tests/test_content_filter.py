"""Tests for post-decode sensitive wording redaction."""

from __future__ import annotations

from nano_llm.inference.content_filter import redact_sensitive_output


def test_redact_empty() -> None:
    assert redact_sensitive_output("") == ""


def test_redact_case_insensitive() -> None:
    out = redact_sensitive_output("PORN and SeX")
    assert "[redacted]" in out
    assert "porn" not in out.lower()
    assert "sex" not in out.lower()


def test_redact_phrase_before_word() -> None:
    # "porn star" should be caught as a phrase.
    out = redact_sensitive_output("a porn star appears")
    assert "[redacted]" in out


def test_preserves_other_text() -> None:
    out = redact_sensitive_output("Sussex is sunny.")
    assert "Sussex" in out
