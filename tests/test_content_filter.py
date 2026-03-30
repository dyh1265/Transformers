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

"""Tests for post-decode sensitive wording redaction."""

from nano_llm.inference.content_filter import redact_sensitive_output


def test_redact_empty_unchanged() -> None:
    assert redact_sensitive_output("") == ""
    assert redact_sensitive_output("hello") == "hello"


def test_redact_sex_and_porn_phrases() -> None:
    assert "sex" not in redact_sensitive_output("they had sex with").lower()
    assert "[redacted]" in redact_sensitive_output("they had sex with")
    assert "[redacted]" in redact_sensitive_output("a porn star here")
    assert "[redacted]" in redact_sensitive_output("its pornographical way")


def test_redact_preserves_place_names() -> None:
    assert "Sussex" in redact_sensitive_output("We visited Sussex.")
    assert "Middlesex" in redact_sensitive_output("Middlesex county")


def test_redact_case_insensitive() -> None:
    out = redact_sensitive_output("PORN and SeX")
    assert "porn" not in out.lower()
    assert "sex" not in out.lower()
