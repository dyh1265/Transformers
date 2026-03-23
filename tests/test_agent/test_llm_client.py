"""Tests for llm_client (mocked)."""

from unittest.mock import patch

import pytest

from nano_llm.agent.llm_client import parse_config_from_response


def test_parse_config_extracts_json() -> None:
    text = 'Here is the config:\n```json\n{"d_model": 128, "num_layers": 4}\n```'
    out = parse_config_from_response(text)
    assert out is not None
    assert out["d_model"] == 128
    assert out["num_layers"] == 4


def test_parse_config_plain_json() -> None:
    text = '{"d_model": 64, "batch_size": 32}'
    out = parse_config_from_response(text)
    assert out == {"d_model": 64, "batch_size": 32}


def test_parse_config_invalid_returns_none() -> None:
    text = "This is not JSON at all"
    assert parse_config_from_response(text) is None
