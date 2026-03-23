"""Ollama/Qwen client for HPO agent (OpenAI-compatible API)."""

from __future__ import annotations

import json
import re

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:11434/v1/"
DEFAULT_MODEL = "llama3.2"


def create_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="ollama")


def chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Send chat completion request and return assistant message content."""
    client = create_client(base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def parse_config_from_response(text: str) -> dict | None:
    """Extract JSON config from LLM response. Returns None if not found."""
    text = text.strip()
    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try first {...} span
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
