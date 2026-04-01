"""Filesystem-backed inference worker."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_llm.inference.generate import generate, generate_both_heads

logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    """Single inference request payload."""

    job_id: str
    prompt: str
    both_reviews: bool = False
    head_id: int | None = None
    shared_head: bool = False
    max_tokens: int = 100
    method: str = "greedy"
    top_k: int = 40
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_stop_newline: bool = False
    stop_sequence: str | None = None
    seed: int | None = None
    no_sanitize: bool = False
    no_censor: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceJob":
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Job is missing required field: prompt")

        job_id = str(data.get("job_id") or "")
        if not job_id:
            raise ValueError("Job is missing required field: job_id")

        return cls(
            job_id=job_id,
            prompt=prompt,
            both_reviews=bool(data.get("both_reviews", False)),
            head_id=data.get("head_id"),
            shared_head=bool(data.get("shared_head", False)),
            max_tokens=int(data.get("max_tokens", 100)),
            method=str(data.get("method", "greedy")),
            top_k=int(data.get("top_k", 40)),
            top_p=float(data.get("top_p", 0.9)),
            temperature=float(data.get("temperature", 1.0)),
            repetition_penalty=float(data.get("repetition_penalty", 1.0)),
            no_stop_newline=bool(data.get("no_stop_newline", False)),
            stop_sequence=data.get("stop_sequence"),
            seed=data.get("seed"),
            no_sanitize=bool(data.get("no_sanitize", False)),
            no_censor=bool(data.get("no_censor", False)),
        )


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _ensure_tarnet_model(model: Any) -> None:
    if not bool(getattr(model, "tarnet_two_heads", False)):
        raise ValueError(
            "Inference worker is TARNet-only and requires a two-head checkpoint "
            "(train with --tarnet-two-heads)."
        )


def _process_job(
    model: Any,
    tokenizer: Any,
    job: InferenceJob,
    max_context: int,
) -> dict[str, Any]:
    if job.both_reviews:
        out0, out1 = generate_both_heads(
            model,
            tokenizer,
            prompt=job.prompt,
            max_new_tokens=job.max_tokens,
            max_context=max_context,
            method=job.method,
            top_k=job.top_k,
            top_p=job.top_p,
            temperature=job.temperature,
            repetition_penalty=job.repetition_penalty,
            stop_at_newline=not job.no_stop_newline,
            stop_sequence=job.stop_sequence,
            seed=job.seed,
            sanitize=not job.no_sanitize,
            censor_adult=not job.no_censor,
        )
        return {"output_y0": out0, "output_y1": out1}

    out = generate(
        model,
        tokenizer,
        prompt=job.prompt,
        head_id=job.head_id,
        shared_head=job.shared_head,
        max_new_tokens=job.max_tokens,
        max_context=max_context,
        method=job.method,
        top_k=job.top_k,
        top_p=job.top_p,
        temperature=job.temperature,
        repetition_penalty=job.repetition_penalty,
        stop_at_newline=not job.no_stop_newline,
        stop_sequence=job.stop_sequence,
        seed=job.seed,
        sanitize=not job.no_sanitize,
        censor_adult=not job.no_censor,
    )
    return {"output": out}


def process_single_request(
    model: Any,
    tokenizer: Any,
    request_path: Path,
    response_dir: Path,
    max_context: int,
) -> tuple[str, bool]:
    """Process one request JSON file and write a response JSON file."""
    try:
        payload = _safe_read_json(request_path)
        response = process_request_payload(
            model=model, tokenizer=tokenizer, payload=payload, max_context=max_context
        )
        ok = bool(response.get("ok"))
    except Exception as exc:  # noqa: BLE001
        job_id = request_path.stem
        response = {"job_id": job_id, "ok": False, "error": str(exc)}
        ok = False

    response_path = response_dir / f"{response['job_id']}.json"
    _write_json(response_path, response)
    request_path.unlink(missing_ok=True)
    return response_path.name, ok


def process_request_payload(
    model: Any,
    tokenizer: Any,
    payload: dict[str, Any],
    max_context: int,
) -> dict[str, Any]:
    """Process one in-memory request payload and return response payload."""
    _ensure_tarnet_model(model)
    job = InferenceJob.from_dict(payload)
    output_payload = _process_job(model=model, tokenizer=tokenizer, job=job, max_context=max_context)
    return {"job_id": job.job_id, "ok": True, **output_payload}


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def process_openai_chat_payload(
    model: Any,
    tokenizer: Any,
    payload: dict[str, Any],
    max_context: int,
    default_model_name: str = "nano-llm-tarnet",
) -> dict[str, Any]:
    """Process OpenAI-style /v1/chat/completions payload."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array")

    user_texts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() != "user":
            continue
        text = _coerce_message_content(msg.get("content"))
        if text.strip():
            user_texts.append(text.strip())
    if not user_texts:
        raise ValueError("No usable user message content found")

    both_reviews = bool(payload.get("both_reviews", False))
    max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 100)))
    temperature = float(payload.get("temperature", 1.0))
    top_p = float(payload.get("top_p", 0.9))
    job_payload: dict[str, Any] = {
        "job_id": str(payload.get("user") or f"chat-{int(time.time())}"),
        "prompt": user_texts[-1],
        "both_reviews": both_reviews,
        "max_tokens": max_tokens,
        "method": "top_p" if top_p < 1.0 else "greedy",
        "top_p": top_p,
        "temperature": temperature,
        "no_stop_newline": True,
    }
    # Optional passthrough knobs
    for key in (
        "top_k",
        "repetition_penalty",
        "stop_sequence",
        "seed",
        "no_sanitize",
        "no_censor",
        "head_id",
        "shared_head",
    ):
        if key in payload:
            job_payload[key] = payload[key]

    out = process_request_payload(model=model, tokenizer=tokenizer, payload=job_payload, max_context=max_context)
    created = int(time.time())
    model_name = str(payload.get("model") or default_model_name)
    if both_reviews:
        y0 = str(out["output_y0"])
        y1 = str(out["output_y1"])
        choices = [
            {"index": 0, "message": {"role": "assistant", "content": y0}, "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": y1}, "finish_reason": "stop"},
        ]
        completion_tokens = len(tokenizer.encode(y0)) + len(tokenizer.encode(y1))
    else:
        text = str(out["output"])
        choices = [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ]
        completion_tokens = len(tokenizer.encode(text))
    prompt_tokens = len(tokenizer.encode(user_texts[-1]))
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def run_worker_loop(
    model: Any,
    tokenizer: Any,
    request_dir: Path,
    response_dir: Path,
    max_context: int,
    poll_interval: float = 1.0,
    max_jobs: int | None = None,
) -> int:
    """Run worker loop until max_jobs reached (or forever when None)."""
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    while True:
        request_files = sorted(request_dir.glob("*.json"))
        if not request_files:
            time.sleep(poll_interval)
            continue

        for request_path in request_files:
            response_name, ok = process_single_request(
                model=model,
                tokenizer=tokenizer,
                request_path=request_path,
                response_dir=response_dir,
                max_context=max_context,
            )
            status = "ok" if ok else "error"
            logger.info("Processed %s -> %s (%s)", request_path.name, response_name, status)
            processed += 1
            if max_jobs is not None and processed >= max_jobs:
                return processed
