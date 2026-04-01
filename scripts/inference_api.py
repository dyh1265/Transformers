#!/usr/bin/env python3
"""Serve TARNet inference over localhost HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nano_llm.inference import load_model_and_tokenizer
from nano_llm.inference.worker import process_openai_chat_payload, process_request_payload


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    try:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def _parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b""
    if not raw:
        text = "{}"
    else:
        # utf-8-sig strips BOM (PowerShell Set-Content -Encoding UTF8 often writes one)
        text = raw.decode("utf-8-sig").strip()
        if not text:
            text = "{}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        head = text[:120].replace("\n", " ")
        raise ValueError(
            f"Invalid JSON body ({exc}). Start of body (check PowerShell quoting): {head!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def make_handler(
    model: Any,
    tokenizer: Any,
    max_context: int,
    *,
    log_requests: bool,
    on_cuda: bool,
) -> type[BaseHTTPRequestHandler]:
    class InferenceApiHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _path_only(self) -> str:
            return urlparse(self.path).path.rstrip("/") or "/"

        def do_GET(self) -> None:  # noqa: N802
            path = self._path_only()
            if path == "/health":
                _json_response(self, 200, {"ok": True, "service": "tarnet-inference-api"})
                return
            _json_response(self, 404, {"ok": False, "error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            t0 = time.perf_counter()
            path = self._path_only()
            # PowerShell / .NET clients often send Expect: 100-continue; stdlib http.server
            # must acknowledge before reading the body or the client may close the socket.
            expect = (self.headers.get("Expect") or "").lower()
            if expect == "100-continue":
                self.send_response_only(HTTPStatus.CONTINUE)
                self.end_headers()
                try:
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                if log_requests:
                    pacing = "GPU" if on_cuda else "CPU (can be slow)"
                    print(f"[inference_api] POST {path} … (generating; {pacing})", flush=True)
                payload = _parse_json_body(self)
                if path == "/generate":
                    response = process_request_payload(
                        model=model,
                        tokenizer=tokenizer,
                        payload=payload,
                        max_context=max_context,
                    )
                    _json_response(self, 200, response)
                    if log_requests:
                        print(
                            f"[inference_api] POST {path} -> 200 in {time.perf_counter() - t0:.2f}s",
                            flush=True,
                        )
                    return
                if path == "/v1/chat/completions":
                    response = process_openai_chat_payload(
                        model=model,
                        tokenizer=tokenizer,
                        payload=payload,
                        max_context=max_context,
                    )
                    _json_response(self, 200, response)
                    if log_requests:
                        print(
                            f"[inference_api] POST {path} -> 200 in {time.perf_counter() - t0:.2f}s",
                            flush=True,
                        )
                    return
                _json_response(self, 404, {"ok": False, "error": "Not found"})
                if log_requests:
                    print(
                        f"[inference_api] POST {path} -> 404 in {time.perf_counter() - t0:.2f}s",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                try:
                    _json_response(self, 400, {"ok": False, "error": str(exc)})
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                if log_requests:
                    print(
                        f"[inference_api] POST {path} -> 400 in {time.perf_counter() - t0:.2f}s ({exc!r})",
                        flush=True,
                    )

        def log_message(self, format: str, *args: Any) -> None:
            # Keep stdout clean; request logs are optional for local use.
            return

    return InferenceApiHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve TARNet inference API on localhost")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/counterfactual_repeat_20m/best.pt"),
        help="Path to TARNet two-head checkpoint",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Port to bind (default 18080 to avoid clashes with other tools on 8000)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Override context length (default: checkpoint seq_len)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: auto",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-request timing to stderr",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        rebuild_tokenizer_from_corpus=False,
    )
    if not bool(getattr(model, "tarnet_two_heads", False)):
        print(
            "Error: API server is TARNet-only. Use a checkpoint trained with --tarnet-two-heads.",
            file=sys.stderr,
        )
        return 2

    max_context = args.max_context if args.max_context is not None else int(config.get("seq_len", 128))
    device_str = str(next(model.parameters()).device)
    on_cuda = device_str.startswith("cuda")
    handler = make_handler(
        model=model,
        tokenizer=tokenizer,
        max_context=max_context,
        log_requests=not args.quiet,
        on_cuda=on_cuda,
    )
    # Single-threaded server: one request at a time (avoids concurrent forwards on
    # a shared PyTorch module, which is unsafe and can crash the process on some platforms).
    # Use stdlib defaults for SO_REUSEADDR; forcing reuse=True on Windows can allow a second
    # process to bind the same port and produce "empty reply" intermittently.
    server = HTTPServer((args.host, args.port), handler)
    print(
        f"TARNet API listening on http://{args.host}:{args.port} (pid {os.getpid()})",
        flush=True,
    )
    print(f"Model device: {device_str}", flush=True)
    if on_cuda:
        print(
            "Each completion blocks until generation finishes (GPU; long dual-head runs can still take a bit).",
            flush=True,
        )
    else:
        print(
            "Each completion blocks until generation finishes; on CPU, 120 tokens × two heads can take minutes.",
            flush=True,
        )
    print("POST /generate, POST /v1/chat/completions, GET /health", flush=True)
    if sys.platform == "win32":
        print(
            rf"If curl reports empty replies, check only this PID owns port {args.port}: "
            rf"netstat -ano | findstr :{args.port}",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
