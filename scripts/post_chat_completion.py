#!/usr/bin/env python3
"""POST JSON to /v1/chat/completions (stdin or file). Avoids Windows PowerShell + curl -d mangling."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="POST JSON body to inference chat completions API")
    p.add_argument(
        "json_path",
        nargs="?",
        help="Path to JSON file (omit to read stdin)",
    )
    p.add_argument(
        "--url",
        default="http://127.0.0.1:18080/v1/chat/completions",
        help="Full URL for chat completions",
    )
    args = p.parse_args()

    if args.json_path:
        raw = Path(args.json_path).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    req = urllib.request.Request(
        args.url,
        data=raw.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            sys.stdout.buffer.write(resp.read())
            sys.stdout.buffer.write(b"\n")
    except urllib.error.HTTPError as e:
        sys.stdout.buffer.write(e.read())
        sys.stdout.buffer.write(b"\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
