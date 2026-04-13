from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quarry.config import Settings  # noqa: E402
from quarry.hosted_auth import build_openai_compatible_headers  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple OpenRouter Q&A smoke test using current Quarry config."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="用一句话介绍你自己。",
        help="Question to send to OpenRouter.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to Quarry TOML config file.",
    )
    parser.add_argument("--model", help="Override model from config.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--site-url",
        default=os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        help="Value for OpenRouter HTTP-Referer header.",
    )
    parser.add_argument(
        "--app-name",
        default=os.getenv("OPENROUTER_APP_NAME", "Quarry OpenRouter QA Test"),
        help="Value for OpenRouter X-Title header.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print full JSON response for debugging.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    settings = Settings.from_env(config_path=args.config)
    base_url = (settings.llm_base_url or "").rstrip("/")
    api_key = settings.llm_api_key or ""
    model = args.model or settings.llm_model

    if not base_url:
        print("[ERROR] Missing hosted.llm_base_url in config.", file=sys.stderr)
        return 2
    if not api_key:
        print("[ERROR] Missing hosted.llm_api_key in config/environment.", file=sys.stderr)
        return 2

    endpoint = f"{base_url}/chat/completions"
    headers = build_openai_compatible_headers(base_url, api_key)
    headers["HTTP-Referer"] = args.site_url
    headers["X-Title"] = args.app_name
    payload = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": args.question},
        ],
    }

    print(f"[INFO] Endpoint: {endpoint}")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Question: {args.question}")

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        print(f"[ERROR] HTTP {exc.response.status_code}: {exc.response.reason_phrase}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
        return 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = response.json()

    if args.raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"[INFO] Latency: {elapsed_ms:.1f} ms")
        return 0

    answer = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        answer = str(message.get("content", "")).strip()

    usage = data.get("usage", {})

    print("\n=== ANSWER ===")
    print(answer or "[EMPTY RESPONSE]")
    if usage:
        print("\n=== USAGE ===")
        print(json.dumps(usage, ensure_ascii=False, indent=2))
    print(f"\n[INFO] Latency: {elapsed_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
