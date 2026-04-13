from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_azure_openai_base_url(base_url: str) -> str | None:
    parsed = urlparse(base_url.strip())
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")

    if not parsed.scheme or not hostname:
        return None

    if hostname.endswith(".openai.azure.com") or hostname.endswith(".cognitiveservices.azure.com"):
        if path in {"", "/openai"}:
            path = "/openai/v1"
        elif not path.endswith("/openai/v1"):
            return None
    elif hostname.endswith(".services.ai.azure.com"):
        if path in {"", "/openai"}:
            path = "/openai/v1"
        elif not path.endswith("/openai/v1"):
            return None
    else:
        return None

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def is_azure_openai_base_url(base_url: str) -> bool:
    return normalize_azure_openai_base_url(base_url) is not None


def build_openai_compatible_headers(base_url: str, api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if is_azure_openai_base_url(base_url):
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
