from __future__ import annotations

import asyncio
import json

import httpx

from quarry.adapters.production import HostedEmbeddingClient, OpenAICompatibleLLM, _http_error_log_fields
from quarry.hosted_auth import build_openai_compatible_headers, normalize_azure_openai_base_url


class DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class DummyAsyncClient:
    def __init__(self, captured: dict[str, object], payload: dict[str, object]) -> None:
        self.captured = captured
        self.payload = payload

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> DummyResponse:
        self.captured["url"] = url
        self.captured["headers"] = headers
        self.captured["json"] = json
        return DummyResponse(self.payload)


def test_build_openai_compatible_headers_uses_api_key_for_azure_routes() -> None:
    resource_headers = build_openai_compatible_headers(
        "https://example.openai.azure.com/openai/v1",
        "azure-key",
    )
    cognitive_headers = build_openai_compatible_headers(
        "https://example.cognitiveservices.azure.com/openai/v1",
        "azure-key",
    )
    project_headers = build_openai_compatible_headers(
        "https://example.services.ai.azure.com/api/projects/demo/openai/v1",
        "azure-key",
    )

    assert resource_headers == {"Content-Type": "application/json", "api-key": "azure-key"}
    assert cognitive_headers == {"Content-Type": "application/json", "api-key": "azure-key"}
    assert project_headers == {"Content-Type": "application/json", "api-key": "azure-key"}


def test_normalize_azure_openai_base_url_accepts_root_and_v1_routes() -> None:
    assert (
        normalize_azure_openai_base_url(
            "https://example.cognitiveservices.azure.com"
        )
        == "https://example.cognitiveservices.azure.com/openai/v1"
    )
    assert (
        normalize_azure_openai_base_url(
            "https://example.cognitiveservices.azure.com/openai/v1/"
        )
        == "https://example.cognitiveservices.azure.com/openai/v1"
    )


def test_openai_compatible_llm_uses_bearer_header_for_standard_openai(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "quarry.adapters.production.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(
            captured,
            {"choices": [{"message": {"content": "hello"}}]},
        ),
    )

    llm = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="openai-key",
        model="gpt-4o-mini",
    )
    result = asyncio.run(llm.complete("Test prompt"))

    assert result == "hello"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer openai-key",
    }
    assert captured["json"]["temperature"] == 0.1


def test_openai_compatible_llm_omits_temperature_for_gpt5_family(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "quarry.adapters.production.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(
            captured,
            {"choices": [{"message": {"content": "hello"}}]},
        ),
    )

    llm = OpenAICompatibleLLM(
        base_url="https://example.cognitiveservices.azure.com/openai/v1",
        api_key="azure-key",
        model="gpt-5.2-chat",
    )
    result = asyncio.run(llm.complete("Test prompt", temperature=0.2))

    assert result == "hello"
    assert captured["json"]["model"] == "gpt-5.2-chat"
    assert "temperature" not in captured["json"]


def test_http_error_log_fields_include_status_url_and_body() -> None:
    request = httpx.Request("POST", "https://example.cognitiveservices.azure.com/openai/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "Unsupported value for temperature.",
                "param": "temperature",
            }
        },
    )
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    fields = _http_error_log_fields(exc)

    assert fields["http_status_code"] == 400
    assert fields["request_url"] == "https://example.cognitiveservices.azure.com/openai/v1/chat/completions"
    assert json.loads(fields["error_body"])["error"]["param"] == "temperature"


def test_hosted_embedding_client_uses_api_key_header_for_azure(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "quarry.adapters.production.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(
            captured,
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        ),
    )

    client = HostedEmbeddingClient(
        base_url="https://example.openai.azure.com/openai/v1",
        api_key="azure-key",
        model="text-embedding-3-small-deployment",
        dimensions=3,
    )
    result = asyncio.run(client.embed_texts(["hello"]))

    assert result == [[0.1, 0.2, 0.3]]
    assert captured["url"] == "https://example.openai.azure.com/openai/v1/embeddings"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "api-key": "azure-key",
    }
