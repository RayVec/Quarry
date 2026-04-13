from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from quarry.adapters.local_models import LocalSentenceTransformerEmbeddingClient, load_hf_asset_with_cache


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def encode(self, texts, **kwargs):
        with self._state_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(0.05)
        with self._state_lock:
            self.active_calls -= 1
        return np.ones((len(texts), 2), dtype="float32")


def test_local_embedding_client_serializes_concurrent_model_calls() -> None:
    client = LocalSentenceTransformerEmbeddingClient("mock-model", device="mps")
    fake_model = FakeSentenceTransformer()
    client._model = fake_model
    client._load_attempted = True

    async def exercise() -> None:
        await asyncio.gather(
            client.embed_texts(["query one"]),
            client.embed_texts(["query two"]),
            client.embed_texts(["query three"]),
        )

    asyncio.run(exercise())

    assert fake_model.max_active_calls == 1


def test_load_hf_asset_with_cache_prefers_local_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    monkeypatch.setattr("quarry.adapters.local_models.resolve_cached_hf_snapshot_path", lambda model_name: snapshot_path)

    calls: list[str] = []
    loaded = load_hf_asset_with_cache(
        "intfloat/e5-large-v2",
        lambda target: calls.append(target) or {"target": target},
        component="embedding",
    )

    assert loaded == {"target": str(snapshot_path)}
    assert calls == [str(snapshot_path)]


def test_load_hf_asset_with_cache_falls_back_to_repo_id_when_cached_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    monkeypatch.setattr("quarry.adapters.local_models.resolve_cached_hf_snapshot_path", lambda model_name: snapshot_path)

    calls: list[str] = []

    def loader(target: str) -> str:
        calls.append(target)
        if target == str(snapshot_path):
            raise RuntimeError("broken cache")
        return f"loaded:{target}"

    loaded = load_hf_asset_with_cache(
        "intfloat/e5-large-v2",
        loader,
        component="embedding",
    )

    assert loaded == "loaded:intfloat/e5-large-v2"
    assert calls == [str(snapshot_path), "intfloat/e5-large-v2"]


def test_load_hf_asset_with_cache_uses_repo_id_when_no_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quarry.adapters.local_models.resolve_cached_hf_snapshot_path", lambda model_name: None)

    calls: list[str] = []
    loaded = load_hf_asset_with_cache(
        "intfloat/e5-large-v2",
        lambda target: calls.append(target) or {"target": target},
        component="embedding",
    )

    assert loaded == {"target": "intfloat/e5-large-v2"}
    assert calls == ["intfloat/e5-large-v2"]
