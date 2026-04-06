from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from quarry.adapters.local_models import LocalSentenceTransformerEmbeddingClient


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
