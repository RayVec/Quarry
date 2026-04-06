from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Sequence

import numpy as np

from quarry.adapters.in_memory import (
    ConservativeFallbackGenerationClient,
    HashEmbeddingClient,
    HeuristicDecompositionClient,
    HeuristicMetadataEnricher,
    HeuristicNLIClient,
    SimpleCrossEncoderReranker,
    normalize_text,
    tokenize,
)
from quarry.adapters.interfaces import DecompositionClient, EmbeddingClient, GenerationClient, MetadataEnricher, NLIClient, Reranker, Retriever
from quarry.domain.models import ChunkObject, ConfidenceLabel, GenerationRequest, RetrievalFilters, RetrievedPassage, ScoredReference
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed
from quarry.prompts import SHARED_SYSTEM_PROMPT, decomposition_classification_prompt, decomposition_prompt, generation_prompt, metadata_enrichment_prompt, parse_json_response


logger = logger_with_trace(__name__)


def resolve_torch_device(preferred: str = "auto") -> str:
    import torch

    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_embedding_text(text: str, *, model_name: str, is_query: bool) -> str:
    normalized = normalize_text(text)
    lowered = model_name.lower()
    if "e5" in lowered:
        prefix = "query" if is_query else "passage"
        return f"{prefix}: {normalized}"
    return normalized


@dataclass(slots=True)
class FaissIndexBundle:
    embedding_model: str
    dimensions: int
    chunk_ids: list[str]
    vectors: list[list[float]]


def write_faiss_bundle(bundle: FaissIndexBundle, *, index_path: Path, metadata_path: Path) -> None:
    import faiss

    metadata_path.write_text(
        json.dumps(
            {
                "embedding_model": bundle.embedding_model,
                "dimensions": bundle.dimensions,
                "chunk_ids": bundle.chunk_ids,
            },
            indent=2,
        )
    )
    index = faiss.IndexFlatIP(bundle.dimensions)
    if bundle.vectors:
        matrix = np.asarray(bundle.vectors, dtype="float32")
        index.add(matrix)
    faiss.write_index(index, str(index_path))


def load_faiss_bundle(index_path: Path, metadata_path: Path) -> tuple[object, dict[str, object]] | None:
    if not index_path.exists() or not metadata_path.exists():
        return None

    import faiss

    return faiss.read_index(str(index_path)), json.loads(metadata_path.read_text())


class NullConfidenceNLIClient(NLIClient):
    async def score(self, sentence_text: str, chunk_texts: Sequence[str]) -> list[ScoredReference]:
        return [ScoredReference(score=None, label=None) for _ in chunk_texts]


class LocalTextCompletionBackend:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        default_max_new_tokens: int = 768,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.default_max_new_tokens = default_max_new_tokens
        self._load_attempted = False
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _resolve_dtype(self, torch_module):
        if self.dtype == "auto":
            return "auto"
        return getattr(torch_module, self.dtype, "auto")

    def _load(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._device = resolve_torch_device(self.device)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self._resolve_dtype(torch),
                    device_map="auto" if self._device != "cpu" else None,
                )
                if self._device == "cpu":
                    model.to(torch.device(self._device))
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
            except Exception as exc:
                logger.warning("local text model unavailable", extra={"error": str(exc), "model": self.model_name})
                self._tokenizer = None
                self._model = None

    def is_ready(self) -> bool:
        self._load()
        return self._model is not None and self._tokenizer is not None

    def _render_prompt(self, prompt: str) -> str:
        assert self._tokenizer is not None
        messages = [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return (
            f"System: {SHARED_SYSTEM_PROMPT}\n\n"
            f"User: {prompt}\n\nAssistant:"
        )

    def _complete_sync(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        with self._inference_lock:
            rendered_prompt = self._render_prompt(prompt)
            encoded = self._tokenizer(rendered_prompt, return_tensors="pt")
            device = torch.device(self._device or resolve_torch_device(self.device))
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            prompt_tokens = encoded["input_ids"].shape[1]
            new_tokens = output[:, prompt_tokens:]
            return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
        operation: str = "completion",
    ) -> str:
        self._load()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"Local text model {self.model_name} is unavailable.")
        logger.info(
            "local text completion started",
            extra={
                "operation": operation,
                "model": self.model_name,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
                "prompt": prompt,
                "console_visible": False,
            },
        )
        start = timed()
        raw = await asyncio.to_thread(
            self._complete_sync,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
        )
        logger.info(
            "local text completion completed",
            extra={
                "operation": operation,
                "model": self.model_name,
                "response": raw,
                "latency_ms": elapsed_ms(start),
                "console_visible": False,
            },
        )
        return raw


class LocalStructuredDecompositionClient(DecompositionClient):
    def __init__(self, backend: LocalTextCompletionBackend, fallback: DecompositionClient | None = None) -> None:
        self.backend = backend
        self.fallback = fallback or HeuristicDecompositionClient()

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        try:
            raw = await self.backend.complete(decomposition_prompt(query, max_facets), max_new_tokens=256, operation="query_decomposition")
            payload = parse_json_response(raw)
            facets = [str(item) for item in payload.get("facets", []) if str(item).strip()]
            return facets[:max_facets] or [query]
        except Exception:
            logger.warning("local decomposition fell back to heuristic")
            return await self.fallback.decompose_query(query, max_facets)


class LocalStructuredMetadataEnricher(MetadataEnricher):
    def __init__(self, backend: LocalTextCompletionBackend, fallback: MetadataEnricher | None = None) -> None:
        self.backend = backend
        self.fallback = fallback or HeuristicMetadataEnricher()

    async def enrich(self, chunk: ChunkObject) -> ChunkObject:
        try:
            raw = await self.backend.complete(metadata_enrichment_prompt(chunk), max_new_tokens=256, operation="metadata_enrichment")
            payload = parse_json_response(raw)
            return chunk.model_copy(
                update={
                    "metadata_summary": str(payload.get("summary", chunk.metadata_summary)),
                    "metadata_entities": [str(item) for item in payload.get("entities", [])],
                    "metadata_questions": [str(item) for item in payload.get("questions", [])],
                }
            )
        except Exception:
            logger.warning("local metadata enrichment fell back to heuristic")
            return await self.fallback.enrich(chunk)


class LocalStructuredGenerationClient(GenerationClient):
    def __init__(self, backend: LocalTextCompletionBackend, fallback: GenerationClient | None = None) -> None:
        self.backend = backend
        self.fallback = fallback or ConservativeFallbackGenerationClient()

    async def generate(self, request: GenerationRequest) -> str:
        try:
            max_tokens = 384 if request.mode == "regeneration" else self.backend.default_max_new_tokens
            return await self.backend.complete(generation_prompt(request), temperature=0.2, max_new_tokens=max_tokens, operation=f"generation:{request.mode}")
        except Exception:
            logger.warning("local generation fell back to conservative no-ref implementation")
            return await self.fallback.generate(request)


class LocalSentenceTransformerEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 16,
        fallback: EmbeddingClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fallback = fallback or HashEmbeddingClient()
        self._model = None
        self._load_attempted = False
        self.dimensions: int | None = None
        self.active_model_name: str = model_name
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer

                kwargs: dict[str, object] = {"device": resolve_torch_device(self.device)}
                if "nomic" in self.model_name.lower():
                    kwargs["trust_remote_code"] = True
                self._model = SentenceTransformer(self.model_name, **kwargs)
                self.dimensions = int(self._model.get_sentence_embedding_dimension())
                self.active_model_name = self.model_name
            except Exception as exc:
                logger.warning("local embedding model unavailable", extra={"error": str(exc), "model": self.model_name})
                self._model = None
                self.active_model_name = getattr(self.fallback, "model", "hash-embedding-v1")

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        assert self._model is not None
        with self._inference_lock:
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.dimensions = int(vectors.shape[1]) if getattr(vectors, "shape", None) else self.dimensions
            return vectors.astype("float32").tolist()

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            vectors = await self.fallback.embed_texts(texts)
            if vectors:
                self.dimensions = len(vectors[0])
            return vectors
        return await asyncio.to_thread(self._embed_sync, texts)


class LocalBM25Retriever(Retriever):
    def __init__(self, chunks: Sequence[ChunkObject]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        self._tokenized_chunks = [tokenize(chunk.text) or ["_"] for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_chunks) if self._tokenized_chunks else None

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        start = timed()
        if self._bm25 is None:
            return [], {"retriever": "sparse", "result_count": 0, "fallback_used": False, "latency_ms": elapsed_ms(start), "provider": "rank-bm25"}
        query_tokens = tokenize(query)
        if not query_tokens:
            return [], {"retriever": "sparse", "result_count": 0, "fallback_used": False, "latency_ms": elapsed_ms(start), "provider": "rank-bm25"}

        scores = self._bm25.get_scores(query_tokens)
        candidates: list[RetrievedPassage] = []
        for index, score in enumerate(scores):
            chunk = self._chunks[index]
            if filters and filters.document_id and chunk.document_id != filters.document_id:
                continue
            if filters and filters.section_path and chunk.section_path != filters.section_path:
                continue
            if float(score) <= 0:
                continue
            candidates.append(
                RetrievedPassage(
                    chunk=chunk,
                    score=float(score),
                    source_facet=source_facet,
                    rank=0,
                    retriever="sparse" if not filters else "scoped",
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        ranked = [item.model_copy(update={"rank": idx + 1}) for idx, item in enumerate(candidates[:top_k])]
        gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else None
        return ranked, {
            "retriever": "sparse",
            "result_count": len(ranked),
            "fallback_used": False,
            "latency_ms": elapsed_ms(start),
            "top_score_gap": gap,
            "provider": "rank-bm25",
        }


class FaissVectorRetriever(Retriever):
    def __init__(
        self,
        *,
        faiss_index_path: Path,
        metadata_path: Path,
        chunk_lookup: dict[str, ChunkObject],
        embedding_client: EmbeddingClient,
        embedding_model_name: str,
        fallback: Retriever | None = None,
    ) -> None:
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.chunk_lookup = chunk_lookup
        self.embedding_client = embedding_client
        self.embedding_model_name = embedding_model_name
        self.fallback = fallback
        self._index = None
        self._chunk_ids: list[str] = []
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            bundle = load_faiss_bundle(self.faiss_index_path, self.metadata_path)
            if bundle is None:
                return
            self._index, metadata = bundle
            self._chunk_ids = [str(chunk_id) for chunk_id in metadata.get("chunk_ids", [])]
        except Exception as exc:
            logger.warning("faiss index unavailable", extra={"error": str(exc)})
            self._index = None
            self._chunk_ids = []

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        start = timed()
        self._load()
        if self._index is None or not self._chunk_ids:
            if self.fallback is not None:
                results, diagnostic = await self.fallback.search(query, top_k=top_k, source_facet=source_facet, filters=filters)
                diagnostic = diagnostic or {}
                diagnostic.setdefault("fallback_used", True)
                diagnostic.setdefault("latency_ms", elapsed_ms(start))
                diagnostic.setdefault("provider", "fallback-dense")
                return results, diagnostic
            return [], {"retriever": "dense", "result_count": 0, "fallback_used": True, "latency_ms": elapsed_ms(start), "provider": "faiss"}

        prepared_query = prepare_embedding_text(query, model_name=self.embedding_model_name, is_query=True)
        query_vector = np.asarray((await self.embedding_client.embed_texts([prepared_query]))[0], dtype="float32").reshape(1, -1)
        search_k = len(self._chunk_ids) if filters else min(max(top_k * 10, top_k), len(self._chunk_ids))
        scores, indices = self._index.search(query_vector, search_k)
        ranked: list[RetrievedPassage] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0 or index >= len(self._chunk_ids):
                continue
            chunk = self.chunk_lookup.get(self._chunk_ids[index])
            if chunk is None:
                continue
            if filters and filters.document_id and chunk.document_id != filters.document_id:
                continue
            if filters and filters.section_path and chunk.section_path != filters.section_path:
                continue
            ranked.append(
                RetrievedPassage(
                    chunk=chunk,
                    score=float(score),
                    source_facet=source_facet,
                    rank=rank,
                    retriever="dense" if not filters else "scoped",
                )
            )
            if len(ranked) >= top_k:
                break

        gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else None
        return ranked, {
            "retriever": "dense",
            "result_count": len(ranked),
            "fallback_used": False,
            "latency_ms": elapsed_ms(start),
            "top_score_gap": gap,
            "provider": "faiss",
        }


class LocalCrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        fallback: Reranker | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fallback = fallback or SimpleCrossEncoderReranker()
        self._model = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=resolve_torch_device(self.device))
            except Exception as exc:
                logger.warning("local reranker unavailable", extra={"error": str(exc), "model": self.model_name})
                self._model = None

    def _predict_sync(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        assert self._model is not None
        with self._inference_lock:
            scores = self._model.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=False)
            return [float(score) for score in np.asarray(scores).reshape(-1)]

    async def rerank(self, query: str, candidates: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        self._load()
        if self._model is None:
            return await self.fallback.rerank(query, candidates)
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = await asyncio.to_thread(self._predict_sync, pairs)
        reranked = [
            candidate.model_copy(update={"score": score, "retriever": "reranked"})
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


class LocalMNLIClient(NLIClient):
    # When the argmax lands on "neutral" but the raw entailment probability is
    # at or above this threshold, we still treat the pair as SUPPORTED.  This
    # handles paraphrased or slightly reworded sentences where the model splits
    # probability mass between entailment and neutral without a clear winner.
    ENTAILMENT_SOFT_THRESHOLD = 0.35

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        fallback: NLIClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fallback = fallback or NullConfidenceNLIClient()
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._device = resolve_torch_device(self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.to(torch.device(self._device))
                self._model.eval()
            except Exception as exc:
                logger.warning("local nli model unavailable", extra={"error": str(exc), "model": self.model_name})
                self._model = None
                self._tokenizer = None

    def _score_sync(self, sentence_text: str, chunk_texts: Sequence[str]) -> list[ScoredReference]:
        import torch

        assert self._model is not None
        assert self._tokenizer is not None

        labels = {int(index): str(label).lower() for index, label in self._model.config.id2label.items()}
        entail_idx = next((idx for idx, label in labels.items() if "entail" in label), None)
        neutral_idx = next((idx for idx, label in labels.items() if "neutral" in label), None)
        contradiction_idx = next((idx for idx, label in labels.items() if "contrad" in label), None)
        if entail_idx is None or neutral_idx is None or contradiction_idx is None:
            raise RuntimeError(f"Unexpected label mapping for NLI model: {labels}")

        scored: list[ScoredReference] = []
        device = torch.device(self._device or resolve_torch_device(self.device))
        with self._inference_lock:
            for start in range(0, len(chunk_texts), self.batch_size):
                batch = list(chunk_texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch,
                    [sentence_text] * len(batch),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predicted = probabilities.argmax(axis=-1)
                for row, predicted_idx in zip(probabilities, predicted):
                    entailment_score = float(row[entail_idx])
                    if predicted_idx == entail_idx or entailment_score >= self.ENTAILMENT_SOFT_THRESHOLD:
                        label = ConfidenceLabel.SUPPORTED
                    elif predicted_idx == neutral_idx:
                        label = ConfidenceLabel.PARTIALLY_SUPPORTED
                    else:
                        label = ConfidenceLabel.NOT_SUPPORTED
                    scored.append(ScoredReference(score=entailment_score, label=label))
            return scored

    async def score(self, sentence_text: str, chunk_texts: Sequence[str]) -> list[ScoredReference]:
        self._load()
        if self._model is None or self._tokenizer is None:
            return await self.fallback.score(sentence_text, chunk_texts)
        try:
            return await asyncio.to_thread(self._score_sync, sentence_text, chunk_texts)
        except Exception as exc:
            logger.warning("local nli scoring fell back to null-confidence mode", extra={"error": str(exc)})
            return await self.fallback.score(sentence_text, chunk_texts)
