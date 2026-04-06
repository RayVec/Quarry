from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Protocol, Sequence

import httpx

from quarry.adapters.in_memory import (
    ConservativeFallbackGenerationClient,
    DeterministicGenerationClient,
    HashEmbeddingClient,
    HeuristicDecompositionClient,
    HeuristicMetadataEnricher,
    HeuristicNLIClient,
    SemanticDenseRetriever,
    SimpleCrossEncoderReranker,
)
from quarry.adapters.interfaces import DecompositionClient, EmbeddingClient, GenerationClient, MetadataEnricher, NLIClient, Reranker, Retriever
from quarry.adapters.local_models import (
    FaissVectorRetriever,
    LocalBM25Retriever,
    LocalCrossEncoderReranker,
    LocalMNLIClient,
    LocalSentenceTransformerEmbeddingClient,
    LocalStructuredDecompositionClient,
    LocalStructuredGenerationClient,
    LocalStructuredMetadataEnricher,
    LocalTextCompletionBackend,
    NullConfidenceNLIClient,
)
from quarry.adapters.mlx_runtime import (
    AppleMLXModelManager,
    MLXStructuredDecompositionClient,
    MLXStructuredGenerationClient,
    MLXStructuredMetadataEnricher,
    MLXTextCompletionBackend,
)
from quarry.config import Settings, is_local_component_ready
from quarry.domain.models import ChunkObject, ConfidenceLabel, GenerationRequest, RetrievalFilters, RetrievedPassage, RuntimeMode, RuntimeProfile, ScoredReference
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed
from quarry.model_cache import configure_model_cache
from quarry.prompts import (
    SHARED_SYSTEM_PROMPT,
    decomposition_classification_prompt,
    decomposition_prompt,
    generation_prompt,
    metadata_enrichment_prompt,
    parse_json_response,
    with_shared_system_prompt,
)
from quarry.retries import with_retries


logger = logger_with_trace(__name__)


@dataclass(slots=True)
class RuntimeClientProfile:
    generation_provider: str
    parser_provider: str
    runtime_mode: RuntimeMode
    runtime_profile: RuntimeProfile
    local_model_status: dict[str, str]
    active_model_ids: list[str]


class CompletionLLM(Protocol):
    async def complete(self, prompt: str, *, temperature: float = 0.1, operation: str = "completion") -> str:
        ...


class OpenAICompatibleLLM:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete(self, prompt: str, *, temperature: float = 0.1, operation: str = "completion") -> str:
        logger.info(
            "hosted llm request started",
            extra={
                "operation": operation,
                "provider": self.base_url,
                "model": self.model,
                "temperature": temperature,
                "prompt": prompt,
                "console_visible": True,
            },
        )
        start = timed()

        async def request_operation() -> str:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
                return payload["choices"][0]["message"]["content"]

        try:
            raw = await with_retries(request_operation)
        except Exception as exc:
            logger.exception(
                "hosted llm request failed",
                extra={
                    "operation": operation,
                    "provider": self.base_url,
                    "model": self.model,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "latency_ms": elapsed_ms(start),
                    "console_visible": True,
                },
            )
            raise
        logger.info(
            "hosted llm request completed",
            extra={
                "operation": operation,
                "provider": self.base_url,
                "model": self.model,
                "response": raw,
                "latency_ms": elapsed_ms(start),
                "console_visible": True,
            },
        )
        return raw


class GeminiLLM:
    def __init__(self, *, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def complete(self, prompt: str, *, temperature: float = 0.1, operation: str = "completion") -> str:
        logger.info(
            "gemini llm request started",
            extra={
                "operation": operation,
                "provider": "gemini",
                "model": self.model,
                "api_key_configured": bool(self.api_key),
                "temperature": temperature,
                "prompt": prompt,
                "console_visible": True,
            },
        )
        start = timed()

        async def request_operation() -> str:
            def invoke() -> str:
                try:
                    from google import genai
                except Exception as exc:
                    raise RuntimeError(
                        "google-genai is required for hosted.provider=gemini. "
                        "Install dependencies with `pip install -e \".[local]\"` "
                        "or `pip install -U google-genai`."
                    ) from exc

                client_kwargs: dict[str, str] = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                client = genai.Client(**client_kwargs)
                response = client.models.generate_content(
                    model=self.model,
                    contents=with_shared_system_prompt(prompt),
                    config={"temperature": temperature},
                )
                text = getattr(response, "text", None)
                if isinstance(text, str) and text.strip():
                    return text
                if text is not None:
                    rendered = str(text).strip()
                    if rendered:
                        return rendered
                raise RuntimeError("Gemini response did not include text content.")

            return await asyncio.to_thread(invoke)

        try:
            raw = await with_retries(request_operation)
        except Exception as exc:
            logger.exception(
                "gemini llm request failed",
                extra={
                    "operation": operation,
                    "provider": "gemini",
                    "model": self.model,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "latency_ms": elapsed_ms(start),
                    "console_visible": True,
                },
            )
            raise
        logger.info(
            "gemini llm request completed",
            extra={
                "operation": operation,
                "provider": "gemini",
                "model": self.model,
                "response": raw,
                "latency_ms": elapsed_ms(start),
                "console_visible": True,
            },
        )
        return raw


class HostedQueryDecompositionClient(DecompositionClient):
    def __init__(self, llm: OpenAICompatibleLLM, fallback: DecompositionClient | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicDecompositionClient()

    async def classify_query(self, query: str) -> str:
        try:
            raw = await self.llm.complete(decomposition_classification_prompt(query), operation="query_classification")
            return str(parse_json_response(raw)["query_type"])
        except Exception:
            logger.warning("classification fell back to heuristic")
            return await self.fallback.classify_query(query)

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        try:
            raw = await self.llm.complete(decomposition_prompt(query, max_facets), operation="query_decomposition")
            payload = parse_json_response(raw)
            facets = [str(item) for item in payload.get("facets", []) if str(item).strip()]
            return facets[:max_facets] or [query]
        except Exception:
            logger.warning("decomposition fell back to heuristic")
            return await self.fallback.decompose_query(query, max_facets)


class HostedMetadataEnricher(MetadataEnricher):
    def __init__(self, llm: OpenAICompatibleLLM, fallback: MetadataEnricher | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicMetadataEnricher()

    async def enrich(self, chunk: ChunkObject) -> ChunkObject:
        try:
            raw = await self.llm.complete(metadata_enrichment_prompt(chunk), operation="metadata_enrichment")
            payload = parse_json_response(raw)
            return chunk.model_copy(
                update={
                    "metadata_summary": str(payload.get("summary", chunk.metadata_summary)),
                    "metadata_entities": [str(item) for item in payload.get("entities", [])],
                    "metadata_questions": [str(item) for item in payload.get("questions", [])],
                }
            )
        except Exception:
            logger.warning("metadata enrichment fell back to heuristic")
            return await self.fallback.enrich(chunk)


class HostedGenerationClient(GenerationClient):
    def __init__(self, llm: CompletionLLM, fallback: GenerationClient | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or ConservativeFallbackGenerationClient()

    async def generate(self, request: GenerationRequest) -> str:
        try:
            return await self.llm.complete(generation_prompt(request), temperature=0.2, operation=f"generation:{request.mode}")
        except Exception as exc:
            logger.exception(
                "hosted generation failed; falling back to conservative no-ref implementation",
                extra={
                    "mode": request.mode,
                    "provider_class": self.llm.__class__.__name__,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "console_visible": True,
                },
            )
            return await self.fallback.generate(request)


def build_metadata_enricher(
    settings: Settings,
    *,
    text_backend: LocalTextCompletionBackend | None = None,
) -> MetadataEnricher:
    configure_model_cache(settings)
    if settings.use_live_metadata_enrichment and settings.has_live_llm_credentials:
        llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        return HostedMetadataEnricher(llm)
    if settings.use_local_models:
        if settings.uses_mlx_profile and (settings.runtime_mode == RuntimeMode.LOCAL.value or is_local_component_ready(settings, "text")):
            backend = MLXTextCompletionBackend(
                settings.mlx_text_model_name,
                model_manager=AppleMLXModelManager(),
                default_max_new_tokens=min(settings.mlx_max_new_tokens, 384),
            )
            return MLXStructuredMetadataEnricher(
                backend,
                fallback=HeuristicMetadataEnricher() if settings.runtime_mode != RuntimeMode.LOCAL.value else None,
            )
        if settings.uses_mlx_profile:
            return HeuristicMetadataEnricher()
        backend = text_backend or LocalTextCompletionBackend(
            settings.local_text_model_name,
            device=settings.local_model_device,
            dtype=settings.local_text_dtype,
            default_max_new_tokens=min(settings.local_text_max_new_tokens, 384),
        )
        return LocalStructuredMetadataEnricher(backend)
    return HeuristicMetadataEnricher()


class HostedEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        fallback: EmbeddingClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.fallback = fallback or HashEmbeddingClient(dimensions=dimensions)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        async def operation() -> list[list[float]]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": list(texts)},
                )
                response.raise_for_status()
                payload = response.json()
                return [item["embedding"] for item in payload["data"]]

        try:
            return await with_retries(operation)
        except Exception:
            logger.warning("embedding client fell back to hash embeddings")
            return await self.fallback.embed_texts(texts)


class LocalVectorRetriever(Retriever):
    def __init__(
        self,
        *,
        vector_index_path: Path,
        chunk_lookup: dict[str, ChunkObject],
        embedding_client: EmbeddingClient,
    ) -> None:
        self.vector_index_path = vector_index_path
        self.chunk_lookup = chunk_lookup
        self.embedding_client = embedding_client
        self._vectors = self._load_vectors()

    def _load_vectors(self) -> list[tuple[str, list[float]]]:
        if not self.vector_index_path.exists():
            return []
        payload = json.loads(self.vector_index_path.read_text())
        return [(entry["chunk_id"], [float(value) for value in entry["vector"]]) for entry in payload.get("vectors", [])]

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        start = timed()
        if not self._vectors:
            return [], {"retriever": "dense", "result_count": 0, "fallback_used": False, "latency_ms": elapsed_ms(start), "provider": "legacy-vector-json"}
        query_vector = (await self.embedding_client.embed_texts([query]))[0]
        scored: list[RetrievedPassage] = []
        for chunk_id, vector in self._vectors:
            chunk = self.chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            if filters and filters.document_id and chunk.document_id != filters.document_id:
                continue
            if filters and filters.section_path and chunk.section_path != filters.section_path:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector))
            if score <= 0:
                continue
            scored.append(
                RetrievedPassage(
                    chunk=chunk,
                    score=score,
                    source_facet=source_facet,
                    rank=0,
                    retriever="dense" if not filters else "scoped",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        ranked = [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(scored[:top_k])]
        gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else None
        return ranked, {
            "retriever": "dense",
            "result_count": len(ranked),
            "fallback_used": False,
            "latency_ms": elapsed_ms(start),
            "top_score_gap": gap,
            "provider": "legacy-vector-json",
        }


def build_runtime_clients(
    settings: Settings,
    chunk_lookup: dict[str, ChunkObject],
    artifacts_dir: Path,
) -> tuple[DecompositionClient, GenerationClient, Retriever, Retriever, Reranker, NLIClient, RuntimeClientProfile]:
    configure_model_cache(settings)
    use_local_models = getattr(settings, "use_local_models", True)
    try:
        runtime_mode = RuntimeMode(settings.runtime_mode)
    except Exception:
        runtime_mode = RuntimeMode.HYBRID if use_local_models else RuntimeMode.HOSTED
    try:
        runtime_profile = RuntimeProfile(settings.runtime_profile)
    except Exception:
        runtime_profile = RuntimeProfile.APPLE_LITE_MLX if settings.uses_mlx_profile else RuntimeProfile.FULL_LOCAL_TRANSFORMERS
    vector_metadata_path = artifacts_dir / "vector_index_metadata.json"
    indexed_embedding_model = None
    if vector_metadata_path.exists():
        try:
            indexed_embedding_model = json.loads(vector_metadata_path.read_text()).get("embedding_model")
        except Exception:
            indexed_embedding_model = None

    local_status: dict[str, str] = {
        "runtime_profile": settings.runtime_profile,
        "decomposition": "disabled",
        "generation": "disabled",
        "metadata": "disabled",
        "text": "disabled",
        "embedding": "disabled",
        "reranker": "disabled",
        "nli": "disabled",
        "parser": "disabled",
        "parser_provider": settings.parser_provider,
    }

    text_backend: LocalTextCompletionBackend | None = None
    mlx_backend: MLXTextCompletionBackend | None = None
    needs_openai_llm = settings.has_live_llm_credentials and (
        (settings.use_live_generation and settings.llm_provider == "openai_compatible")
        or settings.use_live_decomposition
        or settings.use_live_metadata_enrichment
    )
    live_openai_llm = (
        OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        if needs_openai_llm
        else None
    )
    live_generation_llm: CompletionLLM | None
    if settings.use_live_generation and settings.has_live_generation_credentials:
        if settings.llm_provider == "gemini":
            live_generation_llm = GeminiLLM(api_key=settings.llm_api_key, model=settings.llm_model)
        else:
            live_generation_llm = live_openai_llm or OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
            )
    else:
        live_generation_llm = None

    logger.info(
        "hosted generation configuration",
        extra={
            "runtime_mode": runtime_mode.value,
            "runtime_profile": runtime_profile.value,
            "use_live_generation": settings.use_live_generation,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_base_url_configured": bool(settings.llm_base_url),
            "llm_api_key_configured": bool(settings.llm_api_key),
            "live_generation_enabled": live_generation_llm is not None,
            "local_model_device": getattr(settings, "local_model_device", "auto"),
            "console_visible": True,
        },
    )
    if settings.llm_provider == "gemini" and settings.llm_base_url:
        logger.info(
            "hosted llm base_url is ignored when provider=gemini",
            extra={
                "llm_base_url": settings.llm_base_url,
                "console_visible": True,
            },
        )
    if settings.use_live_generation and live_generation_llm is None:
        if settings.llm_provider == "gemini":
            logger.warning(
                "hosted generation is enabled but gemini credentials are missing; falling back to local/deterministic generation",
                extra={
                    "missing": ["hosted.llm_api_key or GEMINI_API_KEY"],
                    "console_visible": True,
                },
            )
        else:
            missing_fields: list[str] = []
            if not settings.llm_base_url:
                missing_fields.append("hosted.llm_base_url")
            if not settings.llm_api_key:
                missing_fields.append("hosted.llm_api_key")
            logger.warning(
                "hosted generation is enabled but OpenAI-compatible credentials are incomplete; falling back to local/deterministic generation",
                extra={
                    "missing": missing_fields,
                    "console_visible": True,
                },
            )

    def ensure_local_text_backend() -> LocalTextCompletionBackend:
        nonlocal text_backend
        if text_backend is None:
            text_backend = LocalTextCompletionBackend(
                settings.local_text_model_name,
                device=settings.local_model_device,
                dtype=settings.local_text_dtype,
                default_max_new_tokens=settings.local_text_max_new_tokens,
            )
        return text_backend

    def ensure_mlx_backend() -> MLXTextCompletionBackend:
        nonlocal mlx_backend
        if mlx_backend is None:
            mlx_backend = MLXTextCompletionBackend(
                settings.mlx_text_model_name,
                model_manager=AppleMLXModelManager(),
                default_max_new_tokens=settings.mlx_max_new_tokens,
            )
        return mlx_backend

    if settings.use_live_decomposition and live_openai_llm is not None:
        decomposition = HostedQueryDecompositionClient(live_openai_llm)
        local_status["decomposition"] = "hosted"
    elif use_local_models:
        if settings.uses_mlx_profile and (runtime_mode == RuntimeMode.LOCAL or is_local_component_ready(settings, "text")):
            decomposition = MLXStructuredDecompositionClient(
                ensure_mlx_backend(),
                fallback=HeuristicDecompositionClient() if runtime_mode == RuntimeMode.HYBRID else None,
            )
            local_status["decomposition"] = "configured"
            local_status["parser"] = "configured"
        elif settings.uses_mlx_profile:
            decomposition = HeuristicDecompositionClient()
            local_status["decomposition"] = "heuristic"
            local_status["parser"] = "heuristic"
        else:
            decomposition = LocalStructuredDecompositionClient(ensure_local_text_backend())
            local_status["decomposition"] = "configured"
            local_status["parser"] = "configured"
    else:
        decomposition = HeuristicDecompositionClient()
        local_status["decomposition"] = "heuristic"
        local_status["parser"] = "heuristic"

    if settings.use_live_generation and live_generation_llm is not None:
        generation = HostedGenerationClient(live_generation_llm)
        generation_provider = f"hosted:{settings.llm_model}"
        local_status["generation"] = "hosted"
    elif use_local_models:
        if settings.uses_mlx_profile and (runtime_mode == RuntimeMode.LOCAL or is_local_component_ready(settings, "text")):
            generation = MLXStructuredGenerationClient(
                ensure_mlx_backend(),
                fallback=ConservativeFallbackGenerationClient() if runtime_mode == RuntimeMode.HYBRID else None,
            )
            generation_provider = f"mlx:{settings.mlx_text_model_name}"
            local_status["generation"] = "configured"
            local_status["parser"] = "configured"
        elif settings.uses_mlx_profile:
            generation = HostedGenerationClient(live_generation_llm) if live_generation_llm is not None else DeterministicGenerationClient()
            generation_provider = f"hosted:{settings.llm_model}" if live_generation_llm is not None else "fallback:deterministic"
            local_status["generation"] = "hosted" if live_generation_llm is not None else "heuristic"
            local_status["parser"] = "heuristic"
        else:
            generation = LocalStructuredGenerationClient(ensure_local_text_backend())
            generation_provider = f"local:{settings.local_text_model_name}"
            local_status["generation"] = "configured"
            local_status["parser"] = "configured"
    else:
        generation = HostedGenerationClient(live_generation_llm) if live_generation_llm is not None else DeterministicGenerationClient()
        generation_provider = f"hosted:{settings.llm_model}" if live_generation_llm is not None else "fallback:deterministic"
        local_status["generation"] = "hosted" if live_generation_llm is not None else "heuristic"
        local_status["parser"] = "heuristic"

    if settings.use_live_metadata_enrichment and live_openai_llm is not None:
        local_status["metadata"] = "hosted"
    elif use_local_models:
        if settings.uses_mlx_profile and (runtime_mode == RuntimeMode.LOCAL or is_local_component_ready(settings, "text")):
            local_status["metadata"] = "configured"
        elif settings.uses_mlx_profile:
            local_status["metadata"] = "heuristic"
        else:
            local_status["metadata"] = "configured"
    else:
        local_status["metadata"] = "heuristic"

    if local_status["generation"] == local_status["decomposition"]:
        local_status["text"] = local_status["generation"]
    else:
        local_status["text"] = "mixed"

    if settings.use_live_embeddings and settings.embedding_base_url and settings.embedding_api_key:
        embedding_client: EmbeddingClient = HostedEmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        local_status["embedding"] = "hosted"
    elif use_local_models:
        embedding_client = LocalSentenceTransformerEmbeddingClient(
            getattr(settings, "local_embedding_model", "intfloat/e5-large-v2"),
            device=getattr(settings, "local_model_device", "auto"),
            fallback=HashEmbeddingClient(dimensions=settings.embedding_dimensions),
        )
        local_status["embedding"] = "configured"
    else:
        embedding_client = HashEmbeddingClient(dimensions=settings.embedding_dimensions)
        local_status["embedding"] = "heuristic"

    bm25_retriever = LocalBM25Retriever(list(chunk_lookup.values()))
    dense_fallback: Retriever = (
        LocalVectorRetriever(
            vector_index_path=artifacts_dir / "vector_index.json",
            chunk_lookup=chunk_lookup,
            embedding_client=embedding_client,
        )
        if (artifacts_dir / "vector_index.json").exists()
        else SemanticDenseRetriever(type("FallbackStore", (), {"all_chunks": lambda s: list(chunk_lookup.values())})())  # type: ignore[arg-type]
    )
    dense = FaissVectorRetriever(
        faiss_index_path=artifacts_dir / "vector_index.faiss",
        metadata_path=vector_metadata_path,
        chunk_lookup=chunk_lookup,
        embedding_client=embedding_client,
        embedding_model_name=str(
            indexed_embedding_model
            or (
                settings.embedding_model
                if settings.use_live_embeddings and settings.embedding_base_url and settings.embedding_api_key
                else getattr(settings, "local_embedding_model", settings.embedding_model)
            )
        ),
        fallback=dense_fallback,
    )
    sparse = bm25_retriever
    reranker = (
        LocalCrossEncoderReranker(
            getattr(settings, "local_reranker_model", "BAAI/bge-reranker-v2-m3"),
            device=getattr(settings, "local_model_device", "auto"),
            fallback=SimpleCrossEncoderReranker(),
        )
        if use_local_models
        else SimpleCrossEncoderReranker()
    )
    local_status["reranker"] = "configured" if use_local_models else "heuristic"
    nli_client = (
        LocalMNLIClient(
            settings.nli_model_name,
            device=getattr(settings, "local_model_device", "auto"),
            fallback=NullConfidenceNLIClient(),
        )
        if use_local_models
        else HeuristicNLIClient()
    )
    local_status["nli"] = "configured" if use_local_models else "heuristic"
    profile = RuntimeClientProfile(
        generation_provider=generation_provider,
        parser_provider=settings.parser_provider,
        runtime_mode=runtime_mode,
        runtime_profile=runtime_profile,
        local_model_status=local_status,
        active_model_ids=settings.active_model_ids,
    )
    return decomposition, generation, sparse, dense, reranker, nli_client, profile
