from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from quarry.adapters.in_memory import InMemoryChunkStore
from quarry.adapters.production import build_runtime_clients
from quarry.config import Settings
from quarry.logging_utils import configure_logging, elapsed_ms, logger_with_trace, start_trace, timed
from quarry.model_cache import configure_model_cache
from quarry.pipeline.decomposition import QueryDecomposer
from quarry.pipeline.generation import AnswerGenerator, SentenceRegenerator
from quarry.pipeline.retrieval import HybridRetriever
from quarry.pipeline.verification import VerificationService
from quarry.services.pipeline_service import PipelineService
from quarry.services.session_store import SessionStore

from .routes import router


logger = logger_with_trace(__name__)


def _is_console_noisy_request(method: str, path: str) -> bool:
    if method != "GET":
        return False
    return path.startswith("/api/v1/sessions/")


def _ensure_runtime_ready(settings: Settings, local_model_status: dict[str, str]) -> None:
    if settings.runtime_mode != "local":
        return
    status_path = settings.artifacts_dir / "local_model_status.json"
    errors: list[str] = []
    required_components = ("embedding", "reranker", "nli", "decomposition", "generation", "parser")
    if not status_path.exists():
        errors.append("Run `quarry warm-local-models` before starting local mode.")
    else:
        warmed = json.loads(status_path.read_text())
        if warmed.get("runtime_profile") and warmed.get("runtime_profile") != settings.runtime_profile:
            errors.append("Warmup profile does not match the configured runtime profile.")
        for key in required_components:
            if not str(warmed.get(key, "")).startswith("ready:"):
                errors.append(f"Warmup is incomplete for {key}.")
        if warmed.get("parser_provider") and warmed.get("parser_provider") != settings.parser_provider:
            errors.append("Warmup parser provider does not match the configured runtime profile.")
    for key in required_components:
        status = local_model_status.get(key)
        if status in {"disabled", "heuristic", "hosted"}:
            errors.append(f"local mode cannot start with {key} in {status} mode.")
    if errors:
        raise RuntimeError(" ".join(errors))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_model_cache(settings)
    configure_logging(settings.artifacts_dir.parent / "logs", enable_file_logs=settings.trace_logs, category="runtime")
    artifacts_ready = (settings.artifacts_dir / "manifest.json").exists()
    corpus_root = settings.artifacts_dir if artifacts_ready else settings.corpus_dir
    chunk_store = InMemoryChunkStore.from_directory(corpus_root)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunk_store.all_chunks()}
    decomposition_client, generation_client, sparse_retriever, dense_retriever, reranker, nli_client, runtime_profile = build_runtime_clients(
        settings,
        chunk_lookup,
        settings.artifacts_dir,
    )
    _ensure_runtime_ready(settings, runtime_profile.local_model_status)

    pipeline_service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(decomposition_client, max_facets=settings.max_facets),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            reranker=reranker,
            sparse_top_k=settings.sparse_top_k,
            dense_top_k=settings.dense_top_k,
            rerank_top_k=settings.rerank_top_k,
            multihop_anchor_pool_size=settings.multihop_anchor_pool_size,
            multihop_rerank_budget=settings.multihop_rerank_budget,
            rrf_k=settings.retrieval_rrf_k,
        ),
        answer_generator=AnswerGenerator(generation_client),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=nli_client),
        session_store=SessionStore(),
        scoped_top_k=settings.scoped_retrieval_top_k,
        refinement_token_budget=settings.refinement_token_budget,
        ambiguity_gap_threshold=settings.ambiguity_gap_threshold,
        generation_provider=runtime_profile.generation_provider,
        parser_provider=runtime_profile.parser_provider,
        runtime_mode=runtime_profile.runtime_mode.value,
        runtime_profile=runtime_profile.runtime_profile.value,
        local_model_status=runtime_profile.local_model_status,
        active_model_ids=runtime_profile.active_model_ids,
    )

    app = FastAPI(title=settings.app_name, version="0.2.0")
    app.state.query_tasks: set[asyncio.Task] = set()

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        request_start = timed()
        trace_id = start_trace()
        console_visible = not _is_console_noisy_request(request.method, request.url.path)
        logger.info(
            "http request started",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "query_string": request.url.query,
                "console_visible": console_visible,
            },
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "http request failed",
                extra={
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                    "latency_ms": elapsed_ms(request_start),
                    "console_visible": True,
                },
            )
            raise
        logger.info(
            "http request completed",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": elapsed_ms(request_start),
                "console_visible": console_visible,
            },
        )
        response.headers["X-Trace-Id"] = trace_id
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin, "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.pipeline_service = pipeline_service
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("quarry.api.app:app", host="127.0.0.1", port=8000, reload=False, access_log=False)
