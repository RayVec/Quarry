from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from quarry.adapters.in_memory import HashEmbeddingClient
from quarry.adapters.local_models import FaissIndexBundle, LocalSentenceTransformerEmbeddingClient, prepare_embedding_text, write_faiss_bundle
from quarry.adapters.production import HostedEmbeddingClient, build_metadata_enricher
from quarry.config import Settings
from quarry.domain.models import ChunkObject, CorpusManifest, DocumentArtifactSummary, ParsedDocument, StructuralIndexEntry
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed


logger = logger_with_trace(__name__)


def _metadata_provider_name(enricher: object) -> str:
    backend = getattr(enricher, "backend", None)
    llm = getattr(enricher, "llm", None)
    if backend is not None and getattr(backend, "model_name", None):
        return str(getattr(backend, "model_name"))
    if llm is not None and getattr(llm, "model", None):
        return str(getattr(llm, "model"))
    return "heuristic"


async def enrich_chunks(
    chunks: list[ChunkObject],
    settings: Settings,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[ChunkObject]:
    enricher = build_metadata_enricher(settings)
    provider_name = _metadata_provider_name(enricher)
    total = len(chunks)
    if total == 0:
        return []

    batch_size = 25
    enriched: list[ChunkObject] = []
    for start_index in range(0, total, batch_size):
        end_index = min(start_index + batch_size, total)
        batch_number = (start_index // batch_size) + 1
        batch_count = (total + batch_size - 1) // batch_size
        if progress is not None:
            progress(
                f"metadata enrichment chunks {start_index + 1}-{end_index} / {total}"
                f" with {provider_name} (batch={batch_number}/{batch_count}, remaining_chunks={max(total - end_index, 0)})"
            )
        batch = chunks[start_index:end_index]
        enriched.extend([await enricher.enrich(chunk) for chunk in batch])
    return enriched


async def build_vector_index(
    chunks: list[ChunkObject],
    settings: Settings,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    stage_start = timed()
    base_texts = [f"{chunk.text}\nQuestions: {' | '.join(chunk.metadata_questions)}" for chunk in chunks]
    if settings.use_live_embeddings and settings.embedding_base_url and settings.embedding_api_key:
        embedder = HostedEmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        texts = list(base_texts)
        embedding_model = settings.embedding_model
    elif settings.use_local_models:
        embedding_model = settings.local_embedding_model
        texts = [prepare_embedding_text(text, model_name=embedding_model, is_query=False) for text in base_texts]
        embedder = LocalSentenceTransformerEmbeddingClient(
            embedding_model,
            device=settings.local_model_device,
            fallback=HashEmbeddingClient(dimensions=settings.embedding_dimensions),
        )
    else:
        embedder = HashEmbeddingClient(dimensions=settings.embedding_dimensions)
        texts = list(base_texts)
        embedding_model = settings.embedding_model
    if progress is not None:
        progress(
            f"vector embedding {len(texts)} chunk(s) with {embedding_model}"
            f" (remaining_chunks={len(texts)})"
        )
    vectors = await embedder.embed_texts(texts)
    if settings.use_local_models and isinstance(embedder, LocalSentenceTransformerEmbeddingClient):
        embedding_model = embedder.active_model_name
    dimensions = len(vectors[0]) if vectors else settings.embedding_dimensions
    logger.info(
        "vector index embeddings complete",
        extra={
            "chunk_count": len(chunks),
            "embedding_model": embedding_model,
            "dimensions": dimensions,
            "latency_ms": elapsed_ms(stage_start),
        },
    )
    if progress is not None:
        progress(
            f"vector embedding complete for {len(texts)} chunk(s) with {embedding_model}"
            f" (remaining_chunks=0)"
        )
    return {
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "vectors": [{"chunk_id": chunk.chunk_id, "vector": vector} for chunk, vector in zip(chunks, vectors)],
    }


def write_artifacts(
    *,
    settings: Settings,
    parsed_documents: list[ParsedDocument],
    document_chunks: dict[str, list[ChunkObject]],
    structural_index: list[StructuralIndexEntry],
    vector_index: dict[str, object],
    progress: Callable[[str], None] | None = None,
) -> CorpusManifest:
    stage_start = timed()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = settings.artifacts_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[DocumentArtifactSummary] = []

    all_chunks = []
    for parsed_document in parsed_documents:
        document_dir = documents_dir / parsed_document.document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        parsed_document_path = document_dir / "parsed_document.json"
        chunks_path = document_dir / "chunks.json"
        parsed_document_path.write_text(parsed_document.model_dump_json(indent=2))
        chunks = document_chunks[parsed_document.document_id]
        chunks_path.write_text(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
        all_chunks.extend(chunks)
        summaries.append(
            DocumentArtifactSummary(
                document_id=parsed_document.document_id,
                document_title=parsed_document.document_title,
                parsed_document_path=str(parsed_document_path),
                chunks_path=str(chunks_path),
                chunk_count=len(chunks),
            )
        )

    structural_index_path = settings.artifacts_dir / "structural_index.json"
    vector_index_path = settings.artifacts_dir / "vector_index.faiss"
    vector_metadata_path = settings.artifacts_dir / "vector_index_metadata.json"
    legacy_vector_json_path = settings.artifacts_dir / "vector_index.json"
    if progress is not None:
        progress(f"writing artifacts for {len(parsed_documents)} document(s) into {settings.artifacts_dir}")
    structural_index_path.write_text(json.dumps([entry.model_dump() for entry in structural_index], indent=2))
    legacy_vector_json_path.write_text(json.dumps(vector_index, indent=2))
    write_faiss_bundle(
        FaissIndexBundle(
            embedding_model=str(vector_index["embedding_model"]),
            dimensions=int(vector_index["dimensions"]),
            chunk_ids=[entry["chunk_id"] for entry in vector_index.get("vectors", [])],
            vectors=[[float(value) for value in entry["vector"]] for entry in vector_index.get("vectors", [])],
        ),
        index_path=vector_index_path,
        metadata_path=vector_metadata_path,
    )
    local_model_status_path = settings.artifacts_dir / "local_model_status.json"

    manifest = CorpusManifest(
        corpus_id="quarry-local-corpus",
        embedding_model=str(vector_index["embedding_model"]),
        embedding_dimensions=int(vector_index["dimensions"]),
        vector_index_path=str(vector_index_path),
        vector_metadata_path=str(vector_metadata_path),
        structural_index_path=str(structural_index_path),
        local_model_status_path=str(local_model_status_path) if local_model_status_path.exists() else None,
        runtime_profile=settings.runtime_profile,
        parser_provider=settings.parser_provider,
        active_model_ids=settings.active_model_ids,
        documents=summaries,
        chunk_count=len(all_chunks),
    )
    (settings.artifacts_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    logger.info(
        "artifact write complete",
        extra={
            "document_count": len(parsed_documents),
            "chunk_count": len(all_chunks),
            "artifacts_dir": str(settings.artifacts_dir),
            "latency_ms": elapsed_ms(stage_start),
        },
    )
    return manifest
