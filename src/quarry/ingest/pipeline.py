from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Callable

from quarry.adapters.local_models import (
    FaissIndexBundle,
    LocalCrossEncoderReranker,
    LocalMNLIClient,
    LocalSentenceTransformerEmbeddingClient,
    LocalTextCompletionBackend,
    write_faiss_bundle,
)
from quarry.adapters.mlx_runtime import AppleMLXModelManager, MLXTextCompletionBackend
from quarry.adapters.production import build_metadata_enricher
from quarry.domain.models import ChunkObject
from quarry.config import Settings, is_local_component_ready
from quarry.domain.models import ParsedDocument, RetrievedPassage
from quarry.ingest.chunking import chunk_document
from quarry.ingest.indexing import build_vector_index, enrich_chunks, write_artifacts
from quarry.ingest.parsers import (
    BasicTextParser,
    CascadingParserAdapter,
    MinerUPipelineParserAdapter,
    OlmOCRTransformersParserAdapter,
    ParsingPipeline,
    PyMuPDFTextParserAdapter,
    PyPDFTextParserAdapter,
    Qwen3VLMlxParserAdapter,
)
from quarry.model_cache import configure_model_cache
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed


logger = logger_with_trace(__name__)


def ensure_parser_ready(settings: Settings) -> None:
    if not settings.use_local_models or not settings.uses_mlx_profile:
        return
    if is_local_component_ready(settings, "parser"):
        return

    logger.info(
        "ingest parser warmup required before parsing",
        extra={"runtime_profile": settings.runtime_profile, "parser_provider": settings.parser_provider},
    )
    warm_local_models(settings)
    if not is_local_component_ready(settings, "parser"):
        raise RuntimeError(
            "The configured MLX parser is not ready. Run `quarry warm-local-models --profile apple_lite_mlx` "
            "and resolve parser warmup errors before ingesting documents."
        )


def build_parsing_pipeline(settings: Settings) -> ParsingPipeline:
    configure_model_cache(settings)
    if settings.uses_mlx_profile:
        primary = CascadingParserAdapter(
            parser_name="primary_chain",
            adapters=[build_parser_adapter("qwen3_vl_mlx", settings)] if settings.use_local_models else [BasicTextParser()],
        )
        fallback = CascadingParserAdapter(
            parser_name="fallback_chain",
            adapters=[build_parser_adapter(settings.parser_fallback, settings), PyPDFTextParserAdapter(), BasicTextParser()],
        )
        return ParsingPipeline(primary=primary, fallback=fallback)
    primary = CascadingParserAdapter(
        parser_name="primary_chain",
        adapters=[
            build_parser_adapter(settings.parser_primary, settings),
        ],
    )
    fallback = CascadingParserAdapter(
        parser_name="fallback_chain",
        adapters=[
            build_parser_adapter(settings.parser_fallback, settings),
            PyPDFTextParserAdapter(),
            BasicTextParser(),
        ],
    )
    return ParsingPipeline(primary=primary, fallback=fallback)


def build_parser_adapter(name: str, settings: Settings):
    normalized = name.strip().lower()
    if normalized in {"apple_lite_mlx", "qwen3_vl_mlx", "mlx_vlm"}:
        return Qwen3VLMlxParserAdapter(
            model_name=settings.mlx_vision_model_name,
            target_longest_image_dim=settings.mlx_page_image_dim,
            max_new_tokens=min(settings.mlx_max_new_tokens, 768),
            max_pdf_pages_per_batch=settings.mlx_max_pdf_pages_per_batch,
        )
    if normalized in {"olmocr", "olmocr_transformers"}:
        return OlmOCRTransformersParserAdapter(
            model_name=settings.olmocr_model_name,
            target_longest_image_dim=settings.parser_target_longest_image_dim,
        )
    if normalized in {"pymupdf", "pymupdf_text", "fitz_text"}:
        return PyMuPDFTextParserAdapter()
    if normalized in {"pypdf", "pypdf_text"}:
        return PyPDFTextParserAdapter()
    if normalized in {"mineru", "mineru_pipeline"}:
        return MinerUPipelineParserAdapter(
            backend=settings.mineru_backend,
            language=settings.mineru_language,
        )
    return BasicTextParser()


def ingest_documents(
    paths: list[str],
    settings: Settings,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    overall_start = timed()
    configure_model_cache(settings)
    ensure_parser_ready(settings)
    parsing_pipeline = build_parsing_pipeline(settings)
    parsed_documents: list[ParsedDocument] = []
    document_chunks: dict[str, list] = {}
    structural_index = []

    if progress is not None:
        progress(f"ingest starting for {len(paths)} source file(s)")

    for index, source_path in enumerate(paths, start=1):
        document_start = timed()
        source_name = Path(source_path).name
        documents_remaining = max(len(paths) - index, 0)
        if progress is not None:
            progress(f"[{index}/{len(paths)}] parsing {source_name} (documents_remaining={documents_remaining})")
        parsed_document = parsing_pipeline.parse(source_path)
        parsed_documents.append(parsed_document)
        if progress is not None:
            page_summary = ""
            if parsed_document.recovered_pages or parsed_document.skipped_pages:
                page_summary = (
                    f", recovered_pages={parsed_document.recovered_pages},"
                    f" skipped_pages={parsed_document.skipped_pages}"
                )
            progress(
                f"[{index}/{len(paths)}] parsed {source_name} with {parsed_document.parser_used}"
                f" ({len(parsed_document.sections)} section(s), fallback={parsed_document.fallback_used}{page_summary}, documents_remaining={documents_remaining})"
            )
            progress(f"[{index}/{len(paths)}] chunking {source_name}")
        chunks, document_structural_index = chunk_document(parsed_document)
        if progress is not None:
            progress(
                f"[{index}/{len(paths)}] built {len(chunks)} chunk(s) and {len(document_structural_index)} structural entry(ies) for {source_name}"
            )
            progress(f"[{index}/{len(paths)}] enriching metadata for {source_name}")
        enriched_chunks = asyncio.run(
            enrich_chunks(
                chunks,
                settings,
                progress=(lambda message, idx=index, total=len(paths), name=source_name: progress(f"[{idx}/{total}] {name}: {message}"))
                if progress is not None
                else None,
            )
        )
        document_chunks[parsed_document.document_id] = enriched_chunks
        structural_index.extend(document_structural_index)
        if progress is not None:
            progress(
                f"[{index}/{len(paths)}] finished {source_name} in {elapsed_ms(document_start)} ms"
                f" (documents_remaining={documents_remaining})"
            )
        logger.info(
            "document ingest complete",
            extra={
                "document_id": parsed_document.document_id,
                "source_path": source_path,
                "chunk_count": len(enriched_chunks),
                "structural_entries": len(document_structural_index),
                "parser_used": parsed_document.parser_used,
                "fallback_used": parsed_document.fallback_used,
                "recovered_pages": parsed_document.recovered_pages,
                "skipped_pages": parsed_document.skipped_pages,
                "latency_ms": elapsed_ms(document_start),
            },
        )

    all_chunks = [chunk for chunks in document_chunks.values() for chunk in chunks]
    if progress is not None:
        progress(f"building vector index for {len(all_chunks)} total chunk(s)")
    vector_index = asyncio.run(
        build_vector_index(
            all_chunks,
            settings,
            progress=progress,
        )
    )
    manifest = write_artifacts(
        settings=settings,
        parsed_documents=parsed_documents,
        document_chunks=document_chunks,
        structural_index=structural_index,
        vector_index=vector_index,
        progress=progress,
    )
    if progress is not None:
        progress(
            f"ingest complete: {len(parsed_documents)} document(s), {manifest.chunk_count} chunk(s), total {elapsed_ms(overall_start)} ms"
        )
    logger.info(
        "corpus ingest complete",
        extra={
            "document_count": len(parsed_documents),
            "chunk_count": manifest.chunk_count,
            "artifacts_dir": str(settings.artifacts_dir),
            "latency_ms": elapsed_ms(overall_start),
        },
    )
    return {"manifest": manifest}


def rebuild_indexes(settings: Settings) -> dict[str, object]:
    configure_model_cache(settings)
    manifest_path = settings.artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        return {"errors": ["No manifest found. Run ingest first."]}
    manifest = json.loads(manifest_path.read_text())
    all_chunks: list[ChunkObject] = []
    for document in manifest.get("documents", []):
        chunks_path = Path(document["chunks_path"])
        if chunks_path.exists():
            all_chunks.extend(ChunkObject.model_validate(item) for item in json.loads(chunks_path.read_text()))
    if not all_chunks:
        return {"errors": ["No chunk artifacts were found to rebuild indexes from."]}

    vector_index = asyncio.run(build_vector_index(all_chunks, settings))
    legacy_vector_path = settings.artifacts_dir / "vector_index.json"
    faiss_index_path = settings.artifacts_dir / "vector_index.faiss"
    vector_metadata_path = settings.artifacts_dir / "vector_index_metadata.json"

    legacy_vector_path.write_text(json.dumps(vector_index, indent=2))
    write_faiss_bundle(
        FaissIndexBundle(
            embedding_model=str(vector_index["embedding_model"]),
            dimensions=int(vector_index["dimensions"]),
            chunk_ids=[entry["chunk_id"] for entry in vector_index.get("vectors", [])],
            vectors=[[float(value) for value in entry["vector"]] for entry in vector_index.get("vectors", [])],
        ),
        index_path=faiss_index_path,
        metadata_path=vector_metadata_path,
    )
    manifest["embedding_model"] = vector_index["embedding_model"]
    manifest["embedding_dimensions"] = vector_index["dimensions"]
    manifest["vector_index_path"] = str(faiss_index_path)
    manifest["vector_metadata_path"] = str(vector_metadata_path)
    manifest["runtime_profile"] = settings.runtime_profile
    manifest["parser_provider"] = settings.parser_provider
    manifest["active_model_ids"] = settings.active_model_ids
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"rebuilt": True, "chunk_count": len(all_chunks)}


def warm_local_models(settings: Settings) -> dict[str, object]:
    stage_start = timed()
    configure_model_cache(settings)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    status_path = settings.artifacts_dir / "local_model_status.json"
    statuses: dict[str, str] = {}

    try:
        logger.info("warmup starting embedding model", extra={"model": settings.local_embedding_model})
        embedder = LocalSentenceTransformerEmbeddingClient(
            settings.local_embedding_model,
            device=settings.local_model_device,
        )
        asyncio.run(embedder.embed_texts(["passage: Warmup embedding sentence for QUARRY."]))
        statuses["embedding"] = f"ready:{embedder.active_model_name}"
    except Exception as exc:
        statuses["embedding"] = f"error:{exc}"

    try:
        logger.info("warmup starting reranker", extra={"model": settings.local_reranker_model})
        sample_chunk = ChunkObject(
            chunk_id="warmup-chunk",
            document_id="warmup",
            document_title="Warmup",
            text="Warmup passage about modular construction and procurement planning for local reranker initialization.",
            section_heading="Warmup",
            section_path="Warmup",
            page_start=1,
            page_end=1,
        )
        reranker = LocalCrossEncoderReranker(settings.local_reranker_model, device=settings.local_model_device)
        asyncio.run(
            reranker.rerank(
                "modular construction",
                [
                    RetrievedPassage(
                        chunk=sample_chunk,
                        score=1.0,
                        source_facet="warmup",
                        rank=1,
                        retriever="dense",
                    )
                ],
            )
        )
        statuses["reranker"] = f"ready:{settings.local_reranker_model}"
    except Exception as exc:
        statuses["reranker"] = f"error:{exc}"

    try:
        logger.info("warmup starting nli", extra={"model": settings.nli_model_name})
        nli = LocalMNLIClient(settings.nli_model_name, device=settings.local_model_device)
        asyncio.run(
            nli.score(
                "Modular construction reduced schedule risk.",
                ["Modular construction reduced schedule risk across observed projects."],
            )
        )
        statuses["nli"] = f"ready:{settings.nli_model_name}"
    except Exception as exc:
        statuses["nli"] = f"error:{exc}"

    try:
        logger.info(
            "warmup starting text backend",
            extra={"model": settings.mlx_text_model_name if settings.uses_mlx_profile else settings.local_text_model_name},
        )
        if settings.uses_mlx_profile:
            mlx_manager = AppleMLXModelManager()
            text_backend = MLXTextCompletionBackend(
                settings.mlx_text_model_name,
                model_manager=mlx_manager,
                default_max_new_tokens=min(settings.mlx_max_new_tokens, 64),
            )
        else:
            text_backend = LocalTextCompletionBackend(
                settings.local_text_model_name,
                device=settings.local_model_device,
                dtype=settings.local_text_dtype,
                default_max_new_tokens=min(settings.local_text_max_new_tokens, 64),
            )
        asyncio.run(text_backend.complete('Return JSON: {"ok": true}', max_new_tokens=32))
        model_name = settings.mlx_text_model_name if settings.uses_mlx_profile else settings.local_text_model_name
        statuses["text"] = f"ready:{model_name}"
        statuses["decomposition"] = f"ready:{model_name}"
        statuses["generation"] = f"ready:{model_name}"
    except Exception as exc:
        statuses["text"] = f"error:{exc}"
        statuses["decomposition"] = f"error:{exc}"
        statuses["generation"] = f"error:{exc}"

    try:
        logger.info("warmup starting metadata enricher")
        enricher = build_metadata_enricher(settings)
        asyncio.run(
            enricher.enrich(
                ChunkObject(
                    chunk_id="warmup-meta",
                    document_id="warmup",
                    document_title="Warmup",
                    text="Warmup chunk for metadata enrichment.",
                    section_heading="Warmup",
                    section_path="Warmup",
                    page_start=1,
                    page_end=1,
                )
            )
        )
        statuses["metadata"] = "ready"
    except Exception as exc:
        statuses["metadata"] = f"error:{exc}"

    with tempfile.TemporaryDirectory(prefix="quarry-warmup-") as temp_dir:
        pdf_path = Path(temp_dir) / "warmup.pdf"
        pdf_path.write_bytes(_minimal_pdf_bytes("QUARRY warmup page"))
        parser_names = ["qwen3_vl_mlx"] if settings.uses_mlx_profile else [settings.parser_primary, settings.parser_fallback]
        for parser_name in parser_names:
            key = "parser" if settings.uses_mlx_profile else f"parser:{parser_name}"
            try:
                logger.info("warmup starting parser", extra={"parser": parser_name})
                build_parser_adapter(parser_name, settings).parse(str(pdf_path))
                statuses[key] = f"ready:{settings.parser_provider if settings.uses_mlx_profile else parser_name}"
            except Exception as exc:
                statuses[key] = f"error:{exc}"

    payload = {
        "runtime_profile": settings.runtime_profile,
        "parser_provider": settings.parser_provider,
        "active_model_ids": settings.active_model_ids,
        **statuses,
    }
    status_path.write_text(json.dumps(payload, indent=2))
    manifest_path = settings.artifacts_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["local_model_status_path"] = str(status_path)
        manifest["runtime_profile"] = settings.runtime_profile
        manifest["parser_provider"] = settings.parser_provider
        manifest["active_model_ids"] = settings.active_model_ids
        manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "warmup complete",
        extra={
            "runtime_profile": settings.runtime_profile,
            "status_path": str(status_path),
            "latency_ms": elapsed_ms(stage_start),
        },
    )
    return {"warmed": True, "status_path": str(status_path), "statuses": payload}


def _minimal_pdf_bytes(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET"
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(content.encode('utf-8'))} >> stream\n{content}\nendstream endobj".encode("utf-8"),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
        output.extend(b"\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    output.extend(
        (
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )
    return bytes(output)
