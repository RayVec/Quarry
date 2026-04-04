from pathlib import Path
import json

import pytest

from quarry.config import Settings
from quarry.ingest.parsers import PyMuPDFTextParserAdapter, PyPDFTextParserAdapter
from quarry.ingest.pipeline import build_parsing_pipeline, ensure_parser_ready, ingest_documents, validate, warm_local_models
from quarry.ingest.pipeline import _minimal_pdf_bytes


def test_ingest_documents_writes_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "sample_report.txt"
    source.write_text(
        "1 Executive Summary\n"
        "Prefabricated modular approaches led to a measurable schedule decrease across the observed projects.\n\n"
        "2 Procurement Risks\n"
        "Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions because equipment lead times no longer aligned with installation windows.\n"
    )

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        embedding_dimensions=64,
        use_local_models=False,
    )

    result = ingest_documents([str(source)], settings)
    validation = validate(settings)

    assert "manifest" in result
    assert (settings.artifacts_dir / "manifest.json").exists()
    assert validation["valid"] is True


def test_ingest_records_block_and_chunk_provenance(tmp_path: Path) -> None:
    source = tmp_path / "sample_report.txt"
    source.write_text(
        "# Executive Summary\n"
        "Prefabricated modular approaches led to a measurable schedule decrease across the observed projects.\n\n"
        "Table 1 Cost Comparison\n"
        "| Scope | Delta |\n"
        "| Modules | 15 percent |\n"
    )

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        embedding_dimensions=64,
        use_local_models=False,
    )

    ingest_documents([str(source)], settings)
    manifest = json.loads((settings.artifacts_dir / "manifest.json").read_text())
    parsed_document_path = Path(manifest["documents"][0]["parsed_document_path"])
    chunks_path = Path(manifest["documents"][0]["chunks_path"])
    parsed_document = json.loads(parsed_document_path.read_text())
    chunks = json.loads(chunks_path.read_text())

    first_block = parsed_document["sections"][0]["blocks"][0]
    first_chunk = chunks[0]

    assert first_block["block_id"]
    assert first_block["parser_provenance"]
    assert first_chunk["parser_provenance"]
    assert first_chunk["layout_blocks"]
    assert first_chunk["page_spans"]


def test_validate_local_mode_requires_warm_status(tmp_path: Path) -> None:
    source = tmp_path / "sample_report.txt"
    source.write_text("1 Executive Summary\nEvidence about schedule outcomes.\n")

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        embedding_dimensions=64,
        use_local_models=False,
    )

    ingest_documents([str(source)], settings)
    local_mode_settings = Settings(
        corpus_dir=settings.corpus_dir,
        artifacts_dir=settings.artifacts_dir,
        embedding_dimensions=64,
        use_local_models=True,
        runtime_mode="local",
    )

    validation = validate(local_mode_settings)

    assert validation["valid"] is False
    assert any("warm-local-models" in error for error in validation["errors"])


def test_validate_flags_pdf_quality_issues_in_parsed_artifacts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    documents_dir = artifacts_dir / "documents" / "doc"
    documents_dir.mkdir(parents=True, exist_ok=True)
    parsed_document_path = documents_dir / "parsed_document.json"
    chunks_path = documents_dir / "chunks.json"
    structural_index_path = artifacts_dir / "structural_index.json"
    vector_metadata_path = artifacts_dir / "vector_index_metadata.json"
    vector_index_path = artifacts_dir / "vector_index.faiss"
    local_model_status_path = artifacts_dir / "local_model_status.json"

    parsed_document_path.write_text(
        json.dumps(
            {
                "document_id": "doc",
                "document_title": "Doc",
                "source_path": str(tmp_path / "sample.pdf"),
                "parser_used": "qwen3_vl_mlx",
                "fallback_used": False,
                "parser_provenance": ["mlx-community/Qwen3-VL-4B-Instruct-4bit"],
                "sections": [
                    {
                        "section_id": "doc-1",
                        "heading": "R",
                        "path": "R",
                        "depth": 0,
                        "page_start": 1,
                        "page_end": 1,
                        "blocks": [
                            {
                                "block_id": "b1",
                                "text": "R",
                                "page_number": 1,
                                "page_end": 1,
                                "block_type": "heading",
                                "parser_provenance": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                            },
                            {
                                "block_id": "b2",
                                "text": "Executive Summary Contents Chapter Page iii 1 15 19 Appendix A: Score Sheet Appendix B: Examples",
                                "page_number": 1,
                                "page_end": 1,
                                "block_type": "paragraph",
                                "parser_provenance": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                            },
                        ],
                    }
                ],
                "figure_captions": [],
                "table_titles": [],
            },
            indent=2,
        )
    )
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "doc-doc-1-l1-1",
                    "document_id": "doc",
                    "document_title": "Doc",
                    "text": "Executive Summary Contents Chapter Page iii 1 15 19 Appendix A: Score Sheet Appendix B: Examples",
                    "level": 1,
                    "section_heading": "R",
                    "section_path": "R",
                    "section_depth": 0,
                    "page_start": 1,
                    "page_end": 7,
                    "metadata_summary": "",
                    "metadata_entities": [],
                    "metadata_questions": [],
                    "source_path": str(tmp_path / "sample.pdf"),
                    "parser_provenance": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                    "layout_blocks": ["b1", "b2"],
                    "page_spans": [[1, 7]],
                    "table_ids": [],
                    "figure_ids": [],
                }
            ],
            indent=2,
        )
    )
    structural_index_path.write_text(json.dumps([{"chunk_id": "doc-doc-1-l1-1", "document_id": "doc", "section_heading": "R", "section_path": "R", "section_depth": 0, "page_range": [1, 7], "covered": False}], indent=2))
    vector_metadata_path.write_text(json.dumps({"embedding_model": "hash-embedding-v1", "dimensions": 64, "chunk_ids": ["doc-doc-1-l1-1"]}, indent=2))
    vector_index_path.write_bytes(b"faiss")
    local_model_status_path.write_text(json.dumps({}))
    (artifacts_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": "quarry-local-corpus",
                "embedding_model": "hash-embedding-v1",
                "embedding_dimensions": 64,
                "vector_index_path": str(vector_index_path),
                "vector_metadata_path": str(vector_metadata_path),
                "structural_index_path": str(structural_index_path),
                "runtime_profile": "full_local_transformers",
                "parser_provider": "allenai/olmOCR-7B-0725-FP8",
                "active_model_ids": [],
                "documents": [
                    {
                        "document_id": "doc",
                        "document_title": "Doc",
                        "parsed_document_path": str(parsed_document_path),
                        "chunks_path": str(chunks_path),
                        "chunk_count": 1,
                    }
                ],
                "chunk_count": 1,
            },
            indent=2,
        )
    )

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=artifacts_dir,
        embedding_dimensions=64,
        use_local_models=False,
        runtime_profile="full_local_transformers",
    )

    validation = validate(settings)

    assert validation["valid"] is False
    assert any("single-character heading" in error for error in validation["errors"])
    assert any("TOC-like" in error for error in validation["errors"])


def test_apple_lite_ingest_records_runtime_profile_and_parser_provider(tmp_path: Path) -> None:
    source = tmp_path / "sample_report.txt"
    source.write_text("# Executive Summary\nFactory fabrication reduced schedule variance.\n")

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        embedding_dimensions=64,
        use_local_models=False,
        runtime_profile="apple_lite_mlx",
    )

    ingest_documents([str(source)], settings)
    manifest = json.loads((settings.artifacts_dir / "manifest.json").read_text())
    parsed_document_path = Path(manifest["documents"][0]["parsed_document_path"])
    parsed_document = json.loads(parsed_document_path.read_text())

    assert manifest["runtime_profile"] == "apple_lite_mlx"
    assert manifest["parser_provider"] == settings.mlx_vision_model_name
    assert parsed_document["parser_used"] == "basic_text"
    assert parsed_document["parser_provenance"] == ["basic_text"]


def test_apple_lite_parsing_pipeline_prefers_mlx_parser_before_basic_fallback(tmp_path: Path) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_profile="apple_lite_mlx",
        runtime_mode="hybrid",
        use_local_models=True,
    )

    parsing_pipeline = build_parsing_pipeline(settings)

    assert parsing_pipeline.primary.adapters[0].parser_name == "qwen3_vl_mlx"
    assert parsing_pipeline.fallback.adapters[0].parser_name == settings.parser_fallback
    assert parsing_pipeline.fallback.adapters[1].parser_name == "pypdf_text"
    assert parsing_pipeline.fallback.adapters[2].parser_name == "basic_text"


def test_pymupdf_text_parser_extracts_recoverable_pdf_text(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(_minimal_pdf_bytes("Fallback text from PyMuPDF"))

    parsed_document = PyMuPDFTextParserAdapter().parse(str(source))

    assert parsed_document.parser_used == "pymupdf_text"
    assert "Fallback text from PyMuPDF" in "\n".join(
        block.text for section in parsed_document.sections for block in section.blocks
    )


def test_pypdf_text_parser_extracts_recoverable_pdf_text(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(_minimal_pdf_bytes("Fallback text from pypdf"))

    parsed_document = PyPDFTextParserAdapter().parse(str(source))

    assert parsed_document.parser_used == "pypdf_text"
    assert "Fallback text from pypdf" in "\n".join(
        block.text for section in parsed_document.sections for block in section.blocks
    )


def test_ensure_parser_ready_auto_warms_apple_profile(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_profile="apple_lite_mlx",
        runtime_mode="hybrid",
        use_local_models=True,
    )

    calls = {"warm": 0, "ready_checks": 0}

    def fake_ready(current_settings: Settings, component: str) -> bool:
        assert current_settings is settings
        assert component == "parser"
        calls["ready_checks"] += 1
        return calls["ready_checks"] > 1

    def fake_warm(current_settings: Settings) -> dict[str, object]:
        assert current_settings is settings
        calls["warm"] += 1
        return {"warmed": True}

    monkeypatch.setattr("quarry.ingest.pipeline.is_local_component_ready", fake_ready)
    monkeypatch.setattr("quarry.ingest.pipeline.warm_local_models", fake_warm)

    ensure_parser_ready(settings)

    assert calls["warm"] == 1


def test_ensure_parser_ready_raises_if_parser_remains_unavailable(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_profile="apple_lite_mlx",
        runtime_mode="hybrid",
        use_local_models=True,
    )

    monkeypatch.setattr("quarry.ingest.pipeline.is_local_component_ready", lambda settings, component: False)
    monkeypatch.setattr("quarry.ingest.pipeline.warm_local_models", lambda settings: {"warmed": False})

    with pytest.raises(RuntimeError, match="MLX parser is not ready"):
        ensure_parser_ready(settings)


def test_warm_local_models_records_apple_profile_status(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_profile="apple_lite_mlx",
        use_local_models=True,
    )

    class FakeEmbedder:
        def __init__(self, *args, **kwargs) -> None:
            self.active_model_name = "intfloat/e5-large-v2"

        async def embed_texts(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class FakeReranker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def rerank(self, query, candidates):
            return list(candidates)

    class FakeNLI:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def score(self, sentence_text, chunk_texts):
            return []

    class FakeTextBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def complete(self, prompt, *, temperature=0.1, max_new_tokens=None):
            return '{"ok": true}'

    class FakeEnricher:
        async def enrich(self, chunk):
            return chunk

    class FakeParser:
        def parse(self, source_path: str):
            return None

    monkeypatch.setattr("quarry.ingest.pipeline.LocalSentenceTransformerEmbeddingClient", FakeEmbedder)
    monkeypatch.setattr("quarry.ingest.pipeline.LocalCrossEncoderReranker", FakeReranker)
    monkeypatch.setattr("quarry.ingest.pipeline.LocalMNLIClient", FakeNLI)
    monkeypatch.setattr("quarry.ingest.pipeline.MLXTextCompletionBackend", FakeTextBackend)
    monkeypatch.setattr("quarry.ingest.pipeline.build_metadata_enricher", lambda settings: FakeEnricher())
    monkeypatch.setattr("quarry.ingest.pipeline.build_parser_adapter", lambda name, settings: FakeParser())

    result = warm_local_models(settings)
    payload = json.loads(Path(result["status_path"]).read_text())

    assert payload["runtime_profile"] == "apple_lite_mlx"
    assert payload["parser_provider"] == settings.mlx_vision_model_name
    assert payload["text"].startswith("ready:")
    assert payload["decomposition"].startswith("ready:")
    assert payload["generation"].startswith("ready:")
    assert payload["parser"].startswith("ready:")
    assert settings.mlx_text_model_name in payload["active_model_ids"]
