from pathlib import Path
import json

import pytest

from quarry import logging_utils
from quarry.adapters.production import HostedGenerationClient, build_runtime_clients
from quarry.api.app import create_app
from quarry.config import Settings
from quarry.startup import prepare_backend


def test_create_app_refuses_unwarmed_local_runtime(tmp_path: Path) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_mode="local",
        use_local_models=True,
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError):
        create_app(settings)


def test_create_app_refuses_unwarmed_local_apple_profile_without_parser_status(tmp_path: Path) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_mode="local",
        runtime_profile="apple_silicon",
        use_local_models=True,
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "local_model_status.json").write_text(
        json.dumps(
            {
                "runtime_profile": "apple_silicon",
                "parser_provider": settings.parser_provider,
                "embedding": "ready:intfloat/e5-large-v2",
                "reranker": "ready:BAAI/bge-reranker-v2-m3",
                "nli": "ready:khalidalt/DeBERTa-v3-large-mnli",
                "text": "ready:mlx-community/Qwen3.5-4B-MLX-4bit",
                "decomposition": "ready:mlx-community/Qwen3.5-4B-MLX-4bit",
                "generation": "ready:mlx-community/Qwen3.5-4B-MLX-4bit",
            }
        )
    )

    with pytest.raises(RuntimeError, match="parser"):
        create_app(settings)


def test_runtime_hosts_generation_while_leaving_decomposition_local(tmp_path: Path) -> None:
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        runtime_profile="apple_silicon",
        runtime_mode="hybrid",
        use_local_models=True,
        use_live_generation=True,
        use_live_decomposition=False,
        use_live_metadata_enrichment=False,
        llm_base_url="https://example.com/v1",
        llm_api_key="test-key",
    )
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    decomposition, generation, *_rest, runtime_profile = build_runtime_clients(settings, {}, settings.artifacts_dir)

    assert isinstance(generation, HostedGenerationClient)
    assert decomposition.__class__.__name__ == "HeuristicDecompositionClient"
    assert runtime_profile.generation_provider == "hosted:gpt-4o-mini"
    assert runtime_profile.local_model_status["generation"] == "hosted"
    assert runtime_profile.local_model_status["decomposition"] == "heuristic"
    assert runtime_profile.local_model_status["metadata"] == "heuristic"


def test_prepare_backend_writes_live_corpus_progress_to_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_TEST_LOG_DIR", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(logging_utils, "running_under_pytest", lambda: False)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLER", None)
    monkeypatch.setattr(logging_utils, "_FILE_PATH", None)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLERS", {})
    monkeypatch.setattr(logging_utils, "_FILE_PATHS", {})

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source = sources_dir / "sample_report.txt"
    source.write_text(
        "1 Executive Summary\n"
        "Prefabricated modular approaches led to a measurable schedule decrease across the observed projects.\n\n"
        "2 Procurement Risks\n"
        "Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions.\n"
    )

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        embedding_dimensions=64,
        use_local_models=False,
    )
    terminal_lines: list[str] = []

    prepare_backend(settings, sources_dir=sources_dir, run_corpus=True, echo=terminal_lines.append)

    corpus_log = logging_utils.current_log_file("corpus")
    assert corpus_log is not None
    log_text = corpus_log.read_text()

    assert "ingest starting for 1 source file(s)" in log_text
    assert "[1/1] parsing sample_report.txt (documents_remaining=0)" in log_text
    assert "metadata enrichment chunks" in log_text
    assert "with heuristic" in log_text
    assert "remaining_chunks=0" in log_text
    assert "building vector index for" in log_text
    assert "vector embedding complete for" in log_text
    assert "remaining_chunks=0" in log_text
    assert "writing artifacts for 1 document(s) into" in log_text
    assert any("[1/1] parsing sample_report.txt" in line for line in terminal_lines)
