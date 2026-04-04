import asyncio
from pathlib import Path

from quarry import logging_utils
from quarry.ingest.parsers import ParserUnavailableError
from quarry.adapters.mlx_runtime import AppleMLXModelManager, MLXStructuredDecompositionClient, render_parser_prompt
from quarry.ingest.parsers import Qwen3VLMlxParserAdapter


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
        operation: str = "completion",
    ) -> str:
        return self.responses.pop(0)


def test_mlx_decomposition_retries_invalid_json() -> None:
    backend = FakeBackend(["not valid json", '{"query_type": "multi_hop"}'])
    client = MLXStructuredDecompositionClient(backend)

    result = asyncio.run(client.classify_query("Compare schedule and cost impacts"))

    assert result == "multi_hop"


def test_qwen3_vl_mlx_parser_normalizes_structured_blocks(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"png")
    adapter = Qwen3VLMlxParserAdapter(model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")

    monkeypatch.setattr(adapter, "_render_pdf_pages", lambda source_path: (tmp_path, [(1, image_path)]))
    monkeypatch.setattr(
        adapter,
        "_parse_page_blocks",
        lambda *, page_number, image_path, max_new_tokens=None: [
            {"block_type": "heading", "text": "Executive Summary", "section_depth": 0},
            {"block_type": "paragraph", "text": "Schedule performance improved after prefabrication."},
            {"block_type": "table_title", "text": "Table 1 Cost Summary"},
            {"block_type": "table", "text": "| Scope | Cost |"},
        ],
    )

    parsed = adapter.parse(str(tmp_path / "sample.pdf"))

    assert parsed.parser_used == "qwen3_vl_mlx"
    assert parsed.parser_provenance == ["mlx-community/Qwen3-VL-4B-Instruct-4bit"]
    assert parsed.sections[0].heading == "Executive Summary"
    assert parsed.sections[0].blocks[1].block_type == "paragraph"
    assert parsed.table_titles == ["Table 1 Cost Summary"]


def test_render_parser_prompt_includes_exclusion_rules() -> None:
    prompt = render_parser_prompt(3)

    assert "Ignore table of contents pages and table of contents entries." in prompt
    assert "Ignore page headers, page footers, page numbers, and repeated running headers." in prompt
    assert "Do not emit single-character headings or heading fragments." in prompt
    assert "If a token is visibly split across adjacent lines, merge it back together" in prompt


def test_mlx_model_manager_prefers_cached_snapshot_path(monkeypatch, tmp_path: Path) -> None:
    model_id = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    cached_snapshot = tmp_path / "snapshot"
    cached_snapshot.mkdir()
    load_calls: list[str] = []

    class FakeLoaded:
        config = None

    def fake_load(path_or_repo: str):
        load_calls.append(path_or_repo)
        return FakeLoaded(), FakeLoaded()

    manager = AppleMLXModelManager()
    monkeypatch.setattr(manager, "_import_generate_dependencies", lambda: (fake_load, object(), None))
    monkeypatch.setattr("quarry.adapters.mlx_runtime.resolve_cached_hf_snapshot_path", lambda repo: cached_snapshot)

    manager._load_model_locked(model_id)

    assert load_calls == [str(cached_snapshot)]


def test_mlx_model_manager_falls_back_to_repo_id_when_cached_snapshot_fails(monkeypatch, tmp_path: Path) -> None:
    model_id = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    cached_snapshot = tmp_path / "snapshot"
    cached_snapshot.mkdir()
    load_calls: list[str] = []

    class FakeLoaded:
        config = None

    def fake_load(path_or_repo: str):
        load_calls.append(path_or_repo)
        if path_or_repo == str(cached_snapshot):
            raise RuntimeError("broken snapshot")
        return FakeLoaded(), FakeLoaded()

    manager = AppleMLXModelManager()
    monkeypatch.setattr(manager, "_import_generate_dependencies", lambda: (fake_load, object(), None))
    monkeypatch.setattr("quarry.adapters.mlx_runtime.resolve_cached_hf_snapshot_path", lambda repo: cached_snapshot)

    manager._load_model_locked(model_id)

    assert load_calls == [str(cached_snapshot), model_id]


def test_qwen3_vl_mlx_parser_logs_page_level_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_TEST_LOG_DIR", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(logging_utils, "running_under_pytest", lambda: False)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLER", None)
    monkeypatch.setattr(logging_utils, "_FILE_PATH", None)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLERS", {})
    monkeypatch.setattr(logging_utils, "_FILE_PATHS", {})
    logging_utils.configure_logging(tmp_path / "logs", category="corpus")

    render_dir = tmp_path / "rendered-pages"
    render_dir.mkdir()
    page_one = render_dir / "page-1.png"
    page_two = render_dir / "page-2.png"
    page_one.write_bytes(b"png")
    page_two.write_bytes(b"png")

    adapter = Qwen3VLMlxParserAdapter(model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    monkeypatch.setattr(adapter, "_render_pdf_pages", lambda source_path: (render_dir, [(1, page_one), (2, page_two)]))
    monkeypatch.setattr(
        adapter,
        "_parse_page_blocks",
        lambda *, page_number, image_path, max_new_tokens=None: [
            {"block_type": "heading", "text": f"Section {page_number}", "section_depth": 0},
            {"block_type": "paragraph", "text": f"Page {page_number} body."},
        ],
    )

    adapter.parse(str(tmp_path / "sample.pdf"))

    corpus_log = logging_utils.current_log_file("corpus")
    assert corpus_log is not None
    log_text = corpus_log.read_text()
    assert "mlx parser rasterized pdf pages" in log_text
    assert "mlx parsing sample.pdf page 1/2 with mlx-community/Qwen3-VL-4B-Instruct-4bit (remaining_pages=1)" in log_text
    assert "mlx parsed sample.pdf page 2/2 with mlx-community/Qwen3-VL-4B-Instruct-4bit (remaining_pages=0)" in log_text


def test_qwen3_vl_mlx_parser_retries_page_before_success(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"png")
    adapter = Qwen3VLMlxParserAdapter(model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    attempts: list[int] = []

    monkeypatch.setattr(adapter, "_render_pdf_pages", lambda source_path: (tmp_path, [(1, image_path)]))

    def fake_parse_page_blocks(*, page_number, image_path, max_new_tokens=None):
        attempts.append(max_new_tokens)
        if len(attempts) < 3:
            raise ParserUnavailableError("invalid json")
        return [{"block_type": "paragraph", "text": "Recovered after retry."}]

    monkeypatch.setattr(adapter, "_parse_page_blocks", fake_parse_page_blocks)

    parsed = adapter.parse(str(tmp_path / "sample.pdf"))

    assert attempts == [768, 640, 512]
    assert parsed.page_parse_statuses[0].attempts == 3
    assert parsed.page_parse_statuses[0].outcome == "parsed"
    assert parsed.recovered_pages == []
    assert parsed.skipped_pages == []


def test_qwen3_vl_mlx_parser_records_page_recovery(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"png")
    adapter = Qwen3VLMlxParserAdapter(model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")

    monkeypatch.setattr(adapter, "_render_pdf_pages", lambda source_path: (tmp_path, [(1, image_path)]))
    monkeypatch.setattr(adapter, "_parse_page_blocks", lambda **kwargs: (_ for _ in ()).throw(ParserUnavailableError("bad page")))
    monkeypatch.setattr(
        adapter,
        "_recover_page_blocks",
        lambda **kwargs: (
            "pymupdf_text",
            [{"block_type": "paragraph", "text": "Recovered text.", "__parser_provenance": "pymupdf_text"}],
        ),
    )

    parsed = adapter.parse(str(tmp_path / "sample.pdf"))

    assert parsed.fallback_used is True
    assert parsed.recovered_pages == [1]
    assert parsed.skipped_pages == []
    assert parsed.page_parse_statuses[0].outcome == "recovered"
    assert parsed.page_parse_statuses[0].parser_used == "pymupdf_text"
    assert parsed.sections[0].blocks[0].parser_provenance == "pymupdf_text"


def test_qwen3_vl_mlx_parser_records_skipped_page_when_fallbacks_fail(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"png")
    adapter = Qwen3VLMlxParserAdapter(model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")

    monkeypatch.setattr(adapter, "_render_pdf_pages", lambda source_path: (tmp_path, [(1, image_path)]))
    monkeypatch.setattr(adapter, "_parse_page_blocks", lambda **kwargs: (_ for _ in ()).throw(ParserUnavailableError("bad page")))
    monkeypatch.setattr(adapter, "_recover_page_blocks", lambda **kwargs: None)

    parsed = adapter.parse(str(tmp_path / "sample.pdf"))

    assert parsed.fallback_used is True
    assert parsed.recovered_pages == []
    assert parsed.skipped_pages == [1]
    assert parsed.page_parse_statuses[0].outcome == "skipped"
    assert parsed.sections == []
