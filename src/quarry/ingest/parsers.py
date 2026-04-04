from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Sequence

from quarry.adapters.interfaces import ParserAdapter
from quarry.adapters.mlx_runtime import AppleMLXModelManager, parse_mlx_page_blocks
from quarry.domain.models import PageParseStatus, ParsedBlock, ParsedDocument, ParsedSection
from quarry.ingest.normalize import normalize_parsed_document
from quarry.logging_utils import logger_with_trace


NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\s+(?P<title>.+)$")
MARKDOWN_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+)$")
logger = logger_with_trace(__name__)


class ParserUnavailableError(RuntimeError):
    """Raised when a parser cannot run in the current environment."""


def _document_title_from_path(path: Path) -> str:
    return path.stem or path.name


class ShellParserAdapter(ParserAdapter):
    def __init__(self, parser_name: str, command: list[str]) -> None:
        self.parser_name = parser_name
        self.command = command

    def parse(self, source_path: str) -> ParsedDocument:
        if not self.command:
            raise ParserUnavailableError(f"{self.parser_name} is not configured.")
        completed = subprocess.run([*self.command, source_path], check=True, capture_output=True, text=True)
        return parse_text_document(source_path, completed.stdout, parser_name=self.parser_name)


class BasicTextParser(ParserAdapter):
    parser_name = "basic_text"

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ParserUnavailableError(f"{self.parser_name} only supports markdown and text files.")
        return parse_text_document(source_path, path.read_text(), parser_name=self.parser_name)


def _normalize_extracted_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_page_text_with_pymupdf(source_path: str, page_number: int) -> str:
    try:
        fitz = importlib.import_module("fitz")
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ParserUnavailableError("PyMuPDF text fallback is unavailable.") from exc

    document = fitz.open(source_path)
    try:
        page = document.load_page(page_number - 1)
        blocks = page.get_text("blocks") or []
        page_lines: list[str] = []
        for block in sorted(blocks, key=lambda item: (float(item[1]), float(item[0]))):
            if len(block) > 6 and int(block[6]) != 0:
                continue
            block_text = _normalize_extracted_text(str(block[4]))
            if block_text:
                page_lines.append(block_text)
        return "\n".join(page_lines)
    finally:
        document.close()


def _extract_page_text_with_pypdf(source_path: str, page_number: int) -> str:
    try:
        pypdf = importlib.import_module("pypdf")
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ParserUnavailableError("pypdf text fallback is unavailable.") from exc

    reader = pypdf.PdfReader(source_path)
    if page_number < 1 or page_number > len(reader.pages):
        raise ParserUnavailableError(f"Page {page_number} is out of range for pypdf extraction.")
    return _normalize_extracted_text(reader.pages[page_number - 1].extract_text() or "")


def _page_text_to_raw_blocks(
    source_path: str,
    *,
    page_number: int,
    parser_name: str,
    page_text: str,
) -> list[dict[str, object]]:
    parsed_page = parse_text_document(
        source_path,
        f"[[PAGE {page_number}]]\n{page_text}",
        parser_name=parser_name,
    )
    raw_blocks: list[dict[str, object]] = []
    for section in parsed_page.sections:
        for block in section.blocks:
            raw_blocks.append(
                {
                    "block_type": block.block_type,
                    "text": block.text,
                    "section_depth": section.depth if block.block_type == "heading" else None,
                    "__parser_provenance": parser_name,
                }
            )
    return raw_blocks


class PyMuPDFTextParserAdapter(ParserAdapter):
    parser_name = "pymupdf_text"

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() != ".pdf":
            raise ParserUnavailableError(f"{self.parser_name} only supports PDFs.")

        page_texts: list[str] = []
        try:
            fitz = importlib.import_module("fitz")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("PyMuPDF text fallback is unavailable.") from exc

        document = fitz.open(source_path)
        try:
            for page_index in range(document.page_count):
                page_text = _extract_page_text_with_pymupdf(source_path, page_index + 1)
                if page_text:
                    page_texts.append(f"[[PAGE {page_index + 1}]]\n{page_text}")
        finally:
            document.close()

        if not page_texts:
            raise ParserUnavailableError("PyMuPDF did not extract any text from the PDF.")
        return parse_text_document(source_path, "\n\n".join(page_texts), parser_name=self.parser_name)


class PyPDFTextParserAdapter(ParserAdapter):
    parser_name = "pypdf_text"

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() != ".pdf":
            raise ParserUnavailableError(f"{self.parser_name} only supports PDFs.")

        page_texts: list[str] = []
        try:
            pypdf = importlib.import_module("pypdf")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("pypdf text fallback is unavailable.") from exc

        reader = pypdf.PdfReader(source_path)
        for page_index, _page in enumerate(reader.pages, start=1):
            page_text = _extract_page_text_with_pypdf(source_path, page_index)
            if page_text:
                page_texts.append(f"[[PAGE {page_index}]]\n{page_text}")

        if not page_texts:
            raise ParserUnavailableError("pypdf did not extract any text from the PDF.")
        return parse_text_document(source_path, "\n\n".join(page_texts), parser_name=self.parser_name)


class OlmOCRTransformersParserAdapter(ParserAdapter):
    parser_name = "olmocr_transformers"

    def __init__(self, *, model_name: str, target_longest_image_dim: int = 1024) -> None:
        self.model_name = model_name
        self.target_longest_image_dim = target_longest_image_dim

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() != ".pdf":
            raise ParserUnavailableError(f"{self.parser_name} only supports PDFs.")

        try:
            pypdf = importlib.import_module("pypdf")
            runner = importlib.import_module("olmocr.bench.runners.run_transformers")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("olmOCR local transformers backend is unavailable.") from exc

        reader = pypdf.PdfReader(source_path)
        page_texts: list[str] = []
        for page_number, _ in enumerate(reader.pages, start=1):
            page_text = runner.run_transformers(
                source_path,
                page_num=page_number,
                model_name=self.model_name,
                target_longest_image_dim=self.target_longest_image_dim,
                prompt_template="yaml",
                response_template="yaml",
            )
            page_texts.append(f"[[PAGE {page_number}]]\n{page_text or ''}")
        return parse_text_document(source_path, "\n\n".join(page_texts), parser_name=self.parser_name)


class MinerUPipelineParserAdapter(ParserAdapter):
    parser_name = "mineru_pipeline"

    def __init__(self, *, backend: str = "pipeline", language: str = "en") -> None:
        self.backend = backend
        self.language = language

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() != ".pdf":
            raise ParserUnavailableError(f"{self.parser_name} only supports PDFs.")

        try:
            pypdf = importlib.import_module("pypdf")
            common = importlib.import_module("mineru.cli.common")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("MinerU local backend is unavailable.") from exc

        reader = pypdf.PdfReader(source_path)
        pdf_bytes = path.read_bytes()
        pdf_file_name = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem) or "document"
        page_texts: list[str] = []

        with tempfile.TemporaryDirectory(prefix="quarry-mineru-") as temp_dir:
            for page_index in range(len(reader.pages)):
                output_dir = Path(temp_dir) / f"page-{page_index + 1}"
                common.do_parse(
                    str(output_dir),
                    [pdf_file_name],
                    [pdf_bytes],
                    [self.language],
                    backend=self.backend,
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_md=True,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=False,
                    start_page_id=page_index,
                    end_page_id=page_index,
                )
                markdown_path = next(output_dir.rglob("*.md"), None)
                if markdown_path is None:
                    continue
                page_texts.append(f"[[PAGE {page_index + 1}]]\n{markdown_path.read_text()}")

        if not page_texts:
            raise ParserUnavailableError("MinerU did not produce markdown output.")
        return parse_text_document(source_path, "\n\n".join(page_texts), parser_name=self.parser_name)


class Qwen3VLMlxParserAdapter(ParserAdapter):
    parser_name = "qwen3_vl_mlx"

    def __init__(
        self,
        *,
        model_name: str,
        target_longest_image_dim: int = 1024,
        max_new_tokens: int = 768,
        max_pdf_pages_per_batch: int = 1,
        model_manager: AppleMLXModelManager | None = None,
    ) -> None:
        self.model_name = model_name
        self.target_longest_image_dim = target_longest_image_dim
        self.max_new_tokens = max_new_tokens
        self.max_pdf_pages_per_batch = max(1, max_pdf_pages_per_batch)
        self.model_manager = model_manager or AppleMLXModelManager()
        self.max_parse_attempts = 3

    def parse(self, source_path: str) -> ParsedDocument:
        path = Path(source_path)
        if path.suffix.lower() != ".pdf":
            raise ParserUnavailableError(f"{self.parser_name} only supports PDFs.")

        temp_dir, pages = self._render_pdf_pages(source_path)
        try:
            logger.info(
                "mlx parser rasterized pdf pages",
                extra={
                    "source_path": source_path,
                    "page_count": len(pages),
                },
            )
            page_blocks: list[tuple[int, list[dict[str, object]]]] = []
            page_statuses: list[PageParseStatus] = []
            parser_provenance: list[str] = [self.model_name]
            for start in range(0, len(pages), self.max_pdf_pages_per_batch):
                batch = pages[start : start + self.max_pdf_pages_per_batch]
                for page_number, image_path in batch:
                    remaining_pages = max(len(pages) - page_number, 0)
                    logger.info(
                        f"mlx parsing {path.name} page {page_number}/{len(pages)} with {self.model_name} (remaining_pages={remaining_pages})",
                        extra={
                            "source_path": source_path,
                            "page_number": page_number,
                            "page_count": len(pages),
                            "parser_provider": self.model_name,
                            "remaining_pages": remaining_pages,
                        },
                    )
                    try:
                        blocks, page_status = self._parse_page_with_retries(
                            source_path=source_path,
                            page_number=page_number,
                            page_count=len(pages),
                            image_path=image_path,
                        )
                    finally:
                        image_path.unlink(missing_ok=True)
                    page_statuses.append(page_status)
                    if page_status.parser_used and page_status.parser_used not in parser_provenance:
                        parser_provenance.append(page_status.parser_used)
                    if blocks:
                        page_blocks.append((page_number, blocks))
                    if page_status.outcome == "parsed":
                        logger.info(
                            f"mlx parsed {path.name} page {page_number}/{len(pages)} with {self.model_name} (remaining_pages={remaining_pages})",
                            extra={
                                "source_path": source_path,
                                "page_number": page_number,
                                "page_count": len(pages),
                                "block_count": len(blocks),
                                "attempts": page_status.attempts,
                                "parser_provider": self.model_name,
                                "remaining_pages": remaining_pages,
                            },
                        )
                    elif page_status.outcome == "recovered":
                        logger.warning(
                            f"mlx recovered {path.name} page {page_number}/{len(pages)} with {page_status.parser_used} (remaining_pages={remaining_pages})",
                            extra={
                                "source_path": source_path,
                                "page_number": page_number,
                                "page_count": len(pages),
                                "block_count": len(blocks),
                                "parser_used": page_status.parser_used,
                                "attempts": page_status.attempts,
                                "error": page_status.error,
                                "parser_provider": self.model_name,
                                "remaining_pages": remaining_pages,
                            },
                        )
                    else:
                        logger.error(
                            f"mlx skipped {path.name} page {page_number}/{len(pages)} after fallback failures (remaining_pages={remaining_pages})",
                            extra={
                                "source_path": source_path,
                                "page_number": page_number,
                                "page_count": len(pages),
                                "attempts": page_status.attempts,
                                "error": page_status.error,
                                "parser_provider": self.model_name,
                                "remaining_pages": remaining_pages,
                            },
                        )
            recovered_pages = [status.page_number for status in page_statuses if status.outcome == "recovered"]
            skipped_pages = [status.page_number for status in page_statuses if status.outcome == "skipped"]
            if recovered_pages or skipped_pages:
                logger.warning(
                    "mlx document completed with page-level fallbacks",
                    extra={
                        "source_path": source_path,
                        "recovered_pages": recovered_pages,
                        "skipped_pages": skipped_pages,
                    },
                )
            return _build_mlx_document_from_blocks(
                source_path,
                page_blocks,
                parser_name=self.parser_name,
                parser_provenance=parser_provenance,
                fallback_used=bool(recovered_pages or skipped_pages),
                page_parse_statuses=page_statuses,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _render_pdf_pages(self, source_path: str) -> tuple[Path, list[tuple[int, Path]]]:
        try:
            fitz = importlib.import_module("fitz")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("PyMuPDF is not installed.") from exc

        rendered: list[tuple[int, Path]] = []
        document = fitz.open(source_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="quarry-mlx-pages-"))
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                rect = page.rect
                longest_dim = max(float(rect.width), float(rect.height), 1.0)
                scale = self.target_longest_image_dim / longest_dim
                matrix = fitz.Matrix(scale, scale)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = temp_dir / f"page-{page_index + 1}.png"
                pixmap.save(str(image_path))
                rendered.append((page_index + 1, image_path))
                logger.info(
                    f"mlx rasterized {Path(source_path).name} page {page_index + 1}/{document.page_count} with PyMuPDF (remaining_pages={max(document.page_count - (page_index + 1), 0)})",
                    extra={
                        "source_path": source_path,
                        "page_number": page_index + 1,
                        "page_count": document.page_count,
                        "rasterizer": "PyMuPDF",
                        "remaining_pages": max(document.page_count - (page_index + 1), 0),
                    },
                )
            return temp_dir, rendered
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ParserUnavailableError("Unable to rasterize the PDF for MLX parsing.") from exc
        finally:
            document.close()

    def _parse_page_blocks(
        self,
        *,
        page_number: int,
        image_path: Path,
        max_new_tokens: int | None = None,
    ) -> list[dict[str, object]]:
        try:
            return asyncio.run(
                parse_mlx_page_blocks(
                    model_manager=self.model_manager,
                    model_name=self.model_name,
                    image_path=str(image_path),
                    page_number=page_number,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                )
            )
        except Exception as exc:
            raise ParserUnavailableError(f"MLX page parsing failed for page {page_number}: {exc}") from exc

    def _parse_page_with_retries(
        self,
        *,
        source_path: str,
        page_number: int,
        page_count: int,
        image_path: Path,
    ) -> tuple[list[dict[str, object]], PageParseStatus]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_parse_attempts + 1):
            attempt_max_tokens = max(256, self.max_new_tokens - ((attempt - 1) * 128))
            try:
                blocks = self._parse_page_blocks(
                    page_number=page_number,
                    image_path=image_path,
                    max_new_tokens=attempt_max_tokens,
                )
                annotated_blocks = [
                    {**block, "__parser_provenance": self.model_name}
                    for block in blocks
                ]
                return annotated_blocks, PageParseStatus(
                    page_number=page_number,
                    outcome="parsed",
                    parser_used=self.model_name,
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_parse_attempts:
                    logger.warning(
                        f"mlx parse failed for page {page_number}/{page_count} with {self.model_name}; retrying attempt {attempt + 1}/{self.max_parse_attempts} (remaining_pages={max(page_count - page_number, 0)})",
                        extra={
                            "source_path": source_path,
                            "page_number": page_number,
                            "page_count": page_count,
                            "attempt": attempt,
                            "max_attempts": self.max_parse_attempts,
                            "error": str(exc),
                            "parser_provider": self.model_name,
                            "remaining_pages": max(page_count - page_number, 0),
                        },
                    )

        recovery_result = self._recover_page_blocks(source_path=source_path, page_number=page_number)
        if recovery_result is not None:
            parser_name, blocks = recovery_result
            return blocks, PageParseStatus(
                page_number=page_number,
                outcome="recovered",
                parser_used=parser_name,
                attempts=self.max_parse_attempts,
                error=str(last_error) if last_error is not None else None,
            )

        return [], PageParseStatus(
            page_number=page_number,
            outcome="skipped",
            parser_used=None,
            attempts=self.max_parse_attempts,
            error=str(last_error) if last_error is not None else "Unknown MLX parser failure.",
        )

    def _recover_page_blocks(self, *, source_path: str, page_number: int) -> tuple[str, list[dict[str, object]]] | None:
        recovery_errors: list[str] = []
        for parser_name, extractor in (
            ("pymupdf_text", _extract_page_text_with_pymupdf),
            ("pypdf_text", _extract_page_text_with_pypdf),
        ):
            try:
                page_text = extractor(source_path, page_number)
                if not page_text:
                    raise ParserUnavailableError(f"{parser_name} did not extract any text.")
                return parser_name, _page_text_to_raw_blocks(
                    source_path,
                    page_number=page_number,
                    parser_name=parser_name,
                    page_text=page_text,
                )
            except Exception as exc:
                recovery_errors.append(f"{parser_name}: {exc}")
                logger.warning(
                    f"mlx page fallback failed with {parser_name}",
                    extra={
                        "source_path": source_path,
                        "page_number": page_number,
                        "fallback_parser": parser_name,
                        "error": str(exc),
                    },
                )
        if recovery_errors:
            logger.error(
                "mlx page fallback chain exhausted",
                extra={
                    "source_path": source_path,
                    "page_number": page_number,
                    "errors": recovery_errors,
                },
            )
        return None


class TableMergePostProcessor:
    def merge(self, parsed_document: ParsedDocument) -> ParsedDocument:
        merged_sections: list[ParsedSection] = []
        for section in parsed_document.sections:
            merged_blocks: list[ParsedBlock] = []
            previous_table: ParsedBlock | None = None
            for block in section.blocks:
                if self._should_merge_tables(previous_table, block):
                    previous_table = previous_table.model_copy(
                        update={
                            "text": f"{previous_table.text}\n{block.text}",
                            "page_end": block.page_end or block.page_number,
                        }
                    )
                    merged_blocks[-1] = previous_table
                    continue
                merged_blocks.append(block)
                if block.block_type == "table":
                    previous_table = block
                else:
                    previous_table = None
            merged_sections.append(section.model_copy(update={"blocks": merged_blocks}))
        return parsed_document.model_copy(update={"sections": merged_sections})

    def _should_merge_tables(self, previous_table: ParsedBlock | None, block: ParsedBlock) -> bool:
        if previous_table is None or block.block_type != "table" or previous_table.block_type != "table":
            return False
        header_match = previous_table.text.splitlines()[:1] == block.text.splitlines()[:1]
        same_table_id = previous_table.table_id and previous_table.table_id == block.table_id
        adjacent_pages = (previous_table.page_end or previous_table.page_number) + 1 >= block.page_number
        return adjacent_pages and bool(header_match or same_table_id)


class ParsingPipeline:
    def __init__(self, primary: ParserAdapter, fallback: ParserAdapter, post_processor: TableMergePostProcessor | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.post_processor = post_processor or TableMergePostProcessor()

    def parse(self, source_path: str) -> ParsedDocument:
        try:
            parsed = self.primary.parse(source_path)
        except Exception as exc:
            logger.warning(
                "primary parser chain failed; falling back to backup parser chain",
                extra={
                    "source_path": source_path,
                    "primary_parser": getattr(self.primary, "parser_name", "primary_chain"),
                    "fallback_parser": getattr(self.fallback, "parser_name", "fallback_chain"),
                    "error": str(exc),
                },
            )
            parsed = self.fallback.parse(source_path).model_copy(update={"fallback_used": True})
            logger.warning(
                "document parsed with fallback parser chain",
                extra={
                    "source_path": source_path,
                    "parser_used": parsed.parser_used,
                    "fallback_used": True,
                },
            )
        merged = self.post_processor.merge(parsed)
        normalized = normalize_parsed_document(merged)
        return normalized


class CascadingParserAdapter(ParserAdapter):
    def __init__(self, parser_name: str, adapters: list[ParserAdapter]) -> None:
        self.parser_name = parser_name
        self.adapters = adapters

    def parse(self, source_path: str) -> ParsedDocument:
        last_error: Exception | None = None
        for adapter in self.adapters:
            try:
                parsed = adapter.parse(source_path)
                return parsed.model_copy(update={"parser_used": adapter.parser_name})
            except Exception as exc:
                last_error = exc
        if last_error is None:  # pragma: no cover
            raise ParserUnavailableError("No parsers configured.")
        raise last_error


def parse_text_document(source_path: str, text: str, *, parser_name: str) -> ParsedDocument:
    path = Path(source_path)
    document_id = path.stem.lower().replace(" ", "-")
    document_title = _document_title_from_path(path)
    sections: list[ParsedSection] = []
    current_heading = "Introduction"
    current_path = current_heading
    current_depth = 0
    current_blocks: list[ParsedBlock] = []
    current_page = 1
    section_start_page = 1
    heading_stack: list[tuple[int, str]] = [(0, current_heading)]
    block_counter = 0
    table_counter = 0
    figure_counter = 0
    pending_table_id: str | None = None

    def make_block(block_text: str, *, block_type: str, page_number: int | None = None, table_id: str | None = None, figure_id: str | None = None) -> ParsedBlock:
        nonlocal block_counter
        block_counter += 1
        return ParsedBlock(
            block_id=f"{document_id}-block-{block_counter}",
            text=block_text,
            page_number=page_number or current_page,
            page_end=page_number or current_page,
            block_type=block_type,
            parser_provenance=parser_name,
            table_id=table_id,
            figure_id=figure_id,
        )

    def set_heading(line: str, *, depth: int) -> None:
        nonlocal current_heading, current_path, current_depth, heading_stack, current_blocks, section_start_page, pending_table_id
        if depth <= 0:
            heading_stack = [(0, line)]
        else:
            heading_stack = heading_stack[:depth]
            heading_stack.append((depth, line))
        current_heading = line
        current_path = " > ".join(item[1] for item in heading_stack)
        current_depth = depth
        current_blocks = [make_block(line, block_type="heading")]
        section_start_page = current_page
        pending_table_id = None

    def flush_section(page_end: int) -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        sections.append(
            ParsedSection(
                section_id=f"{document_id}-{len(sections) + 1}",
                heading=current_heading,
                path=current_path,
                depth=current_depth,
                page_start=section_start_page,
                page_end=max(page_end, section_start_page),
                blocks=current_blocks,
            )
        )
        current_blocks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        page_match = re.match(r"^\[\[PAGE (\d+)\]\]$", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        markdown_heading = MARKDOWN_HEADING_RE.match(line)
        numbered_heading = NUMBERED_HEADING_RE.match(line)
        is_short_upper_heading = line.isupper() and len(line.split()) <= 8
        if markdown_heading or numbered_heading or is_short_upper_heading:
            flush_section(current_page)
            if markdown_heading:
                depth = max(len(markdown_heading.group("hashes")) - 1, 0)
                title = markdown_heading.group("title").strip()
            elif numbered_heading:
                depth = numbered_heading.group("number").count(".")
                title = line
            else:
                depth = 0
                title = line
            set_heading(title, depth=depth)
            continue

        lowered = line.lower()
        if line.startswith("|"):
            block_type = "table"
        elif lowered.startswith("table"):
            block_type = "table_title"
        elif lowered.startswith("figure"):
            block_type = "figure_caption"
        else:
            block_type = "paragraph"
        table_id = None
        figure_id = None
        if block_type == "table_title":
            table_counter += 1
            pending_table_id = f"{document_id}-table-{table_counter}"
            table_id = pending_table_id
        elif block_type == "table":
            table_id = pending_table_id
        elif block_type == "figure_caption":
            figure_counter += 1
            figure_id = f"{document_id}-figure-{figure_counter}"
            pending_table_id = None
        else:
            pending_table_id = None
        current_blocks.append(make_block(line, block_type=block_type, table_id=table_id, figure_id=figure_id))

    flush_section(current_page)
    return ParsedDocument(
        document_id=document_id,
        document_title=document_title,
        source_path=source_path,
        parser_used=parser_name,
        parser_provenance=[parser_name],
        sections=sections,
        figure_captions=[block.text for section in sections for block in section.blocks if block.block_type == "figure_caption"],
        table_titles=[block.text for section in sections for block in section.blocks if block.block_type == "table_title"],
    )


def _build_mlx_document_from_blocks(
    source_path: str,
    page_blocks: Sequence[tuple[int, list[dict[str, object]]]],
    *,
    parser_name: str,
    parser_provenance: list[str],
    fallback_used: bool = False,
    page_parse_statuses: Sequence[PageParseStatus] | None = None,
) -> ParsedDocument:
    path = Path(source_path)
    document_id = path.stem.lower().replace(" ", "-")
    document_title = _document_title_from_path(path)
    sections: list[ParsedSection] = []
    current_heading = "Introduction"
    current_path = current_heading
    current_depth = 0
    current_blocks: list[ParsedBlock] = []
    section_start_page = page_blocks[0][0] if page_blocks else 1
    heading_stack: list[tuple[int, str]] = [(0, current_heading)]
    block_counter = 0
    table_counter = 0
    figure_counter = 0
    pending_table_id: str | None = None

    def make_block(
        block_text: str,
        *,
        page_number: int,
        block_type: str,
        table_id: str | None = None,
        figure_id: str | None = None,
    ) -> ParsedBlock:
        nonlocal block_counter
        block_counter += 1
        return ParsedBlock(
            block_id=f"{document_id}-block-{block_counter}",
            text=block_text,
            page_number=page_number,
            page_end=page_number,
            block_type=block_type,
            parser_provenance=None,
            table_id=table_id,
            figure_id=figure_id,
        )

    def set_heading(line: str, *, depth: int, page_number: int) -> None:
        nonlocal current_heading, current_path, current_depth, heading_stack, current_blocks, section_start_page, pending_table_id
        if depth <= 0:
            heading_stack = [(0, line)]
        else:
            heading_stack = heading_stack[:depth]
            heading_stack.append((depth, line))
        current_heading = line
        current_path = " > ".join(item[1] for item in heading_stack)
        current_depth = depth
        current_blocks = [make_block(line, block_type="heading", page_number=page_number)]
        section_start_page = page_number
        pending_table_id = None

    def flush_section(page_end: int) -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        sections.append(
            ParsedSection(
                section_id=f"{document_id}-{len(sections) + 1}",
                heading=current_heading,
                path=current_path,
                depth=current_depth,
                page_start=section_start_page,
                page_end=max(page_end, section_start_page),
                blocks=current_blocks,
            )
        )
        current_blocks = []

    for page_number, blocks in page_blocks:
        for raw_block in blocks:
            text = " ".join(str(raw_block.get("text", "")).split()).strip()
            if not text:
                continue
            block_type = str(raw_block.get("block_type", "paragraph")).lower()
            if block_type not in {"heading", "paragraph", "table", "figure_caption", "table_title"}:
                block_type = "paragraph"

            if block_type == "heading":
                flush_section(page_number)
                raw_depth = raw_block.get("section_depth")
                depth = raw_depth if isinstance(raw_depth, int) and raw_depth >= 0 else 0
                set_heading(text, depth=depth, page_number=page_number)
                continue

            table_id = None
            figure_id = None
            if block_type == "table_title":
                table_counter += 1
                pending_table_id = f"{document_id}-table-{table_counter}"
                table_id = pending_table_id
            elif block_type == "table":
                table_id = pending_table_id
            elif block_type == "figure_caption":
                figure_counter += 1
                figure_id = f"{document_id}-figure-{figure_counter}"
                pending_table_id = None
            else:
                pending_table_id = None
            current_blocks.append(
                make_block(
                    text,
                    page_number=page_number,
                    block_type=block_type,
                    table_id=table_id,
                    figure_id=figure_id,
                ).model_copy(update={"parser_provenance": str(raw_block.get("__parser_provenance", parser_provenance[0] if parser_provenance else parser_name))})
            )

    last_page = page_blocks[-1][0] if page_blocks else 1
    flush_section(last_page)
    page_statuses = list(page_parse_statuses or [])
    recovered_pages = [status.page_number for status in page_statuses if status.outcome == "recovered"]
    skipped_pages = [status.page_number for status in page_statuses if status.outcome == "skipped"]
    return ParsedDocument(
        document_id=document_id,
        document_title=document_title,
        source_path=source_path,
        parser_used=parser_name,
        fallback_used=fallback_used,
        parser_provenance=parser_provenance,
        sections=sections,
        figure_captions=[block.text for section in sections for block in section.blocks if block.block_type == "figure_caption"],
        table_titles=[block.text for section in sections for block in section.blocks if block.block_type == "table_title"],
        recovered_pages=recovered_pages,
        skipped_pages=skipped_pages,
        page_parse_statuses=page_statuses,
    )
