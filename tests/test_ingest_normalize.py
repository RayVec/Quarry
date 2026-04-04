from pathlib import Path

from quarry.domain.models import ParsedBlock, ParsedDocument, ParsedSection
from quarry.ingest.normalize import detect_quality_issues, normalize_parsed_document


def _block(block_id: str, text: str, *, page: int, block_type: str = "paragraph") -> ParsedBlock:
    return ParsedBlock(
        block_id=block_id,
        text=text,
        page_number=page,
        page_end=page,
        block_type=block_type,
        parser_provenance="mlx-community/Qwen3-VL-4B-Instruct-4bit",
    )


def test_normalize_parsed_document_merges_false_single_character_heading_section() -> None:
    document = ParsedDocument(
        document_id="doc",
        document_title="Doc",
        source_path="/tmp/sample.pdf",
        parser_used="qwen3_vl_mlx",
        parser_provenance=["mlx-community/Qwen3-VL-4B-Instruct-4bit"],
        sections=[
            ParsedSection(
                section_id="doc-1",
                heading="Executive Summary",
                path="Executive Summary",
                depth=0,
                page_start=1,
                page_end=1,
                blocks=[
                    _block("b1", "Executive Summary", page=1, block_type="heading"),
                    _block("b2", "Projects improved schedule certainty during front end planning.", page=1),
                ],
            ),
            ParsedSection(
                section_id="doc-2",
                heading="R",
                path="R",
                depth=0,
                page_start=2,
                page_end=2,
                blocks=[
                    _block("b3", "R", page=2, block_type="heading"),
                    _block("b4", "T-361 validated the new additions through additional research.", page=2),
                ],
            ),
        ],
    )

    normalized = normalize_parsed_document(document)

    assert len(normalized.sections) == 1
    assert normalized.sections[0].blocks[-1].text.startswith("RT-361 validated")


def test_normalize_parsed_document_strips_inline_toc_noise() -> None:
    document = ParsedDocument(
        document_id="doc",
        document_title="Doc",
        source_path="/tmp/sample.pdf",
        parser_used="qwen3_vl_mlx",
        parser_provenance=["mlx-community/Qwen3-VL-4B-Instruct-4bit"],
        sections=[
            ParsedSection(
                section_id="doc-1",
                heading="Executive Summary",
                path="Executive Summary",
                depth=0,
                page_start=1,
                page_end=1,
                blocks=[
                    _block("b1", "Executive Summary", page=1, block_type="heading"),
                    _block(
                        "b2",
                        "RT-361 validated the new additions by conducting additional research. Executive Summary Contents Chapter Page iii 1 15 19 Appendix A: Score Sheet Appendix B: Examples",
                        page=1,
                    ),
                ],
            )
        ],
    )

    normalized = normalize_parsed_document(document)

    assert len(normalized.sections) == 1
    assert "Contents Chapter Page" not in normalized.sections[0].blocks[1].text
    assert normalized.sections[0].blocks[1].text.endswith("additional research")


def test_detect_quality_issues_flags_suspicious_heading_and_toc_contamination() -> None:
    document = ParsedDocument(
        document_id="doc",
        document_title="Doc",
        source_path="/tmp/sample.pdf",
        parser_used="qwen3_vl_mlx",
        parser_provenance=["mlx-community/Qwen3-VL-4B-Instruct-4bit"],
        sections=[
            ParsedSection(
                section_id="doc-1",
                heading="R",
                path="R",
                depth=0,
                page_start=1,
                page_end=1,
                blocks=[
                    _block("b1", "R", page=1, block_type="heading"),
                    _block("b2", "Executive Summary Contents Chapter Page iii 1 15 19 Appendix A: Score Sheet Appendix B: Examples", page=1),
                ],
            )
        ],
    )

    issues = detect_quality_issues(document)

    assert any("single-character heading" in issue for issue in issues)
    assert any("TOC-like" in issue for issue in issues)
