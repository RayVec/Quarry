from __future__ import annotations

from collections import defaultdict
import re

from quarry.domain.models import ParsedBlock, ParsedDocument, ParsedSection


SINGLE_CHAR_HEADING_RE = re.compile(r"^[A-Z]$")
BROKEN_TOKEN_CONTINUATION_RE = re.compile(r"^[A-Z]-\d+\b")
TOC_HEADING_RE = re.compile(r"^(contents|table of contents)$", re.IGNORECASE)
TOC_INLINE_START_RE = re.compile(r"\b(contents|table of contents|chapter page)\b", re.IGNORECASE)
APPENDIX_RE = re.compile(r"\bappendix\s+[a-z]\b", re.IGNORECASE)


def normalize_parsed_document(parsed_document: ParsedDocument) -> ParsedDocument:
    repeated_noise_texts = find_repeated_header_footer_texts(parsed_document)
    normalized_sections: list[ParsedSection] = []

    for section in parsed_document.sections:
        cleaned_blocks: list[ParsedBlock] = []
        for block in section.blocks:
            text = _normalize_whitespace(block.text)
            if not text:
                continue

            if _is_repeated_noise_block(text, block, repeated_noise_texts):
                continue

            if block.block_type in {"heading", "paragraph"}:
                stripped = strip_inline_toc_noise(text)
                if stripped != text:
                    text = stripped
                if not text:
                    continue

            if block.block_type == "heading" and looks_like_toc_text(text):
                continue
            if block.block_type == "paragraph" and looks_like_toc_text(text):
                continue

            cleaned_blocks.append(block.model_copy(update={"text": text}))

        if not cleaned_blocks:
            continue

        rebuilt = _rebuild_section(section, cleaned_blocks)
        if _is_toc_section(rebuilt):
            continue
        normalized_sections.append(rebuilt)

    normalized_sections = _merge_false_heading_sections(normalized_sections)
    normalized_sections = [_merge_split_paragraphs(section) for section in normalized_sections]
    normalized_sections = [section for section in normalized_sections if section.blocks]

    return parsed_document.model_copy(
        update={
            "sections": normalized_sections,
            "figure_captions": [
                block.text
                for section in normalized_sections
                for block in section.blocks
                if block.block_type == "figure_caption"
            ],
            "table_titles": [
                block.text
                for section in normalized_sections
                for block in section.blocks
                if block.block_type == "table_title"
            ],
        }
    )


def detect_quality_issues(parsed_document: ParsedDocument) -> list[str]:
    issues: list[str] = []
    source_path = parsed_document.source_path.lower()
    if not source_path.endswith(".pdf"):
        return issues

    repeated_noise_texts = find_repeated_header_footer_texts(parsed_document)
    if repeated_noise_texts:
        samples = ", ".join(sorted(list(repeated_noise_texts))[:3])
        issues.append(f"Repeated header/footer-like text detected across pages: {samples}")

    for section in parsed_document.sections:
        if is_suspicious_single_char_heading(section.heading):
            issues.append(f"Suspicious single-character heading detected: {section.heading!r}")
        if _is_toc_section(section):
            issues.append(f"Table-of-contents-like section detected: {section.heading!r}")
        for block in section.blocks:
            if block.block_type in {"heading", "paragraph"}:
                if looks_like_toc_text(block.text):
                    issues.append(f"TOC-like block text remained in parsed output: {block.text[:80]!r}")
                stripped = strip_inline_toc_noise(block.text)
                if stripped != _normalize_whitespace(block.text):
                    issues.append(f"Inline TOC-like text remained in parsed output: {block.text[:80]!r}")
    return issues


def find_repeated_header_footer_texts(parsed_document: ParsedDocument) -> set[str]:
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for section in parsed_document.sections:
        for block in section.blocks:
            text = _normalize_whitespace(block.text)
            if not _looks_like_short_running_text(text):
                continue
            pages_by_text[text.lower()].add(block.page_number)
    return {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= 3
    }


def looks_like_toc_text(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    lowered = normalized.lower()
    if not normalized:
        return False
    if TOC_HEADING_RE.match(normalized):
        return True
    if "table of contents" in lowered:
        return True
    if "contents chapter page" in lowered:
        return True
    appendix_count = len(APPENDIX_RE.findall(lowered))
    digit_count = len(re.findall(r"\b\d+\b", normalized))
    chapter_like = "chapter page" in lowered or ("contents" in lowered and "page" in lowered)
    if appendix_count >= 2 and digit_count >= 2:
        return True
    if chapter_like and digit_count >= 4:
        return True
    if lowered.startswith("executive summary contents"):
        return True
    return False


def strip_inline_toc_noise(text: str) -> str:
    normalized = _normalize_whitespace(text)
    for match in TOC_INLINE_START_RE.finditer(normalized):
        if match.start() < 32:
            continue
        tail = normalized[match.start() :].strip()
        if looks_like_toc_text(tail):
            prefix = normalized[: match.start()].rstrip(" .;:-")
            prefix = re.sub(r"([.!?])\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}$", r"\1", prefix)
            return prefix.rstrip(" .;:-")
    return normalized


def is_suspicious_single_char_heading(text: str) -> bool:
    return bool(SINGLE_CHAR_HEADING_RE.fullmatch(_normalize_whitespace(text)))


def _merge_false_heading_sections(sections: list[ParsedSection]) -> list[ParsedSection]:
    merged: list[ParsedSection] = []
    for section in sections:
        first_body = next((block for block in section.blocks if block.block_type != "heading"), None)
        if (
            merged
            and is_suspicious_single_char_heading(section.heading)
            and first_body is not None
            and BROKEN_TOKEN_CONTINUATION_RE.match(first_body.text)
        ):
            updated_blocks: list[ParsedBlock] = []
            body_merged = False
            for block in section.blocks:
                if block.block_type == "heading" and _normalize_whitespace(block.text) == _normalize_whitespace(section.heading):
                    continue
                if not body_merged and block.block_id == first_body.block_id:
                    updated_blocks.append(
                        block.model_copy(update={"text": f"{section.heading}{block.text}"})
                    )
                    body_merged = True
                else:
                    updated_blocks.append(block)
            previous = merged[-1]
            merged[-1] = _rebuild_section(
                previous,
                [*previous.blocks, *updated_blocks],
                page_end=max(previous.page_end, section.page_end),
            )
            continue
        merged.append(section)
    return merged


def _merge_split_paragraphs(section: ParsedSection) -> ParsedSection:
    merged_blocks: list[ParsedBlock] = []
    for block in section.blocks:
        if merged_blocks and _should_merge_paragraphs(merged_blocks[-1], block):
            previous = merged_blocks[-1]
            separator = "" if previous.text.endswith("-") else " "
            merged_blocks[-1] = previous.model_copy(
                update={
                    "text": f"{previous.text.rstrip('-')}{separator}{block.text}" if previous.text.endswith("-") else f"{previous.text}{separator}{block.text}",
                    "page_end": block.page_end or block.page_number,
                }
            )
            continue
        merged_blocks.append(block)
    return _rebuild_section(section, merged_blocks)


def _should_merge_paragraphs(previous: ParsedBlock, current: ParsedBlock) -> bool:
    if previous.block_type != "paragraph" or current.block_type != "paragraph":
        return False
    if current.page_number - (previous.page_end or previous.page_number) > 1:
        return False
    if previous.text.endswith("-"):
        return True
    if previous.text.endswith((".", "!", "?", ":", ";")):
        return False
    if current.text[:1].islower():
        return True
    if current.text.startswith(("and ", "or ", "but ", "because ", "which ", "that ", "when ", "where ", "while ", ",", ";", ")")):
        return True
    return False


def _rebuild_section(
    original: ParsedSection,
    blocks: list[ParsedBlock],
    *,
    page_end: int | None = None,
) -> ParsedSection:
    first_heading = next((block.text for block in blocks if block.block_type == "heading"), original.heading)
    page_start = min(block.page_number for block in blocks)
    resolved_page_end = page_end if page_end is not None else max(block.page_end or block.page_number for block in blocks)
    return original.model_copy(
        update={
            "heading": first_heading,
            "path": _replace_last_path_segment(original.path, first_heading),
            "page_start": page_start,
            "page_end": resolved_page_end,
            "blocks": blocks,
        }
    )


def _replace_last_path_segment(path: str, heading: str) -> str:
    parts = [part.strip() for part in path.split(">")]
    if not parts:
        return heading
    parts[-1] = heading
    return " > ".join(parts)


def _is_toc_section(section: ParsedSection) -> bool:
    if looks_like_toc_text(section.heading):
        return True
    paragraph_blocks = [block for block in section.blocks if block.block_type in {"heading", "paragraph"}]
    if not paragraph_blocks:
        return False
    toc_like_blocks = sum(1 for block in paragraph_blocks if looks_like_toc_text(block.text))
    return toc_like_blocks == len(paragraph_blocks)


def _is_repeated_noise_block(text: str, block: ParsedBlock, repeated_noise_texts: set[str]) -> bool:
    if block.block_type not in {"heading", "paragraph"}:
        return False
    return text.lower() in repeated_noise_texts


def _looks_like_short_running_text(text: str) -> bool:
    if not text or len(text) > 80:
        return False
    return len(text.split()) <= 12


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split()).strip()
