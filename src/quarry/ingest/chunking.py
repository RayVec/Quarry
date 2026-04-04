from __future__ import annotations

import re

from quarry.domain.models import ChunkObject, ParsedBlock, ParsedDocument, StructuralIndexEntry


def token_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]


def chunk_document(parsed_document: ParsedDocument) -> tuple[list[ChunkObject], list[StructuralIndexEntry]]:
    chunks: list[ChunkObject] = []
    for section in parsed_document.sections:
        chunks.extend(_level1_chunks(parsed_document, section))
        chunks.extend(_level2_chunks(parsed_document, section))
    structural_index = [
        StructuralIndexEntry(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_heading=chunk.section_heading,
            section_path=chunk.section_path,
            section_depth=chunk.section_depth,
            page_range=(chunk.page_start, chunk.page_end),
        )
        for chunk in chunks
    ]
    return chunks, structural_index


def _level1_chunks(parsed_document: ParsedDocument, section) -> list[ChunkObject]:
    paragraphs = [block for block in section.blocks if block.block_type in {"paragraph", "table", "figure_caption", "table_title"}]
    return _chunk_blocks(parsed_document, section, paragraphs, level=1, min_tokens=128, max_tokens=256)


def _level2_chunks(parsed_document: ParsedDocument, section) -> list[ChunkObject]:
    paragraphs = [block for block in section.blocks if block.block_type != "heading"]
    return _chunk_blocks(parsed_document, section, paragraphs, level=2, min_tokens=512, max_tokens=1024)


def _chunk_blocks(parsed_document: ParsedDocument, section, blocks: list[ParsedBlock], *, level: int, min_tokens: int, max_tokens: int) -> list[ChunkObject]:
    results: list[ChunkObject] = []
    accumulator: list[ParsedBlock] = []
    sequence = 1
    for block in blocks:
        accumulator.append(block)
        joined = " ".join(item.text for item in accumulator)
        if token_count(joined) >= min_tokens:
            emitted = _emit_chunk_or_split(parsed_document, section, accumulator, level=level, max_tokens=max_tokens, sequence_start=sequence)
            results.extend(emitted)
            sequence += len(emitted)
            accumulator = []
    if accumulator:
        emitted = _emit_chunk_or_split(parsed_document, section, accumulator, level=level, max_tokens=max_tokens, sequence_start=sequence)
        results.extend(emitted)
    return results


def _emit_chunk_or_split(parsed_document: ParsedDocument, section, blocks: list[ParsedBlock], *, level: int, max_tokens: int, sequence_start: int) -> list[ChunkObject]:
    text = " ".join(block.text for block in blocks)
    if token_count(text) <= max_tokens:
        return [_build_chunk(parsed_document, section, blocks, text, level, sequence=sequence_start)]

    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        joined = " ".join(current)
        if token_count(joined) >= max_tokens:
            chunks.append(joined)
            current = []
    if current:
        chunks.append(" ".join(current))
    return [
        _build_chunk(parsed_document, section, blocks, chunk_text, level, sequence=sequence_start + index - 1)
        for index, chunk_text in enumerate(chunks, start=1)
    ]


def _build_chunk(parsed_document: ParsedDocument, section, blocks: list[ParsedBlock], text: str, level: int, *, sequence: int) -> ChunkObject:
    chunk_id = f"{parsed_document.document_id}-{section.section_id}-l{level}-{sequence}"
    page_spans = sorted({(block.page_number, block.page_end or block.page_number) for block in blocks})
    table_ids = sorted({block.table_id for block in blocks if block.table_id})
    figure_ids = sorted({block.figure_id for block in blocks if block.figure_id})
    return ChunkObject(
        chunk_id=chunk_id,
        document_id=parsed_document.document_id,
        document_title=parsed_document.document_title,
        text=text,
        level=level,
        section_heading=section.heading,
        section_path=section.path,
        section_depth=section.depth,
        page_start=min(block.page_number for block in blocks),
        page_end=max(block.page_number for block in blocks),
        source_path=parsed_document.source_path,
        parser_provenance=(parsed_document.parser_provenance[0] if parsed_document.parser_provenance else parsed_document.parser_used),
        layout_blocks=[block.block_id for block in blocks],
        page_spans=page_spans,
        table_ids=table_ids,
        figure_ids=figure_ids,
    )
