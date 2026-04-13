from __future__ import annotations

import asyncio

from quarry.adapters.in_memory import InMemoryChunkStore
from quarry.domain.models import (
    CitationIndexEntry,
    CitationReplacementRequest,
    ChunkObject,
    ParsedSentence,
    Reference,
    RetrievedPassage,
    SentenceStatus,
    SentenceType,
    SessionState,
)
from quarry.services.review_service import ReviewService
from quarry.services.session_store import SessionStore


class StaticHybridRetriever:
    def __init__(self, passage: RetrievedPassage) -> None:
        self._passage = passage

    async def retrieve(self, *, original_query: str, facets: list[str], query_type=None):
        return [self._passage], []


def _build_chunk(chunk_id: str, text: str) -> ChunkObject:
    return ChunkObject(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_title="Doc",
        text=text,
        section_heading="Section",
        section_path="Section",
        page_start=1,
        page_end=1,
    )


def _build_session(base_chunk: ChunkObject) -> SessionState:
    return SessionState(
        session_id="review-session",
        original_query="What is the key risk?",
        facets=["What is the key risk?"],
        citation_index=[
            CitationIndexEntry(
                citation_id=1,
                chunk_id=base_chunk.chunk_id,
                text=base_chunk.text,
                document_id=base_chunk.document_id,
                document_title=base_chunk.document_title,
                section_heading=base_chunk.section_heading,
                section_path=base_chunk.section_path,
                page_number=base_chunk.page_start,
                page_end=base_chunk.page_end,
                retrieval_score=0.9,
                source_facet="facet",
                source_facets=["facet"],
            )
        ],
        parsed_sentences=[
            ParsedSentence(
                sentence_index=0,
                sentence_text="The key risk is delayed procurement sequencing.",
                sentence_type=SentenceType.CLAIM,
                references=[
                    Reference(
                        reference_quote="Delayed procurement sequencing increased schedule risk.",
                        citation_id=1,
                        matched_chunk_id=base_chunk.chunk_id,
                        verified=True,
                    )
                ],
                status=SentenceStatus.VERIFIED,
            )
        ],
    )


def test_get_citation_alternatives_persists_candidates_and_replace_updates_reference() -> None:
    base_chunk = _build_chunk("chunk-base", "Delayed procurement sequencing increased schedule risk.")
    alt_chunk = _build_chunk("chunk-alt", "Procurement sequencing strongly affected schedule volatility.")

    chunk_store = InMemoryChunkStore([base_chunk, alt_chunk])
    retriever = StaticHybridRetriever(
        RetrievedPassage(
            chunk=alt_chunk,
            score=0.8,
            source_facet="facet",
            source_facets=["facet"],
            rank=1,
            retriever="reranked",
        )
    )
    session_store = SessionStore()
    session_store.save(_build_session(base_chunk))

    service = ReviewService(
        session_store=session_store,
        chunk_store=chunk_store,
        hybrid_retriever=retriever,
        scoped_top_k=3,
        ambiguity_gap_threshold=0.05,
    )

    envelope = asyncio.run(service.get_citation_alternatives("review-session", 1))
    assert envelope.citations

    persisted_session = session_store.get("review-session")
    assert any(entry.chunk_id == alt_chunk.chunk_id for entry in persisted_session.citation_index)

    replacement_id = envelope.citations[0].citation_id
    updated = service.replace_with_alternative("review-session", 0, 1, replacement_id)

    replaced_citation = next(entry for entry in updated.citation_index if entry.citation_id == 1)
    assert replaced_citation.chunk_id == alt_chunk.chunk_id

    updated_reference = updated.parsed_sentences[0].references[0]
    assert updated_reference.matched_chunk_id == alt_chunk.chunk_id
    assert updated_reference.replacement_pending is True
    assert any(repl.citation_id == 1 for repl in updated.feedback.citation_replacements)


def test_replace_citation_records_replacement_and_updates_sentence_reference() -> None:
    base_chunk = _build_chunk("chunk-base", "Delayed procurement sequencing increased schedule risk.")
    alt_chunk = _build_chunk("chunk-alt", "Procurement sequencing strongly affected schedule volatility.")

    chunk_store = InMemoryChunkStore([base_chunk, alt_chunk])
    retriever = StaticHybridRetriever(
        RetrievedPassage(
            chunk=alt_chunk,
            score=0.7,
            source_facet="facet",
            source_facets=["facet"],
            rank=1,
            retriever="reranked",
        )
    )
    session_store = SessionStore()
    session_store.save(_build_session(base_chunk))

    service = ReviewService(
        session_store=session_store,
        chunk_store=chunk_store,
        hybrid_retriever=retriever,
        scoped_top_k=3,
        ambiguity_gap_threshold=0.05,
    )

    updated = service.replace_citation(
        "review-session",
        sentence_index=0,
        citation_id=1,
        request=CitationReplacementRequest(
            sentence_index=0,
            replacement_chunk_id=alt_chunk.chunk_id,
        ),
    )

    replaced_citation = next(entry for entry in updated.citation_index if entry.citation_id == 1)
    assert replaced_citation.chunk_id == alt_chunk.chunk_id
    assert replaced_citation.reviewer_note == "Reviewer selected replacement passage."

    updated_reference = updated.parsed_sentences[0].references[0]
    assert updated_reference.matched_chunk_id == alt_chunk.chunk_id
    assert updated_reference.replacement_pending is True
    assert len(updated.feedback.citation_replacements) == 1
    assert updated.feedback.citation_replacements[0].replacement_chunk_id == alt_chunk.chunk_id
