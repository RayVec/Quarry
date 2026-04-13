from __future__ import annotations

from quarry.adapters.interfaces import ChunkStore
from quarry.domain.models import (
    CitationFeedback,
    CitationFeedbackType,
    CitationIndexEntry,
    CitationReplacement,
    CitationReplacementRequest,
    MatchQuality,
    ReviewComment,
    ReviewCommentRequest,
    ScopedRetrievalEnvelope,
    SessionState,
)
from quarry.pipeline.retrieval import HybridRetriever, build_citation_index
from quarry.services.session_store import SessionStore


class ReviewService:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        chunk_store: ChunkStore,
        hybrid_retriever: HybridRetriever,
        scoped_top_k: int,
        ambiguity_gap_threshold: float,
    ) -> None:
        self.session_store = session_store
        self.chunk_store = chunk_store
        self.hybrid_retriever = hybrid_retriever
        self.scoped_top_k = scoped_top_k
        self.ambiguity_gap_threshold = ambiguity_gap_threshold
        self._chunk_lookup = {chunk.chunk_id: chunk for chunk in chunk_store.all_chunks()}

    def get_session(self, session_id: str) -> SessionState:
        return self.session_store.get(session_id)

    def review_snapshot(self, session_id: str) -> SessionState:
        return self.session_store.get(session_id)

    def close_session(self, session_id: str) -> None:
        self.session_store.delete(session_id)

    def add_review_comment(self, session_id: str, request: ReviewCommentRequest) -> SessionState:
        session = self.session_store.get(session_id)
        session.feedback.comments.append(
            ReviewComment(
                text_selection=request.text_selection.strip(),
                char_start=request.char_start,
                char_end=request.char_end,
                comment_text=request.comment_text.strip(),
            )
        )
        return self.session_store.save(session)

    def update_review_comment(self, session_id: str, comment_id: str, comment_text: str) -> SessionState:
        session = self.session_store.get(session_id)
        for comment in session.feedback.comments:
            if comment.comment_id == comment_id:
                comment.comment_text = comment_text.strip()
                break
        return self.session_store.save(session)

    def delete_review_comment(self, session_id: str, comment_id: str) -> SessionState:
        session = self.session_store.get(session_id)
        session.feedback.comments = [comment for comment in session.feedback.comments if comment.comment_id != comment_id]
        session.feedback.resolved_comments = [comment for comment in session.feedback.resolved_comments if comment.comment_id != comment_id]
        return self.session_store.save(session)

    def set_citation_feedback(
        self,
        session_id: str,
        sentence_index: int,
        citation_id: int,
        feedback_type: CitationFeedbackType,
    ) -> SessionState:
        session = self.session_store.get(session_id)
        existing_feedback = next(
            (
                fb
                for fb in session.feedback.citation_feedback
                if fb.sentence_index == sentence_index and fb.citation_id == citation_id
            ),
            None,
        )
        if existing_feedback:
            if feedback_type == CitationFeedbackType.NEUTRAL:
                session.feedback.citation_feedback = [
                    fb
                    for fb in session.feedback.citation_feedback
                    if not (fb.sentence_index == sentence_index and fb.citation_id == citation_id)
                ]
            else:
                existing_feedback.feedback_type = feedback_type
        elif feedback_type != CitationFeedbackType.NEUTRAL:
            session.feedback.citation_feedback.append(
                CitationFeedback(
                    sentence_index=sentence_index,
                    citation_id=citation_id,
                    feedback_type=feedback_type,
                )
            )
        return self.session_store.save(session)

    async def get_citation_alternatives(self, session_id: str, citation_id: int) -> ScopedRetrievalEnvelope:
        session = self.session_store.get(session_id)
        target_citation = next((entry for entry in session.citation_index if entry.citation_id == citation_id), None)
        if target_citation is None:
            return ScopedRetrievalEnvelope(citations=[])

        sentence = None
        for parsed_sentence in session.parsed_sentences:
            sentence_citation_ids = [ref.citation_id for ref in parsed_sentence.references if ref.citation_id is not None]
            if citation_id in sentence_citation_ids:
                sentence = parsed_sentence
                break

        facets = [sentence.sentence_text] if sentence and sentence.sentence_text.strip() else session.facets
        passages, _diagnostics = await self.hybrid_retriever.retrieve(
            original_query=session.original_query,
            facets=facets or session.facets,
            query_type=session.query_type,
        )

        alternative_citations = build_citation_index(
            passages,
            starting_id=max((entry.citation_id for entry in session.citation_index), default=0) + 1,
            ambiguity_gap_threshold=self.ambiguity_gap_threshold,
        )

        existing_chunk_ids = {entry.chunk_id for entry in session.citation_index}
        filtered = [entry for entry in alternative_citations if entry.chunk_id not in existing_chunk_ids][: self.scoped_top_k]
        if filtered:
            return ScopedRetrievalEnvelope(citations=filtered)

        fallback_candidates = [
            chunk
            for chunk in self._chunk_lookup.values()
            if chunk.chunk_id not in existing_chunk_ids and chunk.chunk_id != target_citation.chunk_id
        ][: self.scoped_top_k]

        fallback = [
            CitationIndexEntry(
                citation_id=max((entry.citation_id for entry in session.citation_index), default=0) + index + 1,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section_heading=chunk.section_heading,
                section_path=chunk.section_path,
                page_number=chunk.page_start,
                page_end=chunk.page_end,
                retrieval_score=0.0,
                source_facet="alternative_fallback",
                source_facets=["alternative_fallback"],
            )
            for index, chunk in enumerate(fallback_candidates)
        ]
        return ScopedRetrievalEnvelope(citations=fallback)

    def replace_with_alternative(
        self,
        session_id: str,
        sentence_index: int,
        citation_id: int,
        replacement_citation_id: int,
    ) -> SessionState:
        session = self.session_store.get(session_id)
        target = next((entry for entry in session.citation_index if entry.citation_id == citation_id), None)
        replacement = next((entry for entry in session.citation_index if entry.citation_id == replacement_citation_id), None)
        if target is None or replacement is None:
            return self.session_store.save(session)

        target.chunk_id = replacement.chunk_id
        target.text = replacement.text
        target.document_id = replacement.document_id
        target.document_title = replacement.document_title
        target.section_heading = replacement.section_heading
        target.section_path = replacement.section_path
        target.page_number = replacement.page_number
        target.page_end = replacement.page_end
        target.replacement_pending = True
        target.reviewer_note = "Replaced via citation feedback."

        session.feedback.citation_feedback = [
            fb
            for fb in session.feedback.citation_feedback
            if not (fb.sentence_index == sentence_index and fb.citation_id == citation_id)
        ]
        return self.session_store.save(session)

    async def scoped_retrieval(self, session_id: str, sentence_index: int, citation_id: int) -> ScopedRetrievalEnvelope:
        session = self.session_store.get(session_id)
        if sentence_index < 0 or sentence_index >= len(session.parsed_sentences):
            return ScopedRetrievalEnvelope(citations=[])
        sentence = session.parsed_sentences[sentence_index]
        seed = next((entry for entry in session.citation_index if entry.citation_id == citation_id), None)
        if seed is None:
            return ScopedRetrievalEnvelope(citations=[])

        facets = [sentence.sentence_text] if sentence.sentence_text.strip() else session.facets
        passages, _diagnostics = await self.hybrid_retriever.retrieve(
            original_query=session.original_query,
            facets=facets or session.facets,
            query_type=session.query_type,
        )
        scoped = build_citation_index(
            passages,
            starting_id=max((entry.citation_id for entry in session.citation_index), default=0) + 1,
            ambiguity_gap_threshold=self.ambiguity_gap_threshold,
        )
        existing_chunk_ids = {entry.chunk_id for entry in session.citation_index}
        filtered = [entry for entry in scoped if entry.chunk_id not in existing_chunk_ids][: self.scoped_top_k]
        if filtered:
            return ScopedRetrievalEnvelope(citations=filtered)

        fallback_candidates = [
            chunk
            for chunk in self._chunk_lookup.values()
            if chunk.chunk_id not in existing_chunk_ids and chunk.chunk_id != seed.chunk_id
        ][: self.scoped_top_k]
        fallback = [
            CitationIndexEntry(
                citation_id=max((entry.citation_id for entry in session.citation_index), default=0) + index + 1,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section_heading=chunk.section_heading,
                section_path=chunk.section_path,
                page_number=chunk.page_start,
                page_end=chunk.page_end,
                retrieval_score=0.0,
                source_facet="scoped_fallback",
                source_facets=["scoped_fallback"],
            )
            for index, chunk in enumerate(fallback_candidates)
        ]
        return ScopedRetrievalEnvelope(citations=fallback)

    def replace_citation(
        self,
        session_id: str,
        sentence_index: int,
        citation_id: int,
        request: CitationReplacementRequest,
    ) -> SessionState:
        session = self.session_store.get(session_id)
        replacement = self._chunk_lookup.get(request.replacement_chunk_id)
        if replacement is None:
            return self.session_store.save(session)

        target = next((entry for entry in session.citation_index if entry.citation_id == citation_id), None)
        if target is not None:
            target.chunk_id = replacement.chunk_id
            target.text = replacement.text
            target.document_id = replacement.document_id
            target.document_title = replacement.document_title
            target.section_heading = replacement.section_heading
            target.section_path = replacement.section_path
            target.page_number = replacement.page_start
            target.page_end = replacement.page_end
            target.replacement_pending = True
            target.reviewer_note = "Reviewer selected replacement passage."

        if 0 <= sentence_index < len(session.parsed_sentences):
            sentence = session.parsed_sentences[sentence_index]
            for reference in sentence.references:
                if reference.citation_id == citation_id:
                    reference.matched_chunk_id = replacement.chunk_id
                    reference.document_id = replacement.document_id
                    reference.document_title = replacement.document_title
                    reference.section_heading = replacement.section_heading
                    reference.section_path = replacement.section_path
                    reference.page_number = replacement.page_start
                    reference.replacement_pending = True
            sentence.match_quality = MatchQuality.PARTIAL

        if not any(replacement.citation_id == citation_id for replacement in session.feedback.citation_replacements):
            session.feedback.citation_replacements.append(
                CitationReplacement(citation_id=citation_id, replacement_chunk_id=request.replacement_chunk_id)
            )
        return self.session_store.save(session)

    def undo_replacement(self, session_id: str, citation_id: int) -> SessionState:
        session = self.session_store.get(session_id)
        for entry in session.citation_index:
            if entry.citation_id == citation_id:
                entry.replacement_pending = False
                entry.reviewer_note = None
        for sentence in session.parsed_sentences:
            for reference in sentence.references:
                if reference.citation_id == citation_id:
                    reference.replacement_pending = False
        session.feedback.citation_replacements = [
            replacement for replacement in session.feedback.citation_replacements if replacement.citation_id != citation_id
        ]
        return self.session_store.save(session)
