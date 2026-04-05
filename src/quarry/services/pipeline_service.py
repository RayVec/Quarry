from __future__ import annotations

from collections import Counter
from uuid import uuid4

from quarry.adapters.interfaces import ChunkStore
from quarry.domain.models import (
    CitationIndexEntry,
    CitationReplacement,
    CitationReplacementRequest,
    ConfidenceLabel,
    FeedbackState,
    GenerationRequest,
    MatchQuality,
    ParsedSentence,
    QueryRequest,
    QueryProgressStage,
    QueryRunStatus,
    ResponseMode,
    ReviewWarning,
    RetrievalFilters,
    ScopedRetrievalEnvelope,
    QueryType,
    ReviewComment,
    ReviewCommentRequest,
    SessionState,
    SentenceStatus,
    SentenceType,
    UIMessage,
    UIMessageLevel,
)
from quarry.pipeline.decomposition import QueryDecomposer
from quarry.pipeline.generation import AnswerGenerator, SentenceRegenerator
from quarry.pipeline.parsing import parse_generated_response, render_parsed_sentences
from quarry.pipeline.retrieval import HybridRetriever, build_citation_index
from quarry.pipeline.verification import VerificationService
from quarry.services.session_store import SessionStore
from quarry.services.session_store import SessionNotFoundError
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed


logger = logger_with_trace(__name__)


class PipelineService:
    SINGLE_HOP_GENERATION_MAX_CITATIONS = 8
    REGENERATION_SIMILARITY_THRESHOLD = 0.8
    REGENERATION_MIN_QUOTE_WORDS = 8
    REFINE_PROMPT_WINDOW_RATIO = 0.8

    def __init__(
        self,
        *,
        chunk_store: ChunkStore,
        query_decomposer: QueryDecomposer,
        hybrid_retriever: HybridRetriever,
        answer_generator: AnswerGenerator,
        sentence_regenerator: SentenceRegenerator,
        verifier: VerificationService,
        session_store: SessionStore,
        scoped_top_k: int,
        refinement_token_budget: int,
        ambiguity_gap_threshold: float = 0.05,
        generation_provider: str = "unknown",
        parser_provider: str = "unknown",
        runtime_mode: str = "hybrid",
        runtime_profile: str = "gpu",
        local_model_status: dict[str, str] | None = None,
        active_model_ids: list[str] | None = None,
    ) -> None:
        self.chunk_store = chunk_store
        self.query_decomposer = query_decomposer
        self.hybrid_retriever = hybrid_retriever
        self.answer_generator = answer_generator
        self.sentence_regenerator = sentence_regenerator
        self.verifier = verifier
        self.session_store = session_store
        self.scoped_top_k = scoped_top_k
        self.refinement_token_budget = refinement_token_budget
        self.ambiguity_gap_threshold = ambiguity_gap_threshold
        self.generation_provider = generation_provider
        self.parser_provider = parser_provider
        self.runtime_mode = runtime_mode
        self.runtime_profile = runtime_profile
        self.local_model_status = local_model_status or {}
        self.active_model_ids = active_model_ids or []

    def begin_query(self, request: QueryRequest) -> SessionState:
        session = self._create_base_session(str(uuid4()), request.query)
        return self._save_stage(
            session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.UNDERSTANDING,
            label="Reading your question",
            detail="I'm getting clear on what you want to know.",
        )

    async def run_query_for_session(self, session_id: str, request: QueryRequest) -> SessionState:
        try:
            return await self.run_query(request, session_id=session_id)
        except Exception as exc:
            try:
                session = self.session_store.get(session_id)
            except SessionNotFoundError:
                session = self._create_base_session(session_id, request.query)
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.ERROR,
                    code="query_failed",
                    message="Something went wrong while preparing the answer.",
                )
            )
            self._save_stage(
                session,
                status=QueryRunStatus.FAILED,
                stage=QueryProgressStage.FAILED,
                label="I hit a problem",
                detail="The answer could not be completed this time.",
            )
            logger.exception(
                "query failed",
                extra={
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            return session

    async def run_query(self, request: QueryRequest, *, session_id: str | None = None) -> SessionState:
        query_start = timed()
        session_id = session_id or str(uuid4())
        try:
            base_session = self.session_store.get(session_id)
        except SessionNotFoundError:
            base_session = self._create_base_session(session_id, request.query)
        base_session.original_query = request.query
        base_session.query_type = None
        base_session.facets = []
        base_session.citation_index = []
        base_session.generated_response = ""
        base_session.parsed_sentences = []
        base_session.feedback = FeedbackState()
        base_session.retrieval_diagnostics = []
        base_session.ui_messages = []
        base_session.removed_ungrounded_claim_count = 0
        base_session.response_mode = ResponseMode.RESPONSE_REVIEW
        base_session.clarification_suggestions = []
        logger.info(
            "=== QUERY START ===",
            extra={
                "session_id": session_id,
                "query_preview": request.query[:200],
                "runtime_mode": self.runtime_mode,
                "runtime_profile": self.runtime_profile,
            },
        )
        logger.info(
            "query received",
            extra={
                "session_id": session_id,
                "query": request.query,
                "query_preview": request.query[:200],
                "runtime_mode": self.runtime_mode,
                "runtime_profile": self.runtime_profile,
                "generation_provider": self.generation_provider,
                "console_visible": False,
            },
        )
        self._save_stage(
            base_session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.UNDERSTANDING,
            label="Reading your question",
            detail="I'm getting clear on what you want to know.",
        )
        decomposition_start = timed()
        decomposition = await self.query_decomposer.decompose(request.query)
        base_session.query_type = decomposition.query_type
        base_session.facets = decomposition.facets
        logger.info(
            "query decomposition complete",
            extra={
                "session_id": session_id,
                "query_type": decomposition.query_type.value,
                "clarification_required": decomposition.clarification_required,
                "facets": decomposition.facets,
                "facet_count": len(decomposition.facets),
                "latency_ms": elapsed_ms(decomposition_start),
            },
        )

        if decomposition.clarification_required:
            base_session.response_mode = ResponseMode.CLARIFICATION_REQUIRED
            base_session.clarification_suggestions = self._build_clarification_suggestions(request.query)
            base_session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.INFO,
                    code="clarification_required",
                    message="Please provide a more specific query with at least one concrete topic or metric.",
                )
            )
            self._save_stage(
                base_session,
                status=QueryRunStatus.COMPLETED,
                stage=QueryProgressStage.CLARIFICATION,
                label="I need a little more detail",
                detail="I can't search well yet because the question is still too broad or unclear.",
            )
            logger.info(
                "query halted for clarification",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            logger.info(
                "=== QUERY END ===",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            return self.session_store.save(base_session)

        retrieval_start = timed()
        self._save_stage(
            base_session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.SEARCHING,
            label="Looking through the reports",
            detail="I'm finding the parts of the documents that seem most relevant.",
        )
        retrieved, diagnostics = await self.hybrid_retriever.retrieve(
            original_query=request.query,
            facets=decomposition.facets,
            query_type=decomposition.query_type,
        )
        citation_index = build_citation_index(retrieved, ambiguity_gap_threshold=self.ambiguity_gap_threshold)
        base_session.retrieval_diagnostics = diagnostics
        logger.info(
            "retrieval complete",
            extra={
                "session_id": session_id,
                "facet_count": len(decomposition.facets),
                "retrieved_passage_count": len(retrieved),
                "citation_count": len(citation_index),
                "diagnostics": [diagnostic.model_dump() for diagnostic in diagnostics],
                "latency_ms": elapsed_ms(retrieval_start),
            },
        )

        if not citation_index:
            base_session.response_mode = ResponseMode.GENERATION_FAILED
            base_session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.WARNING,
                    code="no_retrieval_hits",
                    message="I couldn't find strong enough evidence in the reports. Try a narrower question or add more detail.",
                )
            )
            self._save_stage(
                base_session,
                status=QueryRunStatus.COMPLETED,
                stage=QueryProgressStage.COMPLETED,
                label="I finished looking",
                detail="I couldn't find strong enough passages for a grounded answer.",
            )
            logger.info(
                "query completed with no retrieval hits",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            logger.info(
                "=== QUERY END ===",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            return self.session_store.save(base_session)

        generation_citations = self._trim_generation_citations(citation_index, decomposition.query_type)
        self._save_stage(
            base_session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.EVIDENCE,
            label="Pulling together the best evidence",
            detail="I'm narrowing down to the passages I trust most for this answer.",
        )
        generation_request = GenerationRequest(
            original_query=request.query,
            facets=decomposition.facets,
            citation_index=generation_citations,
        )
        logger.info(
            "generation started",
            extra={
                "session_id": session_id,
                "mode": generation_request.mode,
                "citation_count": len(generation_request.citation_index),
            },
        )
        self._save_stage(
            base_session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.WRITING,
            label="Writing the answer",
            detail="I'm turning the evidence into a clear response.",
        )
        raw_response = await self._generate_with_retry(generation_request)
        if raw_response is None:
            base_session.citation_index = citation_index
            base_session.response_mode = ResponseMode.GENERATION_FAILED
            base_session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.ERROR,
                    code="generation_failed",
                    message="I found relevant evidence, but I couldn't turn it into a verified answer this time.",
                )
            )
            self._save_stage(
                base_session,
                status=QueryRunStatus.COMPLETED,
                stage=QueryProgressStage.COMPLETED,
                label="I finished looking",
                detail="I found passages, but I couldn't turn them into a verified answer this time.",
            )
            logger.info(
                "generation failed",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "citation_count": len(citation_index),
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            logger.info(
                "=== QUERY END ===",
                extra={
                    "session_id": session_id,
                    "response_mode": base_session.response_mode.value,
                    "latency_ms": elapsed_ms(query_start),
                },
            )
            return self.session_store.save(base_session)
        self._save_stage(
            base_session,
            status=QueryRunStatus.RUNNING,
            stage=QueryProgressStage.CHECKING,
            label="Checking the answer against the reports",
            detail="I'm making sure the wording still matches the source text before I show it.",
        )
        final_response, parsed_sentences, citation_index, removed_sentence_count = await self._process_response(raw_response, citation_index)
        base_session.generated_response = final_response
        base_session.parsed_sentences = parsed_sentences
        base_session.citation_index = citation_index
        base_session.removed_ungrounded_claim_count = removed_sentence_count
        base_session.response_mode = self._determine_response_mode(parsed_sentences)
        if removed_sentence_count:
            base_session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.WARNING,
                    code="removed_unverified_claims",
                    message=self._removed_unverified_sentences_message(removed_sentence_count),
                )
            )
        if base_session.response_mode == ResponseMode.GENERATION_FAILED:
            base_session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.WARNING,
                    code="generation_unusable",
                    message="The draft answer could not be grounded strongly enough to show as a verified response.",
                )
            )
        self._save_stage(
            base_session,
            status=QueryRunStatus.COMPLETED,
            stage=QueryProgressStage.COMPLETED,
            label="Answer ready",
            detail="I've finished preparing the response.",
        )
        logger.info(
            "query completed",
            extra={
                "session_id": session_id,
                "response_mode": base_session.response_mode.value,
                "sentence_summary": self._sentence_status_summary(parsed_sentences),
                "citation_count": len(citation_index),
                "latency_ms": elapsed_ms(query_start),
            },
        )
        logger.info(
            "=== QUERY END ===",
            extra={
                "session_id": session_id,
                "response_mode": base_session.response_mode.value,
                "latency_ms": elapsed_ms(query_start),
            },
        )
        return self.session_store.save(base_session)

    def get_session(self, session_id: str) -> SessionState:
        return self.session_store.get(session_id)

    def review_snapshot(self, session_id: str) -> SessionState:
        return self.session_store.get(session_id)

    def close_session(self, session_id: str) -> None:
        self.session_store.delete(session_id)

    def add_review_comment(self, session_id: str, request: ReviewCommentRequest) -> SessionState:
        session = self.session_store.get(session_id)
        sentence_type = None
        sentence_text = None
        if request.sentence_index is not None:
            sentence = session.parsed_sentences[request.sentence_index]
            sentence.disagreement_flagged = True
            sentence_type = sentence.sentence_type
            sentence_text = sentence.sentence_text
        session.feedback.comments.append(
            ReviewComment(
                sentence_index=request.sentence_index,
                sentence_type=sentence_type,
                sentence_text=sentence_text,
                comment=request.comment.strip(),
            )
        )
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
            for chunk in self.chunk_store.all_chunks()
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
        replacement = next((entry for entry in self.chunk_store.all_chunks() if entry.chunk_id == request.replacement_chunk_id), None)
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

        if not any(
            replacement.citation_id == citation_id for replacement in session.feedback.citation_replacements
        ):
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

    async def refine(self, session_id: str) -> SessionState:
        refine_start = timed()
        session = self.session_store.get(session_id)
        prior_response = session.generated_response
        working_citations = list(session.citation_index)
        sentence_comments = [comment for comment in session.feedback.comments if comment.sentence_index is not None]
        response_comments = [comment.comment for comment in session.feedback.comments if comment.sentence_index is None]
        replacement_sentence_comments = {
            comment.sentence_index: comment.comment
            for comment in sentence_comments
            if comment.sentence_index is not None
        }
        logger.info(
            "refinement started",
            extra={
                "session_id": session_id,
                "comment_count": len(session.feedback.comments),
                "sentence_comment_count": len(sentence_comments),
            },
        )
        try:
            step2_rewrites = 0
            step2_rewrites = 0
            if replacement_sentence_comments:
                comment_lines = [f"Sentence {idx}: {note}" for idx, note in sorted(replacement_sentence_comments.items())]
                refine_request = GenerationRequest(
                    original_query=session.original_query,
                    facets=session.facets,
                    citation_index=self._trim_citations_to_budget(working_citations),
                    mode="refinement",
                    existing_response=render_parsed_sentences(session.parsed_sentences),
                    sentence_comments=[
                        ReviewComment(
                            sentence_index=idx,
                            sentence_type=session.parsed_sentences[idx].sentence_type if idx < len(session.parsed_sentences) else None,
                            sentence_text=session.parsed_sentences[idx].sentence_text if idx < len(session.parsed_sentences) else None,
                            comment=note,
                        )
                        for idx, note in sorted(replacement_sentence_comments.items())
                    ],
                    disagreement_notes=comment_lines,
                    disagreement_contexts=[],
                )
                self._check_refinement_prompt_budget(refine_request, session.parsed_sentences)
                refined_raw = await self._generate_with_retry(refine_request)
                if refined_raw is not None:
                    next_sentences = parse_generated_response(refined_raw)
                    changed_indices = self._diff_changed_sentence_indices(session.parsed_sentences, next_sentences)
                    session.parsed_sentences = next_sentences
                    if changed_indices:
                        session.parsed_sentences, working_citations = await self._verify_and_reindex_changed_sentences(
                            session.parsed_sentences,
                            working_citations,
                            changed_indices,
                        )
                        step2_rewrites = len(changed_indices)

            step3_additions = 0
            if response_comments:
                passages, diagnostics = await self.hybrid_retriever.retrieve(
                    original_query=session.original_query,
                    facets=response_comments,
                )
                session.retrieval_diagnostics.extend(diagnostics)
                working_citations = self._append_unique_citations(
                    working_citations,
                    build_citation_index(
                        passages,
                        starting_id=max((entry.citation_id for entry in working_citations), default=0) + 1,
                        ambiguity_gap_threshold=self.ambiguity_gap_threshold,
                    ),
                )
                supplement_request = GenerationRequest(
                    original_query=session.original_query,
                    facets=session.facets,
                    citation_index=self._trim_citations_to_budget(working_citations),
                    mode="supplement",
                    existing_response=render_parsed_sentences(session.parsed_sentences),
                    response_comments=response_comments,
                    selected_facets=response_comments,
                )
                supplemental_raw = await self._generate_with_retry(supplement_request)
                if supplemental_raw is not None:
                    supplemental_parsed = parse_generated_response(supplemental_raw)
                    offset = len(session.parsed_sentences)
                    for sentence in supplemental_parsed:
                        sentence.sentence_index += offset
                    session.parsed_sentences.extend(supplemental_parsed)
                    changed_indices = set(range(offset, len(session.parsed_sentences)))
                    if changed_indices:
                        session.parsed_sentences, working_citations = await self._verify_and_reindex_changed_sentences(
                            session.parsed_sentences,
                            working_citations,
                            changed_indices,
                        )
                        step3_additions = len(changed_indices)
        except Exception:
            session.generated_response = prior_response
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.ERROR,
                    code="refinement_failed",
                    message="Refinement was unsuccessful; the previous response has been preserved.",
                )
            )
            logger.info(
                "refinement failed",
                extra={"session_id": session_id, "latency_ms": elapsed_ms(refine_start)},
            )
            return self.session_store.save(session)

        session.parsed_sentences, removed_sentence_count = self._apply_lingering_grounding_policy(session.parsed_sentences)
        self._annotate_match_quality(session.parsed_sentences)
        session.generated_response = render_parsed_sentences(session.parsed_sentences)
        session.citation_index = working_citations
        session.removed_ungrounded_claim_count = removed_sentence_count
        session.feedback = FeedbackState()
        session.refinement_count += 1
        session.response_mode = self._determine_response_mode(session.parsed_sentences)
        summary_parts: list[str] = []
        if step2_rewrites:
            summary_parts.append(f"rewrote {step2_rewrites} sentences")
        if step3_additions:
            summary_parts.append(f"added {step3_additions} new sentences")
        if summary_parts:
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.INFO,
                    code="refine_summary",
                    message=", ".join(summary_parts).capitalize() + ".",
                )
            )
        if removed_sentence_count:
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.WARNING,
                    code="removed_unverified_claims",
                    message=self._removed_unverified_sentences_message(removed_sentence_count),
                )
            )
        logger.info(
            "refinement complete",
            extra={
                "session_id": session_id,
                "response_mode": session.response_mode.value,
                "sentence_summary": self._sentence_status_summary(session.parsed_sentences),
                "citation_count": len(working_citations),
                "latency_ms": elapsed_ms(refine_start),
            },
        )
        return self.session_store.save(session)

    async def _process_response(
        self,
        raw_response: str,
        citation_index: list[CitationIndexEntry],
    ) -> tuple[str, list, list[CitationIndexEntry], int]:
        process_start = timed()
        parsed_sentences = parse_generated_response(raw_response)
        logger.info(
            "response parsed",
            extra={
                "sentence_summary": self._sentence_type_summary(parsed_sentences),
                "raw_response": raw_response,
                "console_visible": False,
            },
        )
        verification = self.verifier.verify_exact_matches(parsed_sentences, list(citation_index))
        parsed_sentences = verification.parsed_sentences
        updated_citations = verification.citation_index
        logger.info(
            "exact match verification complete",
            extra={
                "sentence_summary": self._sentence_status_summary(parsed_sentences),
                "citation_count": len(updated_citations),
            },
        )

        exhausted_regeneration_indices: set[int] = set()
        failed_regeneration_outputs: dict[int, str] = {}
        for _ in range(2):
            pending = [
                sentence
                for sentence in parsed_sentences
                if self._needs_regeneration(sentence) and sentence.sentence_index not in exhausted_regeneration_indices
            ]
            if not pending:
                break
            changed_indices: set[int] = set()
            regeneration_inputs: dict[int, str] = {}
            for sentence in pending:
                regeneration_request = self.sentence_regenerator.regenerate(
                    sentence,
                    updated_citations,
                    failed_regeneration_response=failed_regeneration_outputs.get(sentence.sentence_index),
                )
                regeneration_inputs[sentence.sentence_index] = sentence.sentence_text
                logger.info(
                    "sentence regeneration started",
                    extra={
                        "sentence_index": sentence.sentence_index,
                        "sentence_type": sentence.sentence_type.value,
                        "sentence_text": sentence.sentence_text,
                        "console_visible": False,
                    },
                )
                try:
                    regenerated_raw = await self._generate_with_retry(regeneration_request)
                    if regenerated_raw is None:
                        raise RuntimeError("regeneration failed")
                except Exception:
                    regenerated_raw = self.sentence_regenerator.deterministic_rewrite(sentence, updated_citations)
                replacement = parse_generated_response(regenerated_raw)
                replacement_sentence = self._select_regeneration_replacement(replacement)
                if replacement_sentence is None:
                    fallback_raw = self.sentence_regenerator.deterministic_rewrite(sentence, updated_citations)
                    replacement_sentence = self._select_regeneration_replacement(parse_generated_response(fallback_raw))
                if replacement_sentence is None:
                    continue
                for reference in replacement_sentence.references:
                    reference.minimum_quote_words = self.REGENERATION_MIN_QUOTE_WORDS
                replacement_sentence.sentence_index = sentence.sentence_index
                parsed_sentences[sentence.sentence_index] = replacement_sentence
                changed_indices.add(sentence.sentence_index)
                logger.info(
                    "sentence regeneration complete",
                    extra={
                        "sentence_index": sentence.sentence_index,
                        "replacement_text": replacement_sentence.sentence_text,
                        "replacement_status": replacement_sentence.status.value,
                        "console_visible": False,
                    },
                )
            if changed_indices:
                verification = self.verifier.verify_exact_matches(
                    parsed_sentences,
                    updated_citations,
                    sentence_indices=changed_indices,
                )
                parsed_sentences = verification.parsed_sentences
                updated_citations = verification.citation_index
                logger.info(
                    "post-regeneration verification complete",
                    extra={
                        "sentence_summary": self._sentence_status_summary(parsed_sentences),
                        "reverified_sentence_indices": sorted(changed_indices),
                        "console_visible": False,
                    },
                )
                for sentence_index in changed_indices:
                    sentence = parsed_sentences[sentence_index]
                    if sentence.status != SentenceStatus.UNGROUNDED:
                        failed_regeneration_outputs.pop(sentence_index, None)
                        exhausted_regeneration_indices.discard(sentence_index)
                        continue
                    previous_attempt = regeneration_inputs.get(sentence_index)
                    if previous_attempt is None:
                        continue
                    failed_regeneration_outputs[sentence_index] = sentence.sentence_text
                    similarity = self._sentence_similarity(sentence.sentence_text, previous_attempt)
                    if similarity >= self.REGENERATION_SIMILARITY_THRESHOLD:
                        exhausted_regeneration_indices.add(sentence_index)
                        logger.info(
                            "sentence regeneration halted after similar failed rewrite",
                            extra={
                                "sentence_index": sentence_index,
                                "similarity": round(similarity, 3),
                                "threshold": self.REGENERATION_SIMILARITY_THRESHOLD,
                                "console_visible": False,
                            },
                        )

        parsed_sentences = await self.verifier.score_confidence(parsed_sentences)
        parsed_sentences, removed_sentence_count = self._apply_lingering_grounding_policy(parsed_sentences)
        self._annotate_match_quality(parsed_sentences)
        logger.info(
            "confidence scoring complete",
            extra={
                "sentence_summary": self._sentence_status_summary(parsed_sentences),
                "removed_ungrounded_claim_count": removed_sentence_count,
                "latency_ms": elapsed_ms(process_start),
            },
        )
        return render_parsed_sentences(parsed_sentences), parsed_sentences, updated_citations, removed_sentence_count

    def _select_regeneration_replacement(self, replacements: list[ParsedSentence]) -> ParsedSentence | None:
        usable = [sentence for sentence in replacements if sentence.sentence_text.strip()]
        if not usable:
            return None
        selected = max(
            usable,
            key=lambda sentence: (
                sentence.sentence_type != SentenceType.STRUCTURE,
                bool(sentence.references) or sentence.status == SentenceStatus.NO_REF,
                len(sentence.sentence_text.strip()),
            ),
        )
        return selected.model_copy(deep=True)

    def _needs_regeneration(self, sentence) -> bool:
        return sentence.status == SentenceStatus.UNGROUNDED and sentence.sentence_type in {SentenceType.CLAIM, SentenceType.SYNTHESIS}

    def _sentence_similarity(self, current_text: str, previous_text: str) -> float:
        current_tokens = self._normalized_tokens(current_text)
        previous_tokens = self._normalized_tokens(previous_text)
        if not current_tokens and not previous_tokens:
            return 1.0
        if not current_tokens or not previous_tokens:
            return 0.0
        overlap = sum((Counter(current_tokens) & Counter(previous_tokens)).values())
        return overlap / max(len(current_tokens), len(previous_tokens), 1)

    def _normalized_tokens(self, text: str) -> list[str]:
        cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
        return [token for token in cleaned.split() if token]

    def _append_unique_citations(self, existing: list[CitationIndexEntry], candidate_citations: list[CitationIndexEntry]) -> list[CitationIndexEntry]:
        merged = list(existing)
        seen_chunk_ids = {citation.chunk_id for citation in existing}
        next_id = max((citation.citation_id for citation in existing), default=0) + 1
        for citation in candidate_citations:
            if citation.chunk_id in seen_chunk_ids:
                continue
            merged.append(citation.model_copy(update={"citation_id": next_id}))
            seen_chunk_ids.add(citation.chunk_id)
            next_id += 1
        return merged

    def _trim_citations_to_budget(self, citations: list[CitationIndexEntry]) -> list[CitationIndexEntry]:
        prioritized = sorted(citations, key=lambda citation: (citation.source_facet == "quote_discovery", -citation.retrieval_score))
        selected: list[CitationIndexEntry] = []
        used_tokens = 0
        for citation in prioritized:
            token_estimate = max(len(citation.text.split()), 1)
            if used_tokens + token_estimate > self.refinement_token_budget:
                continue
            selected.append(citation)
            used_tokens += token_estimate
        return selected or citations[: min(len(citations), 10)]

    def _trim_generation_citations(
        self,
        citations: list[CitationIndexEntry],
        query_type: QueryType,
    ) -> list[CitationIndexEntry]:
        if query_type != QueryType.SINGLE_HOP or len(citations) <= self.SINGLE_HOP_GENERATION_MAX_CITATIONS:
            return citations

        selected: list[CitationIndexEntry] = []
        seen_sections: set[tuple[str, str]] = set()
        for citation in citations:
            key = (citation.document_id, citation.section_path)
            if key in seen_sections:
                continue
            selected.append(citation)
            seen_sections.add(key)
            if len(selected) >= self.SINGLE_HOP_GENERATION_MAX_CITATIONS:
                return selected

        for citation in citations:
            if citation in selected:
                continue
            selected.append(citation)
            if len(selected) >= self.SINGLE_HOP_GENERATION_MAX_CITATIONS:
                break
        return selected

    def _create_base_session(self, session_id: str, original_query: str) -> SessionState:
        session = SessionState(
            session_id=session_id,
            original_query=original_query,
            generation_provider=self.generation_provider,
            parser_provider=self.parser_provider,
            runtime_mode=self.runtime_mode,
            runtime_profile=self.runtime_profile,
            local_model_status=self.local_model_status,
            active_model_ids=self.active_model_ids,
        )
        return self.session_store.save(session)

    def _save_stage(
        self,
        session: SessionState,
        *,
        status: QueryRunStatus,
        stage: QueryProgressStage,
        label: str,
        detail: str,
    ) -> SessionState:
        session.query_status = status
        session.query_stage = stage
        session.query_stage_label = label
        session.query_stage_detail = detail
        return self.session_store.save(session)

    def _determine_response_mode(self, parsed_sentences: list) -> ResponseMode:
        visible_claims = [sentence for sentence in parsed_sentences if sentence.sentence_type == SentenceType.CLAIM]
        visible_synthesis = [sentence for sentence in parsed_sentences if sentence.sentence_type == SentenceType.SYNTHESIS]
        if visible_synthesis:
            return ResponseMode.RESPONSE_REVIEW
        if not visible_claims:
            return ResponseMode.GENERATION_FAILED
        if all(sentence.status in {SentenceStatus.UNGROUNDED, SentenceStatus.NO_REF} for sentence in visible_claims):
            return ResponseMode.GENERATION_FAILED
        return ResponseMode.RESPONSE_REVIEW

    def _apply_lingering_grounding_policy(self, parsed_sentences: list) -> tuple[list, int]:
        removed_sentence_count = 0
        retained_sentences: list = []
        for sentence in parsed_sentences:
            if sentence.status in {SentenceStatus.UNGROUNDED, SentenceStatus.NO_REF} and sentence.sentence_type in {
                SentenceType.CLAIM,
                SentenceType.SYNTHESIS,
            }:
                removed_sentence_count += 1
                continue
            retained_sentences.append(sentence)

        for index, sentence in enumerate(retained_sentences):
            sentence.sentence_index = index

        if removed_sentence_count:
            logger.info(
                "ungrounded sentences removed from final response",
                extra={
                    "removed_ungrounded_claim_count": removed_sentence_count,
                    "remaining_sentence_count": len(retained_sentences),
                    "console_visible": False,
                },
            )

        return retained_sentences, removed_sentence_count

    async def _generate_with_retry(self, request: GenerationRequest) -> str | None:
        attempts = 0
        current_request = request
        while attempts < 2:
            attempts += 1
            logger.info(
                "generation attempt started",
                extra={
                    "mode": current_request.mode,
                    "attempt": attempts,
                    "citation_count": len(current_request.citation_index),
                    "selected_facets": current_request.selected_facets,
                    "console_visible": False,
                },
            )
            try:
                raw_response = await self.answer_generator.generate(current_request)
            except Exception:
                raw_response = ""
            if self._is_valid_generation(raw_response):
                logger.info(
                    "generation attempt succeeded",
                    extra={
                        "mode": current_request.mode,
                        "attempt": attempts,
                        "raw_response": raw_response,
                        "console_visible": False,
                    },
                )
                return raw_response
            logger.info(
                "generation attempt failed validation",
                extra={
                    "mode": current_request.mode,
                    "attempt": attempts,
                    "raw_response": raw_response,
                    "console_visible": False,
                },
            )
            current_request = current_request.model_copy(update={"repair_prior_response": raw_response})
        return None

    def _is_valid_generation(self, raw_response: str) -> bool:
        if not raw_response or not raw_response.strip():
            return False
        parsed = parse_generated_response(raw_response)
        if not parsed:
            return False
        if all(sentence.sentence_type == SentenceType.STRUCTURE for sentence in parsed):
            return False
        if all(not sentence.raw_text for sentence in parsed):
            return False
        return True

    async def _verify_and_reindex_changed_sentences(
        self,
        parsed_sentences: list[ParsedSentence],
        citation_index: list[CitationIndexEntry],
        changed_indices: set[int],
    ) -> tuple[list[ParsedSentence], list[CitationIndexEntry]]:
        verification = self.verifier.verify_exact_matches(
            parsed_sentences,
            list(citation_index),
            sentence_indices=changed_indices,
        )
        verified_sentences = verification.parsed_sentences
        verified_citations = verification.citation_index
        scored_sentences = await self.verifier.score_confidence(verified_sentences)
        for index, sentence in enumerate(scored_sentences):
            sentence.sentence_index = index
        return scored_sentences, verified_citations

    def _diff_changed_sentence_indices(
        self,
        prior_sentences: list[ParsedSentence],
        next_sentences: list[ParsedSentence],
    ) -> set[int]:
        aligned_prior: set[int] = set()
        changed_next_indices: set[int] = set()
        for next_index, next_sentence in enumerate(next_sentences):
            best_prior_index = -1
            best_similarity = 0.0
            for prior_index, prior_sentence in enumerate(prior_sentences):
                if prior_index in aligned_prior:
                    continue
                similarity = self._sentence_similarity(next_sentence.sentence_text, prior_sentence.sentence_text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_prior_index = prior_index
            if best_prior_index < 0 or best_similarity < self.REGENERATION_SIMILARITY_THRESHOLD:
                changed_next_indices.add(next_index)
                continue
            aligned_prior.add(best_prior_index)
        for prior_index, _ in enumerate(prior_sentences):
            if prior_index not in aligned_prior and prior_index < len(next_sentences):
                changed_next_indices.add(prior_index)
        return changed_next_indices

    def _check_refinement_prompt_budget(
        self,
        request: GenerationRequest,
        parsed_sentences: list[ParsedSentence],
    ) -> None:
        passage_tokens = sum(len(citation.text.split()) for citation in request.citation_index)
        response_tokens = sum(len(sentence.sentence_text.split()) for sentence in parsed_sentences)
        comment_tokens = sum(len(comment.comment.split()) for comment in request.sentence_comments)
        total_estimate = passage_tokens + response_tokens + comment_tokens
        threshold = int(self.refinement_token_budget * self.REFINE_PROMPT_WINDOW_RATIO)
        if total_estimate > threshold:
            logger.info(
                "refine prompt approaching window limit",
                extra={
                    "estimated_tokens": total_estimate,
                    "threshold": threshold,
                    "sentence_count": len(parsed_sentences),
                    "comment_count": len(request.sentence_comments),
                    "console_visible": False,
                },
            )

    def _build_clarification_suggestions(self, query: str) -> list[str]:
        normalized = " ".join(query.split()).strip().rstrip("?")
        if not normalized:
            return [
                "What findings do the CII reports have about schedule risk?",
                "How does procurement planning affect project outcomes in the CII reports?",
                "What recommendations do the CII reports make about modular construction?",
            ]

        lowered = normalized.lower()
        token_count = len(normalized.split())
        if token_count <= 2:
            return [
                f"What findings do the CII reports have about {lowered}?",
                f"How does {lowered} affect project cost, schedule, or safety?",
                f"What recommendations do the CII reports make about {lowered}?",
            ]

        return [
            f"What specific outcome or metric are you asking about in {normalized}?",
            f"What aspect of {normalized} should QUARRY focus on: cost, schedule, safety, or implementation?",
            f"Can you rewrite {normalized} with a concrete topic, project phase, or report section?",
        ]

    def _removed_unverified_sentences_message(self, removed_sentence_count: int) -> str:
        if removed_sentence_count == 1:
            return "One sentence was removed because it could not be verified against the source documents."
        return f"{removed_sentence_count} sentences were removed because they could not be verified against the source documents."

    def _annotate_match_quality(self, parsed_sentences: list[ParsedSentence]) -> None:
        for sentence in parsed_sentences:
            if sentence.sentence_type == SentenceType.STRUCTURE or sentence.status == SentenceStatus.NO_REF:
                sentence.match_quality = MatchQuality.NONE
                continue

            verified_refs = [reference for reference in sentence.references if reference.verified]
            if not verified_refs:
                sentence.match_quality = MatchQuality.NONE
                continue

            has_short_anchor = any(
                reference.minimum_quote_words is not None
                and reference.minimum_quote_words < self.verifier.DEFAULT_MIN_QUOTE_WORDS
                for reference in verified_refs
            )
            has_partial_label = any(reference.confidence_label == ConfidenceLabel.PARTIALLY_SUPPORTED for reference in verified_refs)
            has_unknown_confidence = any(reference.confidence_label is None or reference.confidence_unknown for reference in verified_refs)

            if sentence.status == SentenceStatus.VERIFIED and not has_short_anchor and not has_partial_label and not has_unknown_confidence:
                sentence.match_quality = MatchQuality.STRONG
            else:
                sentence.match_quality = MatchQuality.PARTIAL

    def _sentence_type_summary(self, parsed_sentences: list) -> dict[str, int]:
        summary = {"claim": 0, "synthesis": 0, "structure": 0, "no_ref": 0}
        for sentence in parsed_sentences:
            summary[sentence.sentence_type.value] = summary.get(sentence.sentence_type.value, 0) + 1
            if sentence.status == SentenceStatus.NO_REF:
                summary["no_ref"] += 1
        return summary

    def _sentence_status_summary(self, parsed_sentences: list) -> dict[str, int]:
        summary: dict[str, int] = {}
        for sentence in parsed_sentences:
            key = sentence.status.value
            summary[key] = summary.get(key, 0) + 1
        return summary
