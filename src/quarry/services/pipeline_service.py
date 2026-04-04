from __future__ import annotations

from collections import Counter
from uuid import uuid4

from quarry.adapters.interfaces import ChunkStore
from quarry.domain.models import (
    CitationIndexEntry,
    CitationMismatch,
    CitationMismatchRequest,
    CitationReplacement,
    CitationReplacementRequest,
    ClaimDisagreement,
    ClaimDisagreementRequest,
    FacetGapRequest,
    FeedbackState,
    GenerationRequest,
    QueryRequest,
    QueryProgressStage,
    QueryRunStatus,
    ResponseMode,
    ReviewWarning,
    RetrievalFilters,
    QueryType,
    ScopedRetrievalRequest,
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
        runtime_profile: str = "full_local_transformers",
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

    def add_citation_mismatch(self, session_id: str, request: CitationMismatchRequest) -> SessionState:
        session = self.session_store.get(session_id)
        session.feedback.citation_mismatches.append(CitationMismatch(citation_id=request.citation_id, reviewer_note=request.reviewer_note))
        return self.session_store.save(session)

    def add_claim_disagreement(self, session_id: str, request: ClaimDisagreementRequest) -> SessionState:
        session = self.session_store.get(session_id)
        sentence = session.parsed_sentences[request.sentence_index]
        sentence.disagreement_flagged = True
        session.feedback.claim_disagreements.append(
            ClaimDisagreement(
                sentence_index=request.sentence_index,
                sentence_text=sentence.sentence_text,
                reviewer_note=request.reviewer_note,
            )
        )
        return self.session_store.save(session)

    def add_facet_gaps(self, session_id: str, request: FacetGapRequest) -> SessionState:
        session = self.session_store.get(session_id)
        for facet in request.facets:
            if facet not in session.feedback.facet_gaps:
                session.feedback.facet_gaps.append(facet)
        return self.session_store.save(session)

    async def scoped_retrieval(self, session_id: str, request: ScopedRetrievalRequest) -> list[CitationIndexEntry]:
        scoped_start = timed()
        session = self.session_store.get(session_id)
        sentence = session.parsed_sentences[request.sentence_index]
        citation = next((entry for entry in session.citation_index if entry.citation_id == request.citation_id), None)
        if citation is None:
            return []
        passages, _ = await self.hybrid_retriever.scoped_retrieve(
            query=sentence.sentence_text,
            source_facet=citation.source_facet,
            filters=RetrievalFilters(document_id=citation.document_id, section_path=citation.section_path),
            top_k=request.top_k or self.scoped_top_k,
        )
        starting_id = max((entry.citation_id for entry in session.citation_index), default=0) + 1
        citations = build_citation_index(passages, starting_id=starting_id, ambiguity_gap_threshold=self.ambiguity_gap_threshold)
        logger.info(
            "scoped retrieval complete",
            extra={
                "session_id": session_id,
                "sentence_index": request.sentence_index,
                "citation_id": request.citation_id,
                "candidate_count": len(citations),
                "latency_ms": elapsed_ms(scoped_start),
            },
        )
        return citations

    def replace_citation(self, session_id: str, request: CitationReplacementRequest) -> SessionState:
        session = self.session_store.get(session_id)
        replacement_chunk = self.chunk_store.get_chunk(request.replacement_chunk_id)
        citation = next((entry for entry in session.citation_index if entry.citation_id == request.citation_id), None)
        if replacement_chunk is None or citation is None:
            return session

        original_chunk_id = citation.chunk_id
        citation.chunk_id = replacement_chunk.chunk_id
        citation.text = replacement_chunk.text
        citation.document_id = replacement_chunk.document_id
        citation.document_title = replacement_chunk.document_title
        citation.section_heading = replacement_chunk.section_heading
        citation.section_path = replacement_chunk.section_path
        citation.page_number = replacement_chunk.page_start
        citation.page_end = replacement_chunk.page_end
        citation.replacement_pending = True
        citation.reviewer_note = request.reviewer_note

        sentence = session.parsed_sentences[request.sentence_index]
        for reference in sentence.references:
            if reference.citation_id == request.citation_id:
                reference.matched_chunk_id = replacement_chunk.chunk_id
                reference.verified = False
                reference.confidence_score = None
                reference.confidence_label = None
                reference.replacement_pending = True
                reference.document_id = replacement_chunk.document_id
                reference.document_title = replacement_chunk.document_title
                reference.section_heading = replacement_chunk.section_heading
                reference.section_path = replacement_chunk.section_path
                reference.page_number = replacement_chunk.page_start
        if ReviewWarning.REPLACEMENT_PENDING not in sentence.warnings:
            sentence.warnings.append(ReviewWarning.REPLACEMENT_PENDING)
        session.feedback.citation_replacements.append(
            CitationReplacement(
                sentence_index=request.sentence_index,
                citation_id=request.citation_id,
                original_chunk_id=original_chunk_id,
                replacement_chunk_id=replacement_chunk.chunk_id,
                reviewer_note=request.reviewer_note,
            )
        )
        return self.session_store.save(session)

    def undo_citation_replacement(self, session_id: str, citation_id: int) -> SessionState:
        session = self.session_store.get(session_id)
        replacement = next((item for item in reversed(session.feedback.citation_replacements) if item.citation_id == citation_id), None)
        citation = next((entry for entry in session.citation_index if entry.citation_id == citation_id), None)
        if replacement is None or citation is None:
            return session
        original_chunk = self.chunk_store.get_chunk(replacement.original_chunk_id)
        if original_chunk is None:
            return session

        citation.chunk_id = original_chunk.chunk_id
        citation.text = original_chunk.text
        citation.document_id = original_chunk.document_id
        citation.document_title = original_chunk.document_title
        citation.section_heading = original_chunk.section_heading
        citation.section_path = original_chunk.section_path
        citation.page_number = original_chunk.page_start
        citation.page_end = original_chunk.page_end
        citation.replacement_pending = False
        citation.reviewer_note = None

        sentence = session.parsed_sentences[replacement.sentence_index]
        for reference in sentence.references:
            if reference.citation_id == citation_id:
                reference.matched_chunk_id = original_chunk.chunk_id
                reference.verified = False
                reference.confidence_score = None
                reference.confidence_label = None
                reference.replacement_pending = False
                reference.document_id = original_chunk.document_id
                reference.document_title = original_chunk.document_title
                reference.section_heading = original_chunk.section_heading
                reference.section_path = original_chunk.section_path
                reference.page_number = original_chunk.page_start
        sentence.warnings = [warning for warning in sentence.warnings if warning != ReviewWarning.REPLACEMENT_PENDING]

        session.feedback.citation_replacements = [item for item in session.feedback.citation_replacements if item is not replacement]
        return self.session_store.save(session)

    async def supplement_response(self, session_id: str, request: FacetGapRequest) -> SessionState:
        supplement_start = timed()
        session = self.session_store.get(session_id)
        if not request.facets:
            return session

        new_passages, diagnostics = await self.hybrid_retriever.retrieve(original_query=session.original_query, facets=request.facets)
        session.retrieval_diagnostics.extend(diagnostics)
        logger.info(
            "supplement retrieval complete",
            extra={
                "session_id": session_id,
                "selected_facets": request.facets,
                "diagnostics": [diagnostic.model_dump() for diagnostic in diagnostics],
                "console_visible": False,
            },
        )
        new_citations = self._append_unique_citations(
            session.citation_index,
            build_citation_index(
                new_passages,
                starting_id=max((entry.citation_id for entry in session.citation_index), default=0) + 1,
                ambiguity_gap_threshold=self.ambiguity_gap_threshold,
            ),
        )
        generation_request = GenerationRequest(
            original_query=session.original_query,
            facets=session.facets,
            citation_index=new_citations,
            mode="supplement",
            existing_response=session.generated_response,
            selected_facets=request.facets,
        )
        logger.info(
            "supplement generation started",
            extra={
                "session_id": session_id,
                "selected_facets": request.facets,
                "citation_count": len(new_citations),
            },
        )
        supplemental_raw = await self._generate_with_retry(generation_request)
        if supplemental_raw is None:
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.ERROR,
                    code="supplement_failed",
                    message="Supplementary generation was unsuccessful; the current response was preserved.",
                )
            )
            logger.info(
                "supplement generation failed",
                extra={"session_id": session_id, "latency_ms": elapsed_ms(supplement_start)},
            )
            return self.session_store.save(session)
        supplemental_response, supplemental_parsed, updated_citations, removed_sentence_count = await self._process_response(supplemental_raw, new_citations)
        offset = len(session.parsed_sentences)
        for sentence in supplemental_parsed:
            sentence.sentence_index += offset
        session.generated_response = "\n\n".join([part for part in [session.generated_response, supplemental_response] if part.strip()])
        session.parsed_sentences.extend(supplemental_parsed)
        session.citation_index = updated_citations
        session.removed_ungrounded_claim_count += removed_sentence_count
        if removed_sentence_count:
            session.ui_messages.append(
                UIMessage(
                    level=UIMessageLevel.WARNING,
                    code="removed_unverified_claims",
                    message=self._removed_unverified_sentences_message(removed_sentence_count),
                )
            )
        for facet in request.facets:
            if facet not in session.feedback.facet_gaps:
                session.feedback.facet_gaps.append(facet)
        logger.info(
            "supplement complete",
            extra={
                "session_id": session_id,
                "selected_facets": request.facets,
                "appended_sentence_count": len(supplemental_parsed),
                "citation_count": len(updated_citations),
                "latency_ms": elapsed_ms(supplement_start),
            },
        )
        return self.session_store.save(session)

    async def refine(self, session_id: str) -> SessionState:
        refine_start = timed()
        session = self.session_store.get(session_id)
        prior_response = session.generated_response
        extra_citations = list(session.citation_index)
        logger.info(
            "refinement started",
            extra={
                "session_id": session_id,
                "mismatch_count": len(session.feedback.citation_mismatches),
                "disagreement_count": len(session.feedback.claim_disagreements),
                "facet_gap_count": len(session.feedback.facet_gaps),
                "citation_count": len(extra_citations),
            },
        )

        for disagreement in session.feedback.claim_disagreements:
            if not disagreement.reviewer_note:
                continue
            passages, diagnostics = await self.hybrid_retriever.retrieve(
                original_query=f"{disagreement.sentence_text} {disagreement.reviewer_note}",
                facets=[f"{disagreement.sentence_text} {disagreement.reviewer_note}"],
            )
            session.retrieval_diagnostics.extend(diagnostics)
            disagreement.contradicting_passages = [passage.chunk.chunk_id for passage in passages[:3]]
            extra_citations = self._append_unique_citations(
                extra_citations,
                build_citation_index(
                    passages,
                    starting_id=max((entry.citation_id for entry in extra_citations), default=0) + 1,
                    ambiguity_gap_threshold=self.ambiguity_gap_threshold,
                ),
            )

        if session.feedback.facet_gaps:
            passages, diagnostics = await self.hybrid_retriever.retrieve(
                original_query=session.original_query,
                facets=session.feedback.facet_gaps,
            )
            session.retrieval_diagnostics.extend(diagnostics)
            extra_citations = self._append_unique_citations(
                extra_citations,
                build_citation_index(
                    passages,
                    starting_id=max((entry.citation_id for entry in extra_citations), default=0) + 1,
                    ambiguity_gap_threshold=self.ambiguity_gap_threshold,
                ),
            )

        mismatch_ids = [mismatch.citation_id for mismatch in session.feedback.citation_mismatches]
        budgeted_citations = self._trim_citations_to_budget(extra_citations)
        generation_request = GenerationRequest(
            original_query=session.original_query,
            facets=session.facets,
            citation_index=budgeted_citations,
            mode="refinement",
            existing_response=session.generated_response,
            mismatch_citation_ids=mismatch_ids,
            disagreement_notes=[disagreement.reviewer_note for disagreement in session.feedback.claim_disagreements if disagreement.reviewer_note],
            disagreement_contexts=self._build_disagreement_contexts(session),
        )
        try:
            raw_response = await self._generate_with_retry(generation_request)
            if raw_response is None:
                raise RuntimeError("refinement generation failed")
            final_response, parsed_sentences, updated_citations, removed_sentence_count = await self._process_response(raw_response, budgeted_citations)
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

        session.generated_response = final_response
        session.parsed_sentences = parsed_sentences
        session.citation_index = updated_citations
        session.removed_ungrounded_claim_count = removed_sentence_count
        session.feedback = FeedbackState()
        session.refinement_count += 1
        session.response_mode = self._determine_response_mode(parsed_sentences)
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
                "sentence_summary": self._sentence_status_summary(parsed_sentences),
                "citation_count": len(updated_citations),
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
                if not replacement:
                    replacement = parse_generated_response(self.sentence_regenerator.deterministic_rewrite(sentence, updated_citations))
                if not replacement:
                    continue
                replacement_sentence = replacement[-1]
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
        logger.info(
            "confidence scoring complete",
            extra={
                "sentence_summary": self._sentence_status_summary(parsed_sentences),
                "removed_ungrounded_claim_count": removed_sentence_count,
                "latency_ms": elapsed_ms(process_start),
            },
        )
        return render_parsed_sentences(parsed_sentences), parsed_sentences, updated_citations, removed_sentence_count

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
            if sentence.status == SentenceStatus.UNGROUNDED and sentence.sentence_type in {SentenceType.CLAIM, SentenceType.SYNTHESIS}:
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

    def _build_disagreement_contexts(self, session: SessionState) -> list[str]:
        contexts: list[str] = []
        for disagreement in session.feedback.claim_disagreements:
            if not disagreement.reviewer_note:
                continue
            contradicting_texts: list[str] = []
            for chunk_id in disagreement.contradicting_passages or []:
                chunk = self.chunk_store.get_chunk(chunk_id)
                if chunk is not None:
                    contradicting_texts.append(chunk.text)
            if contradicting_texts:
                contexts.append(
                    f"Sentence {disagreement.sentence_index}: {disagreement.reviewer_note}. Contradicting evidence: {' || '.join(contradicting_texts[:2])}"
                )
            else:
                contexts.append(f"Sentence {disagreement.sentence_index}: {disagreement.reviewer_note}")
        return contexts

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
