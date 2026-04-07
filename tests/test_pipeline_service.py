import asyncio

from quarry.adapters.in_memory import (
    DeterministicGenerationClient,
    HeuristicDecompositionClient,
    HeuristicNLIClient,
    InMemoryChunkStore,
    KeywordSparseRetriever,
    SemanticDenseRetriever,
    SimpleCrossEncoderReranker,
)
from quarry.domain.models import (
    ChunkObject,
    ConfidenceLabel,
    CitationFeedbackType,
    FeedbackState,
    MatchQuality,
    QueryProgressStage,
    QueryRequest,
    QueryRunStatus,
    RetrievedPassage,
    SessionState,
)
from quarry.pipeline.decomposition import QueryDecomposer
from quarry.pipeline.generation import AnswerGenerator, SentenceRegenerator
from quarry.pipeline.retrieval import HybridRetriever
from quarry.pipeline.verification import VerificationService
from quarry.services.pipeline_service import PipelineService
from quarry.services.session_store import SessionStore


class FlakyGenerationClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        if self.calls == 1:
            return ""
        return '[CLAIM] Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects, especially when design coordination was completed before fabrication mobilization began. [REF: "Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects, especially when design coordination was completed before fabrication mobilization began."]'


class BrokenGenerationClient:
    async def generate(self, request):
        return "[STRUCTURE] This response is malformed for grounded output."


class CapturingGenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        quote = request.citation_index[0].text
        return f'[CLAIM] {quote} [REF: "{quote}"]'


class SimilarFailureGenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.mode == "initial":
            return (
                '[CLAIM] FEED maturity elements are a subset of the elements that make up the entire PDRI. '
                '[REF: "This exact quote does not appear anywhere in the report corpus for verification purposes"]'
            )
        if request.mode == "regeneration":
            return (
                '[CLAIM] FEED maturity elements are a subset of the elements making up the entire PDRI. '
                '[REF: "This exact quote still does not appear anywhere in the report corpus for verification purposes"]'
            )
        return '[NO_REF]'


class LingeringUngroundedGenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.mode == "initial":
            return (
                '[CLAIM] PDRI maturity is always scored with a five-level external certification rubric. '
                '[REF: "This quote does not exist anywhere in the source documents and should fail verification immediately"]\n\n'
                '[SYNTHESIS] PDRI maturity appears to connect front-end planning completeness with broader readiness decisions across the reports. '
                '[REF: "This second quote is also missing from the source documents and should fail verification immediately"] '
                '[REF: "Another missing quote that should not be verified anywhere in the local corpus"]'
            )
        if request.mode == "regeneration":
            if "five-level external certification rubric" in (request.failed_sentence_text or ""):
                return (
                    '[CLAIM] PDRI maturity is usually described through a five-level external certification rubric. '
                    '[REF: "A regenerated quote that still does not exist anywhere in the source documents for testing"]'
                )
            return (
                '[SYNTHESIS] PDRI maturity seems to connect front-end planning completeness with broader readiness decisions across the reports. '
                '[REF: "A regenerated synthesis quote that still does not exist anywhere in the corpus for testing"] '
                '[REF: "Another regenerated synthesis quote that still cannot be verified anywhere in the corpus"]'
            )
        return '[NO_REF]'


class UntaggedRegenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.mode == "initial":
            return (
                '[CLAIM] Executive leadership can better assess where and how to commit limited resources to enhance project performance. '
                '[REF: "This quote does not exist anywhere in the source documents and should fail verification immediately"]'
            )
        if request.mode == "regeneration":
            return (
                'Using this output, executive leadership (e.g., project sponsor, executive steering committees) '
                'can better assess where and how to commit limited resources to enhance project performance. '
                '[REF: "Using this output, executive leadership (e.g., project sponsor, executive steering committees) '
                'can better assess where and how to commit limited resources to enhance project performance."]'
            )
        return "[NO_REF]"


class RaisingRegenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.mode == "initial":
            return (
                '[CLAIM] FEED maturity elements are a subset of the elements making up the entire PDRI. '
                '[REF: "This exact quote does not appear anywhere in the report corpus for verification purposes"]'
            )
        if request.mode == "regeneration":
            raise RuntimeError("regeneration backend unavailable")
        return "[NO_REF]"


class ShortQuoteRegenerationClient:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.mode == "initial":
            return (
                '[CLAIM] FEED maturity elements are a subset of the elements that make up the entire PDRI. '
                '[REF: "This exact quote does not appear anywhere in the report corpus for verification purposes"]'
            )
        if request.mode == "regeneration":
            return (
                '[CLAIM] FEED maturity elements are a subset of the entire PDRI. '
                '[REF: "FEED maturity elements are a subset of the elements making"]'
            )
        return "[NO_REF]"


class StaticHybridRetriever:
    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages
        self.calls = []

    async def retrieve(self, *, original_query: str, facets: list[str], query_type=None):
        self.calls.append({"original_query": original_query, "facets": list(facets), "query_type": query_type})
        return list(self.passages), []


class RaisingDecomposer:
    async def decompose(self, query: str):
        raise RuntimeError("boom")


def build_chunks() -> list[ChunkObject]:
    return [
        ChunkObject(
            chunk_id="cii-001",
            document_id="cii-report-2024",
            document_title="CII Modular Delivery Report 2024",
            text="Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects, especially when design coordination was completed before fabrication mobilization began.",
            section_heading="3.2 Schedule Outcomes",
            section_path="Chapter 3 > 3.2 Schedule Outcomes",
            page_start=18,
            page_end=18,
            metadata_summary="Modular construction shortened schedules when front-end coordination was complete.",
            metadata_entities=["Phase III", "modular construction", "schedule duration"],
            metadata_questions=["How did modular construction affect schedule duration?"],
        ),
        ChunkObject(
            chunk_id="cii-003",
            document_id="cii-report-2023",
            document_title="CII Risk Alignment Study 2023",
            text="Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions because equipment lead times no longer aligned with installation windows.",
            section_heading="2.4 Procurement Risks",
            section_path="Chapter 2 > 2.4 Procurement Risks",
            page_start=11,
            page_end=11,
            metadata_summary="Late procurement lock-in increased disruption risk.",
            metadata_entities=["procurement packages", "design milestone"],
            metadata_questions=["How does late procurement planning affect project risk?"],
        ),
    ]


def build_many_chunks(count: int) -> list[ChunkObject]:
    chunks: list[ChunkObject] = []
    for index in range(count):
        chunks.append(
            ChunkObject(
                chunk_id=f"cii-{index:03d}",
                document_id="cii-report-2024",
                document_title="CII Modular Delivery Report 2024",
                text=(
                    f"PDRI maturity evidence sample {index} explains how the project definition package was assessed "
                    f"with detailed planning, schedule alignment, scope completeness, and front end planning rigor."
                ),
                section_heading=f"Section {index}",
                section_path=f"Chapter 1 > Section {index}",
                page_start=index + 1,
                page_end=index + 1,
            )
        )
    return chunks


def build_regeneration_chunk() -> ChunkObject:
    return ChunkObject(
        chunk_id="cii-regen-001",
        document_id="cii-report-regen",
        document_title="CII Regeneration Report",
        text=(
            "FEED maturity elements are a subset of the elements making up the entire PDRI. "
            "This chapter explains how the full PDRI structure is organized during front end planning."
        ),
        section_heading="1.2 FEED Maturity",
        section_path="Chapter 1 > 1.2 FEED Maturity",
        page_start=5,
        page_end=5,
    )


def build_untagged_regeneration_chunk() -> ChunkObject:
    return ChunkObject(
        chunk_id="cii-regen-untagged-001",
        document_id="cii-report-regen-untagged",
        document_title="CII Regeneration Report Untagged",
        text=(
            "Using this output, executive leadership (e.g., project sponsor, executive steering committees) "
            "can better assess where and how to commit limited resources to enhance project performance. "
            "The discussion focuses on planning resources during front end planning."
        ),
        section_heading="5.1 Executive Leadership",
        section_path="Chapter 5 > 5.1 Executive Leadership",
        page_start=9,
        page_end=9,
    )


def test_pipeline_runs_end_to_end_with_sample_components() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
        generation_provider="mlx:mlx-community/Qwen3.5-4B-MLX-4bit",
        parser_provider="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        runtime_mode="hybrid",
        runtime_profile="apple_silicon",
        local_model_status={"text": "configured", "parser": "configured"},
        active_model_ids=[
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        ],
    )

    session = asyncio.run(
        service.run_query(
            QueryRequest(query="How do modular construction and procurement planning affect schedule risk?")
        )
    )

    assert session.facets
    assert session.citation_index
    assert session.parsed_sentences
    assert session.runtime_profile.value == "apple_silicon"
    assert session.parser_provider == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert session.active_model_ids[0] == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert all(sentence.match_quality in {MatchQuality.STRONG, MatchQuality.PARTIAL, MatchQuality.NONE} for sentence in session.parsed_sentences)


def test_begin_query_creates_running_session_with_initial_stage() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = service.begin_query(QueryRequest(query="What is PDRI maturity?"))

    assert session.query_status == QueryRunStatus.RUNNING
    assert session.query_stage == QueryProgressStage.UNDERSTANDING
    assert session.query_stage_label == "Reading your question"
    assert service.get_session(session.session_id).session_id == session.session_id


def test_pipeline_assigns_partial_match_quality_for_partial_confidence() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(
        service.run_query(QueryRequest(query="How do modular construction and procurement planning affect schedule risk?"))
    )
    claim = next(sentence for sentence in session.parsed_sentences if sentence.sentence_type.value in {"claim", "synthesis"})
    claim.status = claim.status
    if claim.references:
        claim.references[0].verified = True
        claim.references[0].confidence_label = ConfidenceLabel.PARTIALLY_SUPPORTED
    service._annotate_match_quality(session.parsed_sentences)
    assert claim.match_quality == MatchQuality.PARTIAL


def test_run_query_for_session_marks_failed_stage_on_unhandled_error() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    session_store = SessionStore()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=RaisingDecomposer(),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=session_store,
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    pending = service.begin_query(QueryRequest(query="Why did it happen?"))
    session = asyncio.run(service.run_query_for_session(pending.session_id, QueryRequest(query="Why did it happen?")))

    assert session.query_status == QueryRunStatus.FAILED
    assert session.query_stage == QueryProgressStage.FAILED
    assert any(message.code == "query_failed" for message in session.ui_messages)


def test_generation_retries_on_malformed_first_response() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    flaky = FlakyGenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(flaky),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="How do modular construction and procurement planning affect schedule risk?")))

    assert flaky.calls == 2
    assert session.response_mode.value == "response_review"


def test_generation_failure_returns_generation_failed_state() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(BrokenGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="How do modular construction and procurement planning affect schedule risk?")))

    assert session.response_mode.value == "generation_failed"
    assert any(message.code == "generation_failed" for message in session.ui_messages)


def test_similar_failed_regeneration_is_not_retried() -> None:
    chunk = build_regeneration_chunk()
    chunk_store = InMemoryChunkStore([chunk])
    retriever = StaticHybridRetriever(
        [RetrievedPassage(chunk=chunk, score=0.9, source_facet="What is PDRI maturity?", rank=1, retriever="reranked")]
    )
    generator = SimilarFailureGenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What is PDRI maturity?")))

    regeneration_requests = [request for request in generator.requests if request.mode == "regeneration"]
    assert len(regeneration_requests) == 1
    assert session.response_mode.value == "generation_failed"


def test_ungrounded_claims_and_synthesis_are_removed_from_final_response() -> None:
    chunk = build_regeneration_chunk()
    chunk_store = InMemoryChunkStore([chunk])
    retriever = StaticHybridRetriever(
        [RetrievedPassage(chunk=chunk, score=0.9, source_facet="What is PDRI maturity?", rank=1, retriever="reranked")]
    )
    generator = LingeringUngroundedGenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What is PDRI maturity?")))

    assert session.removed_ungrounded_claim_count == 2
    assert session.response_mode.value == "generation_failed"
    assert session.parsed_sentences == []
    assert session.generated_response == ""
    assert "five-level external certification rubric" not in session.generated_response
    assert any(message.code == "removed_unverified_claims" for message in session.ui_messages)


def test_regeneration_prefers_non_empty_sentence_from_untagged_rewrite() -> None:
    chunk = build_untagged_regeneration_chunk()
    chunk_store = InMemoryChunkStore([chunk])
    retriever = StaticHybridRetriever(
        [RetrievedPassage(chunk=chunk, score=0.92, source_facet="What helps executive leadership?", rank=1, retriever="reranked")]
    )
    generator = UntaggedRegenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What helps executive leadership?")))

    regeneration_requests = [request for request in generator.requests if request.mode == "regeneration"]
    assert len(regeneration_requests) == 1
    assert all(sentence.sentence_text.strip() for sentence in session.parsed_sentences)
    assert "[CLAIM] [REF:" not in session.generated_response
    assert any(
        sentence.sentence_text.startswith("Using this output, executive leadership")
        for sentence in session.parsed_sentences
    )


def test_regeneration_fallback_marks_sentence_no_ref_instead_of_fabricating_chunk_text() -> None:
    chunk = build_regeneration_chunk()
    chunk_store = InMemoryChunkStore([chunk])
    retriever = StaticHybridRetriever(
        [RetrievedPassage(chunk=chunk, score=0.9, source_facet="What is PDRI maturity?", rank=1, retriever="reranked")]
    )
    generator = RaisingRegenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What is PDRI maturity?")))

    regeneration_requests = [request for request in generator.requests if request.mode == "regeneration"]
    assert len(regeneration_requests) == 2
    assert session.removed_ungrounded_claim_count == 1
    assert session.response_mode.value == "generation_failed"
    assert session.generated_response == ""
    assert session.parsed_sentences == []


def test_regeneration_accepts_shorter_exact_quote_for_clean_sentence_repair() -> None:
    chunk = build_regeneration_chunk()
    chunk_store = InMemoryChunkStore([chunk])
    retriever = StaticHybridRetriever(
        [RetrievedPassage(chunk=chunk, score=0.9, source_facet="What is PDRI maturity?", rank=1, retriever="reranked")]
    )
    generator = ShortQuoteRegenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What is PDRI maturity?")))

    regenerated_sentence = session.parsed_sentences[0]

    assert any(request.mode == "regeneration" for request in generator.requests)
    assert session.response_mode.value == "response_review"
    assert session.removed_ungrounded_claim_count == 0
    assert regenerated_sentence.sentence_text == "FEED maturity elements are a subset of the entire PDRI."
    assert regenerated_sentence.references[0].reference_quote == "FEED maturity elements are a subset of the elements making"
    assert regenerated_sentence.references[0].minimum_quote_words == 8
    assert regenerated_sentence.references[0].verified is True


def test_query_with_vague_wording_still_processes() -> None:
    """Vague queries now default to multi_hop and are processed normally."""
    chunk_store = InMemoryChunkStore(build_chunks())
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="schedule")))

    # No longer triggers clarification - processes as multi_hop
    assert session.response_mode.value == "response_review"
    assert session.query_type.value in ("single_hop", "multi_hop")

def test_single_hop_generation_request_uses_trimmed_citation_budget() -> None:
    chunks = build_many_chunks(12)
    chunk_store = InMemoryChunkStore(chunks)
    retriever = StaticHybridRetriever(
        [
            RetrievedPassage(chunk=chunk, score=float(100 - index), source_facet="What is PDRI maturity?", rank=index + 1, retriever="reranked")
            for index, chunk in enumerate(chunks)
        ]
    )
    generator = CapturingGenerationClient()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=retriever,
        answer_generator=AnswerGenerator(generator),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=SessionStore(),
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )

    session = asyncio.run(service.run_query(QueryRequest(query="What is PDRI maturity?")))

    assert retriever.calls[0]["query_type"].value == "single_hop"
    assert len(generator.requests) == 1
    assert len(generator.requests[0].citation_index) == 8
    assert len(session.citation_index) == 12


def test_citation_feedback_is_scoped_to_sentence_index() -> None:
    chunk_store = InMemoryChunkStore(build_chunks())
    session_store = SessionStore()
    service = PipelineService(
        chunk_store=chunk_store,
        query_decomposer=QueryDecomposer(HeuristicDecompositionClient(), max_facets=4),
        hybrid_retriever=HybridRetriever(
            sparse_retriever=KeywordSparseRetriever(chunk_store),
            dense_retriever=SemanticDenseRetriever(chunk_store),
            reranker=SimpleCrossEncoderReranker(),
            sparse_top_k=30,
            dense_top_k=30,
            rerank_top_k=20,
            rrf_k=60,
        ),
        answer_generator=AnswerGenerator(DeterministicGenerationClient()),
        sentence_regenerator=SentenceRegenerator(),
        verifier=VerificationService(chunk_store=chunk_store, nli_client=HeuristicNLIClient()),
        session_store=session_store,
        scoped_top_k=3,
        refinement_token_budget=8000,
        ambiguity_gap_threshold=0.05,
    )
    session_store.save(
        SessionState(
            session_id="scope-feedback",
            original_query="test",
            feedback=FeedbackState(),
        )
    )

    service.set_citation_feedback("scope-feedback", 0, 7, CitationFeedbackType.LIKE)
    updated = service.set_citation_feedback("scope-feedback", 1, 7, CitationFeedbackType.DISLIKE)

    assert len(updated.feedback.citation_feedback) == 2
    assert {
        (fb.sentence_index, fb.citation_id, fb.feedback_type.value)
        for fb in updated.feedback.citation_feedback
    } == {
        (0, 7, "like"),
        (1, 7, "dislike"),
    }

    cleared = service.set_citation_feedback("scope-feedback", 0, 7, CitationFeedbackType.NEUTRAL)
    assert {
        (fb.sentence_index, fb.citation_id, fb.feedback_type.value)
        for fb in cleared.feedback.citation_feedback
    } == {(1, 7, "dislike")}
