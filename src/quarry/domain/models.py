from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class QueryType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class SentenceType(str, Enum):
    CLAIM = "claim"
    SYNTHESIS = "synthesis"
    STRUCTURE = "structure"


class SentenceStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNGROUNDED = "ungrounded"
    NO_REF = "no_ref"
    UNCHECKED = "unchecked"


class ConfidenceLabel(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class ResponseMode(str, Enum):
    RESPONSE_REVIEW = "response_review"
    CLARIFICATION_REQUIRED = "clarification_required"
    GENERATION_FAILED = "generation_failed"


class QueryRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryProgressStage(str, Enum):
    QUEUED = "queued"
    UNDERSTANDING = "understanding"
    SEARCHING = "searching"
    EVIDENCE = "evidence"
    WRITING = "writing"
    CHECKING = "checking"
    CLARIFICATION = "clarification"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeMode(str, Enum):
    LOCAL = "local"
    HYBRID = "hybrid"
    HOSTED = "hosted"


class RuntimeProfile(str, Enum):
    APPLE_LITE_MLX = "apple_silicon"
    FULL_LOCAL_TRANSFORMERS = "gpu"


class UIMessageLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ReviewWarning(str, Enum):
    STRUCTURAL_FACT = "structural_fact"
    OVER_CITED = "over_cited"
    REPLACEMENT_PENDING = "replacement_pending"
    DISAGREEMENT_FLAGGED = "disagreement_flagged"
    CONFIDENCE_UNKNOWN = "confidence_unknown"


class LayoutBox(BaseModel):
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float


class ChunkObject(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    level: int = 1
    section_heading: str
    section_path: str
    section_depth: int = 0
    page_start: int = 1
    page_end: int = 1
    metadata_summary: str = ""
    metadata_entities: list[str] = Field(default_factory=list)
    metadata_questions: list[str] = Field(default_factory=list)
    source_path: str | None = None
    parser_provenance: str | None = None
    layout_blocks: list[str] = Field(default_factory=list)
    page_spans: list[tuple[int, int]] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)


class CitationIndexEntry(BaseModel):
    citation_id: int
    chunk_id: str
    text: str
    document_id: str
    document_title: str
    section_heading: str
    section_path: str
    page_number: int
    page_end: int | None = None
    retrieval_score: float
    source_facet: str
    replacement_pending: bool = False
    reviewer_note: str | None = None
    ambiguity_review_required: bool = False
    ambiguity_gap: float | None = None
    retrieval_scores: dict[str, float] = Field(default_factory=dict)


class Reference(BaseModel):
    reference_quote: str
    matched_chunk_id: str | None = None
    verified: bool = False
    confidence_score: float | None = None
    confidence_label: ConfidenceLabel | None = None
    citation_id: int | None = None
    document_id: str | None = None
    document_title: str | None = None
    section_heading: str | None = None
    section_path: str | None = None
    page_number: int | None = None
    replacement_pending: bool = False
    confidence_unknown: bool = False


class ParsedSentence(BaseModel):
    sentence_index: int
    sentence_text: str
    sentence_type: SentenceType
    references: list[Reference] = Field(default_factory=list)
    status: SentenceStatus = SentenceStatus.UNCHECKED
    warnings: list[ReviewWarning] = Field(default_factory=list)
    raw_text: str = ""
    disagreement_flagged: bool = False

    @property
    def structural_warning(self) -> bool:
        return ReviewWarning.STRUCTURAL_FACT in self.warnings

    @property
    def over_cited(self) -> bool:
        return ReviewWarning.OVER_CITED in self.warnings


class StructuralIndexEntry(BaseModel):
    chunk_id: str
    document_id: str
    section_heading: str
    section_path: str
    section_depth: int
    page_range: tuple[int, int]
    covered: bool = False


class CitationMismatch(BaseModel):
    citation_id: int
    reviewer_note: str | None = None


class CitationReplacement(BaseModel):
    sentence_index: int
    citation_id: int
    original_chunk_id: str
    replacement_chunk_id: str
    reviewer_note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ClaimDisagreement(BaseModel):
    sentence_index: int
    sentence_text: str
    reviewer_note: str | None = None
    contradicting_passages: list[str] | None = None

class FeedbackState(BaseModel):
    citation_mismatches: list[CitationMismatch] = Field(default_factory=list)
    claim_disagreements: list[ClaimDisagreement] = Field(default_factory=list)
    facet_gaps: list[str] = Field(default_factory=list)
    citation_replacements: list[CitationReplacement] = Field(default_factory=list)


class UIMessage(BaseModel):
    level: UIMessageLevel
    code: str
    message: str


class RetrieverDiagnostic(BaseModel):
    retriever: str
    result_count: int
    latency_ms: float | None = None
    fallback_used: bool = False
    error: str | None = None
    provider: str | None = None


class FacetRetrievalDiagnostic(BaseModel):
    facet: str
    sparse: RetrieverDiagnostic
    dense: RetrieverDiagnostic
    fused_count: int = 0
    reranked_count: int
    top_score_gap: float | None = None
    top_rerank_score: float | None = None
    degraded_mode: bool = False


class SessionState(BaseModel):
    session_id: str
    original_query: str
    query_type: QueryType | None = None
    facets: list[str] = Field(default_factory=list)
    citation_index: list[CitationIndexEntry] = Field(default_factory=list)
    generated_response: str = ""
    parsed_sentences: list[ParsedSentence] = Field(default_factory=list)
    feedback: FeedbackState = Field(default_factory=FeedbackState)
    refinement_count: int = 0
    retrieval_diagnostics: list[FacetRetrievalDiagnostic] = Field(default_factory=list)
    ui_messages: list[UIMessage] = Field(default_factory=list)
    removed_ungrounded_claim_count: int = 0
    response_mode: ResponseMode = ResponseMode.RESPONSE_REVIEW
    clarification_suggestions: list[str] = Field(default_factory=list)
    generation_provider: str = "unknown"
    parser_provider: str = "unknown"
    runtime_mode: RuntimeMode = RuntimeMode.HYBRID
    runtime_profile: RuntimeProfile = RuntimeProfile.FULL_LOCAL_TRANSFORMERS
    local_model_status: dict[str, str] = Field(default_factory=dict)
    active_model_ids: list[str] = Field(default_factory=list)
    query_status: QueryRunStatus = QueryRunStatus.RUNNING
    query_stage: QueryProgressStage = QueryProgressStage.QUEUED
    query_stage_label: str = "Getting started"
    query_stage_detail: str = "I'm getting ready to work on your question."


class DecompositionResult(BaseModel):
    query_type: QueryType
    facets: list[str]
    clarification_required: bool = False


class RetrievalFilters(BaseModel):
    document_id: str | None = None
    section_path: str | None = None


class RetrievedPassage(BaseModel):
    chunk: ChunkObject
    score: float
    source_facet: str
    rank: int
    retriever: Literal["sparse", "dense", "reranked", "scoped"] = "sparse"


class VerificationResult(BaseModel):
    parsed_sentences: list[ParsedSentence]
    citation_index: list[CitationIndexEntry]


class GenerationRequest(BaseModel):
    original_query: str
    facets: list[str]
    citation_index: list[CitationIndexEntry]
    mode: Literal["initial", "supplement", "refinement", "regeneration"] = "initial"
    existing_response: str | None = None
    selected_facets: list[str] = Field(default_factory=list)
    mismatch_citation_ids: list[int] = Field(default_factory=list)
    disagreement_notes: list[str] = Field(default_factory=list)
    disagreement_contexts: list[str] = Field(default_factory=list)
    failed_sentence_text: str | None = None
    failed_regeneration_response: str | None = None
    max_regeneration_quotes: int = 2
    repair_prior_response: str | None = None


class ScoredReference(BaseModel):
    score: float | None
    label: ConfidenceLabel | None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class CitationMismatchRequest(BaseModel):
    citation_id: int
    reviewer_note: str | None = None


class ClaimDisagreementRequest(BaseModel):
    sentence_index: int
    reviewer_note: str | None = None


class FacetGapRequest(BaseModel):
    facets: list[str] = Field(default_factory=list)


class ScopedRetrievalRequest(BaseModel):
    sentence_index: int
    citation_id: int
    top_k: int = Field(default=3, ge=1, le=10)


class CitationReplacementRequest(BaseModel):
    sentence_index: int
    citation_id: int
    replacement_chunk_id: str
    reviewer_note: str | None = None


class SessionEnvelope(BaseModel):
    session: SessionState


class ParsedBlock(BaseModel):
    block_id: str
    text: str
    page_number: int
    page_end: int | None = None
    block_type: Literal["heading", "paragraph", "table", "figure_caption", "table_title"] = "paragraph"
    parser_provenance: str | None = None
    layout_bbox: LayoutBox | None = None
    table_id: str | None = None
    figure_id: str | None = None


class ParsedSection(BaseModel):
    section_id: str
    heading: str
    path: str
    depth: int
    page_start: int
    page_end: int
    blocks: list[ParsedBlock] = Field(default_factory=list)


class PageParseStatus(BaseModel):
    page_number: int
    outcome: Literal["parsed", "recovered", "skipped"] = "parsed"
    parser_used: str | None = None
    attempts: int = 1
    error: str | None = None


class ParsedDocument(BaseModel):
    document_id: str
    document_title: str
    source_path: str
    parser_used: str
    fallback_used: bool = False
    parser_provenance: list[str] = Field(default_factory=list)
    sections: list[ParsedSection] = Field(default_factory=list)
    figure_captions: list[str] = Field(default_factory=list)
    table_titles: list[str] = Field(default_factory=list)
    recovered_pages: list[int] = Field(default_factory=list)
    skipped_pages: list[int] = Field(default_factory=list)
    page_parse_statuses: list[PageParseStatus] = Field(default_factory=list)


class DocumentArtifactSummary(BaseModel):
    document_id: str
    document_title: str
    parsed_document_path: str
    chunks_path: str
    chunk_count: int


class CorpusManifest(BaseModel):
    corpus_id: str
    created_at: datetime = Field(default_factory=utc_now)
    embedding_model: str
    embedding_dimensions: int
    vector_index_path: str
    vector_metadata_path: str | None = None
    structural_index_path: str
    local_model_status_path: str | None = None
    runtime_profile: RuntimeProfile | None = None
    parser_provider: str | None = None
    active_model_ids: list[str] = Field(default_factory=list)
    documents: list[DocumentArtifactSummary] = Field(default_factory=list)
    chunk_count: int = 0
