from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field


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


class CitationFeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    NEUTRAL = "neutral"


class MatchQuality(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    NONE = "none"


class ConfidenceLabel(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class ResponseMode(str, Enum):
    RESPONSE_REVIEW = "response_review"
    GENERATION_FAILED = "generation_failed"


class ResponseBasis(str, Enum):
    SOCIAL = "social"
    THREAD_CONTEXT_ONLY = "thread_context_only"
    CORPUS_SEARCH = "corpus_search"


class ConversationAction(str, Enum):
    RESPOND = "respond"
    SEARCH = "search"


class QueryRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryProgressStage(str, Enum):
    QUEUED = "queued"
    ORCHESTRATING = "orchestrating"
    UNDERSTANDING = "understanding"
    SEARCHING = "searching"
    EVIDENCE = "evidence"
    WRITING = "writing"
    COVERAGE_CHECK = "coverage_check"
    FOLLOWUP_RETRIEVAL = "followup_retrieval"
    CHECKING = "checking"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryStageDescriptor(BaseModel):
    key: QueryProgressStage
    label: str
    detail: str


VISIBLE_QUERY_PROGRESS_STAGES: tuple[tuple[QueryProgressStage, str, str], ...] = (
    (
        QueryProgressStage.ORCHESTRATING,
        "Deciding whether to search",
        "I'm deciding whether this needs report search or a direct response.",
    ),
    (
        QueryProgressStage.UNDERSTANDING,
        "Reading your question",
        "I'm getting clear on what you want to know.",
    ),
    (
        QueryProgressStage.SEARCHING,
        "Looking through the reports",
        "I'm finding the parts of the documents that seem most relevant.",
    ),
    (
        QueryProgressStage.EVIDENCE,
        "Pulling together the best evidence",
        "I'm narrowing down to the passages I trust most for this answer.",
    ),
    (
        QueryProgressStage.COVERAGE_CHECK,
        "Checking evidence coverage",
        "I'm checking whether each facet is supported by cited evidence.",
    ),
    (
        QueryProgressStage.FOLLOWUP_RETRIEVAL,
        "Retrieving additional evidence",
        "I'm pulling in more evidence for an uncovered facet.",
    ),
    (
        QueryProgressStage.WRITING,
        "Writing the answer",
        "I'm turning the evidence into a clear response.",
    ),
    (
        QueryProgressStage.CHECKING,
        "Checking the answer against the reports",
        "I'm making sure the wording still matches the source text before I show it.",
    ),
)


def default_query_stage_catalog() -> list[QueryStageDescriptor]:
    return [
        QueryStageDescriptor(key=key, label=label, detail=detail)
        for key, label, detail in VISIBLE_QUERY_PROGRESS_STAGES
    ]


def resolve_query_stage_descriptor(stage: QueryProgressStage) -> QueryStageDescriptor | None:
    for key, label, detail in VISIBLE_QUERY_PROGRESS_STAGES:
        if key == stage:
            return QueryStageDescriptor(key=key, label=label, detail=detail)
    return None


class RuntimeMode(str, Enum):
    LOCAL = "local"
    HYBRID = "hybrid"
    HOSTED = "hosted"


class RuntimeProfile(str, Enum):
    APPLE_LITE_MLX = "apple_silicon"
    FULL_LOCAL_TRANSFORMERS = "gpu"


class HostedProviderPreset(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"


class UIMessageLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RefinementScope(str, Enum):
    NONE = "none"
    LOCAL = "local"
    GLOBAL = "global"


class CommentIntent(str, Enum):
    AFFIRMATION = "affirmation"
    MINOR_EDIT = "minor_edit"
    SUBSTANTIVE_EDIT = "substantive_edit"
    REWRITE_REQUEST = "rewrite_request"


class ReviewWarning(str, Enum):
    STRUCTURAL_FACT = "structural_fact"
    OVER_CITED = "over_cited"
    REPLACEMENT_PENDING = "replacement_pending"
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
    source_facets: list[str] = Field(default_factory=list)
    replacement_pending: bool = False
    reviewer_note: str | None = None
    ambiguity_review_required: bool = False
    ambiguity_gap: float | None = None
    retrieval_scores: dict[str, float] = Field(default_factory=dict)


class Reference(BaseModel):
    reference_quote: str
    minimum_quote_words: int | None = None
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
    match_quality: MatchQuality = MatchQuality.NONE
    paragraph_index: int = 0
    warnings: list[ReviewWarning] = Field(default_factory=list)
    raw_text: str = ""

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




class ReviewComment(BaseModel):
    comment_id: str = Field(default_factory=lambda: str(uuid4()))
    text_selection: str = ""
    char_start: int = 0
    char_end: int = 0
    comment_text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("comment_text", "comment"),
        serialization_alias="comment_text",
    )
    resolved: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class SelectionCommentEdit(BaseModel):
    comment_id: str
    comment_text: str
    text_selection: str
    char_start: int
    char_end: int
    created_at: datetime
    resolved: bool = False


class CitationReplacement(BaseModel):
    sentence_index: int = -1
    citation_id: int
    replacement_chunk_id: str


class ClaimDisagreement(BaseModel):
    sentence_index: int
    sentence_text: str
    reviewer_note: str | None = None
    contradicting_passages: list[str] | None = None

class SentenceCitationPair(BaseModel):
    sentence_index: int
    citation_id: int


class CitationFeedback(BaseModel):
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    sentence_index: int = -1
    citation_id: int
    feedback_type: CitationFeedbackType
    created_at: datetime = Field(default_factory=utc_now)


class FeedbackState(BaseModel):
    comments: list[ReviewComment] = Field(default_factory=list)
    resolved_comments: list[SelectionCommentEdit] = Field(default_factory=list)
    citation_replacements: list[CitationReplacement] = Field(default_factory=list)
    citation_feedback: list[CitationFeedback] = Field(default_factory=list)


class RefinementCommentDecision(BaseModel):
    comment_id: str
    intent: CommentIntent
    scope: RefinementScope
    target_sentence_indices: list[int] = Field(default_factory=list)
    summary: str = ""


class RefinementPlan(BaseModel):
    overall_scope: RefinementScope = RefinementScope.NONE
    comment_decisions: list[RefinementCommentDecision] = Field(default_factory=list)
    target_sentence_indices: list[int] = Field(default_factory=list)
    change_summary: str = ""


class UIMessage(BaseModel):
    level: UIMessageLevel
    code: str
    message: str


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, str | int | float | bool | None] | None = None


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
    source_message: str | None = None
    resolved_query: str | None = None
    derived_from_session_id: str | None = None
    query_type: QueryType | None = None
    facets: list[str] = Field(default_factory=list)
    citation_index: list[CitationIndexEntry] = Field(default_factory=list)
    generated_response: str = ""
    parsed_sentences: list[ParsedSentence] = Field(default_factory=list)
    feedback: FeedbackState = Field(default_factory=FeedbackState)
    refinement_count: int = 0
    refinement_scope: RefinementScope | None = None
    change_summary: str | None = None
    retrieval_diagnostics: list[FacetRetrievalDiagnostic] = Field(default_factory=list)
    ui_messages: list[UIMessage] = Field(default_factory=list)
    removed_ungrounded_claim_count: int = 0
    response_mode: ResponseMode = ResponseMode.RESPONSE_REVIEW
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
    query_stage_catalog: list[QueryStageDescriptor] = Field(default_factory=default_query_stage_catalog)


class DecompositionResult(BaseModel):
    query_type: QueryType
    facets: list[str]


class RetrievalFilters(BaseModel):
    document_id: str | None = None
    section_path: str | None = None


class RetrievedPassage(BaseModel):
    chunk: ChunkObject
    score: float
    source_facet: str
    source_facets: list[str] = Field(default_factory=list)
    rank: int
    retriever: Literal["sparse", "dense", "reranked", "scoped"] = "sparse"


class VerificationResult(BaseModel):
    parsed_sentences: list[ParsedSentence]
    citation_index: list[CitationIndexEntry]


class CoverageCheckResult(BaseModel):
    covered_facets: list[str] = Field(default_factory=list)
    gap_facets: list[str] = Field(default_factory=list)
    trigger_followup: bool = False


class GenerationRequest(BaseModel):
    original_query: str
    facets: list[str]
    citation_index: list[CitationIndexEntry]
    mode: Literal[
        "initial",
        "supplement",
        "refinement",
        "regeneration",
        "refinement_planning",
        "sentence_refinement",
    ] = "initial"
    existing_response: str | None = None
    selected_facets: list[str] = Field(default_factory=list)
    mismatch_citation_ids: list[int] = Field(default_factory=list)
    disagreement_notes: list[str] = Field(default_factory=list)
    disagreement_contexts: list[str] = Field(default_factory=list)
    selection_comments: list[ReviewComment] = Field(default_factory=list)
    approved_pairs: list[SentenceCitationPair] = Field(default_factory=list)
    rejected_pairs: list[SentenceCitationPair] = Field(default_factory=list)
    failed_sentence_text: str | None = None
    failed_sentence_comment: str | None = None
    failed_regeneration_response: str | None = None
    max_regeneration_quotes: int = 2
    repair_prior_response: str | None = None
    planned_refinement_scope: RefinementScope | None = None
    target_sentence_indices: list[int] = Field(default_factory=list)
    target_sentence_text: str | None = None
    revision_note: str | None = None


class ScoredReference(BaseModel):
    score: float | None
    label: ConfidenceLabel | None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    source_message: str | None = None
    derived_from_session_id: str | None = None


class ConversationContextTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)
    search_backed: bool = False
    session_id: str | None = None
    derived_from_session_id: str | None = None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)
    context_turns: list[ConversationContextTurn] = Field(default_factory=list)
    latest_grounded_session_id: str | None = None


class ConversationDecision(BaseModel):
    action: ConversationAction
    assistant_text: str | None = None
    search_query: str | None = None
    response_basis: ResponseBasis
    derived_from_session_id: str | None = None


class AssistantTurnState(BaseModel):
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    used_search: bool = False
    response_basis: ResponseBasis
    linked_session_id: str | None = None
    derived_from_session_id: str | None = None


class MessageRunState(BaseModel):
    message_run_id: str
    status: QueryRunStatus = QueryRunStatus.RUNNING
    stage: QueryProgressStage = QueryProgressStage.ORCHESTRATING
    stage_label: str = "Deciding whether to search"
    stage_detail: str = "I'm deciding whether this needs report search or a direct response."
    stage_catalog: list[QueryStageDescriptor] = Field(default_factory=default_query_stage_catalog)
    assistant_turn: AssistantTurnState | None = None
    session: SessionState | None = None


class ReviewCommentRequest(BaseModel):
    text_selection: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    comment_text: str = Field(min_length=1)


class ReviewCommentUpdateRequest(BaseModel):
    comment_text: str = Field(min_length=1)


class CitationReplacementRequest(BaseModel):
    sentence_index: int
    replacement_chunk_id: str = Field(min_length=1)


class CitationFeedbackRequest(BaseModel):
    sentence_index: int
    feedback_type: CitationFeedbackType


class CitationReplaceRequest(BaseModel):
    sentence_index: int
    replacement_citation_id: int


class ScopedRetrievalRequest(BaseModel):
    sentence_index: int = Field(ge=0)


class ScopedRetrievalEnvelope(BaseModel):
    citations: list[CitationIndexEntry]




class SessionEnvelope(BaseModel):
    session: SessionState


class MessageRunEnvelope(BaseModel):
    message_run: MessageRunState


class HostedModelOption(BaseModel):
    id: str
    label: str
    description: str = ""


class HostedProviderDescriptor(BaseModel):
    preset: HostedProviderPreset
    label: str
    provider_family: str
    description: str
    model_label: str
    notes: list[str] = Field(default_factory=list)
    requires_base_url: bool = False
    requires_deployment_name: bool = False
    supports_custom_model: bool = True
    models: list[HostedModelOption] = Field(default_factory=list)


class HostedEnvOverride(BaseModel):
    field: str
    env_var: str


class HostedSettingsState(BaseModel):
    config_path: str
    config_exists: bool
    provider_preset: HostedProviderPreset
    llm_provider: str
    runtime_mode: RuntimeMode
    api_key_configured: bool
    base_url: str | None = None
    selected_model_id: str | None = None
    custom_model_id: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None
    azure_model_family: str | None = None
    custom_base_url: str | None = None
    env_overrides: list[HostedEnvOverride] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    saved_provider_settings: dict[HostedProviderPreset, "HostedSavedProviderState"] = Field(default_factory=dict)


class HostedSavedProviderState(BaseModel):
    api_key_configured: bool = False
    selected_model_id: str | None = None
    custom_model_id: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None
    azure_model_family: str | None = None
    custom_base_url: str | None = None


class HostedSettingsEnvelope(BaseModel):
    settings: HostedSettingsState
    providers: list[HostedProviderDescriptor]


class HostedSettingsUpdateRequest(BaseModel):
    provider_preset: HostedProviderPreset
    selected_model_id: str | None = None
    custom_model_id: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    custom_base_url: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None
    azure_model_family: str | None = None


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
