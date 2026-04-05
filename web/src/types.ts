export type ResponseMode =
  | "response_review"
  | "clarification_required"
  | "generation_failed";

export type QueryType = "single_hop" | "multi_hop";
export type QueryRunStatus = "running" | "completed" | "failed";
export type QueryProgressStage =
  | "queued"
  | "understanding"
  | "searching"
  | "evidence"
  | "writing"
  | "checking"
  | "clarification"
  | "completed"
  | "failed";

export type RuntimeMode = "local" | "hybrid" | "hosted";
export type RuntimeProfile = "apple_silicon" | "gpu";

export type SentenceStatus =
  | "verified"
  | "partially_verified"
  | "ungrounded"
  | "no_ref"
  | "unchecked";

export type MatchQuality = "strong" | "partial" | "none";

export interface CitationIndexEntry {
  citation_id: number;
  chunk_id: string;
  text: string;
  document_id: string;
  document_title: string;
  section_heading: string;
  section_path: string;
  page_number: number;
  page_end?: number | null;
  retrieval_score: number;
  source_facet: string;
  replacement_pending: boolean;
  reviewer_note?: string | null;
  ambiguity_review_required: boolean;
  ambiguity_gap?: number | null;
  retrieval_scores: Record<string, number>;
}

export interface Reference {
  reference_quote: string;
  matched_chunk_id?: string | null;
  verified: boolean;
  confidence_score?: number | null;
  confidence_label?: "supported" | "partially_supported" | "not_supported" | null;
  citation_id?: number | null;
  document_id?: string | null;
  document_title?: string | null;
  section_heading?: string | null;
  section_path?: string | null;
  page_number?: number | null;
  replacement_pending: boolean;
  confidence_unknown: boolean;
}

export interface ParsedSentence {
  sentence_index: number;
  sentence_text: string;
  sentence_type: "claim" | "synthesis" | "structure";
  references: Reference[];
  status: SentenceStatus;
  match_quality: MatchQuality;
  paragraph_index: number;
  warnings: string[];
  raw_text: string;
  disagreement_flagged: boolean;
}

export interface UIMessage {
  level: "info" | "warning" | "error";
  code: string;
  message: string;
}

export interface RetrieverDiagnostic {
  retriever: string;
  result_count: number;
  latency_ms?: number | null;
  fallback_used: boolean;
  error?: string | null;
  provider?: string | null;
}

export interface FacetRetrievalDiagnostic {
  facet: string;
  sparse: RetrieverDiagnostic;
  dense: RetrieverDiagnostic;
  fused_count: number;
  reranked_count: number;
  top_score_gap?: number | null;
  top_rerank_score?: number | null;
  degraded_mode: boolean;
}

export interface FeedbackState {
  comments: Array<{
    sentence_index?: number | null;
    sentence_type?: "claim" | "synthesis" | "structure" | null;
    sentence_text?: string | null;
    comment: string;
  }>;
  citation_replacements: Array<{ citation_id: number; replacement_chunk_id: string }>;
}

export interface SessionState {
  session_id: string;
  original_query: string;
  query_type?: QueryType | null;
  facets: string[];
  citation_index: CitationIndexEntry[];
  generated_response: string;
  parsed_sentences: ParsedSentence[];
  feedback: FeedbackState;
  refinement_count: number;
  retrieval_diagnostics: FacetRetrievalDiagnostic[];
  ui_messages: UIMessage[];
  removed_ungrounded_claim_count: number;
  response_mode: ResponseMode;
  clarification_suggestions: string[];
  generation_provider: string;
  parser_provider: string;
  runtime_mode: RuntimeMode;
  runtime_profile: RuntimeProfile;
  local_model_status: Record<string, string>;
  active_model_ids: string[];
  query_status: QueryRunStatus;
  query_stage: QueryProgressStage;
  query_stage_label: string;
  query_stage_detail: string;
}

export interface SessionEnvelope {
  session: SessionState;
}
