export type ResponseMode = "response_review" | "generation_failed";
export type AssistantMessageSource = "query" | "refinement";

export type QueryType = "single_hop" | "multi_hop";
export type QueryRunStatus = "running" | "completed" | "failed";
export type QueryProgressStage =
  | "queued"
  | "understanding"
  | "searching"
  | "evidence"
  | "writing"
  | "coverage_check"
  | "followup_retrieval"
  | "checking"
  | "completed"
  | "failed";

export type RuntimeMode = "local" | "hybrid" | "hosted";
export type RuntimeProfile = "apple_silicon" | "gpu";
export type HostedProviderPreset =
  | "openai"
  | "openrouter"
  | "azure_openai"
  | "gemini"
  | "custom_openai_compatible";

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
  source_facets: string[];
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
  confidence_label?:
    | "supported"
    | "partially_supported"
    | "not_supported"
    | null;
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
    comment_id: string;
    text_selection: string;
    char_start: number;
    char_end: number;
    comment_text: string;
    resolved: boolean;
  }>;
  resolved_comments: Array<{
    comment_id: string;
    text_selection: string;
    char_start: number;
    char_end: number;
    comment_text: string;
    created_at: string;
    resolved: boolean;
  }>;
  citation_replacements: Array<{
    sentence_index: number;
    citation_id: number;
    replacement_chunk_id: string;
  }>;
  citation_feedback: Array<{
    feedback_id: string;
    sentence_index: number;
    citation_id: number;
    feedback_type: "like" | "dislike" | "neutral";
    created_at: string;
  }>;
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

export interface HostedModelOption {
  id: string;
  label: string;
  description: string;
}

export interface HostedProviderDescriptor {
  preset: HostedProviderPreset;
  label: string;
  provider_family: string;
  description: string;
  model_label: string;
  notes: string[];
  requires_base_url: boolean;
  requires_deployment_name: boolean;
  supports_custom_model: boolean;
  models: HostedModelOption[];
}

export interface HostedEnvOverride {
  field: string;
  env_var: string;
}

export interface HostedSettingsState {
  config_path: string;
  config_exists: boolean;
  provider_preset: HostedProviderPreset;
  llm_provider: string;
  runtime_mode: RuntimeMode;
  api_key_configured: boolean;
  base_url?: string | null;
  selected_model_id?: string | null;
  custom_model_id?: string | null;
  azure_base_url?: string | null;
  azure_deployment_name?: string | null;
  azure_model_family?: string | null;
  custom_base_url?: string | null;
  env_overrides: HostedEnvOverride[];
  notices: string[];
  saved_provider_settings: Partial<
    Record<HostedProviderPreset, HostedSavedProviderState>
  >;
}

export interface HostedSavedProviderState {
  api_key_configured: boolean;
  selected_model_id?: string | null;
  custom_model_id?: string | null;
  azure_base_url?: string | null;
  azure_deployment_name?: string | null;
  azure_model_family?: string | null;
  custom_base_url?: string | null;
}

export interface HostedSettingsEnvelope {
  settings: HostedSettingsState;
  providers: HostedProviderDescriptor[];
}

export interface HostedSettingsUpdatePayload {
  provider_preset: HostedProviderPreset;
  selected_model_id?: string | null;
  custom_model_id?: string | null;
  api_key?: string | null;
  clear_api_key: boolean;
  custom_base_url?: string | null;
  azure_base_url?: string | null;
  azure_deployment_name?: string | null;
  azure_model_family?: string | null;
}
