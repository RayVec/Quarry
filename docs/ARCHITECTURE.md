# QUARRY Architecture

## 1. Purpose

QUARRY is a local-first grey literature retrieval and review system for a small, curated corpus of technical construction reports. It ingests source documents into local artifacts, retrieves supporting passages, generates citation-tagged answers, verifies those answers against the source text, and presents the result in a conversational review UI.

The current system is optimized for:

- local corpora rather than web-scale search
- reviewability over opaque fluency
- evidence-grounded answers rather than one-shot generation
- human correction loops rather than fully automated finality

## 2. Design Principles

### 2.1 Local corpus as source of truth

The approved document set lives under **`paths.corpus_dir`** in `config.toml` (often `data/sources/` in the example config). The `Settings` default before any file config is applied is `data/corpus/`.

Generated artifacts live in:

- **`paths.artifacts_dir`** (typically `data/artifacts/`)

There is no database and no OpenSearch in the current implementation.

### 2.2 Retrieval before generation

QUARRY does not treat the generator as the source of truth. The runtime pipeline is:

1. understand the query
2. retrieve passages
3. build a citation index
4. generate a response from those passages
5. verify quote matches
6. score semantic support
7. expose review actions

### 2.3 Verification over polish

The system prefers a less polished but grounded answer over a fluent but unverifiable one. That is why the current pipeline keeps:

- exact quote verification
- NLI confidence scoring
- disagreement review
- sentence regeneration when grounding fails

### 2.4 Conversational review experience

The frontend no longer begins as a dense dashboard. The current interaction model is:

- ask
- wait with visible real progress
- read
- react

Review tools remain available, but they appear contextually inside the answer experience.

## 3. Repository Structure

| Path                          | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `src/quarry/config.py`        | runtime configuration and TOML/env loading                   |
| `src/quarry/startup.py`       | startup prep, warmup, optional corpus rebuild, server launch |
| `src/quarry/cli.py`           | operational CLI                                              |
| `src/quarry/api/`             | FastAPI app and routes                                       |
| `src/quarry/domain/models.py` | shared typed models and enums                                |
| `src/quarry/adapters/`        | hosted, MLX, local, and heuristic runtime adapters           |
| `src/quarry/ingest/`          | parsing, chunking, and indexing                              |
| `src/quarry/pipeline/`        | decomposition, retrieval, generation parsing, verification   |
| `src/quarry/services/`        | query orchestration and session state                        |
| `web/src/`                    | conversational review UI                                     |
| `tests/`                      | unit, integration, and E2E tests                             |
| `data/sources/` (typical)     | raw input documents; actual path is `paths.corpus_dir`       |
| `data/artifacts/`             | generated corpus artifacts                                   |
| `data/model-cache/`           | project-local model cache                                    |
| `data/logs/runtime/`          | query/API/runtime logs                                       |
| `data/logs/corpus/`           | warmup and ingest logs                                       |
| `data/logs/tests/`            | pytest log files                                             |

## 4. Runtime Architecture

### 4.1 Runtime modes

- `hybrid`
  - local-first, but graceful fallbacks and hosted generation are allowed
- `local`
  - strict local mode; startup fails if required local components are not warmed and ready
- `hosted`
  - intended for externally provided hosted workflows

### 4.2 Runtime profiles

- `apple_silicon`
  - Apple Silicon friendly profile
  - MLX-backed local text and MLX vision parsing
- `gpu`
  - heavier local transformers profile

The runtime mode controls strictness. The runtime profile controls which local model/parser family is preferred.

### 4.3 Current recommended split

The current design goal is best reliable answers, not “local everything” and not “host everything.”

The recommended mixed strategy is:

- local decomposition (optional: `hosted.use_live_decomposition` with OpenAI-compatible credentials)
- local retrieval
- local reranking
- local verification (DeBERTa MNLI when `use_local_models` is on; see section 7.7)
- local parsing
- hosted answer generation / refinement / regeneration (`hosted.use_live_generation`; Gemini or OpenAI-compatible)

Optional knobs also include hosted metadata enrichment and hosted embeddings (`hosted.use_live_metadata_enrichment`, `hosted.use_live_embeddings`). See `docs/MODELS_PROMPTS.md` for provider constraints (for example, Gemini is generation-only).

This keeps trust-critical steps local while using a stronger hosted model for difficult synthesis.

## 5. Configuration and Startup

### 5.1 Settings

`src/quarry/config.py` defines `Settings`, which is the central configuration object used by startup, ingest, runtime selection, and the API.

Important configuration groups:

- runtime mode and runtime profile
- corpus and artifact directories
- retrieval budgets
- parser settings
- hosted provider settings
- local/MLX model IDs

Config resolution order:

1. defaults
2. `config.toml`
3. environment variable overrides

### 5.2 Startup flow

The preferred launcher is:

- `python start_backend.py`

It delegates to `src/quarry/startup.py`.

Current startup flow:

1. load config
2. configure model cache
3. configure logging
4. check warmup readiness
5. optionally warm local models
6. optionally rebuild corpus from the configured `paths.corpus_dir`
7. start Uvicorn

The launcher asks whether to rebuild the corpus. This keeps “serve now” and “refresh sources first” in one entrypoint.

## 6. Ingest Architecture

The ingest system lives under `src/quarry/ingest/`.

Main modules:

- `parsers.py`
- `normalize.py`
- `chunking.py`
- `indexing.py`
- `pipeline.py`

### 6.1 Source of truth and outputs

Raw source files:

- directory configured as `paths.corpus_dir` (commonly `data/sources/`)

Generated outputs:

- `data/artifacts/manifest.json`
- `data/artifacts/documents/<document_id>/parsed_document.json`
- `data/artifacts/documents/<document_id>/chunks.json`
- `data/artifacts/vector_index.faiss`
- `data/artifacts/vector_index_metadata.json`
- `data/artifacts/local_model_status.json`

`data/corpus/` is only a fallback/sample path when no artifact manifest exists.

### 6.2 Parsing

Parsing depends on runtime profile.

`apple_silicon`:

- rasterize PDF pages locally
- parse with `mlx-community/Qwen3-VL-4B-Instruct-4bit` via the `qwen3_vl_mlx` adapter
- parser prompt explicitly tells the model to ignore TOC pages, headers, footers, page numbers, and single-character heading fragments
- normalize into structured blocks such as heading, paragraph, table, table title, and figure caption
- when `use_local_models` is true, the ingest **primary** PDF chain is **always** this MLX vision parser (`build_parsing_pipeline` in `ingest/pipeline.py`); **`[parser] primary` does not select a different PDF engine on this profile** — that setting applies on the GPU profile. The **fallback** chain still honors `[parser] fallback` (for example `pymupdf_text`, then `pypdf_text`)
- ingest ensures the parser is warmed before document parsing starts
- `MinerU` remains installed and configurable, but it is **not** in the default fallback chain
- markdown and text files use `basic_text`
- fallback usage is logged explicitly

`gpu`:

- PDF path is driven by `[parser] primary` / `[parser] fallback` (typical default: `olmocr_transformers` with `pymupdf_text` and lighter fallbacks — see `Settings` and `config.example.toml`)

### 6.2.1 Post-parse normalization

Parsed output is now cleaned before chunking.

Current deterministic normalization rules include:

- strip obvious table-of-contents fragments
- drop repeated header/footer-like short text that appears across multiple pages
- merge false single-character heading sections such as `R` + `T-361 ...` into `RT-361 ...`
- merge obviously broken paragraph continuations

This creates a parser-agnostic quality layer that protects downstream chunking and retrieval even when the upstream parser is imperfect.

### 6.3 Chunking

Chunking is structure-aware, not pure semantic chunking.

Current behavior:

- preserve section and block boundaries where possible
- build level 1 and level 2 chunks
- use size thresholds to group or split text
- split at sentence boundaries only when a block is too large

This design preserves provenance and reviewability better than a pure semantic chunker.

Because chunking is downstream of parsing, QUARRY now treats normalized parsed output as the canonical input to chunk construction.

### 6.4 Metadata enrichment

Chunks can be enriched with:

- one-sentence summary
- entity list
- suggested retrieval questions

In `hybrid` mode, metadata enrichment can fall back to heuristic enrichment if the prompt-driven path fails.

This stage does not create the original parsing error, but it can amplify bad chunks if parser quality is poor. That is why normalization now runs before enrichment and indexing.

## 7. Query Pipeline

The runtime query pipeline is orchestrated in:

- `src/quarry/services/pipeline_service.py`

### 7.1 Session model

Each submitted query creates a backend session.

Current behavior:

- one backend session per query
- session state lives in memory
- review mutations update that session
- frontend conversation history is separate and local to the browser

### 7.2 Async query start and real progress

The frontend now uses:

- `POST /api/v1/query/start`

The backend immediately creates a session and starts the query asynchronously.

The frontend then polls:

- `GET /api/v1/sessions/{session_id}`

This supports real backend-driven waiting states through these `SessionState` fields:

- `query_status`
- `query_stage`
- `query_stage_label`
- `query_stage_detail`

Additional stages for multi-hop coverage-repair flow:

- `coverage_check` — label: "Checking evidence coverage"
- `followup_retrieval` — label: "Retrieving additional evidence"

### 7.3 Query understanding

`src/quarry/pipeline/decomposition.py` implements heuristic-only classification.

Current behavior:

- heuristic patterns identify obvious single-hop queries (factoids, definitions)
- heuristic patterns identify obvious multi-hop queries (comparisons, multi-aspect questions)
- queries that don't match any heuristic pattern default to multi_hop
- MLX model is used only for facet generation in multi-hop queries, not for classification

This approach eliminates classification delays (~3-5s) and avoids low-confidence model judgments on edge cases.

### 7.4 Retrieval

`src/quarry/pipeline/retrieval.py` implements hybrid sparse+dense retrieval.

Current steps:

- sparse retrieval
- dense retrieval
- reciprocal rank fusion
- reranking
- citation index construction

For **single-hop** queries, `HybridRetriever._resolve_limits` caps retrieval at **12 / 12 / 8** (sparse / dense / rerank) by taking the **minimum** of those caps and the configured `sparse_top_k`, `dense_top_k`, and `rerank_top_k` (defaults in `Settings` are 30 / 30 / 20).

For **multi-hop** queries, retrieval now keeps a wider anchored pre-rerank pool:

- per-facet sparse/dense stays at configured defaults (typically 30/30)
- fused candidates across all facets are merged into an anchored pool capped by `multihop_anchor_pool_size` (default **40**)
- reranked output used for citation-index construction is capped by `multihop_rerank_budget` (default **20**)
- every passage/citation preserves facet provenance as:
  - `source_facet`: primary facet (backward-compatible)
  - `source_facets`: all facets that surfaced that chunk

This reduces latency and often improves precision by cutting noise on narrow questions.

### 7.5 Generation

Generation is driven by `GenerationRequest` plus `generation_prompt(...)`.

Current supported modes:

- initial answer generation
- supplement generation
- refinement generation
- sentence regeneration

For single-hop questions, the initial generation stage trims citation context to the strongest passages instead of always passing the full citation set.

### 7.6 Parsing generated output

`src/quarry/pipeline/parsing.py` converts the raw generated response into:

- sentence list
- sentence types
- inline references
- sentence status objects

These parsed sentences become the reviewable unit in the UI.

### 7.7 Verification

`src/quarry/pipeline/verification.py` handles:

- exact quote verification
- quote discovery
- sentence status assignment
- NLI confidence scoring
- post-generation facet coverage check (after exact-match quote resolution)

Runtime optimization (behavior-preserving):

- `InMemoryChunkStore` precomputes normalized text per chunk at load time and keeps a `chunk_id -> normalized_text` map.
- `find_chunk_by_quote()` still uses the same substring matching behavior, but now reuses precomputed normalized chunk text instead of normalizing per lookup.
- Scoped lookup (`chunk_ids` provided) now uses direct chunk-id lookup plus precomputed normalized text, avoiding repeated normalization in the hot path.
- `verify_exact_matches()` flow is unchanged: search citation chunks first, then fall back to full corpus only on miss.

The verifier also tracks lightweight lookup metrics (`quote_lookup_metrics`) to support performance observation:

- `scoped_lookups`
- `full_corpus_fallbacks`
- `quote_match_rate`
- `avg_candidates_checked`

The trust model is:

1. the quote must exist in the source text
2. the verified citation must still semantically support the sentence

For multi-hop queries, verification now also computes a facet coverage signal after exact-match resolution: if no resolved citation maps back to chunks surfaced by a facet, that facet is marked as a gap for optional one-round follow-up retrieval.

Default exact-match policy requires at least 10 words for a quoted anchor. Regenerated references can opt into a shorter minimum (currently 8 words) so sentence repair has more room to produce clear prose without losing exact-text verification.

NLI confidence scoring follows exact-quote verification and assigns a `confidence_label` per reference. Wiring in `build_runtime_clients` (`production.py`):

- **`LocalMNLIClient`** when **`use_local_models` is true** (default in `hybrid` / `local`): loads `khalidalt/DeBERTa-v3-large-mnli` (or `settings.nli_model_name`) via Transformers on the configured device — **both** `apple_silicon` and `gpu` profiles use this path when local models are enabled. It maps predicted class to label with a soft threshold (`ENTAILMENT_SOFT_THRESHOLD = 0.35`): if the argmax is `neutral` but the raw entailment probability is still ≥ 0.35, the label is upgraded to `supported`. If the model fails to load or scoring throws, the client falls back to **`NullConfidenceNLIClient`**, which returns `label=None` / `score=None` (verification still runs on exact quotes; semantic labels may be absent).
- **`HeuristicNLIClient`** when **`use_local_models` is false**: token-set overlap as `max(precision, recall)` on token sets derived from sentence and chunk text. Thresholds: ≥ 0.7 → `supported`; ≥ 0.4 → `partially_supported`; else `not_supported`. This is **not** the default for typical laptop `hybrid` setups with local models on.

### 7.8 Regeneration

If a sentence remains ungrounded after initial generation, the service may regenerate that sentence.

Important current optimizations:

- only changed sentences are re-verified after regeneration
- unchanged sentence/reference pairs can reuse cached NLI results
- regeneration halts early when the rewritten sentence is still ungrounded and too similar to the previous failed attempt
- retry prompts now include the previous failed rewrite and explicitly instruct the model to find different evidence or use `[NO_REF]`
- when regeneration cannot produce a grounded sentence, QUARRY falls back to a conservative `[NO_REF]` sentence shell instead of fabricating prose from raw chunk prefixes or bullet fragments
- regeneration prompts explicitly prefer clear natural sentences and allow shorter verbatim quote anchors (**8 to 15 words** in `generation_prompt` for `regeneration` mode)
- after a successful regeneration, post-regeneration verification sets `reference.minimum_quote_words` to **`REGENERATION_MIN_QUOTE_WORDS` (8)** so shorter anchors can still pass exact-match checks than the default 10-word minimum

This avoids wasting time on repeated near-identical failed rewrites.

### 7.9 Lingering ungrounded sentence policy

After regeneration and confidence scoring, QUARRY applies a stricter final response policy:

- lingering ungrounded or `NO_REF` `CLAIM` sentences are removed from the visible response
- lingering ungrounded or `NO_REF` `SYNTHESIS` sentences are also removed from the visible response
- `STRUCTURE` sentences are unchanged

This reflects the difference between:

- unsupported grounded content, which should not remain in a supposedly grounded answer once exact quote verification fails

The session also records how many ungrounded sentences were removed so the reviewer can see that the answer was pruned and investigate through citations or follow-up review if needed.

## 8. Review and Mutation Architecture

The same service object also handles review mutations:

- add selection-anchored review comments
- citation-level like / dislike / neutral feedback
- citation replacement (alternative passage) from the citation drawer
- unified refine orchestration

The authoritative reviewer state lives in `SessionState.feedback`.

Important feedback fields:

- `comments` — text selections with `char_start` / `char_end`, `text_selection`, `comment_text`, `resolved`
- `citation_feedback` — per `(sentence_index, citation_id)` a `feedback_type` of `like`, `dislike`, or `neutral`
- `citation_replacements` — records replacement chunk choices tied to a citation id (used with drawer replacement flows)
- `resolved_comments` — selections that could not be re-anchored after a prior refine

### 8.1 Citation like / dislike and how refine uses them

**Persistence:** The UI posts citation feedback to `POST /sessions/{session_id}/citations/{citation_id}/feedback` with `sentence_index` and `feedback_type`. Entries are stored in `feedback.citation_feedback` (scoped by sentence index so the same numeric `citation_id` in different sentences does not collide).

**Refine gate (`PipelineService.refine`):** A **refinement generation** call (hosted or local, whichever `AnswerGenerator` is configured) runs only when **at least one** of the following is true:

- there is at least one **unresolved** selection comment, or
- there is at least one citation with **`dislike`** feedback.

If the reviewer only left **`like`** feedback (and no unresolved comments or dislikes), refine **does not** call the generator to rewrite the answer; it returns a no-op and does not clear feedback state or increment `refinement_count`.

**Dislike / like → model input:**

- Feedback is handled at the sentence/citation pair level during refinement. The request carries pair-scoped `approved_pairs` and `rejected_pairs` so the prompt can preserve approved relations when still supported.
- Citations whose ids are rejected in the current refine call are removed from the passage list passed into the refinement `GenerationRequest`.
- `mismatch_citation_ids` is kept only as a compatibility bridge for globally rejected citations. It should not be used to represent mixed pair feedback for the same citation id.
- The generation prompt’s **Reviewer Feedback** section lists pair-scoped approvals as soft positive signals, and the refinement instructions say to preserve them only if they still remain supported after rewriting.

**Like semantics:** **`like`** is a soft positive signal. It does not trigger refine by itself, but when refine is already running it is passed into the request as a pair-scoped approval so the model can prefer preserving that sentence/citation relation if it still holds. It never overrides verification failure or an explicit dislike.

**Related:** Choosing a replacement passage in the citation drawer updates `citation_index` (and may set `reviewer_note` / `replacement_pending` on the entry). Refine then uses the updated index when building source passages; that path is separate from the thumbs but composes with dislike-driven removal.

Implementation reference: `PipelineService.refine`, `PipelineService.set_citation_feedback`, `prompts.generation_prompt`, `prompts._format_reviewer_feedback`, `prompts._format_citation_line`.

## 9. API Architecture

The FastAPI app is built in:

- `src/quarry/api/app.py`

Key behavior:

- CORS for local frontend development
- request trace IDs on every HTTP request
- runtime readiness enforcement in `local` mode
- background query task tracking for async `/query/start`

Main routes in:

- `src/quarry/api/routes.py`

Important route groups:

- query start and fetch
- session fetch and close
- feedback mutation (selection comments, citation like/dislike, citation replacement / undo)
- unified refine

## 10. Frontend Architecture

The frontend lives under `web/src/`.

Current frontend dependencies include `lucide-react` for consistent iconography across utility controls (for example, the diagnostics trigger in the thread header).

### 10.1 Shell

`web/src/App.tsx` implements the current shell:

- persistent left sidebar with `New Search` and `Recent Research`
- landing hero and composer on first load
- conversation thread after first query
- docked composer at the bottom of the workspace
- only the newest assistant message remains interactive

Thread history is frontend-local, persisted in browser storage, and consists of:

- user messages
- assistant messages
- pending assistant messages

On refresh, the frontend restores the prior thread from local storage on the same browser/device. This is local persistence only; it is not synced through the backend.

Recent research entries are also persisted in browser storage and are used to repopulate the sidebar independently of the currently visible thread.

### 10.2 Contextual review surfaces

Key UI surfaces:

- `ConversationMessage.tsx`
- `PendingConversationMessage.tsx`
- `ReviewPanel.tsx`
- `CitationDialog.tsx`
- `DiagnosticsDrawer.tsx`

Current behavior:

- user query becomes a bubble immediately
- pending assistant bubble shows real backend progress
- final assistant message supports paragraph-style reading with inline citation and review actions
- citation badges shown to users are renumbered contiguously (`[1]..[N]`) for readability; backend/internal citation IDs remain unchanged for review mutations
- removed-sentence warnings are surfaced to the reviewer in the message and review panel
- older assistant messages remain visible but read-only
- the workspace drawer combines provider settings and diagnostics; the landing gear opens it to settings, and the thread gear opens it to diagnostics
- comment capture uses text selections (character ranges) instead of sentence-index or response-level freeform comments
- refine input is driven by **unresolved selection comments** and/or **disliked citations**; citation **likes** do not trigger a rewrite by themselves
- response-level supplement comments are no longer a separate path
- comment affordance is hidden by default; selecting text reveals a floating comment icon near the selection
- comment composer/editing uses a compact floating card anchored near selection/highlight instead of in-flow expanding panels
- citation tooltip display is suppressed while a selection icon/card is active to avoid interaction overlap

### 10.2.1 Paragraph layout and match quality

Generation and parsing now support lightweight paragraph grouping:

- generator may insert `[PARA]` markers at topic transitions
- parser strips markers and assigns `paragraph_index` on each `ParsedSentence`
- sentence remains the minimum unit for verification, comments, and regeneration

Frontend rendering (`ResponseReview.tsx`) groups by `paragraph_index` and renders sentences in a continuous paragraph flow with inline citation badges.

The left per-sentence confidence rail has been removed from main reading flow. Citation badges now use a richer per-reference display model on the client (`web/src/utils/retrievalDisplay.ts`) that combines retrieval score / ambiguity signals with sentence and reference verification state. The backend sentence-level `match_quality` field still exists, but the frontend uses it as a fallback rather than the sole badge signal.

- unified `strong` / `good` → green badge
- unified `fair` → amber badge
- unified `weak` → no badge shown, unless the citation is pending replacement

Server-side `match_quality` is still computed after verification/NLI scoring:

- `strong`: verified sentence with supported references and no short-anchor downgrade
- `partial`: partially verified support and/or shorter regeneration anchor
- `none`: structure/no-ref sentence (or no verified references after filtering)

Raw verification fields (`status`, `confidence_label`, `confidence_score`) remain in the session payload for diagnostics and logging, and the citation drawer uses the same unified match display model as the inline badges.

Both drawers currently use the shared `SheetContent` close button from `web/src/components/ui/sheet.tsx` (`sheet.module.css` `closeIconButton` plus screen-reader text `Close`); they are not wired through the older `drawer-close-trigger` helper in `web/src/styles/app.css`.

### 10.2.2 Selection comment lifecycle

Review comments are persisted as selection anchors in `feedback.comments`:

- `comment_id`
- `text_selection`
- `char_start` / `char_end`
- `comment_text`
- `resolved`

During refinement, the pipeline attempts to re-anchor each selection in the new response text:

- if the selection can be located, the comment remains active with updated offsets
- if the selection cannot be located, the comment moves to `feedback.resolved_comments`

Frontend currently exposes selection comment creation/edit/delete in the response review surface and summarizes active/resolved counts in the review panel.

Interaction contract:

- no active selection/highlight: no comment controls rendered
- active browser selection: show transient comment icon near the selection
- click icon: open compact comment card (textarea + submit)
- save: clear transient selection UI and render persistent yellow highlight
- click highlight: open comment card for one or more overlapping comments at that location

### 10.3 UI modes

The backend `ResponseMode` enum (`src/quarry/domain/models.py`) and the frontend `ResponseMode` type (`web/src/types.ts`) both expose only:

- `response_review`
- `generation_failed`

`PipelineService._determine_response_mode` sets one of these two values on every completed session.

## 11. Logging and Observability

`src/quarry/logging_utils.py` centralizes logging.

Current logging model:

- console logs are concise
- file logs are verbose
- runtime logs go to `data/logs/runtime/`
- corpus-stage logs go to `data/logs/corpus/`
- pytest logs go to `data/logs/tests/`

The logs are designed to answer:

- what query came in
- what stage the system reached
- what prompt was sent
- what the model returned
- why a fallback happened

## 12. Current Constraints

Important current constraints:

- backend session state is in memory only
- frontend thread history is restored from browser storage on the same device/browser, not from backend persistence
- clearing browser storage removes local thread history
- corpus rebuilds are explicit, not automatic background sync
- performance is still sensitive to hosted provider latency
- the system is optimized for a small curated corpus, not a large enterprise document warehouse
