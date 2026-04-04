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

The approved document set lives in:

- `data/sources/`

Generated artifacts live in:

- `data/artifacts/`

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
- mismatch review
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

| Path | Purpose |
| --- | --- |
| `src/quarry/config.py` | runtime configuration and TOML/env loading |
| `src/quarry/startup.py` | startup prep, warmup, optional corpus rebuild, server launch |
| `src/quarry/cli.py` | operational CLI |
| `src/quarry/api/` | FastAPI app and routes |
| `src/quarry/domain/models.py` | shared typed models and enums |
| `src/quarry/adapters/` | hosted, MLX, local, and heuristic runtime adapters |
| `src/quarry/ingest/` | parsing, chunking, and indexing |
| `src/quarry/pipeline/` | decomposition, retrieval, generation parsing, verification |
| `src/quarry/services/` | query orchestration and session state |
| `web/src/` | conversational review UI |
| `tests/` | unit, integration, and E2E tests |
| `data/sources/` | raw input documents |
| `data/artifacts/` | generated corpus artifacts |
| `data/model-cache/` | project-local model cache |
| `data/logs/runtime/` | query/API/runtime logs |
| `data/logs/corpus/` | warmup and ingest logs |
| `data/logs/tests/` | pytest log files |

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

- local decomposition
- local retrieval
- local reranking
- local verification
- local parsing
- hosted answer generation / refinement / regeneration

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
6. optionally rebuild corpus from `data/sources/`
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

- `data/sources/`

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
- parse with `mlx-community/Qwen3-VL-4B-Instruct-4bit`
- parser prompt now explicitly tells the model to ignore TOC pages, headers, footers, page numbers, and single-character heading fragments
- normalize into structured blocks such as heading, paragraph, table, table title, and figure caption
- when local models are enabled, QUARRY now tries this MLX parser first even in `hybrid` mode
- ingest ensures the parser is warmed before document parsing starts
- the default PDF fallback chain is `pymupdf_text` and then `pypdf_text`
- `MinerU` is still installed and configurable, but it is not used in the default runtime chain as of April 2, 2026
- markdown and text files use `basic_text`
- fallback usage is logged explicitly

`gpu`:

- prefer the heavier local parser stack, with lightweight PDF text fallbacks by default

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

### 7.3 Query understanding

`src/quarry/pipeline/decomposition.py` implements heuristic-first classification.

Current behavior:

- obvious vague or contextless queries become `clarification_required`
- obvious factoid/definition queries become `single_hop`
- obvious comparison/multi-aspect queries become `multi_hop`
- only ambiguous cases escalate to the decomposition model

If the result is `clarification_required`, the session stops early and returns server-generated clarification suggestions.

### 7.4 Retrieval

`src/quarry/pipeline/retrieval.py` implements hybrid sparse+dense retrieval.

Current steps:

- sparse retrieval
- dense retrieval
- reciprocal rank fusion
- reranking
- citation index construction

The system now uses smaller retrieval budgets for obvious single-hop questions:

- sparse top-k: 12
- dense top-k: 12
- rerank top-k: 8

This reduces latency and often improves precision by cutting noise.

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

The trust model is:

1. the quote must exist in the source text
2. the verified citation must still semantically support the sentence

Default exact-match policy requires at least 10 words for a quoted anchor. Regenerated references can opt into a shorter minimum (currently 8 words) so sentence repair has more room to produce clear prose without losing exact-text verification.

NLI confidence scoring follows exact-quote verification and assigns a `confidence_label` per reference:

- **`HeuristicNLIClient`** (Apple Silicon / no-model path): computes token-set overlap as `max(precision, recall)` — where precision is `|sentence ∩ chunk| / |sentence|` and recall is `|sentence ∩ chunk| / |chunk|`. Taking the maximum of both directions means a sentence that is a near-verbatim paraphrase of the source (and thus has high recall) is not unfairly penalised when it contains a few additional connective words not present in the chunk. Thresholds: ≥ 0.7 → `supported`; ≥ 0.4 → `partially_supported`; else `not_supported`.
- **`LocalMNLIClient`** (GPU / full-transformer path): runs an MNLI classifier and maps predicted class to label. An additional soft threshold (`ENTAILMENT_SOFT_THRESHOLD = 0.35`) upgrades a result to `supported` when the argmax lands on neutral but the raw entailment probability is still ≥ 0.35. This handles near-verbatim rewrites where the model splits probability mass between entailment and neutral without a clear winner.

### 7.8 Regeneration

If a sentence remains ungrounded after initial generation, the service may regenerate that sentence.

Important current optimizations:

- only changed sentences are re-verified after regeneration
- unchanged sentence/reference pairs can reuse cached NLI results
- regeneration halts early when the rewritten sentence is still ungrounded and too similar to the previous failed attempt
- retry prompts now include the previous failed rewrite and explicitly instruct the model to find different evidence or use `[NO_REF]`
- when regeneration cannot produce a grounded sentence, QUARRY falls back to a conservative `[NO_REF]` sentence shell instead of fabricating prose from raw chunk prefixes or bullet fragments
- regeneration prompts explicitly prefer clear natural sentences and allow shorter verbatim quote anchors (8 to 10 words)

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

- add mismatch feedback
- add disagreement feedback
- add facet gaps
- replace citation
- undo replacement
- supplement
- refine

The authoritative reviewer state lives in `SessionState.feedback`.

Important feedback fields:

- `citation_mismatches`
- `claim_disagreements`
- `facet_gaps`
- `citation_replacements`

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
- feedback mutation
- supplement and refine
- citation replacement
- scoped retrieval

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
- final assistant message supports inline citation and review actions
- citation badges shown to users are renumbered contiguously (`[1]..[N]`) for readability; backend/internal citation IDs remain unchanged for review mutations
- removed-sentence warnings are surfaced to the reviewer in the message and review panel
- older assistant messages remain visible but read-only

### 10.2.1 Citation drawer and match quality

The citation drawer is `web/src/components/CitationDialog.tsx`. **Match quality** copy is produced by `describeUnifiedMatchQuality()` in `web/src/utils/retrievalDisplay.ts`.

**Inputs from the session citation** (see `CitationIndexEntry` and `build_citation_index` in `src/quarry/pipeline/retrieval.py`):

- `retrieval_score` — final passage score after retrieval/reranking. The UI treats this as a **rough ranking signal**, not a calibrated probability (absolute scale can vary by retriever/reranker).
- `ambiguity_gap` — difference between the top two scores in that retrieval batch (when at least two results exist).
- `ambiguity_review_required` — set when `ambiguity_gap` is below the backend threshold (default `0.05`, `ambiguity_gap_threshold` in `src/quarry/config.py`).
- `reference.confidence_label` / `reference.confidence_unknown` — sentence-to-passage verification output from the NLI step for the specific citation reference the user clicked.
- `sentence.status` — sentence-level verification status (`verified`, `partially_verified`, `ungrounded`, etc.), used as a guardrail so drawer copy stays aligned with the visible answer quality.

**Policy (four user-facing levels):** `strong`, `good`, `fair`, `weak` (headlines: Strong / Good / Fair / Weak match).

1. Map `retrieval_score` to an internal retrieval tier: ≥ `0.9`, ≥ `0.72`, ≥ `0.5`, else lowest.
2. If a comparable `ambiguity_gap` exists, only keep the top retrieval tier when the top result has a clear lead (≥ `0.05`); otherwise cap that step at the next tier down. Missing `ambiguity_gap` does not block a high-scoring citation from reaching the top retrieval tier.
3. **Tight race:** if `ambiguity_review_required` or `ambiguity_gap` &lt; `0.05`, drop one retrieval tier (not below weak).
4. Mix in verification signals for the clicked citation:
   - `sentence.status in {ungrounded, no_ref}` or `reference.confidence_label == not_supported` → `weak`
   - `sentence.status == partially_verified` or `reference.confidence_label == partially_supported` → cap at `fair`
   - `sentence.status == unchecked` or `reference.confidence_unknown == true` → cap at `good`
   - `verified` / `supported` leaves the retrieval-derived tier unchanged
5. **Detail text**:

| Level   | Headline       | Detail (base)                                      |
|---------|----------------|----------------------------------------------------|
| strong  | Strong match   | This passage clearly supports the sentence.        |
| good    | Good match     | This passage supports the sentence.                |
| fair    | Fair match     | This passage supports only part of the sentence.   |
| weak    | Weak match     | This passage does not support the sentence.        |

6. **Invalid score:** non-finite `retrieval_score` → weak, headline “Match quality unknown”, detail: `No retrieval score was available.`

The diagnostics drawer close button uses the same icon-only pattern and `drawer-close-trigger` styling as the citation drawer.

### 10.3 UI modes

The UI handles multiple response modes:

- `response_review`
- `clarification_required`
- `generation_failed`

Clarification suggestions are generated server-side and can be clicked to rerun the query.

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
