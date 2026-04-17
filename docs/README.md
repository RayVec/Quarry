# QUARRY

QUARRY is a local-first grey literature retrieval system for small curated corpora of technical construction reports. It ingests source documents into local artifacts, retrieves evidence with a sparse+dense pipeline, generates citation-tagged answers, verifies the grounding, and presents the result in a conversational review UI.

The current implementation is intentionally lightweight:

- local source files under `data/sources/`
- generated artifacts under `data/artifacts/`
- FastAPI backend
- React + Vite frontend
- `lucide-react` icon library for frontend controls and status glyphs
- in-memory query sessions
- no database
- no OpenSearch

## Docs

- `docs/README.md`: setup and run guide
- `docs/DESIGN.md`: visual design system and layout rules
- `docs/ARCHITECTURE.md`: system architecture, modules, pipeline, API, and UI design
- `docs/MODELS_PROMPTS.md`: runtime model inventory and prompt reference

## Quick Start

Backend environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,local]"
```

Frontend:

```bash
cd web
npm install
```

Normal local workflow:

1. Put source documents in `data/sources/`
2. Fill in `config.toml`
3. Start the backend
4. Start the frontend

Backend:

```bash
source .venv/bin/activate
python start_backend.py
```

`start_backend.py` will ask:

```text
Run corpus rebuild from data/sources before starting backend? [y/N]:
```

- `y`: rebuild artifacts from `data/sources/`, then start the API
- anything else: skip rebuild and start from existing artifacts (no startup validation gate)

Frontend:

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

- `http://127.0.0.1:5173`

Python/runtime note:

- the current project metadata targets Python 3.13+
- the April 6, 2026 Apple Silicon crash was initially misdiagnosed as a Python 3.13 compatibility problem
- the corrected root cause was concurrent local torch/MPS access during multi-facet retrieval
- current code serializes access to shared local model instances; if you still see native MPS crashes, set `runtime.local_model_device = "cpu"` as a fallback

## Configuration

QUARRY reads local config from:

- `config.toml`

Environment variables still work, but the TOML file is the intended day-to-day setup path.

Important config sections:

- `[runtime]`
  - `mode = "hybrid" | "local" | "hosted"`
  - `profile = "apple_silicon" | "gpu"`
- `[hosted]`
  - `provider` (`openai_compatible` or `gemini`)
  - `llm_base_url`
  - `llm_api_key`
  - `llm_model`
  - `use_live_generation`
  - `use_live_decomposition`
  - `use_live_metadata_enrichment`
  - `use_live_embeddings`
  - `embedding_base_url`
  - `embedding_api_key`
  - `embedding_model`
  - `embedding_dimensions`
- `[mlx]`
  - `text_model`
  - `vision_model`
  - `max_new_tokens`
  - `page_image_dim`
- `[local_models]`
  - `embedding_model` (local embedding model)
  - `reranker_model`
  - `nli_model`

Embedding path note:

- local embedding uses `[local_models].embedding_model` (default `intfloat/e5-large-v2`)
- hosted embedding uses `[hosted].embedding_model` (default `hash-embedding-v1`) when `hosted.use_live_embeddings = true`

The easiest mixed setup is:

- `runtime.mode = "hybrid"`
- `runtime.profile = "apple_silicon"`
- `hosted.use_live_generation = true`
- `hosted.use_live_decomposition = false`
- `hosted.use_live_metadata_enrichment = false`

That gives you:

- local decomposition
- local retrieval
- local reranking
- local verification
- local parsing
- hosted answer generation / refinement / regeneration

Gemini note:

- `provider = "gemini"` currently applies to hosted generation only
- decomposition and metadata enrichment remain on the existing OpenAI-compatible hosted path (or local/heuristic fallbacks)
- for Gemini keys, QUARRY accepts either `hosted.llm_api_key` or environment variable `GEMINI_API_KEY`

Azure OpenAI note:

- keep `provider = "openai_compatible"`
- set `hosted.llm_base_url = "https://<resource>.openai.azure.com/openai/v1"`
- set `hosted.llm_model` to your Azure deployment name
- for new integrations, use the OpenAI v1-compatible route instead of `https://<resource>.services.ai.azure.com/models`

## Runtime Modes

- `hybrid`
  - local-first, but graceful fallbacks and hosted generation are allowed
- `local`
  - strict local mode; startup fails if required warmed local components are missing
- `hosted`
  - intended for externally provided hosted workflows

## Runtime Profiles

- `apple_silicon`
  - default on Apple Silicon
  - MLX Qwen text + MLX Qwen vision parser
  - parser is model-first for PDFs
- `gpu`
  - for Linux / Windows machines with a GPU
  - standard HuggingFace Qwen 7B text model + olmOCR 7B vision parser
  - models are larger and slower than the MLX equivalents

## Corpus and Artifacts

Use the folders this way:

- `data/sources/`
  - raw PDFs, markdown, and text files you want QUARRY to ingest
- `data/artifacts/`
  - generated corpus artifacts built from those sources
- `data/corpus/`
  - fallback/sample corpus data used only when no generated artifact manifest exists

If you add or change documents, rebuild from `data/sources/` before asking new questions.

Important parser behavior:

- QUARRY does not intentionally skip the Apple MLX parser just because a warmup status file is missing or stale
- when `apple_silicon` is active and local models are enabled, ingest tries the MLX parser first
- direct ingest also ensures the MLX parser is warmed before parsing begins
- the default PDF fallback chain is `pymupdf_text` first, then `pypdf_text`
- those text fallbacks are intentionally lossy and are there to preserve recoverable page text when MLX parsing fails
- `MinerU` remains installed and configurable, but as of April 2, 2026 it is not part of the default runtime fallback chain
- per-page MLX failures, recoveries, and skipped pages are logged to the terminal and corpus log
- corpus progress logs now include model/parser names plus remaining document/page/chunk counts where that information is available
- `parsed_document.json` now records `recovered_pages`, `skipped_pages`, and per-page parse statuses for debugging
- `.md` and `.txt` files use the lightweight `basic_text` parser
- parsed PDF output is cleaned before chunking
- parse normalization strips common TOC/header/footer artifacts and suspicious one-character heading fragments before chunking
- if fallback is used, the backend logs it explicitly

## CLI Commands

QUARRY also exposes a package CLI:

```bash
source .venv/bin/activate
quarry start
quarry serve
quarry ingest data/sources/*.pdf
quarry rebuild-indexes
quarry warm-local-models
```

Useful variants:

```bash
quarry start --skip-corpus
quarry start --profile apple_silicon
quarry start --config /path/to/config.toml
```

## Current Query Experience

The frontend is now conversational rather than dashboard-based.

- first load shows a centered composer
- after submit, the user query becomes a chat bubble immediately
- a pending QUARRY bubble appears right away
- that pending bubble is driven by real backend query stages, not a fake timer
- once the session completes, the pending bubble becomes the final assistant message

Conversation history is frontend-local and browser-persisted:

- each new query creates a new backend session
- the frontend keeps earlier messages visible as read-only snapshots
- only the newest assistant response stays interactive
- the current browser restores the thread after refresh from local storage
- clearing browser storage removes that local history

## Review Features

The current UI keeps review actions but moves them into contextual surfaces:

- paragraph-style reading flow with inline sentence interactions
- text-selection-first comments: no new-comment controls are visible until text is selected
- a floating comment icon appears near the active selection and opens a compact floating card
- saved comments render as inline yellow annotation highlights with margin indicators; clicking either opens comment edit/delete
- citation drawer for quote context
- displayed citation badges are renumbered contiguously (1..N) for readability, while internal citation IDs remain stable for review actions
- review panel beneath the newest assistant message for feedback summary and refine
- workspace drawer with provider settings and runtime diagnostics tabs

Response rendering now uses paragraph grouping markers from generation/parsing:

- generator can emit `[PARA]` between topic shifts
- parser assigns `paragraph_index` per sentence
- frontend renders sentence text continuously within each paragraph

Frontend citation quality display is resolved per reference in the client with `describeUnifiedMatchQuality(...)`, using retrieval score plus verification signals such as exact-quote match, quote coverage, sentence status, and confidence labels. The backend still exposes sentence-level `match_quality = strong | partial | none` as a coarse fallback.

- `strong` / `good` display as a green citation badge
- `fair` displays as an amber citation badge
- `weak` hides the badge unless the citation is in a pending replacement state

Raw `status`, `confidence_label`, and backend `match_quality` remain in diagnostics/session payloads for debugging and logs.

The backend now exposes a unified review action model built around text selections:

- selection comments with `text_selection`, `char_start`, `char_end`, and `comment_text`
- citation-level **like / dislike / neutral** via `POST .../citations/{id}/feedback` (scoped by `sentence_index`)
- citation replacement/undo actions from the citation drawer
- single refine endpoint that applies selection comments and citation-driven refinement decisions
- resolved comment tracking when a prior selection no longer anchors in the refined response

**Refine and citation thumbs:** Running refine triggers a full-response regeneration only when there is at least one **unresolved selection comment** or at least one **disliked** citation. **Disliked** citations are treated as hard negatives and are removed from the passages shown to the model for that call. **Likes** are soft positives: they do not trigger refine, but when refine is already happening they are passed through as pair-scoped approval signals so the backend can prefer preserving still-supported sentence/citation pairs. See `docs/ARCHITECTURE.md` (section 8.1) for the full pipeline description.

## Query Pipeline Highlights

Important current behavior:

- query classification is heuristic-only; unmatched shapes default to `multi_hop`
- single-hop queries use smaller retrieval budgets than multi-hop queries
- multi-hop retrieval now keeps a wider anchored candidate pool (`retrieval.multihop_anchor_pool_size`, default 40) before reranking down to `retrieval.multihop_rerank_budget` (default 20)
- citation entries now preserve facet provenance in both `source_facet` (primary facet) and `source_facets` (all facets that surfaced the chunk)
- single-hop generation also trims citation context to the strongest passages
- exact quote verification still runs after generation
- multi-hop flow now performs a post-exact-match facet coverage check and may run one bounded follow-up retrieval + supplement pass when a facet is uncovered
- NLI confidence scoring still runs after exact-match verification
- quote verification now precomputes normalized chunk text once per in-memory store load and reuses it for scoped/full quote lookup (no behavior change, lower repeated work)
- verification now exposes lightweight lookup telemetry (`scoped_lookups`, `full_corpus_fallbacks`, `quote_match_rate`, `avg_candidates_checked`) for performance monitoring
- when hosted or local generation falls back, QUARRY now emits conservative `[NO_REF]` output instead of fabricating a sentence from raw chunk text
- lingering ungrounded `CLAIM` sentences are removed from the final response instead of being shown as unsupported facts
- lingering ungrounded `SYNTHESIS` sentences are also removed from the final response
- regeneration now avoids repeated retries when the model is converging on the same failed sentence
- retries that do happen now include the previous failed rewrite and an explicit `[NO_REF]` escape path
- default exact-match verification still expects at least 10 words for a quoted anchor, but sentence-repair prompts can use shorter verbatim anchors (8 to 15 words) and post-regeneration verification honors that lower minimum

## Logging

QUARRY writes timestamped logs under:

- `data/logs/runtime/`
- `data/logs/corpus/`

Pytest logs are separated automatically under:

- `data/logs/tests/`

The console is intentionally concise. The file logs are verbose and include:

- incoming query text
- request trace IDs
- stage transitions
- prompts
- raw model responses
- retrieval diagnostics
- verification summaries

## Testing

Backend and shared tests:

```bash
source .venv/bin/activate
pytest -q
```

Frontend build check:

```bash
cd web
npm run build
```

Browser E2E:

```bash
source .venv/bin/activate
python -m playwright install chromium
QUARRY_RUN_E2E=1 pytest -q -m e2e
```
