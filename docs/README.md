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
python3 -m venv .venv
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
2. Fill in `quarry.local.toml`
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

## Configuration

QUARRY reads local config from:

- `quarry.local.toml`

Environment variables still work, but the TOML file is the intended day-to-day setup path.

Important config sections:

- `[runtime]`
  - `mode = "hybrid" | "local" | "hosted"`
  - `profile = "apple_lite_mlx" | "full_local_transformers"`
- `[hosted]`
  - `llm_base_url`
  - `llm_api_key`
  - `llm_model`
  - `use_live_generation`
  - `use_live_decomposition`
  - `use_live_metadata_enrichment`
- `[mlx]`
  - `text_model`
  - `vision_model`
  - `max_new_tokens`
  - `page_image_dim`
- `[local_models]`
  - `embedding_model`
  - `reranker_model`
  - `nli_model`

The easiest mixed setup is:

- `runtime.mode = "hybrid"`
- `runtime.profile = "apple_lite_mlx"`
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

## Runtime Modes

- `hybrid`
  - local-first, but graceful fallbacks and hosted generation are allowed
- `local`
  - strict local mode; startup fails if required warmed local components are missing
- `hosted`
  - intended for externally provided hosted workflows

## Runtime Profiles

- `apple_lite_mlx`
  - default on Apple Silicon
  - MLX Qwen text + MLX Qwen vision parser
  - parser is model-first for PDFs
- `full_local_transformers`
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
- when `apple_lite_mlx` is active and local models are enabled, ingest tries the MLX parser first
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
quarry start --profile apple_lite_mlx
quarry start --config /path/to/quarry.local.toml
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

- inline disagreement flagging under individual sentences
- citation drawer for quote context, mismatch marking, replacements, and scoped retrieval
- displayed citation badges are renumbered contiguously (1..N) for readability, while internal citation IDs remain stable for review actions
- expandable “Review and refine” panel
- hidden diagnostics drawer for runtime and retrieval details

The backend still exposes the same core review actions:

- mismatch feedback
- disagreement feedback
- facet-gap feedback
- supplement
- refine
- citation replacement / undo replacement

## Query Pipeline Highlights

Important current behavior:

- obvious query shapes are classified heuristically first before escalating to a model
- single-hop queries use smaller retrieval budgets than multi-hop queries
- single-hop generation also trims citation context to the strongest passages
- exact quote verification still runs after generation
- NLI confidence scoring still runs after exact-match verification
- lingering ungrounded `CLAIM` sentences are removed from the final response instead of being shown as unsupported facts
- lingering ungrounded `SYNTHESIS` sentences are also removed from the final response
- regeneration now avoids repeated retries when the model is converging on the same failed sentence
- retries that do happen now include the previous failed rewrite and an explicit `[NO_REF]` escape path

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
