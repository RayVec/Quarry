# QUARRY Models and Prompts

This document describes the current runtime models, non-model retrieval components, and prompt-driven tasks used by QUARRY.

## 1. Cache and Warmup

QUARRY keeps model downloads inside the repository:

- `data/model-cache/huggingface/`
- `data/model-cache/modelscope/`
- `data/model-cache/torch/`

Warmup/readiness is recorded in:

- `data/artifacts/local_model_status.json`

That status file is the reliable source for whether a local component is ready.

## 2. Runtime Profiles

### 2.1 `apple_silicon`

Primary local text model:

- `mlx-community/Qwen3.5-4B-MLX-4bit`

Primary local parser model:

- `mlx-community/Qwen3-VL-4B-Instruct-4bit`

Intended use:

- Apple Silicon laptops
- smaller local footprint
- local decomposition
- local metadata enrichment
- local generation when hosted generation is disabled
- MLX-based document parsing

### 2.2 `gpu`

Use this profile on Linux or Windows machines with a GPU. It loads standard HuggingFace models — larger than the MLX equivalents but compatible with any CUDA-capable hardware.

Primary local text model:

- `Qwen/Qwen2.5-7B-Instruct`

Primary parser stack:

- `allenai/olmOCR-7B-0725-FP8`
- lightweight PDF text fallbacks by parser settings
- `MinerU` remains installed and configurable for manual experiments, but it is not part of the default runtime path as of April 2, 2026

Intended use:

- heavier local execution
- stricter all-local setups

## 3. Models by Role

### 3.1 Local text generation and decomposition

Possible models:

- `mlx-community/Qwen3.5-4B-MLX-4bit`
- `Qwen/Qwen2.5-7B-Instruct`

Used for:

- query classification when heuristics do not settle the answer
- query decomposition into facets
- metadata enrichment
- local-only answer generation
- local-only supplement generation
- local-only refinement generation
- local-only sentence regeneration

### 3.2 Local vision and parsing

Primary Apple profile parser:

- `mlx-community/Qwen3-VL-4B-Instruct-4bit`

Used for:

- page-level PDF parsing
- extracting headings, paragraphs, tables, and figure captions
- this is the preferred and required PDF parser path for `apple_silicon`
- ingest warms and checks this parser before parsing starts when local models are enabled
- the prompt explicitly tells the model to ignore TOC pages, page headers, page footers, page numbers, and single-character heading fragments
- when MLX parsing fails, the default lightweight fallbacks are `pymupdf_text` and then `pypdf_text`

Heavier parser path:

- `allenai/olmOCR-7B-0725-FP8`

Installed but not used by default:

- `MinerU`
- kept available for explicit/manual parser experiments only
- not part of the default fallback chain as of April 2, 2026

Lightweight text-file parser:

- `basic_text`
- used only for `.md` and `.txt` inputs

### 3.3 Embedding model

- `intfloat/e5-large-v2`

Used for:

- embedding chunks during indexing
- embedding queries during dense retrieval

### 3.4 Reranker

- `BAAI/bge-reranker-v2-m3`

Used for:

- reranking fused sparse+dense candidates before citation selection

### 3.5 NLI model

- `khalidalt/DeBERTa-v3-large-mnli`

Used for:

- support scoring after exact quote verification
- assigning `supported`, `partially_supported`, or `not_supported`

Scoring policy:

- Argmax over softmax probabilities maps `entailment → supported`, `neutral → partially_supported`, `contradiction → not_supported`.
- **Soft entailment threshold:** if the argmax lands on `neutral` but the raw entailment probability is ≥ `0.35` (`ENTAILMENT_SOFT_THRESHOLD`), the label is upgraded to `supported`. This handles near-verbatim rewrites where probability mass is split between entailment and neutral without a decisive winner.

On Apple Silicon the model is replaced by `HeuristicNLIClient`, which scores token-set overlap as `max(precision, recall)` (see Architecture §7.7).

## 4. Non-model Retrieval Components

### Sparse retrieval

Current sparse retrieval is local BM25 over the ingested chunk corpus.

There is no OpenSearch path in the current implementation.

### Dense retrieval

Dense retrieval is backed by local vector artifacts plus embedding lookup.

Artifacts include:

- `data/artifacts/vector_index.faiss`
- `data/artifacts/vector_index_metadata.json`

### Fusion and reranking

QUARRY uses:

- reciprocal rank fusion
- reranking on the merged candidate set

For obvious single-hop queries, retrieval budgets are smaller than for multi-hop queries.

## 5. Hosted Generation

Hosted generation is optional.

If configured, a hosted model is used for:

- answer generation
- supplement generation
- refinement generation
- sentence regeneration

Supported hosted providers for generation:

- `openai_compatible` (OpenRouter/OpenAI/Azure-compatible endpoints)
- `gemini` (Google AI Studio Gemini API)

Current config knobs:

- `hosted.provider`
- `hosted.llm_base_url`
- `hosted.llm_api_key`
- `hosted.llm_model`
- `hosted.use_live_generation`

Provider-specific behavior:

- `openai_compatible` uses `hosted.llm_base_url` + `hosted.llm_api_key`
- `gemini` uses `hosted.llm_api_key` (or environment variable `GEMINI_API_KEY`)
- Gemini support in this release is generation-only; decomposition and metadata-enrichment hosted paths remain OpenAI-compatible

The recommended mixed strategy is:

- hosted generation on
- local decomposition
- local metadata enrichment
- local retrieval
- local verification

## 6. What Stays Local in Hybrid Mode

In the recommended `hybrid + apple_silicon` setup, these usually stay local:

- query classification
- query decomposition
- metadata enrichment
- PDF parsing
- embeddings
- sparse retrieval
- dense retrieval
- reranking
- NLI confidence scoring

These can be hosted:

- answer generation
- supplement generation
- refinement generation
- sentence regeneration

## 7. Prompted Tasks

QUARRY currently uses prompt-driven model calls for:

- query classification
- query decomposition
- metadata enrichment
- answer generation
- supplement generation
- refinement generation
- sentence regeneration
- JSON repair for structured tasks
- MLX page parsing

Not every query uses every prompt. Some stages now short-circuit heuristically before calling a model.

## 8. Shared System Prompt

The shared system prompt establishes rules that are true across all tasks:

- QUARRY is a research assistant for technical construction reports
- output formats must be followed exactly
- factual statements must be grounded in supplied text
- quotes must be copied verbatim

It is defined in:

- `src/quarry/prompts.py`

It is used in two ways:

- as the true system message for hosted OpenAI-compatible calls and local transformers chat-style calls
- prepended directly into MLX task prompts, because the MLX path does not rely on a separate system-message channel

## 9. Query Classification Prompt (Deprecated)

**Note: As of the latest version, classification is handled entirely by heuristics. The classification model/prompt is no longer used.**

Previously used purpose:

- classify the query as `single_hop` or `multi_hop`

Current behavior:

- Heuristic patterns in `decomposition.py` identify obvious single-hop and multi-hop queries
- Queries that don't match any pattern default to `multi_hop`
- MLX model is used only for facet generation, not classification

## 10. Query Decomposition Prompt

Code:

- `decomposition_prompt(...)`

Purpose:

- turn a multi-hop query into focused, search-ready sub-queries

Current design:

- facets must be self-contained questions
- each facet should target one aspect of the original query
- named entities should be preserved
- overlapping facets are discouraged

Expected JSON:

```json
{ "facets": ["sub-query 1", "sub-query 2"] }
```

## 11. Metadata Enrichment Prompt

Code:

- `metadata_enrichment_prompt(...)`

Purpose:

- enrich a chunk with:
  - `summary`
  - `entities`
  - `questions`

Expected JSON:

```json
{
  "summary": "one sentence",
  "entities": ["term 1", "term 2"],
  "questions": ["question 1", "question 2"]
}
```

Important runtime note:

- in `hybrid`, metadata enrichment can fall back to heuristics if the prompt-driven path fails

## 12. Generation Prompt

Code:

- `generation_prompt(...)`

This is the central prompt for:

- initial answer generation
- supplement generation
- refinement generation
- sentence regeneration

### 12.1 Current structure

The current prompt is intentionally context-first:

1. task framing
2. query
3. information facets
4. source passages
5. existing response context, if applicable
6. reviewer feedback, if applicable
7. previous malformed output, if applicable
8. writing task guidance
9. citation and tagging format rules
10. mode-specific instruction

This ordering is meant to help the model understand what it is answering before it sees output-format constraints.

### 12.2 Sentence tags

The generator can emit:

- `[CLAIM]`
- `[SYNTHESIS]`
- `[STRUCTURE]`
- `[NO_REF]`

Rules:

- `CLAIM`
  - one factual sentence
  - exactly one verbatim quote
  - standard generation uses a 10 to 40 word quote anchor
- `SYNTHESIS`
  - combines evidence across passages
  - at least two verbatim quotes from different passages
- `STRUCTURE`
  - framing or connective text
  - no citation
- `[NO_REF]`
  - used when no supplied passage supports the claim

### 12.3 Supplement mode

Supplement mode tells the model:

- an earlier response already exists
- certain facets need more coverage
- write only the additional content
- do not repeat the existing response

### 12.4 Refinement mode

Refinement mode tells the model:

- regenerate the full response
- avoid flagged passages
- consider reviewer disagreements
- present conflicting evidence when needed
- say so when evidence is insufficient

**How citation dislike reaches this prompt (runtime wiring):**

- Refinement generation only runs when there are unresolved selection comments and/or at least one **disliked** citation (see pipeline refine gate). **Likes are not injected into the prompt.**
- For each refinement call, `mismatch_citation_ids` is set to the list of citation ids the reviewer **disliked**.
- Source passages for the request **exclude** those disliked ids entirely, so their chunk text does not appear under **Source Passages**.
- `## Reviewer Feedback` is built by `_format_reviewer_feedback`: for each id in `mismatch_citation_ids`, it emits a line that the citation was flagged as a mismatch (when the citation is no longer in the trimmed index, the line is generic; if it were still present, it would include section path and note).
- Individual passage lines can be prefixed with `[MISMATCH: …]` via `_format_citation_line` when that passage’s id is in `mismatch_citation_ids` **and** the passage is still included — in the usual dislike path the passage is omitted instead, so the model mainly sees the textual feedback bullets plus the remaining passages.
- Unresolved selection comments are also merged into **Reviewer Feedback** and into `disagreement_notes`-driven lines; refinement mode instructions additionally tell the model to treat selection comments as required edits when present.

### 12.5 Regeneration mode

Regeneration mode is sentence-level repair.

It tells the model:

- which sentence failed citation verification
- to rewrite only that sentence
- to use valid evidence from the supplied passages
- to prefer a clear natural sentence over copied chunk openings, headings, or bullets
- that shorter exact quotes of 8 to 10 words are allowed during sentence repair
- to use `[NO_REF]` if no passage supports the claim

Current runtime policy after regeneration:

- if a regenerated sentence still cannot be grounded, QUARRY falls back to `[NO_REF]` rather than fabricating prose from raw chunk text
- if a `CLAIM` sentence still cannot be grounded after the allowed regeneration attempts, QUARRY removes it from the visible response
- if a `SYNTHESIS` sentence still cannot be grounded after the allowed regeneration attempts, QUARRY also removes it from the visible response

## 13. Regeneration Retry Guidance

Recent behavior change:

- if a regenerated sentence fails verification and another retry is still warranted, the next regeneration prompt includes the failed rewrite itself

This adds:

```text
## Retry Guidance
Your previous rewrite was: "..."
This still could not be verified. Either find a different passage to cite,
or respond with [NO_REF] if no passage supports this claim.
```

Together with the sentence-repair instructions, this prevents the model from simply repeating the same failed rewrite and gives it room to anchor a cleaner sentence on a shorter exact quote.

Related runtime behavior:

- if a failed rewrite is too similar to the previous attempt, QUARRY now halts regeneration rather than retrying again

That stop condition is in the pipeline layer, not the prompt itself.

## 14. Repair Prompt for Malformed Generation

Code:

- `repair_generation_prompt(...)`

Purpose:

- retry malformed generation outputs while keeping task and citation rules intact

This wraps the normal generation prompt and adds a final repair instruction telling the model to return a corrected response with no commentary or markdown fences.

## 15. JSON Repair Prompt

Code path:

- `src/quarry/adapters/mlx_runtime.py`

Purpose:

- when a structured task returns invalid JSON, QUARRY gives the model one repair attempt

Repair instruction:

- previous output was not valid JSON
- return valid JSON only
- no fences
- no commentary

This is used for:

- classification
- decomposition
- metadata enrichment
- MLX page block extraction

## 16. MLX Page Parsing Prompt

Code path:

- `render_parser_prompt(...)` in `src/quarry/adapters/mlx_runtime.py`

Purpose:

- parse a rasterized PDF page into structured blocks

Expected JSON shape:

```json
{
  "blocks": [
    {
      "block_type": "heading" | "paragraph" | "table" | "table_title" | "figure_caption",
      "text": "string",
      "section_depth": 1,
      "section_heading": "optional heading"
    }
  ]
}
```

Key rules:

- preserve reading order
- only use heading for real headings
- ignore TOC pages and TOC entries
- ignore page headers, footers, and page numbers
- do not emit single-character headings or heading fragments
- merge visibly broken tokens instead of splitting them across blocks
- do not invent coordinates

## 17. Prompt Routing by Runtime

### Hosted

Hosted OpenAI-compatible calls:

- use the shared system prompt as a true system message
- send the task prompt as the user message

### Local transformers

Local transformers calls:

- use the shared system prompt in the model’s chat template
- send the task prompt as the user instruction

### MLX

MLX calls:

- inline the shared system prompt directly into the task prompt
- attempt one JSON repair when needed
- may fall back to heuristic behavior in `hybrid`

## 18. Runtime Status and Inspection

To check what the current process is using, inspect:

- `data/artifacts/local_model_status.json`
- session diagnostics in the frontend
- runtime logs in `data/logs/runtime/`
- corpus-stage logs in `data/logs/corpus/`

Important session/runtime fields exposed by the backend:

- `generation_provider`
- `parser_provider`
- `runtime_mode`
- `runtime_profile`
- `local_model_status`
- `active_model_ids`

## 19. Warmup

Warmup command:

```bash
source .venv/bin/activate
quarry warm-local-models
```

Warmup verifies readiness for:

- embedding
- reranker
- NLI
- text
- decomposition
- generation
- parser

In `local` mode, startup enforces these readiness checks before the server is allowed to run.
