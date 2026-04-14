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
- `MinerU` remains installed and configurable for manual experiments, but it is not part of the default fallback chain

Intended use:

- heavier local execution
- stricter all-local setups

## 3. Models by Role

### 3.1 Local text generation and decomposition

Possible models:

- `mlx-community/Qwen3.5-4B-MLX-4bit`
- `Qwen/Qwen2.5-7B-Instruct`

Used for:

- query **decomposition into facets** (only when `QueryDecomposer` classifies the query as **multi-hop**; single-hop skips this call)
- metadata enrichment (when not hosted)
- local-only answer generation
- local-only supplement generation
- local-only refinement generation
- local-only sentence regeneration

Query **classification** (single-hop vs multi-hop) is **heuristic-only** in `QueryDecomposer`; this text model is not used for that step.

### 3.2 Local vision and parsing

Primary Apple profile parser:

- `mlx-community/Qwen3-VL-4B-Instruct-4bit` (via the `qwen3_vl_mlx` / `mlx_vlm` adapter)

Used for:

- page-level PDF parsing
- extracting headings, paragraphs, tables, and figure captions
- when `runtime.profile = apple_silicon` and `use_local_models` is true, ingest **always** uses this MLX vision parser for the PDF primary chain; **`[parser] primary` in config does not switch the Apple PDF path** (that key applies on the GPU profile). The MLX fallback chain still uses `[parser] fallback` (for example `pymupdf_text`) after primary failures
- ingest warms and checks this parser before parsing starts when local models are enabled
- the prompt explicitly tells the model to ignore TOC pages, page headers, page footers, page numbers, and single-character heading fragments
- when MLX parsing fails, the default lightweight fallbacks are `pymupdf_text` and then `pypdf_text`

Heavier parser path (GPU / non-MLX profile):

- `allenai/olmOCR-7B-0725-FP8` — typical default when `parser.primary = olmocr_transformers` in config (see `config.example.toml` and `Settings` defaults)

Installed but not used by default:

- `MinerU`
- kept available for explicit/manual parser experiments only
- not part of the default fallback chain

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

For obvious single-hop queries, retrieval budgets are smaller than for multi-hop queries (12/12/8 caps).

For multi-hop queries, retrieval uses a widened anchored pool strategy:

- per-facet sparse/dense budgets remain the configured defaults (typically 30/30)
- merged fused candidates are capped by `retrieval.multihop_anchor_pool_size` (default 40) before reranking
- final multi-hop rerank output is capped by `retrieval.multihop_rerank_budget` (default 20)
- retrieved passages carry both `source_facet` and `source_facets` provenance for downstream coverage checks

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
- `hosted.use_live_decomposition` (OpenAI-compatible chat completions only; not Gemini)
- `hosted.use_live_metadata_enrichment` (OpenAI-compatible only; not Gemini)
- `hosted.use_live_embeddings` plus `hosted.embedding_base_url`, `hosted.embedding_api_key`, `hosted.embedding_model`, `hosted.embedding_dimensions` for optional hosted embedding during indexing/query (separate from the chat LLM)

Provider-specific behavior:

- `openai_compatible` uses `hosted.llm_base_url` + `hosted.llm_api_key` for chat-style tasks (generation, and optionally decomposition / metadata enrichment when the toggles above are on); Azure OpenAI should use the `https://<resource>.openai.azure.com/openai/v1` route and the deployed model name
- `gemini` uses `hosted.llm_api_key` (or environment variable `GEMINI_API_KEY`) for **generation only** (`use_live_generation`); the Google client prepends `SHARED_SYSTEM_PROMPT` into the same user-facing content via `with_shared_system_prompt` (no separate chat `system` role)
- Hosted decomposition and metadata enrichment, when enabled, use the same OpenAI-compatible HTTP stack as generation (`OpenAICompatibleLLM`), not the Gemini client

The recommended mixed strategy is:

- hosted generation on
- local decomposition
- local metadata enrichment
- local retrieval
- local verification

## 6. What Stays Local in Hybrid Mode

In the recommended `hybrid + apple_silicon` setup, these usually stay local:

- query classification (heuristic-only in `QueryDecomposer`; no model call)
- query decomposition (local MLX or local transformers, unless `use_live_decomposition` is enabled)
- metadata enrichment (local, unless `use_live_metadata_enrichment` is enabled)
- PDF parsing
- embeddings (local, unless `use_live_embeddings` is enabled)
- sparse retrieval
- dense retrieval
- reranking
- NLI confidence scoring

These can be hosted (according to `hosted.*` toggles):

- answer generation (`use_live_generation`)
- supplement generation
- refinement generation
- sentence regeneration
- optionally: decomposition, metadata enrichment, embeddings (each has its own flag; decomposition/metadata require OpenAI-compatible credentials)

## 7. Prompted Tasks

QUARRY currently uses prompt-driven model calls for:

- query decomposition into facets (only when the heuristic classifier yields **multi-hop**; **single-hop** skips the decomposition model and uses the original query as the only facet)
- metadata enrichment
- answer generation
- supplement generation
- refinement generation
- sentence regeneration
- JSON repair for structured MLX outputs (decomposition and metadata enrichment on the MLX path)
- MLX page parsing

**Not model-driven in current runtime:** query **classification** is handled only by heuristics in `QueryDecomposer` (see §9). There is no live call to `decomposition_classification_prompt`.

Not every query uses every prompt. Some stages short-circuit heuristically before calling a model.

The decomposition prompt now includes bridge-aware guidance: when a query links two entities through a relationship, generate one facet per entity plus one facet for the connecting relationship when appropriate.

## 8. Shared System Prompt

The shared system prompt establishes rules that are true across all tasks:

- QUARRY is a grounded research assistant for technical reports
- output formats must be followed exactly
- factual statements must be grounded in supplied text
- the answer should be written in the user's language unless another language is requested
- prose should be clear, natural, concise, and logically ordered
- the user’s question should be answered directly rather than drifting into document narration unless provenance is itself important
- source formatting such as bullets, checklists, field labels, table fragments, and OCR noise should not be imitated
- quotes must be copied verbatim

It is defined in:

- `src/quarry/prompts.py`

It is used in several ways:

- as the true `system` message for hosted **OpenAI-compatible** chat completions and local **transformers** chat-style calls
- prepended into the **Gemini** request body via `with_shared_system_prompt(...)` (single contents string, not a separate API system role)
- prepended directly into **MLX** task prompts, because the MLX path does not rely on a separate system-message channel

## 9. Query Classification (Heuristic-Only) and Legacy Prompt

**Runtime behavior:** `QueryDecomposer` in `src/quarry/pipeline/decomposition.py` classifies every query with **heuristics only** (`_heuristic_classify_query`). There is **no** call to a language model for classification.

- Obvious multi-hop patterns (phrases, comma/`and` structure with certain openers) → `multi_hop`
- Obvious single-hop patterns (prefixes such as “what is …”, metric hints without multi-clause structure) → `single_hop`
- If neither pattern matches → default **`multi_hop`** with source logged as `heuristic_default_multi_hop`
- The old third bucket **`clarification_required`** is not produced by this heuristic path (the prompt text in `decomposition_classification_prompt` still describes it for historical/tests only)

**Legacy code:** `decomposition_classification_prompt(...)` remains in `src/quarry/prompts.py` and still documents JSON for `single_hop` | `multi_hop` | `clarification_required`, but **no production code invokes it** today (`DecompositionClient` only requires `decompose_query`). Tests may still import the function to pin prompt text.

**After classification:** for `multi_hop` only, the configured decomposition client runs **`decomposition_prompt`** (MLX, local transformers, hosted OpenAI-compatible, or heuristic fallback). For `single_hop`, the model decomposition step is skipped and the original query is used as the single facet.

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
9. source-handling guidance for raw evidence
10. citation and tagging format rules
11. mode-specific instruction

This ordering is meant to help the model understand what it is answering before it sees output-format constraints.

Current writing guidance in the prompt emphasizes:

- start with the direct answer or most important finding
- match the answer shape to the user’s request instead of defaulting to a document tour
- treat decomposition facets as internal coverage checks rather than a reader-facing outline
- organize around the smallest clear set of reader-facing themes or steps needed to answer fully
- synthesize aggressively: merge overlapping evidence, summarize categories, and use representative examples unless the user explicitly asks for an exhaustive inventory
- use additional passages only when they add a distinct detail, contrast, or qualification
- avoid provenance-led narration unless source provenance or exact wording is materially important
- keep the response compact and readable with short grouped paragraphs instead of laundry-list structure

### 12.2 Paragraph markers and sentence tags

The generator may insert **`[PARA]`** between topic shifts (formatting only; not a sentence tag and must not carry references). The prompt instructs grouping related sentences into paragraphs and using `[PARA]` when the topic changes.

The generator can emit:

- `[CLAIM]`
- `[SYNTHESIS]`
- `[STRUCTURE]`
- `[NO_REF]`

Rules:

- `CLAIM`
  - one factual sentence
  - exactly one verbatim quote
  - the sentence itself should be natural prose rather than copied source formatting
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

**How citation dislike / like reach this prompt (runtime wiring):**

- Refinement generation only runs when there are unresolved selection comments and/or at least one **disliked** citation. **Likes do not trigger refine by themselves.**
- The refinement request now carries pair-scoped `approved_pairs` and `rejected_pairs` so the prompt can reason about the same citation id in different sentence contexts.
- `mismatch_citation_ids` is still populated for globally rejected citation ids, but only as a compatibility bridge. Mixed feedback for the same citation id should be represented through `approved_pairs` / `rejected_pairs`, not by broad mismatch taint.
- Source passages for the request exclude globally rejected ids entirely, so their chunk text does not appear under **Source Passages**.
- `## Reviewer Feedback` includes soft approval lines for `approved_pairs` and mismatch lines for globally rejected ids. Source passages are rendered as structured evidence blocks (`Passage [id]`, `Section`, optional reviewer note/flag, `Raw evidence`) so the model treats them as evidence rather than answer prose.
- Unresolved selection comments are also merged into **Reviewer Feedback** and into `disagreement_notes`-driven lines; refinement mode instructions additionally tell the model to preserve approved pairs only if they remain supported after rewriting.

### 12.5 Regeneration mode

Regeneration mode is sentence-level repair.

It tells the model:

- which sentence failed citation verification
- to rewrite only that sentence
- to use valid evidence from the supplied passages
- to prefer a clear natural sentence over copied chunk openings, headings, or bullets
- that shorter exact quotes of **8 to 15 words** are allowed during sentence repair (see `generation_prompt` regeneration branch in `prompts.py`)
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

- `src/quarry/adapters/mlx_runtime.py` (`_json_repair_prompt` and callers)

Purpose:

- when an **MLX** structured task returns invalid JSON, QUARRY gives the model one repair attempt before falling back (for decomposition/metadata) or failing the parse path

Repair instruction:

- previous output was not valid JSON
- return valid JSON only
- no fences
- no commentary

This is used for:

- **query decomposition** (MLX): after `decomposition_prompt`, on parse failure
- **metadata enrichment** (MLX): after `metadata_enrichment_prompt`, on parse failure

It is **not** used for query classification in production (classification is heuristic-only). **Local transformers** decomposition and metadata paths (`LocalStructuredDecompositionClient`, `LocalStructuredMetadataEnricher`) call `parse_json_response` once and **fall back to heuristics** on failure without an automatic JSON-repair pass.

MLX page parsing uses separate parsing/retry logic around `parse_mlx_page_blocks` / `render_parser_prompt`; it does not share the same `_json_repair_prompt` helper as decomposition.

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

Hosted **OpenAI-compatible** calls:

- use `SHARED_SYSTEM_PROMPT` as a true chat `system` message
- send the task prompt as the user message

Hosted **Gemini** generation:

- pass a single string built with `with_shared_system_prompt(prompt)` into the Google GenAI `generate_content` call (system rules are inlined, not a separate role)

### Local transformers

Local transformers calls:

- use the shared system prompt in the model’s chat template
- send the task prompt as the user instruction

### MLX

MLX calls:

- inline the shared system prompt directly into the task prompt (`with_shared_system_prompt`)
- attempt one JSON repair for **decomposition** and **metadata enrichment** when `parse_json_response` fails
- may fall back to heuristic decomposition/metadata behavior in `hybrid` when the client provides a heuristic fallback

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
- text (shared backend with decomposition/generation readiness flags in the status file)
- decomposition
- generation
- metadata enrichment
- parser

In `local` mode, startup enforces readiness checks for embedding, reranker, NLI, decomposition, generation, and parser (see `_ensure_runtime_ready` in `src/quarry/api/app.py`); the warmup run still records **metadata** and **text** in `local_model_status.json` for visibility.
