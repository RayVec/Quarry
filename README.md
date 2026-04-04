# QUARRY

QUARRY is a local-first grey literature retrieval and review system for small, curated corpora of technical documents. It ingests PDFs into local artifacts, retrieves evidence with a sparse + dense pipeline, generates citation-tagged answers, verifies their grounding, and presents the result in a conversational review UI.

- **No database** — all indexes are local files
- **No OpenSearch** — sparse retrieval is local BM25
- **No cloud required** — runs fully local on Apple Silicon with MLX models
- **Optional hosted generation** — plug in any OpenAI-compatible API for answer generation when you want better output quality without running large models locally

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.13+ |
| Node.js | 18+ |
| Apple Silicon | Recommended (for `apple_silicon` profile) |
| GPU (CUDA) | Optional (for `gpu` profile) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/quarry.git
cd quarry
```

### 2. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,local]"
```

> The `local` extra pulls in torch, transformers, faiss, mlx-vlm, and other model dependencies. On Apple Silicon this resolves to the MPS-compatible builds.

### 3. Frontend

```bash
cd web
npm install
```

---

## Configuration

QUARRY reads its runtime config from `config.toml`. This file is **gitignored** and must be created locally by each user.

```bash
cp config.example.toml config.toml
```

Then open `config.toml` and fill in the values for your setup. The example file is fully annotated.

### Runtime modes

| Mode | Behaviour |
|---|---|
| `hybrid` | Local-first, but allows hosted generation — **recommended starting point** |
| `local` | Strict local only; startup fails if local models are not warmed |
| `hosted` | Delegates all generation to the hosted LLM |

### Runtime profiles

| Profile | Hardware | Local model stack |
|---|---|---|
| `apple_silicon` | Apple Silicon (M1/M2/M3/M4) | MLX Qwen 4B text + MLX Qwen-VL 4B vision parser |
| `gpu` | Linux / Windows with a GPU | HuggingFace Qwen 7B text + olmOCR 7B vision parser |

> **Auto-detection:** QUARRY defaults to `apple_silicon` on Apple Silicon and `gpu` on everything else. You can override with `runtime.profile` in your TOML.

### Recommended hybrid setup (Apple Silicon + hosted generation)

```toml
[runtime]
mode = "hybrid"
profile = "apple_silicon"

[hosted]
llm_base_url = "https://openrouter.ai/api/v1"
llm_api_key  = "YOUR_API_KEY_HERE"
llm_model    = "openai/gpt-4o-mini"
use_live_generation = true
```

This keeps decomposition, retrieval, reranking, PDF parsing, and NLI verification local while offloading answer generation to the hosted model.

### Fully local setup (Apple Silicon, no API key needed)

```toml
[runtime]
mode = "local"
profile = "apple_silicon"

[hosted]
use_live_generation = false
```

All generation runs through the local MLX text model. Run `quarry warm-local-models` first to download and verify the models.

### Model download sizes

| Component | Model | Approx. size |
|---|---|---|
| Text (MLX) | `mlx-community/Qwen3.5-4B-MLX-4bit` | ~2.5 GB |
| Vision/parser (MLX) | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | ~2.5 GB |
| Embeddings | `intfloat/e5-large-v2` | ~1.3 GB |
| Reranker | `BAAI/bge-reranker-v2-m3` | ~570 MB |
| NLI | `khalidalt/DeBERTa-v3-large-mnli` | ~900 MB |
| Text (transformers) | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB |
| Parser (transformers) | `allenai/olmOCR-7B-0725-FP8` | ~8 GB |

Models are cached under `data/model-cache/` by default (gitignored). Change `paths.model_cache_dir` to redirect to an external drive if disk space is tight.

---

## Running

### 1. Add source documents

Drop your PDFs (or `.md` / `.txt` files) into `data/sources/`.

### 2. Start the backend

```bash
source .venv/bin/activate
python start_backend.py
```

The script will ask whether to rebuild the corpus from `data/sources/` before starting the API. Answer `y` on first run or whenever you add documents.

### 3. Start the frontend

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

### CLI shortcuts

```bash
source .venv/bin/activate

quarry start                          # interactive rebuild prompt + serve
quarry start --skip-corpus            # skip rebuild, serve existing artifacts
quarry start --profile apple_silicon # override profile at launch
quarry ingest data/sources/*.pdf      # ingest only, no server
quarry rebuild-indexes                # rebuild vector + sparse indexes only
quarry warm-local-models              # download and verify all local models
quarry serve                          # serve without corpus prompt
```

---

## Warm up local models (first run)

Before running in `local` mode — or the first time you use a new profile — warm the models:

```bash
source .venv/bin/activate
quarry warm-local-models
```

This downloads all models and writes a readiness record to `data/artifacts/local_model_status.json`.

---

## Data directories

```
data/
  sources/      ← put your PDFs and text files here (gitignored)
  artifacts/    ← generated indexes and parsed chunks (gitignored)
  corpus/       ← optional sample/fallback corpus data (gitignored)
  model-cache/  ← downloaded model weights (gitignored, can be large)
  logs/         ← runtime and corpus logs (gitignored)
```

All `data/` subdirectories except the directories themselves are gitignored. The directories are preserved with `.gitkeep` files.

---

## Testing

```bash
# Backend unit tests
source .venv/bin/activate
pytest -q

# Frontend build check
cd web && npm run build

# Browser E2E (requires a running backend)
source .venv/bin/activate
python -m playwright install chromium
QUARRY_RUN_E2E=1 pytest -q -m e2e
```

---

## Documentation

Full documentation lives in [`docs/`](docs/):

| File | Contents |
|---|---|
| [`docs/README.md`](docs/README.md) | Detailed setup, configuration, and CLI reference |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture, modules, pipeline, and API design |
| [`docs/MODELS_PROMPTS.md`](docs/MODELS_PROMPTS.md) | Model inventory, runtime profiles, and prompt reference |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Visual design system and UI layout rules |
