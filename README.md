# QUARRY

A local-first retrieval and review system for technical grey literature. Ingest your PDF corpus, ask questions, and get answers where every sentence is grounded in a verbatim quote from the source — verified automatically.

Runs on Apple Silicon (MLX) or any CUDA GPU. No database, no cloud dependency. Optionally connect a hosted LLM for better generation quality while keeping retrieval, parsing, and verification fully local.

---

## Requirements

- Python 3.13+
- Node.js 18+
- Mac with Apple Silicon (M1 or later) — or a Linux/Windows machine with a CUDA GPU

---

## 1. Install

```bash
git clone https://github.com/RayVec/Quarry.git
cd Quarry
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,local]"
```

```bash
cd web && npm install && cd ..
```

---

## 2. Configure

```bash
cp config.example.toml config.toml
```

Open `config.toml`. Most users only need to think about these two things:

**1. Choose a mode** — if you're not sure, keep the default:

```toml
[runtime]
mode = "hybrid"
```

- `hybrid`: recommended; local retrieval/parsing/verification plus hosted answer generation
- `local`: no API key; everything runs locally
- `hosted`: use a hosted LLM for generation while keeping the rest of the stack configured through the same app

**2. If you use `hybrid` or `hosted`, add your API key:**

```toml
[hosted]
provider    = "openai_compatible" # or "gemini"
llm_api_key = "YOUR_API_KEY_HERE"
llm_model   = "openai/gpt-4o-mini"
```

Get a free key at [openrouter.ai](https://openrouter.ai) — it gives access to many models including free-tier ones.
For Google AI Studio, set `provider = "gemini"` and use a Gemini model ID (for example `gemini-3-flash-preview`).

`profile` is auto-detected from your hardware, so most users do not need to set it.

---

## 3. Add your documents

Drop your PDFs into `data/sources/`. Any `.md` or `.txt` files work too.

---

## 4. Run

**Terminal 1 — backend:**

```bash
source .venv/bin/activate
python start_backend.py
```

When asked to rebuild the corpus, type `y` on the first run (or any time you add new documents).

**Terminal 2 — frontend:**

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

Open **http://127.0.0.1:5173** and start asking questions.

---

## Advanced configuration

### Run fully local (no API key)

```toml
[runtime]
mode = "local"
```

Then warm the local models before first run (downloads ~8 GB on Apple Silicon):

```bash
source .venv/bin/activate
quarry warm-local-models
```

### Override the auto-detected hardware profile

```toml
[runtime]
profile = "apple_silicon"   # Mac M-series — MLX Qwen 4B models
# profile = "gpu"           # Linux / Windows — HuggingFace Qwen 7B models (~23 GB)
```

### Use a different API provider

Use OpenAI-compatible providers (default):

```toml
[hosted]
provider     = "openai_compatible"
llm_base_url = "https://api.openai.com/v1"      # OpenAI directly
# llm_base_url = "https://openrouter.ai/api/v1" # OpenRouter (default)
llm_api_key  = "YOUR_API_KEY_HERE"
llm_model    = "gpt-4o"
```

Use Google AI Studio (Gemini) for hosted generation:

```toml
[hosted]
provider            = "gemini"
llm_model           = "gemini-3-flash-preview"
use_live_generation = true
```

Set either `hosted.llm_api_key` in `config.toml` or environment variable `GEMINI_API_KEY`.

### Move the model cache off your main drive

Models can reach 10+ GB. Redirect the cache to an external drive:

```toml
[paths]
model_cache_dir = "/Volumes/external/quarry-models"
```

### Offload only generation, keep everything else local

The default `hybrid` mode already does this. You can fine-tune which tasks go hosted:

```toml
[hosted]
use_live_generation          = true   # answer generation → hosted
use_live_decomposition       = false  # query decomposition → local
use_live_metadata_enrichment = false  # chunk enrichment → local
```

---

|                                                    |                                                          |
| -------------------------------------------------- | -------------------------------------------------------- |
| [`docs/README.md`](docs/README.md)                 | Full config reference, CLI commands, and runtime options |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)     | How the pipeline works                                   |
| [`docs/MODELS_PROMPTS.md`](docs/MODELS_PROMPTS.md) | Model details and prompt reference                       |
