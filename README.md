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

Open `config.toml`. There are only two things most users need to set:

**1. Your hardware profile** — already auto-detected, but confirm it matches:
```toml
[runtime]
profile = "apple_silicon"   # Mac M-series
# profile = "gpu"           # Linux / Windows with a CUDA GPU
```

**2. Your API key** — needed for answer generation:
```toml
[hosted]
llm_api_key = "YOUR_API_KEY_HERE"
llm_model   = "openai/gpt-4o-mini"
```

Get a free key at [openrouter.ai](https://openrouter.ai) — it gives access to many models including free-tier ones.

> Want to run fully local with no API key? Set `mode = "local"` in `[runtime]` and run `quarry warm-local-models` first. Models total ~8 GB for Apple Silicon.

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

## Docs

| | |
|---|---|
| [`docs/README.md`](docs/README.md) | Full config reference, CLI commands, and runtime options |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pipeline works |
| [`docs/MODELS_PROMPTS.md`](docs/MODELS_PROMPTS.md) | Model details and prompt reference |
