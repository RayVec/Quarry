from __future__ import annotations

import os
from pathlib import Path

from quarry.config import Settings


def resolve_model_cache_dir(settings: Settings) -> Path:
    base = settings.model_cache_dir or (settings.artifacts_dir.parent / "model-cache")
    return base if base.is_absolute() else (Path.cwd() / base).resolve()


def configure_model_cache(settings: Settings) -> dict[str, str]:
    cache_root = resolve_model_cache_dir(settings)
    hf_home = cache_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    torch_home = cache_root / "torch"
    modelscope_cache = cache_root / "modelscope"

    for path in (cache_root, hf_home, hf_hub_cache, transformers_cache, torch_home, modelscope_cache):
        path.mkdir(parents=True, exist_ok=True)

    values = {
        "QUARRY_MODEL_CACHE_DIR": str(cache_root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "TORCH_HOME": str(torch_home),
        "MODELSCOPE_CACHE": str(modelscope_cache),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def resolve_hf_hub_cache_dir() -> Path:
    explicit_hub_cache = os.getenv("HF_HUB_CACHE")
    if explicit_hub_cache:
        return Path(explicit_hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_hf_snapshot_path(repo_id: str) -> Path | None:
    candidate = Path(repo_id).expanduser()
    if candidate.exists():
        return candidate.resolve()

    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo_id, local_files_only=True)).resolve()
    except Exception:
        pass

    repo_dir = resolve_hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text().strip()
        if snapshot_id:
            snapshot_path = snapshots_dir / snapshot_id
            if snapshot_path.exists():
                return snapshot_path.resolve()

    snapshot_candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshot_candidates:
        return None

    latest_snapshot = max(snapshot_candidates, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.resolve()
