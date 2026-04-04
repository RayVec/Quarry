from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import tomllib


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def getenv_bool_alias(name: str, default: bool, *, alias: str | None = None) -> bool:
    value = os.getenv(name)
    if value is not None:
        return value.lower() in {"1", "true", "yes", "on"}
    if alias:
        legacy_value = os.getenv(alias)
        if legacy_value is not None:
            return legacy_value.lower() in {"1", "true", "yes", "on"}
    return default


def is_apple_silicon_host() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def default_runtime_profile() -> str:
    return "apple_silicon" if is_apple_silicon_host() else "gpu"


def default_config_path() -> Path:
    return Path("config.toml")


def validate_runtime_mode(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    allowed = {"local", "hybrid", "hosted"}
    if normalized in allowed:
        return normalized
    raise ValueError(f"Unsupported QUARRY runtime mode: {value!r}. Use one of: local, hybrid, hosted.")


def load_file_config(config_path: str | Path | None = None) -> dict[str, object]:
    resolved = Path(config_path) if config_path else Path(os.getenv("QUARRY_CONFIG_PATH", default_config_path()))
    if not resolved.exists():
        return {}
    payload = tomllib.loads(resolved.read_text())
    if not isinstance(payload, dict):
        return {}

    flattened: dict[str, object] = {}
    section_mappings = {
        "app": {
            "name": "app_name",
            "cors_origin": "cors_origin",
        },
        "paths": {
            "corpus_dir": "corpus_dir",
            "artifacts_dir": "artifacts_dir",
            "model_cache_dir": "model_cache_dir",
        },
        "runtime": {
            "mode": "runtime_mode",
            "profile": "runtime_profile",
            "use_local_models": "use_local_models",
            "local_model_device": "local_model_device",
            "trace_logs": "trace_logs",
        },
        "retrieval": {
            "sparse_top_k": "sparse_top_k",
            "dense_top_k": "dense_top_k",
            "rerank_top_k": "rerank_top_k",
            "max_facets": "max_facets",
            "retrieval_rrf_k": "retrieval_rrf_k",
            "scoped_retrieval_top_k": "scoped_retrieval_top_k",
            "refinement_token_budget": "refinement_token_budget",
        },
        "thresholds": {
            "support_threshold": "support_threshold",
            "partial_threshold": "partial_threshold",
            "ambiguity_gap_threshold": "ambiguity_gap_threshold",
        },
        "hosted": {
            "llm_base_url": "llm_base_url",
            "llm_api_key": "llm_api_key",
            "llm_model": "llm_model",
            "use_live_generation": "use_live_generation",
            "use_live_decomposition": "use_live_decomposition",
            "use_live_metadata_enrichment": "use_live_metadata_enrichment",
            "embedding_base_url": "embedding_base_url",
            "embedding_api_key": "embedding_api_key",
            "embedding_model": "embedding_model",
            "embedding_dimensions": "embedding_dimensions",
            "use_live_embeddings": "use_live_embeddings",
        },
        "mlx": {
            "text_model": "mlx_text_model_name",
            "vision_model": "mlx_vision_model_name",
            "max_new_tokens": "mlx_max_new_tokens",
            "page_image_dim": "mlx_page_image_dim",
            "max_pdf_pages_per_batch": "mlx_max_pdf_pages_per_batch",
        },
        "local_models": {
            "text_model": "local_text_model_name",
            "text_max_new_tokens": "local_text_max_new_tokens",
            "text_dtype": "local_text_dtype",
            "embedding_model": "local_embedding_model",
            "reranker_model": "local_reranker_model",
            "nli_model": "nli_model_name",
        },
        "parser": {
            "primary": "parser_primary",
            "fallback": "parser_fallback",
            "olmocr_model": "olmocr_model_name",
            "image_dim": "parser_target_longest_image_dim",
            "mineru_backend": "mineru_backend",
            "mineru_language": "mineru_language",
        },
    }

    for key, value in payload.items():
        if not isinstance(value, dict):
            flattened[key] = value
            continue
        mapping = section_mappings.get(key, {})
        for section_key, section_value in value.items():
            flattened[mapping.get(section_key, section_key)] = section_value
    return flattened


def _config_value(config: dict[str, object], key: str, default: object) -> object:
    return config.get(key, default)


def _config_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_local_model_status(artifacts_dir: Path) -> dict[str, object]:
    status_path = artifacts_dir / "local_model_status.json"
    if not status_path.exists():
        return {}
    try:
        import json

        payload = json.loads(status_path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def is_local_component_ready(settings: "Settings", component: str) -> bool:
    payload = load_local_model_status(settings.artifacts_dir)
    if payload.get("runtime_profile") and payload.get("runtime_profile") != settings.runtime_profile:
        return False
    return str(payload.get(component, "")).startswith("ready:")


@dataclass(slots=True)
class Settings:
    app_name: str = "QUARRY"
    corpus_dir: Path = Path("data/corpus")
    artifacts_dir: Path = Path("data/artifacts")
    model_cache_dir: Path | None = None
    sparse_top_k: int = 30
    dense_top_k: int = 30
    rerank_top_k: int = 20
    max_facets: int = 4
    retrieval_rrf_k: int = 60
    scoped_retrieval_top_k: int = 3
    refinement_token_budget: int = 8000
    support_threshold: float = 0.7
    partial_threshold: float = 0.4
    ambiguity_gap_threshold: float = 0.05
    cors_origin: str = "http://127.0.0.1:5173"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    runtime_mode: str = "hybrid"
    runtime_profile: str = default_runtime_profile()
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "hash-embedding-v1"
    embedding_dimensions: int = 192
    use_local_models: bool = True
    local_model_device: str = "auto"
    local_text_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    local_text_max_new_tokens: int = 768
    local_text_dtype: str = "auto"
    mlx_text_model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    mlx_vision_model_name: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    mlx_max_new_tokens: int = 768
    mlx_page_image_dim: int = 1024
    mlx_max_pdf_pages_per_batch: int = 1
    local_embedding_model: str = "intfloat/e5-large-v2"
    local_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_live_generation: bool = False
    use_live_decomposition: bool = False
    use_live_metadata_enrichment: bool = False
    use_live_embeddings: bool = False
    nli_model_name: str = "khalidalt/DeBERTa-v3-large-mnli"
    parser_primary: str = "olmocr_transformers"
    parser_fallback: str = "pymupdf_text"
    olmocr_model_name: str = "allenai/olmOCR-7B-0725-FP8"
    parser_target_longest_image_dim: int = 1024
    mineru_backend: str = "pipeline"
    mineru_language: str = "en"
    trace_logs: bool = True

    def __post_init__(self) -> None:
        self.runtime_mode = validate_runtime_mode(self.runtime_mode)

    @classmethod
    def from_env(cls, *, config_path: str | Path | None = None) -> "Settings":
        file_config = load_file_config(config_path)
        return cls(
            app_name=os.getenv("QUARRY_APP_NAME", str(_config_value(file_config, "app_name", "QUARRY"))),
            corpus_dir=Path(os.getenv("QUARRY_CORPUS_DIR", str(_config_value(file_config, "corpus_dir", "data/corpus")))),
            artifacts_dir=Path(os.getenv("QUARRY_ARTIFACTS_DIR", str(_config_value(file_config, "artifacts_dir", "data/artifacts")))),
            model_cache_dir=Path(os.getenv("QUARRY_MODEL_CACHE_DIR")) if os.getenv("QUARRY_MODEL_CACHE_DIR") else (Path(str(_config_value(file_config, "model_cache_dir", ""))) if _config_value(file_config, "model_cache_dir", "") else None),
            sparse_top_k=int(os.getenv("QUARRY_SPARSE_TOP_K", str(_config_value(file_config, "sparse_top_k", 30)))),
            dense_top_k=int(os.getenv("QUARRY_DENSE_TOP_K", str(_config_value(file_config, "dense_top_k", 30)))),
            rerank_top_k=int(os.getenv("QUARRY_RERANK_TOP_K", str(_config_value(file_config, "rerank_top_k", 20)))),
            max_facets=int(os.getenv("QUARRY_MAX_FACETS", str(_config_value(file_config, "max_facets", 4)))),
            retrieval_rrf_k=int(os.getenv("QUARRY_RRF_K", str(_config_value(file_config, "retrieval_rrf_k", 60)))),
            scoped_retrieval_top_k=int(os.getenv("QUARRY_SCOPED_TOP_K", str(_config_value(file_config, "scoped_retrieval_top_k", 3)))),
            refinement_token_budget=int(os.getenv("QUARRY_REFINEMENT_TOKEN_BUDGET", str(_config_value(file_config, "refinement_token_budget", 8000)))),
            support_threshold=float(os.getenv("QUARRY_SUPPORT_THRESHOLD", str(_config_value(file_config, "support_threshold", 0.7)))),
            partial_threshold=float(os.getenv("QUARRY_PARTIAL_THRESHOLD", str(_config_value(file_config, "partial_threshold", 0.4)))),
            ambiguity_gap_threshold=float(os.getenv("QUARRY_AMBIGUITY_GAP_THRESHOLD", str(_config_value(file_config, "ambiguity_gap_threshold", 0.05)))),
            cors_origin=os.getenv("QUARRY_CORS_ORIGIN", str(_config_value(file_config, "cors_origin", "http://127.0.0.1:5173"))),
            llm_base_url=os.getenv("QUARRY_LLM_BASE_URL", str(_config_value(file_config, "llm_base_url", ""))) or None,
            llm_api_key=os.getenv("QUARRY_LLM_API_KEY", str(_config_value(file_config, "llm_api_key", ""))) or None,
            llm_model=os.getenv("QUARRY_LLM_MODEL", str(_config_value(file_config, "llm_model", "gpt-4o-mini"))),
            runtime_mode=validate_runtime_mode(os.getenv("QUARRY_RUNTIME_MODE", str(_config_value(file_config, "runtime_mode", "hybrid")))),
            runtime_profile=os.getenv("QUARRY_RUNTIME_PROFILE", str(_config_value(file_config, "runtime_profile", default_runtime_profile()))),
            embedding_base_url=os.getenv("QUARRY_EMBEDDING_BASE_URL", str(_config_value(file_config, "embedding_base_url", ""))) or None,
            embedding_api_key=os.getenv("QUARRY_EMBEDDING_API_KEY", str(_config_value(file_config, "embedding_api_key", ""))) or None,
            embedding_model=os.getenv("QUARRY_EMBEDDING_MODEL", str(_config_value(file_config, "embedding_model", "hash-embedding-v1"))),
            embedding_dimensions=int(os.getenv("QUARRY_EMBEDDING_DIMENSIONS", str(_config_value(file_config, "embedding_dimensions", 192)))),
            use_local_models=getenv_bool("QUARRY_USE_LOCAL_MODELS", _config_bool(file_config, "use_local_models", True)),
            local_model_device=os.getenv("QUARRY_LOCAL_MODEL_DEVICE", str(_config_value(file_config, "local_model_device", "auto"))),
            local_text_model_name=os.getenv("QUARRY_LOCAL_TEXT_MODEL", str(_config_value(file_config, "local_text_model_name", "Qwen/Qwen2.5-7B-Instruct"))),
            local_text_max_new_tokens=int(os.getenv("QUARRY_LOCAL_TEXT_MAX_NEW_TOKENS", str(_config_value(file_config, "local_text_max_new_tokens", 768)))),
            local_text_dtype=os.getenv("QUARRY_LOCAL_TEXT_DTYPE", str(_config_value(file_config, "local_text_dtype", "auto"))),
            mlx_text_model_name=os.getenv("QUARRY_MLX_TEXT_MODEL", str(_config_value(file_config, "mlx_text_model_name", "mlx-community/Qwen3.5-4B-MLX-4bit"))),
            mlx_vision_model_name=os.getenv("QUARRY_MLX_VISION_MODEL", str(_config_value(file_config, "mlx_vision_model_name", "mlx-community/Qwen3-VL-4B-Instruct-4bit"))),
            mlx_max_new_tokens=int(os.getenv("QUARRY_MLX_MAX_NEW_TOKENS", str(_config_value(file_config, "mlx_max_new_tokens", 768)))),
            mlx_page_image_dim=int(os.getenv("QUARRY_MLX_PAGE_IMAGE_DIM", os.getenv("QUARRY_PARSER_IMAGE_DIM", str(_config_value(file_config, "mlx_page_image_dim", 1024))))),
            mlx_max_pdf_pages_per_batch=int(os.getenv("QUARRY_MLX_MAX_PDF_PAGES_PER_BATCH", str(_config_value(file_config, "mlx_max_pdf_pages_per_batch", 1)))),
            local_embedding_model=os.getenv("QUARRY_LOCAL_EMBEDDING_MODEL", str(_config_value(file_config, "local_embedding_model", "intfloat/e5-large-v2"))),
            local_reranker_model=os.getenv("QUARRY_LOCAL_RERANKER_MODEL", str(_config_value(file_config, "local_reranker_model", "BAAI/bge-reranker-v2-m3"))),
            use_live_generation=getenv_bool_alias("QUARRY_USE_LIVE_GENERATION", _config_bool(file_config, "use_live_generation", False), alias="QUARRY_USE_LIVE_LLM"),
            use_live_decomposition=getenv_bool("QUARRY_USE_LIVE_DECOMPOSITION", _config_bool(file_config, "use_live_decomposition", False)),
            use_live_metadata_enrichment=getenv_bool("QUARRY_USE_LIVE_METADATA_ENRICHMENT", _config_bool(file_config, "use_live_metadata_enrichment", False)),
            use_live_embeddings=getenv_bool("QUARRY_USE_LIVE_EMBEDDINGS", _config_bool(file_config, "use_live_embeddings", False)),
            nli_model_name=os.getenv("QUARRY_NLI_MODEL", str(_config_value(file_config, "nli_model_name", "khalidalt/DeBERTa-v3-large-mnli"))),
            parser_primary=os.getenv("QUARRY_PRIMARY_PARSER", str(_config_value(file_config, "parser_primary", "olmocr_transformers"))),
            parser_fallback=os.getenv("QUARRY_FALLBACK_PARSER", str(_config_value(file_config, "parser_fallback", "pymupdf_text"))),
            olmocr_model_name=os.getenv("QUARRY_OLMOCR_MODEL", str(_config_value(file_config, "olmocr_model_name", "allenai/olmOCR-7B-0725-FP8"))),
            parser_target_longest_image_dim=int(os.getenv("QUARRY_PARSER_IMAGE_DIM", str(_config_value(file_config, "parser_target_longest_image_dim", 1024)))),
            mineru_backend=os.getenv("QUARRY_MINERU_BACKEND", str(_config_value(file_config, "mineru_backend", "pipeline"))),
            mineru_language=os.getenv("QUARRY_MINERU_LANGUAGE", str(_config_value(file_config, "mineru_language", "en"))),
            trace_logs=getenv_bool("QUARRY_TRACE_LOGS", _config_bool(file_config, "trace_logs", True)),
        )

    @property
    def uses_mlx_profile(self) -> bool:
        return self.runtime_profile == "apple_silicon"

    @property
    def parser_provider(self) -> str:
        if self.uses_mlx_profile:
            return self.mlx_vision_model_name
        return self.olmocr_model_name

    @property
    def has_live_llm_credentials(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)

    @property
    def active_model_ids(self) -> list[str]:
        if not self.use_local_models:
            return []
        if self.uses_mlx_profile:
            return [
                self.mlx_text_model_name,
                self.mlx_vision_model_name,
                self.local_embedding_model,
                self.local_reranker_model,
                self.nli_model_name,
            ]
        return [
            self.local_text_model_name,
            self.olmocr_model_name,
            self.local_embedding_model,
            self.local_reranker_model,
            self.nli_model_name,
        ]
