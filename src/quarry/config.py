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


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    return Path(config_path) if config_path else Path(os.getenv("QUARRY_CONFIG_PATH", default_config_path()))


def validate_runtime_mode(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    allowed = {"local", "hybrid", "hosted"}
    if normalized in allowed:
        return normalized
    raise ValueError(f"Unsupported QUARRY runtime mode: {value!r}. Use one of: local, hybrid, hosted.")


def validate_llm_provider(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    allowed = {"openai_compatible", "gemini"}
    if normalized in allowed:
        return normalized
    raise ValueError(
        f"Unsupported QUARRY hosted LLM provider: {value!r}. "
        "Use one of: openai_compatible, gemini."
    )


def load_raw_file_config(config_path: str | Path | None = None) -> dict[str, object]:
    resolved = resolve_config_path(config_path)
    if not resolved.exists():
        return {}
    payload = tomllib.loads(resolved.read_text())
    if not isinstance(payload, dict):
        return {}
    return payload


def load_file_config(config_path: str | Path | None = None) -> dict[str, object]:
    payload = load_raw_file_config(config_path)

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
            "multihop_anchor_pool_size": "multihop_anchor_pool_size",
            "multihop_rerank_budget": "multihop_rerank_budget",
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
            "provider": "llm_provider",
            "llm_provider": "llm_provider",
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


def _toml_basic_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("TOML does not support NaN or infinite float values.")
        return repr(value)
    if isinstance(value, str):
        return _toml_basic_string(value)
    if isinstance(value, Path):
        return _toml_basic_string(str(value))
    if isinstance(value, list):
        return f"[{', '.join(_toml_value(item) for item in value)}]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _render_toml_table(lines: list[str], table_path: list[str], payload: dict[str, object]) -> None:
    scalar_items: list[tuple[str, object]] = []
    nested_items: list[tuple[str, dict[str, object]]] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if table_path:
        if lines:
            lines.append("")
        lines.append(f"[{'.'.join(table_path)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in nested_items:
        _render_toml_table(lines, [*table_path, key], value)


def render_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    _render_toml_table(lines, [], payload)
    return "\n".join(lines).rstrip() + "\n"


def write_raw_file_config(payload: dict[str, object], config_path: str | Path | None = None) -> Path:
    resolved = resolve_config_path(config_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_suffix(f"{resolved.suffix}.tmp")
    temp_path.write_text(render_toml(payload))
    temp_path.replace(resolved)
    return resolved


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
    multihop_anchor_pool_size: int = 40
    multihop_rerank_budget: int = 20
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
    llm_model: str = "stepfun/step-3.5-flash:free"
    llm_provider: str = "openai_compatible"
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
        self.llm_provider = validate_llm_provider(self.llm_provider)

    @classmethod
    def from_env(cls, *, config_path: str | Path | None = None) -> "Settings":
        file_config = load_file_config(config_path)
        llm_provider = validate_llm_provider(
            os.getenv(
                "QUARRY_LLM_PROVIDER",
                os.getenv(
                    "QUARRY_HOSTED_PROVIDER",
                    str(_config_value(file_config, "llm_provider", "openai_compatible")),
                ),
            )
        )
        configured_llm_api_key = os.getenv("QUARRY_LLM_API_KEY", str(_config_value(file_config, "llm_api_key", ""))).strip()
        gemini_env_api_key = os.getenv("QUARRY_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
        resolved_llm_api_key = configured_llm_api_key or (gemini_env_api_key if llm_provider == "gemini" else "")
        default_llm_model = (
            "gemini-3-flash-preview"
            if llm_provider == "gemini"
            else "stepfun/step-3.5-flash:free"
        )
        return cls(
            app_name=os.getenv("QUARRY_APP_NAME", str(_config_value(file_config, "app_name", "QUARRY"))),
            corpus_dir=Path(os.getenv("QUARRY_CORPUS_DIR", str(_config_value(file_config, "corpus_dir", "data/corpus")))),
            artifacts_dir=Path(os.getenv("QUARRY_ARTIFACTS_DIR", str(_config_value(file_config, "artifacts_dir", "data/artifacts")))),
            model_cache_dir=Path(os.getenv("QUARRY_MODEL_CACHE_DIR")) if os.getenv("QUARRY_MODEL_CACHE_DIR") else (Path(str(_config_value(file_config, "model_cache_dir", ""))) if _config_value(file_config, "model_cache_dir", "") else None),
            sparse_top_k=int(os.getenv("QUARRY_SPARSE_TOP_K", str(_config_value(file_config, "sparse_top_k", 30)))),
            dense_top_k=int(os.getenv("QUARRY_DENSE_TOP_K", str(_config_value(file_config, "dense_top_k", 30)))),
            rerank_top_k=int(os.getenv("QUARRY_RERANK_TOP_K", str(_config_value(file_config, "rerank_top_k", 20)))),
            multihop_anchor_pool_size=int(
                os.getenv(
                    "QUARRY_MULTIHOP_ANCHOR_POOL_SIZE",
                    str(_config_value(file_config, "multihop_anchor_pool_size", 40)),
                )
            ),
            multihop_rerank_budget=int(
                os.getenv(
                    "QUARRY_MULTIHOP_RERANK_BUDGET",
                    str(_config_value(file_config, "multihop_rerank_budget", 20)),
                )
            ),
            max_facets=int(os.getenv("QUARRY_MAX_FACETS", str(_config_value(file_config, "max_facets", 4)))),
            retrieval_rrf_k=int(os.getenv("QUARRY_RRF_K", str(_config_value(file_config, "retrieval_rrf_k", 60)))),
            scoped_retrieval_top_k=int(os.getenv("QUARRY_SCOPED_TOP_K", str(_config_value(file_config, "scoped_retrieval_top_k", 3)))),
            refinement_token_budget=int(os.getenv("QUARRY_REFINEMENT_TOKEN_BUDGET", str(_config_value(file_config, "refinement_token_budget", 8000)))),
            support_threshold=float(os.getenv("QUARRY_SUPPORT_THRESHOLD", str(_config_value(file_config, "support_threshold", 0.7)))),
            partial_threshold=float(os.getenv("QUARRY_PARTIAL_THRESHOLD", str(_config_value(file_config, "partial_threshold", 0.4)))),
            ambiguity_gap_threshold=float(os.getenv("QUARRY_AMBIGUITY_GAP_THRESHOLD", str(_config_value(file_config, "ambiguity_gap_threshold", 0.05)))),
            cors_origin=os.getenv("QUARRY_CORS_ORIGIN", str(_config_value(file_config, "cors_origin", "http://127.0.0.1:5173"))),
            llm_base_url=os.getenv("QUARRY_LLM_BASE_URL", str(_config_value(file_config, "llm_base_url", ""))) or None,
            llm_api_key=resolved_llm_api_key or None,
            llm_model=os.getenv("QUARRY_LLM_MODEL", str(_config_value(file_config, "llm_model", default_llm_model))),
            llm_provider=llm_provider,
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
    def has_live_generation_credentials(self) -> bool:
        if self.llm_provider == "gemini":
            return bool(self.llm_api_key)
        return self.has_live_llm_credentials

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
