from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gc
import importlib
import inspect
import json
from pathlib import Path
import threading
from typing import Sequence

from quarry.adapters.in_memory import HeuristicDecompositionClient, HeuristicMetadataEnricher
from quarry.adapters.interfaces import DecompositionClient, GenerationClient, MetadataEnricher
from quarry.domain.models import ChunkObject, GenerationRequest
from quarry.logging_utils import logger_with_trace
from quarry.model_cache import resolve_cached_hf_snapshot_path
from quarry.prompts import (
    SHARED_SYSTEM_PROMPT,
    decomposition_classification_prompt,
    decomposition_prompt,
    generation_prompt,
    metadata_enrichment_prompt,
    parse_json_response,
    with_shared_system_prompt,
)


logger = logger_with_trace(__name__)


class MlxRuntimeUnavailableError(RuntimeError):
    """Raised when mlx-vlm is not available in the current environment."""


def _json_repair_prompt(prompt: str, raw_response: str) -> str:
    return (
        f"{prompt}\n\n"
        "The previous output was not valid JSON.\n"
        "Return valid JSON only, without markdown fences or commentary.\n"
        "The first character of your response must be '{' and there must be no prefix text.\n"
        f"Previous output:\n{raw_response.strip()}"
    )


@dataclass(slots=True)
class _LoadedMlxModel:
    model_id: str
    model: object
    processor: object
    config: object | None = None


class AppleMLXModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: _LoadedMlxModel | None = None

    def _import_generate_dependencies(self) -> tuple[object, object, object | None]:
        try:
            mlx_vlm = importlib.import_module("mlx_vlm")
        except Exception as exc:  # pragma: no cover - depends on environment
            raise MlxRuntimeUnavailableError("mlx-vlm is not installed.") from exc

        load = getattr(mlx_vlm, "load", None)
        generate = getattr(mlx_vlm, "generate", None)
        if load is None or generate is None:
            raise MlxRuntimeUnavailableError("mlx-vlm load/generate APIs are unavailable.")

        prompt_utils = None
        for module_name in ("mlx_vlm.prompt_utils", "mlx_vlm.utils"):
            try:
                prompt_utils = importlib.import_module(module_name)
                break
            except Exception:
                continue
        return load, generate, prompt_utils

    def _clear_mlx_cache(self) -> None:
        try:  # pragma: no cover - depends on environment
            mx = importlib.import_module("mlx.core")
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
                return

            metal = getattr(mx, "metal", None)
            if metal is not None and hasattr(metal, "clear_cache"):
                metal.clear_cache()
        except Exception:
            pass

    def _unload_model(self) -> None:
        self._loaded = None
        gc.collect()
        self._clear_mlx_cache()

    def _load_model_locked(self, model_id: str) -> _LoadedMlxModel:
        if self._loaded is not None and self._loaded.model_id == model_id:
            return self._loaded
        self._unload_model()
        load, _, _ = self._import_generate_dependencies()
        cached_model_path = resolve_cached_hf_snapshot_path(model_id)
        load_target = str(cached_model_path) if cached_model_path is not None else model_id
        try:
            model, processor = load(load_target)
        except Exception:
            if cached_model_path is None or load_target == model_id:
                raise
            logger.info(
                "mlx cached snapshot load failed; retrying via repo id",
                extra={
                    "model": model_id,
                    "cached_path": str(cached_model_path),
                    "console_visible": False,
                },
            )
            model, processor = load(model_id)
        else:
            if cached_model_path is not None and load_target != model_id:
                logger.info(
                    "mlx model loaded from local snapshot",
                    extra={
                        "model": model_id,
                        "cached_path": str(cached_model_path),
                        "console_visible": False,
                    },
                )
        config = getattr(model, "config", None) or getattr(processor, "config", None)
        self._loaded = _LoadedMlxModel(model_id=model_id, model=model, processor=processor, config=config)
        return self._loaded

    def _apply_chat_template(
        self,
        *,
        loaded: _LoadedMlxModel,
        prompt: str,
        num_images: int,
        enable_thinking: bool | None = None,
    ) -> str:
        _, _, prompt_utils = self._import_generate_dependencies()
        if prompt_utils is None:
            return prompt
        apply_chat_template = getattr(prompt_utils, "apply_chat_template", None)
        if apply_chat_template is None:
            return prompt

        attempts: list[object] = []
        if enable_thinking is not None:
            attempts.extend(
                [
                    lambda: apply_chat_template(
                        loaded.processor,
                        loaded.config,
                        prompt,
                        num_images=num_images,
                        enable_thinking=enable_thinking,
                    ),
                    lambda: apply_chat_template(
                        loaded.processor,
                        loaded.config,
                        prompt,
                        enable_thinking=enable_thinking,
                    ),
                    lambda: apply_chat_template(
                        loaded.processor,
                        prompt,
                        num_images=num_images,
                        enable_thinking=enable_thinking,
                    ),
                    lambda: apply_chat_template(
                        loaded.processor,
                        prompt,
                        enable_thinking=enable_thinking,
                    ),
                ]
            )
        attempts.extend(
            [
                lambda: apply_chat_template(loaded.processor, loaded.config, prompt, num_images=num_images),
                lambda: apply_chat_template(loaded.processor, loaded.config, prompt),
                lambda: apply_chat_template(loaded.processor, prompt, num_images=num_images),
                lambda: apply_chat_template(loaded.processor, prompt),
            ]
        )
        for attempt in attempts:
            try:
                rendered = attempt()
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                continue
        return prompt

    def _extract_text(self, result: object) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("text", "response", "output"):
                value = result.get(key)
                if isinstance(value, str):
                    return value.strip()
        value = getattr(result, "text", None)
        if isinstance(value, str):
            return value.strip()
        return str(result).strip()

    def _generate_sync(
        self,
        *,
        model_id: str,
        prompt: str,
        image_paths: Sequence[str] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        enable_thinking: bool | None = None,
    ) -> str:
        with self._lock:
            loaded = self._load_model_locked(model_id)
            _, generate, _ = self._import_generate_dependencies()
            rendered_prompt = self._apply_chat_template(
                loaded=loaded,
                prompt=prompt,
                num_images=len(image_paths or []),
                enable_thinking=enable_thinking,
            )
            images: str | list[str] | None
            if image_paths:
                images = list(image_paths)
                if len(image_paths) == 1:
                    images = image_paths[0]
            else:
                images = None

            kwargs: dict[str, object] = {"max_tokens": max_tokens, "verbose": False}
            signature = inspect.signature(generate)
            if "temperature" in signature.parameters:
                kwargs["temperature"] = temperature
            elif "temp" in signature.parameters:
                kwargs["temp"] = temperature

            if images is None:
                result = generate(loaded.model, loaded.processor, prompt=rendered_prompt, **kwargs)
            else:
                if "image" in signature.parameters:
                    result = generate(loaded.model, loaded.processor, prompt=rendered_prompt, image=images, **kwargs)
                else:
                    result = generate(loaded.model, loaded.processor, rendered_prompt, images, **kwargs)
            return self._extract_text(result)

    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        enable_thinking: bool | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            model_id=model_id,
            prompt=prompt,
            image_paths=None,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

    async def generate_with_images(
        self,
        model_id: str,
        prompt: str,
        image_paths: Sequence[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        enable_thinking: bool | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            model_id=model_id,
            prompt=prompt,
            image_paths=list(image_paths),
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

    def is_ready(self, model_id: str) -> bool:
        try:
            with self._lock:
                self._load_model_locked(model_id)
            return True
        except Exception as exc:
            logger.warning("mlx model unavailable", extra={"error": str(exc), "model": model_id})
            return False

    def unload(self) -> None:
        with self._lock:
            self._unload_model()


class MLXTextCompletionBackend:
    def __init__(
        self,
        model_name: str,
        *,
        model_manager: AppleMLXModelManager | None = None,
        default_max_new_tokens: int = 768,
    ) -> None:
        self.model_name = model_name
        self.model_manager = model_manager or AppleMLXModelManager()
        self.default_max_new_tokens = default_max_new_tokens

    def is_ready(self) -> bool:
        return self.model_manager.is_ready(self.model_name)

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
        operation: str = "completion",
        enable_thinking: bool | None = None,
    ) -> str:
        rendered_prompt = with_shared_system_prompt(prompt)
        logger.info(
            "mlx text completion started",
            extra={
                "operation": operation,
                "model": self.model_name,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
                "prompt": rendered_prompt,
                "console_visible": False,
            },
        )
        start = asyncio.get_running_loop().time()
        raw = await self.model_manager.generate_text(
            self.model_name,
            rendered_prompt,
            max_tokens=max_new_tokens or self.default_max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        latency_ms = round((asyncio.get_running_loop().time() - start) * 1000, 3)
        logger.info(
            "mlx text completion completed",
            extra={
                "operation": operation,
                "model": self.model_name,
                "response": raw,
                "latency_ms": latency_ms,
                "console_visible": False,
            },
        )
        return raw


class MLXStructuredDecompositionClient(DecompositionClient):
    def __init__(self, backend: MLXTextCompletionBackend, fallback: DecompositionClient | None = None) -> None:
        self.backend = backend
        self.fallback = fallback

    async def _complete_json(self, prompt: str, *, max_new_tokens: int) -> dict[str, object]:
        raw = await self.backend.complete(prompt, max_new_tokens=max_new_tokens, operation="json_task")
        try:
            return parse_json_response(raw)
        except Exception:
            logger.warning(
                "mlx json parse failed; attempting repair",
                extra={"raw_response": raw, "console_visible": False},
            )
            repaired = await self.backend.complete(_json_repair_prompt(prompt, raw), max_new_tokens=max_new_tokens, operation="json_repair")
            return parse_json_response(repaired)

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        try:
            prompt = decomposition_prompt(query, max_facets)
            raw = await self.backend.complete(
                prompt,
                max_new_tokens=256,
                operation="query_decomposition",
                enable_thinking=False,
            )
            try:
                payload = parse_json_response(raw)
            except Exception:
                logger.warning(
                    "mlx decomposition json parse failed; attempting repair",
                    extra={"raw_response": raw, "console_visible": False},
                )
                repaired = await self.backend.complete(
                    _json_repair_prompt(prompt, raw),
                    max_new_tokens=256,
                    operation="query_decomposition_repair",
                    enable_thinking=False,
                )
                payload = parse_json_response(repaired)
            facets = [str(item) for item in payload.get("facets", []) if str(item).strip()]
            return facets[:max_facets] or [query]
        except Exception:
            if self.fallback is None:
                raise
            logger.warning("mlx decomposition fell back to heuristic")
            return await self.fallback.decompose_query(query, max_facets)


class MLXStructuredMetadataEnricher(MetadataEnricher):
    def __init__(self, backend: MLXTextCompletionBackend, fallback: MetadataEnricher | None = None) -> None:
        self.backend = backend
        self.fallback = fallback

    async def enrich(self, chunk: ChunkObject) -> ChunkObject:
        prompt = metadata_enrichment_prompt(chunk)
        try:
            raw = await self.backend.complete(
                prompt,
                max_new_tokens=256,
                operation="metadata_enrichment",
                enable_thinking=False,
            )
            try:
                payload = parse_json_response(raw)
            except Exception:
                logger.warning(
                    "mlx metadata json parse failed; attempting repair",
                    extra={"raw_response": raw, "console_visible": False},
                )
                repaired = await self.backend.complete(
                    _json_repair_prompt(prompt, raw),
                    max_new_tokens=256,
                    operation="metadata_enrichment_repair",
                    enable_thinking=False,
                )
                payload = parse_json_response(repaired)
            return chunk.model_copy(
                update={
                    "metadata_summary": str(payload.get("summary", chunk.metadata_summary)),
                    "metadata_entities": [str(item) for item in payload.get("entities", [])],
                    "metadata_questions": [str(item) for item in payload.get("questions", [])],
                }
            )
        except Exception:
            if self.fallback is None:
                raise
            logger.warning("mlx metadata enrichment fell back to heuristic")
            return await self.fallback.enrich(chunk)


class MLXStructuredGenerationClient(GenerationClient):
    def __init__(self, backend: MLXTextCompletionBackend, fallback: GenerationClient | None = None) -> None:
        self.backend = backend
        self.fallback = fallback

    async def generate(self, request: GenerationRequest) -> str:
        try:
            max_tokens = 384 if request.mode == "regeneration" else self.backend.default_max_new_tokens
            return await self.backend.complete(generation_prompt(request), temperature=0.2, max_new_tokens=max_tokens, operation=f"generation:{request.mode}")
        except Exception:
            if self.fallback is None:
                raise
            logger.warning("mlx generation fell back to conservative no-ref implementation")
            return await self.fallback.generate(request)


def render_parser_prompt(page_number: int) -> str:
    return (
        f"{SHARED_SYSTEM_PROMPT}\n\n"
        "You are extracting a structured document page into JSON blocks.\n"
        "Return valid JSON only with the schema:\n"
        '{"blocks": [{"block_type": "heading" | "paragraph" | "table" | "table_title" | "figure_caption", "text": string, '
        '"section_depth": integer | null, "section_heading": string | null}]}\n'
        "Rules:\n"
        "- Preserve the page reading order.\n"
        "- Use heading blocks only for true section headings.\n"
        "- Use table_title only for table labels like 'Table 1 ...'.\n"
        "- Use figure_caption only for figure labels like 'Figure 2 ...'.\n"
        "- Ignore table of contents pages and table of contents entries.\n"
        "- Ignore page headers, page footers, page numbers, and repeated running headers.\n"
        "- Do not emit single-character headings or heading fragments.\n"
        "- If a token is visibly split across adjacent lines, merge it back together instead of separating it into different blocks.\n"
        "- Do not mix navigation text like contents listings with body paragraphs.\n"
        "- Do not invent coordinates.\n"
        f"Page number: {page_number}"
    )


async def parse_mlx_page_blocks(
    *,
    model_manager: AppleMLXModelManager,
    model_name: str,
    image_path: str,
    page_number: int,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    prompt = render_parser_prompt(page_number)
    raw = await model_manager.generate_with_images(
        model_name,
        prompt,
        [image_path],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    try:
        payload = parse_json_response(raw)
    except Exception:
        repaired = await model_manager.generate_with_images(
            model_name,
            _json_repair_prompt(prompt, raw),
            [image_path],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        payload = parse_json_response(repaired)
    blocks = payload.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("mlx parser did not return a blocks list")
    return [block for block in blocks if isinstance(block, dict)]
