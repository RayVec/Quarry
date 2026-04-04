from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys())
_OMIT_CONSOLE_KEYS = {
    "prompt",
    "response",
    "raw_response",
    "request_body",
    "generated_response",
    "citation_index",
    "parsed_sentences",
    "existing_response",
    "reviewer_feedback",
    "diagnostics",
    "meta",
    "top_chunk_ids",
    "filters",
    "active_model_ids",
}
_FILE_HANDLER: logging.Handler | None = None
_FILE_PATH: Path | None = None
_FILE_HANDLERS: dict[str, logging.Handler] = {}
_FILE_PATHS: dict[str, Path] = {}
_ACTIVE_LOG_CATEGORY = "runtime"


class ConsoleVisibilityFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console_visible", True))


class QuarryFormatter(logging.Formatter):
    def __init__(self, *, include_verbose: bool) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s trace=%(trace_id)s %(message)s")
        self.include_verbose = include_verbose

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = self._extract_context(record)
        if not context:
            return base
        return f"{base} {json.dumps(context, ensure_ascii=False, default=str)}"

    def _extract_context(self, record: logging.LogRecord) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_KEYS or key in {"message", "asctime", "trace_id"}:
                continue
            if not self.include_verbose and key in _OMIT_CONSOLE_KEYS:
                continue
            payload[key] = value
        return payload


def configure_logging(log_dir: Path | str, *, enable_file_logs: bool = True, category: str = "runtime") -> Path | None:
    global _FILE_HANDLER, _FILE_PATH, _ACTIVE_LOG_CATEGORY

    _ACTIVE_LOG_CATEGORY = category
    log_dir = resolve_log_dir(log_dir, category=category)
    log_dir.mkdir(parents=True, exist_ok=True)

    target_path: Path | None = None
    if enable_file_logs:
        if category not in _FILE_PATHS:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _FILE_PATHS[category] = log_dir / f"quarry-{timestamp}.log"
        target_path = _FILE_PATHS[category]

    _FILE_PATH = target_path

    for name, candidate in logging.Logger.manager.loggerDict.items():
        if isinstance(candidate, logging.Logger) and name.startswith("quarry"):
            _configure_logger(candidate, target_path)
    return target_path


def resolve_log_dir(log_dir: Path | str, *, category: str = "runtime") -> Path:
    explicit_dir = os.getenv("QUARRY_LOG_DIR")
    if explicit_dir:
        return Path(explicit_dir)

    base_dir = Path(log_dir)
    if running_under_pytest():
        return Path(os.getenv("QUARRY_TEST_LOG_DIR", str(base_dir / "tests")))
    category_env = os.getenv(f"QUARRY_{category.upper()}_LOG_DIR")
    if category_env:
        return Path(category_env)
    return base_dir / category


def current_log_file(category: str | None = None) -> Path | None:
    if category is None:
        return _FILE_PATH
    return _FILE_PATHS.get(category)


def running_under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def set_trace(trace_id: str) -> None:
    TRACE_ID.set(trace_id)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    _configure_logger(logger, _FILE_PATH)
    return logger


def _configure_logger(logger: logging.Logger, file_path: Path | None) -> None:
    if not any(isinstance(handler, logging.StreamHandler) and getattr(handler, "_quarry_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._quarry_console = True  # type: ignore[attr-defined]
        handler.setFormatter(QuarryFormatter(include_verbose=False))
        handler.addFilter(ConsoleVisibilityFilter())
        logger.addHandler(handler)

    global _FILE_HANDLER
    for handler in list(logger.handlers):
        if getattr(handler, "_quarry_file_path", None) is not None:
            logger.removeHandler(handler)

    if file_path is not None:
        handler_key = str(file_path)
        if handler_key not in _FILE_HANDLERS:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler._quarry_file_path = handler_key  # type: ignore[attr-defined]
            file_handler.setFormatter(QuarryFormatter(include_verbose=True))
            _FILE_HANDLERS[handler_key] = file_handler
        _FILE_HANDLER = _FILE_HANDLERS[handler_key]
        logger.addHandler(_FILE_HANDLER)
    else:
        _FILE_HANDLER = None

    logger.setLevel(logging.INFO)
    logger.propagate = False


class TraceAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("trace_id", TRACE_ID.get() or "-")
        return msg, kwargs


def logger_with_trace(name: str) -> TraceAdapter:
    return TraceAdapter(get_logger(name), {})


def start_trace() -> str:
    trace_id = str(uuid4())
    TRACE_ID.set(trace_id)
    return trace_id


def timed() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
