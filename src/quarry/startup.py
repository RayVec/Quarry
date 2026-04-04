from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import uvicorn

from quarry.api.app import create_app
from quarry.config import Settings, load_local_model_status
from quarry.ingest.pipeline import ingest_documents, warm_local_models
from quarry.logging_utils import configure_logging, current_log_file, elapsed_ms, logger_with_trace, start_trace, timed
from quarry.model_cache import configure_model_cache


SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".md", ".txt"}
REQUIRED_WARM_COMPONENTS = ("embedding", "reranker", "nli", "text", "decomposition", "generation", "parser")
logger = logger_with_trace(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_sources_dir(sources_dir: str | Path | None) -> Path:
    path = Path(sources_dir) if sources_dir else _project_root() / "data" / "sources"
    return path.resolve()


def _source_files(sources_dir: Path) -> list[Path]:
    if not sources_dir.exists():
        return []
    return sorted(path for path in sources_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES)


def _warmup_reasons(settings: Settings) -> tuple[bool, list[str]]:
    if not settings.use_local_models:
        return False, []

    payload = load_local_model_status(settings.artifacts_dir)
    reasons: list[str] = []
    if not payload:
        reasons.append("missing local_model_status.json")
        return True, reasons

    if payload.get("runtime_profile") not in (None, settings.runtime_profile):
        reasons.append("runtime profile mismatch")
    if payload.get("parser_provider") not in (None, settings.parser_provider):
        reasons.append("parser provider mismatch")
    for key in REQUIRED_WARM_COMPONENTS:
        if not str(payload.get(key, "")).startswith("ready:"):
            reasons.append(f"{key} not ready")
    return bool(reasons), reasons


def _print_warmup_summary(settings: Settings, echo: Callable[[str], None]) -> None:
    payload = load_local_model_status(settings.artifacts_dir)
    if not payload:
        echo("Warmup status file is still missing.")
        return

    echo("Warmup status summary:")
    for key in ("embedding", "reranker", "nli", "text", "decomposition", "generation", "metadata", "parser"):
        if key in payload:
            echo(f"  {key}: {payload[key]}")
    echo("")


def _build_progress_reporter(echo: Callable[[str], None]) -> Callable[[str], None]:
    def report(message: str) -> None:
        text = str(message)
        echo(text)
        for line in text.splitlines():
            if line.strip():
                logger.info(line, extra={"console_visible": False})

    return report


def prepare_backend(
    settings: Settings,
    *,
    sources_dir: str | Path | None = None,
    run_corpus: bool = True,
    echo: Callable[[str], None] = print,
) -> dict[str, object]:
    overall_start = timed()
    trace_id = start_trace()
    configure_model_cache(settings)
    configure_logging(settings.artifacts_dir.parent / "logs", enable_file_logs=settings.trace_logs, category="corpus")
    report = _build_progress_reporter(echo)
    resolved_sources_dir = _resolve_sources_dir(sources_dir)
    resolved_sources_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "startup preparation begin",
        extra={
            "trace_id": trace_id,
            "runtime_mode": settings.runtime_mode,
            "runtime_profile": settings.runtime_profile,
            "sources_dir": str(resolved_sources_dir),
        },
    )

    needs_warmup, reasons = _warmup_reasons(settings)
    if not settings.use_local_models:
        report("Local model warmup skipped because use_local_models=false.")
        report("")
    elif needs_warmup:
        logger.info("startup warmup required", extra={"reasons": ", ".join(reasons)})
        report("Local warmup is missing or incomplete.")
        report(", ".join(reasons))
        report("")
        report("Running quarry warm-local-models...")
        report("")
        result = warm_local_models(settings)
        report(json.dumps(result, indent=2, default=str))
        report("")
        _print_warmup_summary(settings, report)
    else:
        logger.info("startup warmup already satisfied")
        report("Local warmup already available.")
        report("")

    source_files = _source_files(resolved_sources_dir)
    if run_corpus:
        if not source_files:
            raise RuntimeError(
                f"No source files found in {resolved_sources_dir}. "
                "Add PDFs, markdown, or text files there before starting the backend."
            )

        report(f"Rebuilding QUARRY corpus from {resolved_sources_dir}...")
        report(f"Found {len(source_files)} source file(s).")
        report("")
        logger.info("startup ingest begin", extra={"source_count": len(source_files)})
        ingest_result = ingest_documents([str(path) for path in source_files], settings, progress=report)
        manifest = ingest_result.get("manifest")
        if manifest is not None:
            report(f"Ingest complete. Built {getattr(manifest, 'chunk_count', 'unknown')} chunk(s).")
            report("")

    else:
        report("Skipping corpus rebuild.")
        logger.info("startup corpus rebuild skipped")
        manifest_path = settings.artifacts_dir / "manifest.json"
        if manifest_path.exists():
            report("Using existing artifacts without validation.")
            report("")
        else:
            report("No artifact manifest found. Backend will use fallback corpus_dir data if available.")
            report("")
    logger.info(
        "startup preparation complete",
        extra={
            "document_count": len(source_files),
            "manifest_path": str((settings.artifacts_dir / "manifest.json").resolve()),
            "run_corpus": run_corpus,
            "latency_ms": elapsed_ms(overall_start),
        },
    )
    return {
        "sources_dir": str(resolved_sources_dir),
        "source_files": [str(path) for path in source_files],
        "manifest_path": str((settings.artifacts_dir / "manifest.json").resolve()),
    }


def serve_backend(
    settings: Settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    sources_dir: str | Path | None = None,
    run_corpus: bool = True,
    config_path: str | Path | None = None,
    echo: Callable[[str], None] = print,
) -> None:
    server_start = timed()
    prep = prepare_backend(settings, sources_dir=sources_dir, run_corpus=run_corpus, echo=echo)
    configure_logging(settings.artifacts_dir.parent / "logs", enable_file_logs=settings.trace_logs, category="runtime")
    echo("Starting QUARRY backend...")
    echo(f"Project: {_project_root()}")
    display_config = Path(config_path).resolve() if config_path else (_project_root() / "quarry.local.toml").resolve()
    echo(f"Config: {display_config}")
    echo(f"Sources: {prep['sources_dir']}")
    runtime_log = current_log_file("runtime")
    corpus_log = current_log_file("corpus")
    if runtime_log is not None:
        echo(f"Runtime logs: {runtime_log}")
    if corpus_log is not None:
        echo(f"Corpus logs: {corpus_log}")
    echo("")
    logger.info(
        "startup launching api server",
        extra={"host": host, "port": port, "latency_ms": elapsed_ms(server_start)},
    )
    uvicorn.run(create_app(settings), host=host, port=port, reload=False)


def build_start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python start_backend.py")
    parser.add_argument("--config", help="Path to a TOML config file. Defaults to quarry.local.toml when present.")
    parser.add_argument(
        "--profile",
        choices=["apple_silicon", "gpu"],
        help="Select the QUARRY runtime profile for this run.",
    )
    parser.add_argument("--sources", default="data/sources", help="Directory containing source PDFs/markdown/text files.")
    parser.add_argument("--skip-corpus", action="store_true", help="Skip corpus rebuild and start from existing artifacts.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_start_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env(config_path=args.config)
    if args.profile:
        settings.runtime_profile = args.profile
    serve_backend(
        settings,
        host=args.host,
        port=args.port,
        sources_dir=args.sources,
        run_corpus=not args.skip_corpus,
        config_path=args.config,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
