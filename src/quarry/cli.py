from __future__ import annotations

import argparse
import json

import uvicorn

from quarry.api.app import create_app
from quarry.config import Settings
from quarry.ingest.pipeline import ingest_documents, rebuild_indexes, warm_local_models
from quarry.model_cache import configure_model_cache
from quarry.startup import serve_backend


def add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=["apple_silicon", "gpu"],
        help="Select the QUARRY runtime profile for this command.",
    )


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="Path to a TOML config file. Defaults to config.toml when present.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quarry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one or more source documents into artifact files.")
    add_config_argument(ingest)
    add_profile_argument(ingest)
    ingest.add_argument("paths", nargs="+")

    rebuild = subparsers.add_parser("rebuild-indexes", help="Rebuild vector and search artifacts from current corpus files.")
    add_config_argument(rebuild)
    add_profile_argument(rebuild)
    warm = subparsers.add_parser("warm-local-models", help="Download and warm the configured local OCR, embedding, reranker, NLI, and text models.")
    add_config_argument(warm)
    add_profile_argument(warm)
    start = subparsers.add_parser("start", help="Warm local models, optionally rebuild corpus from data/sources, and start the QUARRY API server.")
    add_config_argument(start)
    add_profile_argument(start)
    start.add_argument("--sources", default="data/sources")
    start.add_argument("--skip-corpus", action="store_true")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", default=8000, type=int)
    serve = subparsers.add_parser("serve", help="Start the QUARRY API server.")
    add_config_argument(serve)
    add_profile_argument(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env(config_path=getattr(args, "config", None))
    if getattr(args, "profile", None):
        settings.runtime_profile = args.profile
    configure_model_cache(settings)

    if args.command == "ingest":
        result = ingest_documents(args.paths, settings)
    elif args.command == "rebuild-indexes":
        result = rebuild_indexes(settings)
    elif args.command == "warm-local-models":
        result = warm_local_models(settings)
    elif args.command == "start":
        serve_backend(
            settings,
            host=args.host,
            port=args.port,
            sources_dir=args.sources,
            run_corpus=not args.skip_corpus,
            config_path=getattr(args, "config", None),
        )
        return
    elif args.command == "serve":
        uvicorn.run(
            create_app(settings, config_path=getattr(args, "config", None)),
            host=args.host,
            port=args.port,
            reload=False,
            access_log=False,
        )
        return
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
