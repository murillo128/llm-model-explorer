"""Installable backend process entry point."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local LLM Model Explorer backend")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cors-origin", action="append", default=[], metavar="ORIGIN")
    args = parser.parse_args(argv)

    from .settings import Settings

    try:
        settings = Settings(
            model_root=args.model_root,
            cache_dir=args.cache_dir,
            device=args.device,
            host=args.host,
            port=args.port,
            cors_origins=tuple(args.cors_origin),
        )
    except ValueError as exc:
        parser.error(str(exc))

    from .app import ApplicationServer, create_app

    logging.basicConfig(level=logging.INFO)
    ApplicationServer(create_app(settings), host=settings.host, port=settings.port).run()
