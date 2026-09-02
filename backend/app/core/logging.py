"""Structured logging setup.

All log records are JSON (or console in dev). Agent traces must be *safe
operational summaries only* — never prompts, completions, or chain-of-thought.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Windows consoles often default to cp1252 — force UTF-8 so tracebacks
    # containing Unicode don't crash the logger itself.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

    logging.basicConfig(format="%(message)s", level=level, stream=sys.stdout)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.dict_tracebacks)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
