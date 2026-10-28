"""Logging setup using rich for human-friendly output."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Put one console handler (rich if available) on the root logger.

    File handlers stay: Hydra's writes the run log beside ``.hydra/``.
    httpx logs every request at INFO; usage.jsonl already records each call.
    """
    try:
        from rich.logging import RichHandler
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=True)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    for h in [h for h in root.handlers if not isinstance(h, logging.FileHandler)]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
