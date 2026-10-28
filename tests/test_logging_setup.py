"""Logging setup keeps the run log.

Hydra attaches a file handler that writes ``<run dir>/<script>.log`` beside
``.hydra/``. setup_logging swaps the console handler and must keep that file
handler, or the run directory holds an empty log.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from autodata.utils import setup_logging


@pytest.fixture
def clean_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
        if h not in saved:
            h.close()
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)


def test_run_log_file_survives_setup(clean_root: logging.Logger, tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    clean_root.addHandler(logging.FileHandler(log_file))
    setup_logging("INFO")
    logging.getLogger("autodata.test").info("RUN-LOG-LINE")
    for h in clean_root.handlers:
        h.flush()
    assert "RUN-LOG-LINE" in log_file.read_text()


def test_setup_twice_keeps_one_console_handler(clean_root: logging.Logger) -> None:
    setup_logging("INFO")
    setup_logging("INFO")
    consoles = [h for h in clean_root.handlers if not isinstance(h, logging.FileHandler)]
    assert len(consoles) == 1


def test_http_client_chatter_stays_out(clean_root: logging.Logger) -> None:
    setup_logging("INFO")
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
