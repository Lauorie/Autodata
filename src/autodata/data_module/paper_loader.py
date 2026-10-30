"""Load CS-paper markdown files from disk."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper:
    paper_id: str   # filename without .md
    title: str      # first H1 line, fallback to paper_id
    text: str       # full markdown body
    source_path: Path


def _extract_title(md: str, fallback: str) -> str:
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s.removeprefix("# ").strip()
    return fallback


def load_paper(path: str | Path) -> Paper:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    paper_id = p.stem
    return Paper(paper_id=paper_id, title=_extract_title(text, paper_id),
                 text=text, source_path=p)


def iter_papers(papers_dir: str | Path, limit: int | None = None) -> Iterator[Paper]:
    """Papers in filename order. A blank file gives the challenger nothing to
    read and is skipped; ``limit`` counts the papers returned."""
    d = Path(papers_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"papers dir not found: {d}")
    files = sorted(p for p in d.glob("*.md") if p.is_file())
    logger.info("loading up to %s of %d papers from %s", limit, len(files), d)
    n = 0
    for f in files:
        if limit is not None and n >= limit:
            return
        try:
            paper = load_paper(f)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("skip %s: %s", f.name, e)
            continue
        if not paper.text.strip():
            logger.warning("skip %s: blank", f.name)
            continue
        n += 1
        yield paper


__all__ = ["Paper", "iter_papers", "load_paper"]
