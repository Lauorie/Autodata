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
    d = Path(papers_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"papers dir not found: {d}")
    files = sorted(p for p in d.glob("*.md") if p.is_file())
    if limit is not None:
        files = files[:limit]
    logger.info("loading %d papers from %s", len(files), d)
    for f in files:
        try:
            yield load_paper(f)
        except Exception as e:
            logger.warning("skip %s: %s", f.name, e)


__all__ = ["Paper", "iter_papers", "load_paper"]
