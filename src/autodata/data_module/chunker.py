"""Section-aware chunker for CS paper markdown.

Strategy:
1. Split on top-level (`#` and `##`) headers, treating the header as part of
   the chunk that follows it.
2. Drop sections whose title matches a junk regex (References, Acknowledgements, ...).
3. Hard-cap each chunk by character count; if a section exceeds the cap,
   split on paragraph boundaries.
4. Drop very-small chunks (< 600 chars) since they rarely yield discriminative
   questions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .paper_loader import Paper

logger = logging.getLogger(__name__)


_HEADER_RE = re.compile(r"^(#{1,2})\s+(.*?)\s*$", re.MULTILINE)

_JUNK_TITLES = re.compile(
    r"^(references?|bibliograph(y|ie)|acknowledg(e?ments?)?|appendix|"
    r"author\s+contributions|funding|ethics\s+statement|"
    r"impact\s+statements?|broader\s+impacts?)\b",
    re.IGNORECASE,
)

DEFAULT_MAX_CHARS = 12000
MIN_CHUNK_CHARS = 600


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    title: str            # paper title
    section: str          # section heading
    text: str             # chunk body (includes section heading)
    chunk_id: str         # paper_id::section_idx::sub_idx


def _split_sections(md: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text), ...]. Section text starts with the header line."""
    matches = list(_HEADER_RE.finditer(md))
    if not matches:
        return [("", md)]
    sections: list[tuple[str, str]] = []
    # Lead-in (anything before the first header) -> labelled as "Front Matter"
    lead = md[: matches[0].start()].strip()
    if lead:
        sections.append(("Front Matter", lead))
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[m.start():end].strip()
        sections.append((title, body))
    return sections


def _split_long_section(text: str, max_chars: int) -> list[str]:
    """Split a long section on paragraph boundaries, accumulating up to max_chars."""
    paras = re.split(r"\n\s*\n", text)
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        plen = len(p) + 2
        if cur_len + plen > max_chars and cur:
            out.append("\n\n".join(cur))
            cur, cur_len = [p], plen
        else:
            cur.append(p)
            cur_len += plen
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_paper(paper: Paper, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec_idx, (sec_title, sec_text) in enumerate(_split_sections(paper.text)):
        if _JUNK_TITLES.match(sec_title):
            continue
        # Strip the leading header line for the junk check; keep it inside the body though.
        if len(sec_text) <= max_chars:
            pieces = [sec_text]
        else:
            pieces = _split_long_section(sec_text, max_chars)
        for sub_idx, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                paper_id=paper.paper_id,
                title=paper.title,
                section=sec_title or "(unsectioned)",
                text=piece,
                chunk_id=f"{paper.paper_id}::{sec_idx:02d}::{sub_idx:02d}",
            ))
    logger.debug("paper %s -> %d chunks", paper.paper_id, len(chunks))
    return chunks


__all__ = ["Chunk", "DEFAULT_MAX_CHARS", "MIN_CHUNK_CHARS", "chunk_paper"]
