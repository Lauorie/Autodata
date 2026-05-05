"""Tests for paper section-aware chunker."""

from autodata.data_module.chunker import (
    DEFAULT_MAX_CHARS,
    MIN_CHUNK_CHARS,
    chunk_paper,
)
from autodata.data_module.paper_loader import Paper


def _make_paper(text: str, paper_id: str = "test", title: str = "Test Paper") -> Paper:
    from pathlib import Path
    return Paper(paper_id=paper_id, title=title, text=text, source_path=Path("/dev/null"))


def test_skips_short_chunks() -> None:
    md = "# Tiny\n\nshort.\n"
    chunks = chunk_paper(_make_paper(md))
    assert chunks == []


def test_keeps_longer_chunk() -> None:
    body = "Long paragraph. " * 200  # ~3200 chars
    md = f"# Intro\n\n{body}\n"
    chunks = chunk_paper(_make_paper(md))
    assert len(chunks) == 1
    assert chunks[0].section == "Intro"
    assert "Long paragraph" in chunks[0].text


def test_drops_references() -> None:
    body = "X " * 400
    md = f"# Intro\n\n{body}\n\n# References\n\n{body}\n"
    chunks = chunk_paper(_make_paper(md))
    sections = {c.section for c in chunks}
    assert "Intro" in sections
    assert not any("Reference" in s for s in sections)


def test_splits_long_section() -> None:
    para = ("para text. " * 100) + "\n\n"
    body = para * 25  # well over default max
    md = f"# Mega\n\n{body}"
    chunks = chunk_paper(_make_paper(md), max_chars=4000)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 4000 + 100  # small slop allowed
        assert c.section == "Mega"


def test_chunk_id_unique() -> None:
    body = "X " * 400
    md = f"# A\n\n{body}\n\n# B\n\n{body}\n"
    chunks = chunk_paper(_make_paper(md))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_min_chunk_chars_constant() -> None:
    # Sanity: documented constant is sane.
    assert 100 < MIN_CHUNK_CHARS < DEFAULT_MAX_CHARS
