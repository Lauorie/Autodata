"""Reading the paper corpus.

The SciArena corpus holds a few broken files; one is a single newline. A
blank paper gives the challenger nothing to read, so it is skipped with a
warning, and ``limit`` counts the papers actually returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autodata.data_module.paper_loader import iter_papers


def _corpus(tmp_path: Path, texts: dict[str, str]) -> Path:
    for name, text in texts.items():
        (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
    return tmp_path


def test_title_is_the_first_h1(tmp_path: Path) -> None:
    d = _corpus(tmp_path, {"a": "intro\n# The Title\n\nbody", "b": "no heading"})
    assert [(p.paper_id, p.title) for p in iter_papers(d)] == [("a", "The Title"), ("b", "b")]


def test_blank_paper_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    d = _corpus(tmp_path, {"a": "# A\nbody", "b": "\n", "c": "  \n\t\n", "d": "# D\nbody"})
    assert [p.paper_id for p in iter_papers(d)] == ["a", "d"]
    assert "b.md" in caplog.text and "c.md" in caplog.text


def test_limit_counts_papers_returned(tmp_path: Path) -> None:
    d = _corpus(tmp_path, {"a": "\n", "b": "# B\nx", "c": "# C\nx", "d": "# D\nx"})
    assert [p.paper_id for p in iter_papers(d, limit=2)] == ["b", "c"]
