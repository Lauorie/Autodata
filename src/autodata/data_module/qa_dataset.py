"""Training and test questions from finished runs (arXiv 2606.25996, Sec 3.1).

The paper trains on the questions the loop accepted and the end-of-loop
filter kept, and holds out 100 of them as a test set. Only
``final_qa.jsonl`` is read: a rejected round failed the very criteria that
make a question worth training on.

The split is by paper, so no test question shares a paper with training.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..llm.schemas import QAPair


def load_final_qas(run_dirs: Sequence[str | Path]) -> list[QAPair]:
    """Every row of ``<run>/final_qa.jsonl``, in run order."""
    qas: list[QAPair] = []
    for run in run_dirs:
        path = Path(run) / "final_qa.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                qas.append(QAPair.model_validate(json.loads(line)))
    return qas


def _key(question: str) -> str:
    return " ".join(question.lower().split())


def dedupe(qas: Sequence[QAPair]) -> list[QAPair]:
    """Drop repeats of a question (case and spacing ignored); keep the first."""
    seen: set[str] = set()
    out = []
    for q in qas:
        if (k := _key(q.question)) not in seen:
            seen.add(k)
            out.append(q)
    return out


def split_by_paper(qas: Sequence[QAPair], n_test: int,
                   seed: int) -> tuple[list[QAPair], list[QAPair]]:
    """Move whole papers, in seeded order, into the test set until it holds
    ``n_test`` questions; a paper with several questions can overshoot."""
    if len(qas) <= n_test:
        raise ValueError(f"{len(qas)} questions leave none for training after a "
                         f"{n_test}-question test set")
    by_paper: dict[str, list[QAPair]] = defaultdict(list)
    for q in qas:
        by_paper[q.paper_id].append(q)
    papers = sorted(by_paper)
    random.Random(seed).shuffle(papers)
    test_papers: set[str] = set()
    n = 0
    for p in papers:
        if n >= n_test:
            break
        test_papers.add(p)
        n += len(by_paper[p])
    train = [q for q in qas if q.paper_id not in test_papers]
    test = [q for q in qas if q.paper_id in test_papers]
    return train, test


__all__ = ["dedupe", "load_final_qas", "split_by_paper"]
