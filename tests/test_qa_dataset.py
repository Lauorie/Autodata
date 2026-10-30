"""Training and test data from finished runs (arXiv 2606.25996, Sec 3.1).

The paper trains on the questions the loop accepted and the end-of-loop
filter kept (1.3k), and holds out 100 of them as a test set. Rejected
rounds are never training data. Test and train share no paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodata.data_module.qa_dataset import dedupe, load_final_qas, split_by_paper
from tests.fakes import make_qa


def _write_run(root: Path, name: str, final: list[dict], rounds: list[dict] | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "final_qa.jsonl").write_text("".join(json.dumps(q) + "\n" for q in final))
    (d / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds or []))
    return d


def _qa(paper: str, question: str) -> dict:
    return {**make_qa(question=question).model_dump(), "paper_id": paper}


def test_only_final_questions_are_loaded(tmp_path: Path) -> None:
    rejected = {"qa": _qa("p9", "a rejected question?"), "accepted": False}
    run = _write_run(tmp_path, "r1", [_qa("p1", "kept?")], rounds=[rejected])
    qas = load_final_qas([run])
    assert [q.question for q in qas] == ["kept?"]


def test_a_run_without_final_questions_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_final_qas([tmp_path / "empty"])


def test_duplicate_questions_keep_the_first(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", [_qa("p1", "Same  question?")])
    b = _write_run(tmp_path, "b", [_qa("p2", "same question?"), _qa("p3", "other?")])
    qas = dedupe(load_final_qas([a, b]))
    assert [(q.paper_id, q.question) for q in qas] == [("p1", "Same  question?"), ("p3", "other?")]


def _many(n_papers: int, per_paper: int = 1) -> list:
    return [make_qa(question=f"q{p}-{k}?").model_copy(update={"paper_id": f"p{p}"})
            for p in range(n_papers) for k in range(per_paper)]


def test_split_shares_no_paper() -> None:
    train, test = split_by_paper(_many(20, per_paper=2), n_test=6, seed=1)
    assert len(test) == 6 and len(train) == 34
    assert not {q.paper_id for q in train} & {q.paper_id for q in test}


def test_split_is_deterministic() -> None:
    a = split_by_paper(_many(30), n_test=5, seed=3)
    b = split_by_paper(_many(30), n_test=5, seed=3)
    assert [q.question for q in a[1]] == [q.question for q in b[1]]


def test_split_needs_training_data_left() -> None:
    with pytest.raises(ValueError, match="test"):
        split_by_paper(_many(5), n_test=5, seed=0)
