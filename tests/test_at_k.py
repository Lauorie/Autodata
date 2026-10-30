"""Held-out evaluation: mean@k and best@k (arXiv 2606.25996, Table 2).

Each test question is answered k times with the loop's own solver settings
and graded by the same judge. A question with a sample lost to
infrastructure is left out and counted, never scored as 0.
"""

from __future__ import annotations

import pytest

from autodata.llm import SolverScore
from autodata.llm.prompt_set import PromptSet
from autodata.pipeline.at_k import score_test_set, summarize_at_k
from tests.fakes import WEAK_PER, ScriptedBackend, judge_payload, make_clients, make_qa


def _s(score: float, infra: bool = False, truncated: bool = False) -> SolverScore:
    return SolverScore(answer="a", per_criterion=[score], weighted_score=score,
                       infra_error=infra, truncated=truncated)


def test_mean_and_best_at_k() -> None:
    summary = summarize_at_k([[_s(0.2), _s(0.4), _s(0.6)], [_s(0.0), _s(1.0), _s(0.5)]])
    assert summary.n == 2
    assert summary.mean_at_k == pytest.approx((0.4 + 0.5) / 2)
    assert summary.best_at_k == pytest.approx((0.6 + 1.0) / 2)


def test_question_with_an_infra_sample_is_left_out() -> None:
    summary = summarize_at_k([[_s(0.2), _s(0.4), _s(0.6)], [_s(0.9), _s(0.0, infra=True), _s(0.9)]])
    assert (summary.n, summary.left_out) == (1, 1)
    assert summary.mean_at_k == pytest.approx(0.4)


def test_truncation_rate_counts_scored_samples() -> None:
    summary = summarize_at_k([[_s(0.0, truncated=True), _s(0.5), _s(0.5), _s(0.5)]])
    assert summary.truncation_rate == pytest.approx(0.25)


def test_nothing_scored_gives_no_means() -> None:
    summary = summarize_at_k([[_s(0.0, infra=True)]])
    assert summary.mean_at_k is None and summary.best_at_k is None


def test_test_set_is_scored_k_times_per_question() -> None:
    backend = ScriptedBackend(handlers={"weak_solver": lambda m: "ANSWER",
                                        "judge": lambda m: judge_payload(WEAK_PER)})
    clients = make_clients(backend)
    qas = [make_qa(question=f"q{i}?") for i in range(2)]
    per_qa = score_test_set(qas, clients["weak_solver"], clients["judge"], k=3, retries=0,
                            workers=1, prompts=PromptSet.seed())
    assert [len(s) for s in per_qa] == [3, 3]
    assert sum(r == "weak_solver" for r, _ in backend.calls) == 6
    assert per_qa[0][0].weighted_score == pytest.approx(0.5)
