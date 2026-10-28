"""Regression tests for bugs flagged by the codex review.

Each test pins a single behaviour the corresponding fix introduces.
"""

from __future__ import annotations

import pytest

from autodata.llm.schemas import QAPair, RubricCriterion, RubricSpec
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.evaluate_rubric import _normalise_judge_scores
from autodata.pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_paper
from autodata.pipeline.quality_verifier import verify_quality
from tests.fakes import ScriptedBackend, challenger_payload, make_clients, make_paper, qv_payload

# ---------------------------------------------------------------------------
# Codex MAJOR #4: Verifier verdict parsing must fail closed
# ---------------------------------------------------------------------------

def _qa_for_qv() -> QAPair:
    return QAPair(
        question_type="failure mode prediction", skill_tags=["causal_reasoning"],
        context="ctx", question="q?", reference_answer="a",
        rubric=RubricSpec(criteria=[RubricCriterion(
            criterion="answer mentions key concept", weight=1, category="positive")]),
        paper_id="p",
    )


@pytest.mark.parametrize("payload, expected", [
    (qv_payload("PASS"), True),
    (qv_payload("pass"), True),                                  # case-insensitive
    (qv_payload(" PASS "), True),
    (qv_payload("FAIL"), False),
    # OVERALL says PASS but a check failed: inconsistent verdict -> fail closed.
    (qv_payload("PASS", check_1_verdict="LEAKS_ANSWER"), False),
    (qv_payload("PASS", check_2_verdict="RECALL"), False),
    (qv_payload("PASS", check_2_verdict="TOO_EASY"), False),
    (qv_payload("PASS", check_3_verdict="FAIL"), False),
    (qv_payload("PASS", check_4_verdict="INCONSISTENT"), False),
    (qv_payload("PASS", check_3_verdict=None), False),           # missing check
    (qv_payload("yes"), False),                                  # not the contract
    (qv_payload("true"), False),
    ('{"overall": "PASS"}', False),                              # checks absent
    ('{}', False),
])
def test_qv_verdict_fails_closed(payload: str, expected: bool) -> None:
    backend = ScriptedBackend(handlers={"quality_verifier": [payload]})
    verdict = verify_quality(make_clients(backend)["quality_verifier"], make_paper(), _qa_for_qv())
    assert verdict.passed is expected


def test_qv_failure_lists_failed_checks() -> None:
    payload = qv_payload("FAIL", check_1_verdict="LEAKS_ANSWER", check_3_verdict="FAIL")
    backend = ScriptedBackend(handlers={"quality_verifier": [payload]})
    verdict = verify_quality(make_clients(backend)["quality_verifier"], make_paper(), _qa_for_qv())
    assert "check_1_verdict=LEAKS_ANSWER" in verdict.issues
    assert "check_3_verdict=FAIL" in verdict.issues
    assert not any(i.startswith("check_2") for i in verdict.issues)


# ---------------------------------------------------------------------------
# Codex MAJOR #5: Judge scores must be clamped + length-aligned BEFORE rubric_score
# ---------------------------------------------------------------------------

def test_normalise_clamps_minor_overshoot() -> None:
    # within tolerance -> clamped (judge rounded a bit)
    out = _normalise_judge_scores([1.1, 0.5, 1.05], expected_n=3)
    assert out == [1.0, 0.5, 1.0]


def test_normalise_clamps_minor_undershoot() -> None:
    out = _normalise_judge_scores([-0.1, 0.0, 0.5], expected_n=3)
    assert out == [0.0, 0.0, 0.5]


@pytest.mark.parametrize("raw", [
    [2.0, 0.5, 0.6],          # gross overshoot: judge confused
    [-0.5, 0.5, 0.5],         # gross undershoot
    ["nope", 0.5, 0.5],       # not a number
    [0.7],                    # too few: which item did it skip?
    [0.1, 0.2, 0.3, 0.4],     # too many
])
def test_normalise_rejects_unusable_scores(raw: list) -> None:
    # Unusable output is a judge failure to re-ask about, never a row of
    # zeros: zeros would read as "the weak solver failed" (paper Fig 7).
    with pytest.raises(ValueError):
        _normalise_judge_scores(raw, expected_n=3)


# ---------------------------------------------------------------------------
# Codex MAJOR #1: Inner loop must dereference prompts via the prompts module
# (so a hot-swap in meta-optimization actually takes effect).
# ---------------------------------------------------------------------------

def test_inner_loop_uses_late_bound_prompts(monkeypatch) -> None:
    """When `autodata.llm.prompts.CHALLENGER_SYSTEM` is monkeypatched at
    runtime, the inner loop should pick up the new value rather than a
    snapshot taken at import time. This is the silent-meta bug from codex.
    """
    from autodata.llm import prompts as base_prompts

    backend = ScriptedBackend(handlers={
        "challenger": [challenger_payload()],
        "quality_verifier": [qv_payload("FAIL", check_2_verdict="RECALL")],
    })
    monkeypatch.setattr(base_prompts, "CHALLENGER_SYSTEM", "MUTATED-CHALLENGER-SYSTEM")
    run_inner_loop_for_paper(
        make_paper(), make_clients(backend), AcceptanceCriteria(),
        InnerLoopConfig(max_iterations=1, num_solver_samples=1, parallel_solvers=False),
    )
    assert "MUTATED-CHALLENGER-SYSTEM" in backend.text_seen_by("challenger")
