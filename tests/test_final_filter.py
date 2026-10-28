"""End-of-loop quality filter (Sec 3.1 Data Setup: 2.8k accepted -> 1.3k kept).

A fixed gate after the loop: deterministic Fig 9 check-3 rubric counts, then
one more quality-verifier call. It fails closed.
"""

import pytest

from autodata.llm import RubricSpec
from autodata.llm.schemas import QAPair, VerifierVerdict
from autodata.pipeline.final_filter import RubricShape, final_filter, rubric_shape_issues
from tests.fakes import (
    RUBRIC,
    Handler,
    ScriptedBackend,
    make_clients,
    make_paper,
    make_qa,
    qv_payload,
)

SHAPE = RubricShape()


def _pos(n: int) -> list[dict]:
    return [{"criterion": f"insight number {i}", "weight": 2, "category": "positive"}
            for i in range(n)]


def _neg(n: int) -> list[dict]:
    return [{"criterion": f"error number {i}", "weight": -2, "category": "negative"}
            for i in range(n)]


def test_baseline_rubric_shape_has_no_issues() -> None:
    assert rubric_shape_issues(RubricSpec(criteria=RUBRIC), SHAPE) == []


@pytest.mark.parametrize("items, fragment", [
    (_pos(6) + _neg(3), "total"),        # 9 items
    (_pos(15) + _neg(6), "total"),       # 21 items
    (_pos(8) + _neg(2), "negative"),
    (_pos(3) + _neg(7), "positive"),
])
def test_rubric_shape_issues(items: list[dict], fragment: str) -> None:
    issues = rubric_shape_issues(RubricSpec(criteria=items), SHAPE)
    assert any(fragment in i for i in issues), issues


def _filter(qa: QAPair, qv: Handler) -> tuple[VerifierVerdict, ScriptedBackend]:
    backend = ScriptedBackend(handlers={"quality_verifier": qv})
    verdict = final_filter(make_paper(), qa, make_clients(backend)["quality_verifier"], SHAPE)
    return verdict, backend


def test_malformed_rubric_is_dropped_without_asking_the_verifier() -> None:
    verdict, backend = _filter(make_qa(rubric=_pos(6) + _neg(3)), lambda m: qv_payload())
    assert not verdict.passed
    assert backend.calls == []


def test_verifier_rejection_drops_the_question() -> None:
    verdict, _ = _filter(make_qa(), lambda m: qv_payload("FAIL", check_1_verdict="LEAKS_ANSWER"))
    assert not verdict.passed
    assert "check_1_verdict=LEAKS_ANSWER" in verdict.issues


def test_unreadable_verifier_reply_drops_the_question() -> None:
    verdict, _ = _filter(make_qa(), lambda m: "looks fine to me")
    assert not verdict.passed


def test_clean_question_passes() -> None:
    verdict, backend = _filter(make_qa(), lambda m: qv_payload())
    assert verdict.passed
    assert len(backend.calls) == 1
