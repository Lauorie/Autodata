"""Tests for AutoData acceptance criteria.

The blog specifies (all simultaneously):
  - QV passes
  - weak_avg ≤ 0.65, max(weak) ≤ 0.75, no weak score == 0
  - 0.60 ≤ strong_avg < 0.95
  - gap = strong_avg - weak_avg ≥ 0.20
"""

import pytest

from autodata.pipeline.acceptance import (
    AcceptanceCriteria,
    check_acceptance,
    check_strong,
    check_weak,
)

CRIT = AcceptanceCriteria()


def test_canonical_accept() -> None:
    res = check_acceptance(
        weak_scores=[0.30, 0.40, 0.50],
        strong_scores=[0.80, 0.85, 0.90],
        quality_verifier_passed=True,
        criteria=CRIT,
    )
    assert res.accepted, res.reasons
    assert abs(res.weak_avg - 0.40) < 1e-9
    assert abs(res.strong_avg - 0.85) < 1e-9
    assert abs(res.gap - 0.45) < 1e-9


def test_qv_failure_blocks() -> None:
    res = check_acceptance([0.30], [0.80], quality_verifier_passed=False, criteria=CRIT)
    assert not res.accepted
    assert "quality_verifier_failed" in res.reasons


def test_weak_too_strong() -> None:
    res = check_acceptance([0.70, 0.70], [0.95, 0.95], True, CRIT)
    assert not res.accepted
    assert any("weak_avg" in r for r in res.reasons)


def test_weak_max_cap() -> None:
    res = check_acceptance([0.50, 0.80], [0.85, 0.85], True, CRIT)
    assert not res.accepted
    assert any("weak_max" in r for r in res.reasons)


def test_strong_saturated() -> None:
    res = check_acceptance([0.40, 0.40], [0.97, 0.96], True, CRIT)
    assert not res.accepted
    assert any("saturated" in r or "0.95" in r for r in res.reasons)


def test_strong_too_weak() -> None:
    res = check_acceptance([0.30, 0.30], [0.50, 0.55], True, CRIT)
    assert not res.accepted
    assert any("strong_avg" in r and "<" in r for r in res.reasons)


def test_no_zeros_violated() -> None:
    res = check_acceptance([0.0, 0.40, 0.40], [0.80, 0.80, 0.80], True, CRIT)
    assert not res.accepted
    assert any("zero" in r for r in res.reasons)


def test_gap_too_small() -> None:
    res = check_acceptance([0.55, 0.55], [0.65, 0.65], True, CRIT)
    assert not res.accepted
    assert any("gap" in r for r in res.reasons)


def test_empty_inputs() -> None:
    res = check_acceptance([], [], True, CRIT)
    assert not res.accepted
    assert "empty solver score lists" in res.reasons


# ---------------------------------------------------------------------------
# Stage checks: the weak solver runs first and the strong solver only after
# the weak check passes (Fig 7 steps 4-7, Sec 3.1).
# ---------------------------------------------------------------------------

def test_weak_check_passes_hard_question() -> None:
    res = check_weak([0.30, 0.40, 0.50], CRIT)
    assert res.passed and not res.too_easy and not res.has_zero
    assert res.weak_avg == pytest.approx(0.40)
    assert res.weak_max == 0.50


@pytest.mark.parametrize("scores", [[0.70, 0.70], [0.50, 0.80]])
def test_weak_check_flags_too_easy(scores: list[float]) -> None:
    res = check_weak(scores, CRIT)
    assert not res.passed and res.too_easy and not res.has_zero


def test_weak_zero_is_not_too_easy() -> None:
    res = check_weak([0.0, 0.40, 0.40], CRIT)
    assert not res.passed and res.has_zero and not res.too_easy


def test_weak_zero_allowed_when_rule_off() -> None:
    res = check_weak([0.0, 0.40, 0.40], AcceptanceCriteria(no_zeros=False))
    assert res.passed


def test_weak_check_rejects_empty_scores() -> None:
    assert not check_weak([], CRIT).passed


def test_strong_check_passes_with_gap() -> None:
    res = check_strong(0.30, [0.80, 0.80], CRIT)
    assert res.passed
    assert res.gap == pytest.approx(0.50)


@pytest.mark.parametrize("weak_avg, strong, fragment", [
    (0.40, [0.97, 0.96], "saturated"),
    (0.30, [0.50, 0.55], "strong_avg"),
    (0.55, [0.65, 0.65], "gap"),
])
def test_strong_check_failures(weak_avg: float, strong: list[float], fragment: str) -> None:
    res = check_strong(weak_avg, strong, CRIT)
    assert not res.passed
    assert any(fragment in r for r in res.reasons)
