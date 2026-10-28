"""Fig 7 acceptance criteria (arXiv 2606.25996), all at once:
  - weak_avg ≤ 0.65, max(weak) ≤ 0.75, no weak score == 0
  - 0.60 ≤ strong_avg < 0.95
  - gap = strong_avg - weak_avg ≥ 0.20
"""

import pytest

from autodata.pipeline.acceptance import (
    AcceptanceCriteria,
    check_strong,
    check_weak,
)

CRIT = AcceptanceCriteria()


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
