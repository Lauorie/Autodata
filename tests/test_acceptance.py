"""Tests for AutoData acceptance criteria.

The blog specifies (all simultaneously):
  - QV passes
  - weak_avg ≤ 0.65, max(weak) ≤ 0.75, no weak score == 0
  - 0.60 ≤ strong_avg < 0.95
  - gap = strong_avg - weak_avg ≥ 0.20
"""

from autodata.pipeline.acceptance import AcceptanceCriteria, check_acceptance

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
