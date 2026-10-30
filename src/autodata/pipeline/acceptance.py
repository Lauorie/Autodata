"""Acceptance-criteria check for an Agentic Self-Instruct round.

Fig 7 criteria (arXiv 2606.25996), checked in two stages so the strong
solver runs only on questions the weak solver fails (Sec 3.1):
- weak stage: weak_avg ≤ 0.65, max(weak_samples) ≤ 0.75, no weak score is 0
- strong stage: 0.60 ≤ strong_avg < 0.95, gap = strong_avg - weak_avg ≥ 0.20

The quality verifier's gate comes before both, in the inner loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceCriteria:
    weak_avg_max: float = 0.65
    weak_max_max: float = 0.75
    no_zeros: bool = True
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95
    gap_min: float = 0.20


@dataclass(frozen=True)
class WeakCheck:
    weak_avg: float
    weak_max: float
    too_easy: bool        # weak_avg or weak_max above its ceiling
    has_zero: bool        # a weak attempt scored 0 while the no-zeros rule is on
    reasons: list[str]

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class StrongCheck:
    strong_avg: float
    gap: float
    reasons: list[str]

    @property
    def passed(self) -> bool:
        return not self.reasons


def check_weak(weak_scores: Sequence[float], criteria: AcceptanceCriteria) -> WeakCheck:
    """Fig 7 step 5: WEAK_PASSED (weak_avg ≤ 65%, max_weak ≤ 75%, no zeros)."""
    if not weak_scores:
        return WeakCheck(0.0, 0.0, too_easy=False, has_zero=False,
                         reasons=["empty weak score list"])
    weak_avg = sum(weak_scores) / len(weak_scores)
    weak_max = max(weak_scores)
    reasons: list[str] = []
    if weak_avg > criteria.weak_avg_max:
        reasons.append(f"weak_avg {weak_avg:.2f} > {criteria.weak_avg_max}")
    if weak_max > criteria.weak_max_max:
        reasons.append(f"weak_max {weak_max:.2f} > {criteria.weak_max_max}")
    too_easy = bool(reasons)
    has_zero = criteria.no_zeros and any(s == 0.0 for s in weak_scores)
    if has_zero:
        reasons.append("weak has a zero score (suggests question is broken or unanswerable)")
    return WeakCheck(weak_avg, weak_max, too_easy=too_easy, has_zero=has_zero, reasons=reasons)


def check_strong(weak_avg: float, strong_scores: Sequence[float],
                 criteria: AcceptanceCriteria) -> StrongCheck:
    """Fig 7 step 7: 60% ≤ strong_avg < 95% and gap ≥ 20 points."""
    if not strong_scores:
        return StrongCheck(0.0, 0.0, reasons=["empty strong score list"])
    strong_avg = sum(strong_scores) / len(strong_scores)
    gap = strong_avg - weak_avg
    reasons: list[str] = []
    if strong_avg < criteria.strong_avg_min:
        reasons.append(f"strong_avg {strong_avg:.2f} < {criteria.strong_avg_min}")
    if strong_avg >= criteria.strong_avg_max:
        reasons.append(f"strong_avg {strong_avg:.2f} ≥ {criteria.strong_avg_max} (saturated)")
    if gap < criteria.gap_min:
        reasons.append(f"gap {gap:.2f} < {criteria.gap_min}")
    return StrongCheck(strong_avg, gap, reasons=reasons)


__all__ = [
    "AcceptanceCriteria",
    "StrongCheck",
    "WeakCheck",
    "check_strong",
    "check_weak",
]
