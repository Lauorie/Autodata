"""Acceptance-criteria check for an Agentic Self-Instruct round.

Fig 7 criteria (arXiv 2606.25996), checked in two stages so the strong
solver runs only on questions the weak solver fails (Sec 3.1):
- weak stage: weak_avg ≤ 0.65, max(weak_samples) ≤ 0.75, no weak score is 0
- strong stage: 0.60 ≤ strong_avg < 0.95, gap = strong_avg - weak_avg ≥ 0.20
- Quality Verifier must pass
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceCriteria:
    quality_verifier_must_pass: bool = True
    weak_avg_max: float = 0.65
    weak_max_max: float = 0.75
    no_zeros: bool = True
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95
    gap_min: float = 0.20


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    reasons: list[str]
    weak_avg: float
    weak_max: float
    strong_avg: float
    gap: float

    def to_feedback(self) -> str:
        if self.accepted:
            return "ACCEPTED"
        return "; ".join(self.reasons)


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


def check_acceptance(
    weak_scores: list[float],
    strong_scores: list[float],
    quality_verifier_passed: bool,
    criteria: AcceptanceCriteria,
) -> AcceptanceResult:
    if not weak_scores or not strong_scores:
        return AcceptanceResult(
            accepted=False,
            reasons=["empty solver score lists"],
            weak_avg=0.0, weak_max=0.0, strong_avg=0.0, gap=0.0,
        )
    weak = check_weak(weak_scores, criteria)
    strong = check_strong(weak.weak_avg, strong_scores, criteria)
    reasons: list[str] = []
    if criteria.quality_verifier_must_pass and not quality_verifier_passed:
        reasons.append("quality_verifier_failed")
    reasons += weak.reasons + strong.reasons
    return AcceptanceResult(
        accepted=not reasons,
        reasons=reasons,
        weak_avg=weak.weak_avg, weak_max=weak.weak_max,
        strong_avg=strong.strong_avg, gap=strong.gap,
    )


__all__ = [
    "AcceptanceCriteria",
    "AcceptanceResult",
    "StrongCheck",
    "WeakCheck",
    "check_acceptance",
    "check_strong",
    "check_weak",
]
