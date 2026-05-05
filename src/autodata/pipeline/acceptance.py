"""Acceptance-criteria check for an Agentic Self-Instruct round.

Per AutoData blog:
- weak_avg ≤ 0.65, max(weak_samples) ≤ 0.75, no weak score is exactly 0
- 0.60 ≤ strong_avg < 0.95
- gap = strong_avg - weak_avg ≥ 0.20
- Quality Verifier must pass
"""

from __future__ import annotations

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
    weak_avg = sum(weak_scores) / len(weak_scores)
    weak_max = max(weak_scores)
    strong_avg = sum(strong_scores) / len(strong_scores)
    gap = strong_avg - weak_avg

    reasons: list[str] = []
    if criteria.quality_verifier_must_pass and not quality_verifier_passed:
        reasons.append("quality_verifier_failed")
    if weak_avg > criteria.weak_avg_max:
        reasons.append(f"weak_avg {weak_avg:.2f} > {criteria.weak_avg_max}")
    if weak_max > criteria.weak_max_max:
        reasons.append(f"weak_max {weak_max:.2f} > {criteria.weak_max_max}")
    if criteria.no_zeros and any(s == 0.0 for s in weak_scores):
        reasons.append("weak has a zero score (suggests question is broken or unanswerable)")
    if strong_avg < criteria.strong_avg_min:
        reasons.append(f"strong_avg {strong_avg:.2f} < {criteria.strong_avg_min}")
    if strong_avg >= criteria.strong_avg_max:
        reasons.append(f"strong_avg {strong_avg:.2f} ≥ {criteria.strong_avg_max} (saturated)")
    if gap < criteria.gap_min:
        reasons.append(f"gap {gap:.2f} < {criteria.gap_min}")

    return AcceptanceResult(
        accepted=not reasons,
        reasons=reasons,
        weak_avg=weak_avg, weak_max=weak_max, strong_avg=strong_avg, gap=gap,
    )


__all__ = ["AcceptanceCriteria", "AcceptanceResult", "check_acceptance"]
