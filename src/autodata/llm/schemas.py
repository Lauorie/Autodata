"""Pydantic schemas exchanged between agents.

Mirrors the data shapes implied by the AutoData blog:
- Challenger emits QAPair + RubricSpec
- Quality Verifier emits VerifierVerdict
- Solver answers a QA, scored by Judge -> SolverScore
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RubricCriterion(BaseModel):
    """A single positive criterion the answer should satisfy.

    Per blog: 'positive-only criteria, integer weights ≤ 7'.
    """

    description: str = Field(..., min_length=3)
    weight: int = Field(..., ge=1, le=7)

    @field_validator("description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class RubricSpec(BaseModel):
    """Full rubric attached to a QA pair."""

    criteria: list[RubricCriterion] = Field(..., min_length=1, max_length=10)

    @property
    def total_weight(self) -> int:
        return sum(c.weight for c in self.criteria)


class QAPair(BaseModel):
    """A grounded question/answer with rubric."""

    question: str
    context: str = Field(..., description="Excerpt from the source document the QA is grounded in.")
    reference_answer: str
    rubric: RubricSpec
    paper_id: str
    chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierVerdict(BaseModel):
    """Quality Verifier output."""

    passed: bool
    feedback: str = ""
    issues: list[str] = Field(default_factory=list)


class SolverScore(BaseModel):
    """Per-sample rubric score for one solver attempt."""

    answer: str
    per_criterion: list[float]   # 0..1 per criterion
    weighted_score: float        # 0..1, weighted by criterion weights
    judge_feedback: str = ""

    @field_validator("per_criterion")
    @classmethod
    def _bounds(cls, v: list[float]) -> list[float]:
        return [max(0.0, min(1.0, float(x))) for x in v]


__all__ = [
    "QAPair",
    "RubricCriterion",
    "RubricSpec",
    "SolverScore",
    "VerifierVerdict",
]
