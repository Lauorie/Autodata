"""Pydantic schemas exchanged between agents.

Mirrors the data shapes in the Autodata paper (arXiv 2606.25996, Fig 8-9):
- Challenger emits QAPair (context, question, reference answer, rubric)
- Quality Verifier emits VerifierVerdict
- Solver answers a QA, scored by Judge -> SolverScore
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RubricCriterion(BaseModel):
    """One rubric item: an insight to reward or a reasoning error to penalise.

    Fig 8 baseline: positive items weigh +1..+10, negative items -1..-10.
    Item counts are a quality-verifier rule, not a schema rule.
    """

    criterion: str = Field(..., min_length=3)
    weight: int = Field(..., ge=-10, le=10)
    category: Literal["positive", "negative"]

    @field_validator("criterion")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _sign_matches_category(self) -> RubricCriterion:
        if self.weight == 0:
            raise ValueError("weight must be a nonzero integer")
        if (self.weight > 0) != (self.category == "positive"):
            raise ValueError(
                f"weight {self.weight} does not match category {self.category!r}: "
                "positive items need weight > 0, negative items weight < 0"
            )
        return self


class RubricSpec(BaseModel):
    """Full rubric attached to a QA pair."""

    criteria: list[RubricCriterion] = Field(..., min_length=1)

    @property
    def weights(self) -> list[int]:
        return [c.weight for c in self.criteria]

    def to_prompt_json(self) -> str:
        """The flat item array the challenger wrote, as shown to the QV and judge."""
        return json.dumps([c.model_dump() for c in self.criteria], ensure_ascii=False)


class QAPair(BaseModel):
    """A challenger-written research task.

    ``context`` is written by the challenger and is all the solver sees
    besides the question; it is never the paper text.
    """

    question_type: str
    skill_tags: list[str] = Field(default_factory=list)
    context: str
    question: str
    reference_answer: str
    rubric: RubricSpec
    paper_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierVerdict(BaseModel):
    """Quality Verifier output (Fig 9: four checks plus an overall verdict)."""

    passed: bool
    feedback: str = ""
    issues: list[str] = Field(default_factory=list)
    checks: dict[str, str] = Field(default_factory=dict)


class SolverScore(BaseModel):
    """Per-sample rubric score for one solver attempt."""

    answer: str
    per_criterion: list[float]   # 0..1 per criterion
    weighted_score: float        # 0..1, signed-weight rubric score
    judge_feedback: str = ""
    infra_error: bool = False    # solver failed or judge output unreadable: not a score

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
