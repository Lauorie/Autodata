"""End-of-loop quality filter (arXiv 2606.25996, Sec 3.1 Data Setup).

The paper runs the quality verifier again over the loop-accepted questions
and keeps 1.3k of 2.8k, removing "paper-specific reference leakage, short
contexts, or malformed rubrics". It does not print that prompt. This filter
does two things and fails closed:

1. Deterministic Fig 9 check-3 counts (>= 4 positive, >= 3 negative, 10-20
   total). The schema leaves counts to the verifier prompt so the
   meta-optimizer can rewrite it; this gate stays fixed and treats every data
   source alike.
2. One more quality-verifier call (Fig 9).

There is no "short context" threshold: the paper gives none. The verifier
call always uses the seed prompts, whatever set the loop ran with: the
filter is a fixed gate, outside meta-optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data_module import Paper
from ..llm import LLMClient, QAPair, RubricSpec, VerifierVerdict
from ..llm.prompt_set import PromptSet
from .quality_verifier import VerifierOutputError, verify_quality


@dataclass(frozen=True)
class RubricShape:
    """Fig 9 check 3: rubric criterion counts."""

    min_positive: int = 4
    min_negative: int = 3
    min_total: int = 10
    max_total: int = 20


def rubric_shape_issues(rubric: RubricSpec, shape: RubricShape) -> list[str]:
    total = len(rubric.criteria)
    n_pos = sum(c.category == "positive" for c in rubric.criteria)
    n_neg = total - n_pos
    issues: list[str] = []
    if n_pos < shape.min_positive:
        issues.append(f"positive criteria {n_pos} < {shape.min_positive}")
    if n_neg < shape.min_negative:
        issues.append(f"negative criteria {n_neg} < {shape.min_negative}")
    if not shape.min_total <= total <= shape.max_total:
        issues.append(f"total criteria {total} outside [{shape.min_total}, {shape.max_total}]")
    return issues


def final_filter(paper: Paper, qa: QAPair, qv: LLMClient, shape: RubricShape,
                 retries: int = 0) -> VerifierVerdict:
    issues = rubric_shape_issues(qa.rubric, shape)
    if issues:
        return VerifierVerdict(passed=False, feedback="rubric fails Fig 9 check 3 counts",
                               issues=issues)
    try:
        return verify_quality(qv, paper, qa, retries, prompts=PromptSet.seed())
    except VerifierOutputError as e:
        return VerifierVerdict(passed=False, feedback=str(e),
                               issues=["unreadable verifier reply"])


__all__ = ["RubricShape", "final_filter", "rubric_shape_issues"]
