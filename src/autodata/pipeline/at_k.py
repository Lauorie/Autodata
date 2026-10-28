"""Held-out evaluation: mean@k and best@k (arXiv 2606.25996, Sec 3.1.2, Table 2).

Each test question is answered k times through ``score_solver``, the same
path the loop uses: the solver's role config carries the shared
``solver_generation`` settings, the judge grades the answer without the
reference, and an answer cut at the token budget is scored as the model's
failure. A question with a sample lost to infrastructure is left out and
counted, never scored as 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..llm import LLMClient, QAPair, SolverScore
from ..llm.prompt_set import PromptSet
from .evaluate_rubric import score_solver


@dataclass(frozen=True)
class AtK:
    n: int                      # questions with every sample scored
    left_out: int               # questions with a sample lost to infrastructure
    mean_at_k: float | None     # mean over questions of the mean sample score
    best_at_k: float | None     # mean over questions of the best sample score
    truncation_rate: float      # scored samples cut at the token budget


def summarize_at_k(per_qa: Sequence[Sequence[SolverScore]]) -> AtK:
    complete = [s for s in per_qa if s and not any(x.infra_error for x in s)]
    samples = [x for s in complete for x in s]
    n = len(complete)
    return AtK(
        n=n,
        left_out=len(per_qa) - n,
        mean_at_k=(sum(sum(x.weighted_score for x in s) / len(s) for s in complete) / n
                   if n else None),
        best_at_k=sum(max(x.weighted_score for x in s) for s in complete) / n if n else None,
        truncation_rate=sum(x.truncated for x in samples) / len(samples) if samples else 0.0,
    )


def score_test_set(qas: Sequence[QAPair], solver: LLMClient, judge: LLMClient, k: int,
                   retries: int, workers: int, prompts: PromptSet) -> list[list[SolverScore]]:
    """k graded samples per question, ``workers`` questions at a time."""
    def one(qa: QAPair) -> list[SolverScore]:
        return score_solver(solver, judge, qa, n_samples=k, parallel=True, retries=retries,
                            prompts=prompts)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, qas))


__all__ = ["AtK", "score_test_set", "summarize_at_k"]
