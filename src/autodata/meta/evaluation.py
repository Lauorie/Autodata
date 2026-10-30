"""Score a prompt set: run inner-loop sessions on papers and count the passes.

A session is one inner-loop run on one paper under a round budget. It passes
when a question is accepted within the budget (Sec 4: "the fraction of
generated QA pairs that successfully separate weak and strong solvers") and
then passes the end-of-loop filter. The verifier inside the loop is one of
the prompts under optimization; the filter is not, and it decides what
reaches the dataset. A score that skipped it would reward questions the
dataset drops.

A session stopped by an infrastructure failure, or by a paper too long for
the model, says nothing about the prompts: it leaves the denominator and is
counted apart. So does a filter that could not run. A malformed challenger
reply is the prompts' failure and stays in.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..data_module import Paper
from ..llm import LLMClient, VerifierVerdict
from ..llm.prompt_set import PromptSet
from ..pipeline.acceptance import AcceptanceCriteria
from ..pipeline.final_filter import RubricShape, final_filter
from ..pipeline.inner_loop import (
    InnerLoopConfig,
    RoundRecord,
    run_inner_loop_for_paper,
    stop_reason,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Session:
    paper_id: str
    paper_title: str
    rounds: tuple[RoundRecord, ...]
    passed: bool
    stopped_by: str | None    # the infrastructure outcome that ended it early
    final_verdict: VerifierVerdict | None = None    # the filter's, on an accepted question


@dataclass(frozen=True)
class EvalResult:
    sessions: tuple[Session, ...]

    @property
    def scored(self) -> int:
        return sum(s.stopped_by is None for s in self.sessions)

    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.sessions)

    @property
    def score(self) -> float | None:
        return self.passed / self.scored if self.scored else None

    def is_valid(self, min_scored_fraction: float) -> bool:
        """Enough sessions were scored for the pass rate to mean something."""
        return self.scored > 0 and self.scored >= min_scored_fraction * len(self.sessions)


def _session(paper: Paper, prompts: PromptSet, clients: dict[str, LLMClient],
             criteria: AcceptanceCriteria, inner_cfg: InnerLoopConfig,
             shape: RubricShape) -> Session:
    try:
        qa, rounds = run_inner_loop_for_paper(paper, clients, criteria, inner_cfg, prompts)
    except Exception as e:
        # The loop handles round errors itself; anything here is a crash,
        # and a crash says nothing about the prompts either.
        logger.exception("session on %s crashed: %s", paper.paper_id, e)
        return Session(paper.paper_id, paper.title, (), False, "crash")
    if qa is None:
        return Session(paper.paper_id, paper.title, tuple(rounds), False, stop_reason(rounds))
    try:
        verdict = final_filter(paper, qa, clients["quality_verifier"], shape,
                               inner_cfg.infra_retries)
    except Exception as e:
        logger.exception("final filter on %s failed: %s", paper.paper_id, e)
        return Session(paper.paper_id, paper.title, tuple(rounds), False, "final_filter_error")
    return Session(paper.paper_id, paper.title, tuple(rounds), verdict.passed, None, verdict)


def evaluate(prompts: PromptSet, papers: Sequence[Paper], clients: dict[str, LLMClient],
             criteria: AcceptanceCriteria, inner_cfg: InnerLoopConfig, shape: RubricShape,
             samples_per_paper: int = 1, workers: int = 1) -> EvalResult:
    """Run ``samples_per_paper`` sessions on every paper, ``workers`` at a time."""
    jobs = [p for p in papers for _ in range(samples_per_paper)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        sessions = tuple(pool.map(
            lambda p: _session(p, prompts, clients, criteria, inner_cfg, shape), jobs))
    result = EvalResult(sessions)
    logger.info("evaluated %d sessions: %d passed, %d scored", len(sessions),
                result.passed, result.scored)
    return result


__all__ = ["EvalResult", "Session", "evaluate"]
