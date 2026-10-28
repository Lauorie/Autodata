"""Agentic Self-Instruct inner loop (arXiv 2606.25996, Sec 3.1 and Fig 7).

Repeat until ACCEPTED or the step budget runs out:

  1. Challenger writes context + question + reference answer + rubric
  2. Quality Verifier checks it; on failure -> back to 1
  3. Weak solver (N samples), graded by the judge; if the weak check fails
     -> back to 1 and the strong solver never runs
  4. Strong solver (N samples), graded; strong range + gap check
  5. ACCEPTED if every check passes

Each refinement round shows the challenger every earlier failed question of
the paper, grouped by failure mode, and asks for an entirely new question
from a different angle.

Errors (Fig 7): an infrastructure failure re-runs the evaluation of the same
question, up to ``infra_retries`` times, then stops the paper; it never
enters the history, since it says nothing about the question. A paper too
long for the model stops at once. A solver answer cut at the token budget
is the solver's failure and is scored.

The unit of work is one paper. The challenger and quality verifier read the
whole paper; the solvers see only the challenger-written context and the
question.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..data_module import Paper
from ..llm import LLMClient, QAPair, RubricSpec, SolverScore
from ..llm.client import extract_json
from ..llm.errors import ContextLengthError, OutputTruncatedError
from ..llm.prompt_set import PromptSet
from .acceptance import AcceptanceCriteria, check_strong, check_weak
from .evaluate_rubric import score_solver
from .feedback import FailedAttempt, render_history, snippet
from .quality_verifier import VerifierOutputError, verify_quality

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InnerLoopConfig:
    max_iterations: int = 15
    num_solver_samples: int = 3
    infra_retries: int = 2
    """Re-runs of a failed solver sample, and re-asks of an unusable judge or
    verifier reply, before the paper stops."""
    parallel_solvers: bool = True
    """When True, a solver's N samples run concurrently. Set False in tests
    with order-sensitive fake backends."""


@dataclass
class RoundRecord:
    """One pass through the loop (one Challenger emission).

    ``outcome`` is "accepted", "malformed", "infra_error", "paper_too_long",
    "error", or one of the failure modes in ``feedback.FAILURE_MODES``.
    """

    iteration: int
    outcome: str = ""
    qa: dict[str, Any] | None = None
    qv_verdict: dict[str, Any] | None = None
    weak_samples: list[dict[str, Any]] = field(default_factory=list)
    strong_samples: list[dict[str, Any]] = field(default_factory=list)
    acceptance: dict[str, Any] | None = None
    accepted: bool = False
    duration_s: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

class MalformedOutputError(ValueError):
    """Challenger output was not JSON, or broke the QAPair contract."""


class PaperTooLongError(RuntimeError):
    """The paper does not fit the challenger's or the verifier's context."""


# Outcomes that stop the paper: retrying them, or asking for a new question,
# would fail the same way.
_STOPS_PAPER = frozenset({"infra_error", "paper_too_long", "error"})


def _describe_validation_error(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
    )


def _challenger_step(challenger: LLMClient, paper: Paper, feedback: str,
                     prompts: PromptSet) -> QAPair:
    try:
        raw = challenger.chat([
            {"role": "system", "content": prompts.challenger_system},
            {"role": "user", "content": prompts.render_challenger(
                paper.title, paper.paper_id, paper.text, feedback)},
        ])
    except ContextLengthError as e:
        raise PaperTooLongError(f"challenger: {e}") from e
    except OutputTruncatedError as e:
        raise MalformedOutputError(
            f"your reply hit the output limit (max_tokens={e.max_tokens}) before the JSON "
            "was complete; keep the reasoning focused and the JSON compact") from e
    try:
        obj = extract_json(raw)
    except json.JSONDecodeError as e:
        raise MalformedOutputError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise MalformedOutputError(f"expected a JSON object, got {type(obj).__name__}")
    try:
        return QAPair(
            question_type=obj["question_type"],
            skill_tags=obj.get("skill_tags", []),
            context=obj["context"],
            question=obj["question"],
            reference_answer=obj["reference_answer"],
            rubric=RubricSpec(criteria=obj["rubric"]),
            paper_id=paper.paper_id,
        )
    except KeyError as e:
        raise MalformedOutputError(f"missing key {e}") from e
    except ValidationError as e:
        raise MalformedOutputError(_describe_validation_error(e)) from e


def _first_infra_error(samples: list[SolverScore]) -> str | None:
    return next((s.judge_feedback for s in samples if s.infra_error), None)


def _truncation_note(samples: list[SolverScore], solver: str) -> str:
    n = sum(s.truncated for s in samples)
    return f"; {n}/{len(samples)} {solver} attempts hit the token budget" if n else ""


def _evaluate(qa: QAPair, paper: Paper, clients: dict[str, LLMClient],
              criteria: AcceptanceCriteria, cfg: InnerLoopConfig,
              rec: RoundRecord, prompts: PromptSet) -> FailedAttempt | None:
    """Fig 7 steps 2-8 for one question. Fills ``rec``; returns the history
    entry when the question failed on its merits, else None."""
    try:
        verdict = verify_quality(clients["quality_verifier"], paper, qa, cfg.infra_retries,
                                 prompts=prompts)
    except ContextLengthError as e:
        raise PaperTooLongError(f"quality verifier: {e}") from e
    rec.qv_verdict = verdict.model_dump()
    if not verdict.passed:
        rec.outcome = "failed_qv"
        rec.acceptance = {"accepted": False, "reasons": ["quality_verifier_failed"]}
        return FailedAttempt("failed_qv", qa.question,
                             f"issues: {', '.join(verdict.issues)}; "
                             f"verifier: {snippet(verdict.feedback)}")

    weak = score_solver(clients["weak_solver"], clients["judge"], qa,
                        cfg.num_solver_samples, cfg.parallel_solvers, cfg.infra_retries,
                        prompts=prompts)
    rec.weak_samples = [s.model_dump() for s in weak]
    if (err := _first_infra_error(weak)) is not None:
        rec.outcome, rec.error = "infra_error", f"weak solver: {err}"
        return None
    wc = check_weak([s.weighted_score for s in weak], criteria)
    rec.acceptance = {"accepted": False, "reasons": wc.reasons,
                      "weak_avg": wc.weak_avg, "weak_max": wc.weak_max,
                      "strong_avg": None, "gap": None}
    if not wc.passed:
        rec.outcome = "too_easy" if wc.too_easy else "weak_zero"
        best = max(weak, key=lambda s: s.weighted_score)
        return FailedAttempt(rec.outcome, qa.question,
                             f"weak_avg={wc.weak_avg:.2f} weak_max={wc.weak_max:.2f}"
                             f"{_truncation_note(weak, 'weak')}; "
                             f"judge: {snippet(best.judge_feedback)}")

    strong = score_solver(clients["strong_solver"], clients["judge"], qa,
                          cfg.num_solver_samples, cfg.parallel_solvers, cfg.infra_retries,
                          prompts=prompts)
    rec.strong_samples = [s.model_dump() for s in strong]
    if (err := _first_infra_error(strong)) is not None:
        rec.outcome, rec.error = "infra_error", f"strong solver: {err}"
        return None
    sc = check_strong(wc.weak_avg, [s.weighted_score for s in strong], criteria)
    rec.acceptance.update(accepted=sc.passed, reasons=sc.reasons,
                          strong_avg=sc.strong_avg, gap=sc.gap)
    if sc.passed:
        rec.outcome, rec.accepted = "accepted", True
        return None
    rec.outcome = "failed_on_strong"
    worst = min(strong, key=lambda s: s.weighted_score)
    return FailedAttempt("failed_on_strong", qa.question,
                         f"strong_avg={sc.strong_avg:.2f} weak_avg={wc.weak_avg:.2f} "
                         f"gap={sc.gap:.2f} ({'; '.join(sc.reasons)})"
                         f"{_truncation_note(strong, 'strong')}; "
                         f"judge: {snippet(worst.judge_feedback)}")


def _compose_feedback(history: list[FailedAttempt], malformed_note: str) -> str:
    return "\n\n".join(p for p in (malformed_note, render_history(history)) if p)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_inner_loop_for_paper(
    paper: Paper,
    clients: dict[str, LLMClient],
    criteria: AcceptanceCriteria,
    cfg: InnerLoopConfig,
    prompts: PromptSet | None = None,
) -> tuple[QAPair | None, list[RoundRecord]]:
    """Run the Agentic Self-Instruct loop on one paper.

    ``prompts`` defaults to the seed set. Returns (accepted_QA_or_None,
    list_of_round_records).
    """
    prompts = prompts or PromptSet.seed()
    history: list[FailedAttempt] = []
    malformed_note = ""
    rounds: list[RoundRecord] = []

    for it in range(cfg.max_iterations):
        rec = RoundRecord(iteration=it)
        t0 = time.time()
        feedback = _compose_feedback(history, malformed_note)
        malformed_note = ""
        try:
            qa = _challenger_step(clients["challenger"], paper, feedback, prompts)
            rec.qa = qa.model_dump()
            failure = _evaluate(qa, paper, clients, criteria, cfg, rec, prompts)
        except MalformedOutputError as e:
            # The challenger broke the output contract (bad JSON, weight 0,
            # sign/category mismatch). Burn one round and say what was wrong.
            rec.outcome, rec.error = "malformed", f"{type(e).__name__}: {e}"
            malformed_note = (
                f"[Your previous response was malformed] {e}\n"
                "Reply ONLY with a single valid JSON object as specified, "
                "no prose, no markdown fences, no unterminated strings."
            )
        except VerifierOutputError as e:
            # No usable verdict after re-asks: nothing learned about the
            # question, so it stays out of the history.
            rec.outcome, rec.error = "infra_error", f"quality verifier: {e}"
        except PaperTooLongError as e:
            rec.outcome, rec.error = "paper_too_long", str(e)
        except Exception as e:
            # Network down after client retries, auth, other systematic
            # failures: stop the paper rather than loop on them.
            logger.exception("[%s it=%d] error in round: %s", paper.paper_id, it, e)
            rec.outcome, rec.error = "error", f"{type(e).__name__}: {e}"
        else:
            if failure is not None:
                history.append(failure)

        rec.duration_s = time.time() - t0
        rounds.append(rec)
        reasons = (rec.acceptance or {}).get("reasons") or rec.error or ""
        logger.info("[%s it=%d] %s %s", paper.paper_id, it, rec.outcome, reasons)
        if rec.accepted:
            return qa, rounds
        if rec.outcome in _STOPS_PAPER:
            break

    return None, rounds


__all__ = ["InnerLoopConfig", "MalformedOutputError", "RoundRecord", "run_inner_loop_for_paper"]
