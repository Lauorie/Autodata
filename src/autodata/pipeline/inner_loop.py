"""Agentic Self-Instruct inner loop.

Reproduces the 8-step protocol from the AutoData blog:

  Repeat until ACCEPTED or steps exhausted:
    1. Call Challenger to generate QA + rubric
    2. Call Quality Verifier
    3. If QV fails -> iterate with feedback
    4. Run weak-only solver evaluation
    5. If weak fails -> iterate with feedback
    6. Run strong-only solver evaluation
    7. Check strong criteria + gap; if fails -> iterate
    8. ACCEPTED if all criteria pass

The unit of work is one paper. The challenger and quality verifier read the
whole paper; the solvers see only the challenger-written context and the
question (arXiv 2606.25996, Sec 3.1 and Fig 7-9).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..data_module import Paper
from ..llm import LLMClient, QAPair, RubricSpec, SolverScore, VerifierVerdict
from ..llm import prompts as _prompts_module  # late-bound on every call
from ..llm.client import extract_json
from .acceptance import AcceptanceCriteria, AcceptanceResult, check_acceptance
from .evaluate_rubric import score_solver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InnerLoopConfig:
    max_iterations: int = 6
    num_solver_samples: int = 3
    parallel_solvers: bool = True
    """When True, run weak + strong solvers (and their N samples) concurrently
    via thread pools. Set False in tests with order-sensitive fake backends."""


@dataclass
class RoundRecord:
    """One pass through the loop (one Challenger emission)."""

    iteration: int
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
    """Challenger output parsed as JSON but broke the QAPair contract."""


def _describe_validation_error(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
    )


def _challenger_step(challenger: LLMClient, paper: Paper, feedback: str) -> QAPair:
    raw = challenger.chat([
        {"role": "system", "content": _prompts_module.CHALLENGER_SYSTEM},
        {"role": "user", "content": _prompts_module.challenger_user(
            paper.title, paper.paper_id, paper.text, feedback)},
    ])
    obj = extract_json(raw)
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


# Fig 9 verdict fields and the value each must take for the package to pass.
_QV_PASSING = {
    "check_1_verdict": "NO_LEAKAGE",
    "check_2_verdict": "GOOD",
    "check_3_verdict": "PASS",
    "check_4_verdict": "CONSISTENT",
}


def _verdict_token(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _qv_step(qv: LLMClient, paper: Paper, qa: QAPair) -> VerifierVerdict:
    raw = qv.chat([
        {"role": "system", "content": _prompts_module.QUALITY_VERIFIER_SYSTEM},
        {"role": "user", "content": _prompts_module.quality_verifier_user(
            paper.title, paper.text, qa.question_type, qa.context, qa.question,
            qa.rubric.to_prompt_json())},
    ])
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        obj = {}
    # Fail closed: the package passes only when OVERALL is PASS *and* every
    # check holds its passing value. A malformed or self-contradicting verdict
    # must never let a bad QA pair through the gate.
    checks = {k: _verdict_token(obj.get(k)) for k in _QV_PASSING}
    issues = [f"{k}={v or 'MISSING'}" for k, v in checks.items() if v != _QV_PASSING[k]]
    overall = _verdict_token(obj.get("overall"))
    if overall != "PASS" and not issues:
        issues.append(f"overall={overall or 'MISSING'}")
    feedback = str(obj.get("feedback", ""))
    rubric_issues = obj.get("check_3_issues") or []
    if isinstance(rubric_issues, list) and rubric_issues:
        feedback += "\nRubric issues: " + "; ".join(str(x) for x in rubric_issues)
    if "check_3_counts" in obj:
        checks["check_3_counts"] = str(obj["check_3_counts"])
    return VerifierVerdict(
        passed=overall == "PASS" and not issues,
        feedback=feedback,
        issues=issues,
        checks=checks,
    )


def _weak_too_strong(weak_scores: list[float], crit: AcceptanceCriteria) -> bool:
    """Short-circuit hint: if weak already saturated, no point asking strong solver."""
    if not weak_scores:
        return False
    weak_avg = sum(weak_scores) / len(weak_scores)
    return weak_avg > crit.weak_avg_max + 0.10  # generous slop before short-circuit


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_inner_loop_for_paper(
    paper: Paper,
    clients: dict[str, LLMClient],
    criteria: AcceptanceCriteria,
    cfg: InnerLoopConfig,
) -> tuple[QAPair | None, list[RoundRecord]]:
    """Run the Agentic Self-Instruct loop on one paper.

    Returns (accepted_QA_or_None, list_of_round_records).
    """
    challenger = clients["challenger"]
    qv = clients["quality_verifier"]
    weak = clients["weak_solver"]
    strong = clients["strong_solver"]
    judge = clients["judge"]

    feedback_for_next = ""
    rounds: list[RoundRecord] = []
    accepted_qa: QAPair | None = None

    for it in range(cfg.max_iterations):
        rec = RoundRecord(iteration=it)
        t0 = time.time()
        try:
            # 1. Challenger
            qa = _challenger_step(challenger, paper, feedback_for_next)
            rec.qa = qa.model_dump()

            # 2-3. Quality Verifier
            verdict = _qv_step(qv, paper, qa)
            rec.qv_verdict = verdict.model_dump()
            if not verdict.passed:
                feedback_for_next = (
                    f"[Quality Verifier rejected] issues={verdict.issues}\n{verdict.feedback}"
                )
                rec.acceptance = {"reasons": ["quality_verifier_failed"], "accepted": False}
                rec.duration_s = time.time() - t0
                rounds.append(rec)
                logger.info("[%s it=%d] QV rejected: %s", paper.paper_id, it, verdict.issues)
                continue

            # 4-6. Weak + Strong solvers (parallel for prod; serial for tests).
            if cfg.parallel_solvers:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_weak = pool.submit(score_solver, weak, judge, qa,
                                           cfg.num_solver_samples, True)
                    fut_strong = pool.submit(score_solver, strong, judge, qa,
                                             cfg.num_solver_samples, True)
                    weak_samples = fut_weak.result()
                    strong_samples = fut_strong.result()
            else:
                weak_samples = score_solver(weak, judge, qa,
                                            cfg.num_solver_samples, parallel=False)
                strong_samples = score_solver(strong, judge, qa,
                                              cfg.num_solver_samples, parallel=False)
            rec.weak_samples = [s.model_dump() for s in weak_samples]
            rec.strong_samples = [s.model_dump() for s in strong_samples]
            weak_scores = [s.weighted_score for s in weak_samples]
            strong_scores = [s.weighted_score for s in strong_samples]

            # 7. Acceptance check
            ar = check_acceptance(weak_scores, strong_scores, verdict.passed, criteria)
            rec.acceptance = {
                "accepted": ar.accepted,
                "reasons": ar.reasons,
                "weak_avg": ar.weak_avg, "weak_max": ar.weak_max,
                "strong_avg": ar.strong_avg, "gap": ar.gap,
            }

            if ar.accepted:
                rec.accepted = True
                accepted_qa = qa
                rec.duration_s = time.time() - t0
                rounds.append(rec)
                logger.info("[%s it=%d] ACCEPTED (gap=%.2f weak=%.2f strong=%.2f)",
                            paper.paper_id, it, ar.gap, ar.weak_avg, ar.strong_avg)
                break

            feedback_for_next = _build_iteration_feedback(ar, weak_samples, strong_samples)
            rec.duration_s = time.time() - t0
            rounds.append(rec)
            logger.info("[%s it=%d] rejected: %s", paper.paper_id, it, ar.reasons)

        except (json.JSONDecodeError, MalformedOutputError) as e:
            # Unparseable JSON (challenger / verifier / judge) or a challenger
            # package that breaks the schema (e.g. weight 0, sign/category
            # mismatch). Burn one iteration and feed the problem back rather
            # than aborting the whole paper.
            logger.warning("[%s it=%d] malformed output (will retry next iter): %s",
                           paper.paper_id, it, e)
            rec.error = f"{type(e).__name__}: {e}"
            rec.duration_s = time.time() - t0
            rounds.append(rec)
            feedback_for_next = (
                f"[Your previous response was malformed] {e}\n"
                "Reply ONLY with a single valid JSON object as specified, "
                "no prose, no markdown fences, no unterminated strings."
            )
            continue
        except Exception as e:
            # Other exceptions (network, schema validation, ...) are likely
            # systematic; abort the paper so we don't loop forever.
            logger.exception("[%s it=%d] error in round: %s", paper.paper_id, it, e)
            rec.error = f"{type(e).__name__}: {e}"
            rec.duration_s = time.time() - t0
            rounds.append(rec)
            break

    return accepted_qa, rounds


def _build_iteration_feedback(ar: AcceptanceResult,
                              weak: list[SolverScore],
                              strong: list[SolverScore]) -> str:
    weak_fb = "; ".join(s.judge_feedback[:160] for s in weak[:2])
    strong_fb = "; ".join(s.judge_feedback[:160] for s in strong[:2])
    return (
        "[Acceptance check failed] reasons: " + "; ".join(ar.reasons) + "\n"
        f"weak_avg={ar.weak_avg:.2f} weak_max={ar.weak_max:.2f} "
        f"strong_avg={ar.strong_avg:.2f} gap={ar.gap:.2f}\n"
        f"Sample weak judge feedback: {weak_fb}\n"
        f"Sample strong judge feedback: {strong_fb}\n"
        "Adjust the QA so the weak solver is more likely to fail (require more "
        "paper-specific reasoning) and the strong solver still succeeds."
    )


__all__ = ["InnerLoopConfig", "MalformedOutputError", "RoundRecord", "run_inner_loop_for_paper"]
