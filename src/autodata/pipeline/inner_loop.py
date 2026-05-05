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

We do (4) before (6) to short-circuit on weak-pass / weak-too-strong before
spending strong-solver tokens.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..data_module import Chunk
from ..llm import LLMClient, QAPair, RubricSpec, SolverScore, VerifierVerdict
from ..llm import prompts as _prompts_module      # late-bound on every call
from ..llm.client import extract_json
from .acceptance import AcceptanceCriteria, AcceptanceResult, check_acceptance
from .evaluate_rubric import score_solver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InnerLoopConfig:
    max_iterations: int = 6
    num_solver_samples: int = 3
    qa_per_chunk: int = 1     # how many distinct accepted QAs we want per chunk
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

def _challenger_step(challenger: LLMClient, chunk: Chunk, feedback: str) -> QAPair:
    raw = challenger.chat([
        {"role": "system", "content": _prompts_module.CHALLENGER_SYSTEM},
        {"role": "user", "content": _prompts_module.challenger_user(
            chunk.title, chunk.paper_id, chunk.text, feedback)},
    ])
    obj = extract_json(raw)
    qa = QAPair(
        question=str(obj["question"]),
        context=chunk.text,
        reference_answer=str(obj["reference_answer"]),
        rubric=RubricSpec.model_validate(obj["rubric"]),
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        metadata={"section": chunk.section},
    )
    return qa


def _qv_step(qv: LLMClient, chunk: Chunk, qa: QAPair) -> VerifierVerdict:
    rubric_json = json.dumps(qa.rubric.model_dump(), ensure_ascii=False)
    raw = qv.chat([
        {"role": "system", "content": _prompts_module.QUALITY_VERIFIER_SYSTEM},
        {"role": "user", "content": _prompts_module.quality_verifier_user(
            chunk.title, chunk.text, qa.question, qa.reference_answer, rubric_json)},
    ])
    obj = extract_json(raw)
    raw_passed = obj.get("passed", False)
    # Strict bool. Anything that isn't unambiguously truthy is treated as False
    # (fail-closed). Reasoning: a malformed verifier response should never let a
    # bad QA pair through the gate.
    if isinstance(raw_passed, bool):
        passed = raw_passed
    elif isinstance(raw_passed, str):
        passed = raw_passed.strip().lower() in {"true", "yes"}
    elif raw_passed in (0, 1):
        passed = bool(raw_passed)
    else:
        passed = False
    return VerifierVerdict(
        passed=passed,
        feedback=str(obj.get("feedback", "")),
        issues=[str(x) for x in obj.get("issues", [])],
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

def run_inner_loop_for_chunk(
    chunk: Chunk,
    clients: dict[str, LLMClient],
    criteria: AcceptanceCriteria,
    cfg: InnerLoopConfig,
) -> tuple[QAPair | None, list[RoundRecord]]:
    """Run the Agentic Self-Instruct loop on one chunk.

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
            qa = _challenger_step(challenger, chunk, feedback_for_next)
            rec.qa = qa.model_dump()

            # 2-3. Quality Verifier
            verdict = _qv_step(qv, chunk, qa)
            rec.qv_verdict = verdict.model_dump()
            if not verdict.passed:
                feedback_for_next = (
                    f"[Quality Verifier rejected] issues={verdict.issues}\n{verdict.feedback}"
                )
                rec.acceptance = {"reasons": ["quality_verifier_failed"], "accepted": False}
                rec.duration_s = time.time() - t0
                rounds.append(rec)
                logger.info("[%s it=%d] QV rejected: %s", chunk.chunk_id, it, verdict.issues)
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
                            chunk.chunk_id, it, ar.gap, ar.weak_avg, ar.strong_avg)
                break

            feedback_for_next = _build_iteration_feedback(ar, weak_samples, strong_samples)
            rec.duration_s = time.time() - t0
            rounds.append(rec)
            logger.info("[%s it=%d] rejected: %s", chunk.chunk_id, it, ar.reasons)

        except json.JSONDecodeError as e:
            # Challenger / Verifier / Judge produced unparseable JSON. This is
            # usually transient (high-temperature challenger, occasional
            # malformed string). Burn one iteration and retry rather than
            # aborting the whole chunk.
            logger.warning("[%s it=%d] JSON parse error (will retry next iter): %s",
                           chunk.chunk_id, it, e)
            rec.error = f"JSONDecodeError: {e}"
            rec.duration_s = time.time() - t0
            rounds.append(rec)
            feedback_for_next = (
                "[Your previous response could not be parsed as JSON] "
                "Reply ONLY with a single valid JSON object as specified, "
                "no prose, no markdown fences, no unterminated strings."
            )
            continue
        except Exception as e:
            # Other exceptions (network, schema validation, ...) are likely
            # systematic; abort the chunk so we don't loop forever.
            logger.exception("[%s it=%d] error in round: %s", chunk.chunk_id, it, e)
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


__all__ = ["InnerLoopConfig", "RoundRecord", "run_inner_loop_for_chunk"]
