"""Solver scoring under a rubric.

`score_solver` runs N samples of one solver on a (question, context) pair, then
asks the Judge to grade each sample, returning the per-sample weighted score.

CLI:
    python -m autodata.pipeline.evaluate_rubric \\
        --eval-input outputs/<run>/eval_input.json \\
        --weak-only

The CLI form mirrors the blog's `evaluate_rubric.py --weak-only / --strong-only` knobs.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..llm import LLMClient, QAPair, SolverScore
from ..llm import prompts as _prompts_module    # late-bound
from ..llm.client import extract_json

logger = logging.getLogger(__name__)


def weighted_average(per_criterion: Sequence[float], weights: Sequence[int]) -> float:
    if not per_criterion:
        return 0.0
    if len(per_criterion) != len(weights):
        # Truncate / pad-with-zeros to length-match. We treat missing scores as 0.
        n = min(len(per_criterion), len(weights))
        per_criterion = list(per_criterion[:n]) + [0.0] * (len(weights) - n)
    total_w = sum(weights) or 1
    return sum(s * w for s, w in zip(per_criterion, weights, strict=False)) / total_w


def _ask_solver(solver: LLMClient, qa: QAPair) -> str:
    return solver.chat([
        {"role": "system", "content": _prompts_module.SOLVER_SYSTEM},
        {"role": "user", "content": _prompts_module.solver_user(qa.context, qa.question)},
    ])


_JUDGE_SCORE_TOLERANCE = 0.25  # > tolerance outside [0,1] -> fail-closed


def _normalise_judge_scores(raw_scores: list[float], expected_n: int) -> list[float]:
    """Length-align to ``expected_n`` and clamp scores to [0, 1].

    Fails CLOSED (returns all zeros) if any score is grossly outside the
    allowed range — a judge that emits 2.5 or -0.4 is signalling confusion,
    and we don't want to silently turn that into a "good" 1.0 score that
    could let a bad QA through acceptance.
    """
    floats: list[float] = []
    for s in raw_scores:
        try:
            floats.append(float(s))
        except (TypeError, ValueError):
            return [0.0] * expected_n
    for s in floats:
        if s < (0.0 - _JUDGE_SCORE_TOLERANCE) or s > (1.0 + _JUDGE_SCORE_TOLERANCE):
            return [0.0] * expected_n
    clamped = [max(0.0, min(1.0, s)) for s in floats]
    if len(clamped) < expected_n:
        clamped = clamped + [0.0] * (expected_n - len(clamped))
    elif len(clamped) > expected_n:
        clamped = clamped[:expected_n]
    return clamped


def _ask_judge(judge: LLMClient, qa: QAPair, candidate: str) -> tuple[list[float], str]:
    rubric_json = json.dumps(qa.rubric.model_dump(), ensure_ascii=False)
    raw = judge.chat([
        {"role": "system", "content": _prompts_module.JUDGE_SYSTEM},
        {"role": "user", "content": _prompts_module.judge_user(
            qa.question, qa.reference_answer, candidate, rubric_json)},
    ])
    n_crit = len(qa.rubric.criteria)
    try:
        obj = extract_json(raw)
        raw_scores = obj.get("per_criterion", [])
        if not isinstance(raw_scores, list):
            raise ValueError(f"per_criterion is not a list: {type(raw_scores).__name__}")
        per = _normalise_judge_scores(raw_scores, n_crit)
        fb = str(obj.get("feedback", ""))
        return per, fb
    except Exception as e:
        logger.warning("judge parse failed: %s -- raw=%r", e, raw[:200])
        # treat parse failure as a zero score (don't credit broken evals)
        return [0.0] * n_crit, f"PARSE_FAIL: {e}"


def _score_one(solver: LLMClient, judge: LLMClient,
               qa: QAPair, weights: list[int], idx: int) -> SolverScore:
    try:
        ans = _ask_solver(solver, qa)
    except Exception as e:
        logger.warning("solver call failed sample %d: %s", idx, e)
        return SolverScore(answer=f"ERROR: {e}",
                           per_criterion=[0.0] * len(weights),
                           weighted_score=0.0,
                           judge_feedback="solver_error")
    per, fb = _ask_judge(judge, qa, ans)
    return SolverScore(
        answer=ans,
        per_criterion=per,
        weighted_score=weighted_average(per, weights),
        judge_feedback=fb,
    )


def score_solver(
    solver: LLMClient,
    judge: LLMClient,
    qa: QAPair,
    n_samples: int = 3,
    parallel: bool = True,
) -> list[SolverScore]:
    """Run `n_samples` solver attempts and grade each with the judge.

    When ``parallel`` is True the samples are issued concurrently via a
    ThreadPoolExecutor — this is safe because the proxy backend is just an
    HTTP client and the local HF backend serializes generation internally.
    """
    weights = [c.weight for c in qa.rubric.criteria]
    if not parallel or n_samples <= 1:
        return [_score_one(solver, judge, qa, weights, i) for i in range(n_samples)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_samples) as pool:
        futures = [pool.submit(_score_one, solver, judge, qa, weights, i)
                   for i in range(n_samples)]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# CLI (matches the blog's evaluate_rubric.py interface)
# ---------------------------------------------------------------------------

def _cli() -> int:
    from ..config_runtime import build_role_clients_from_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-input", required=True, help="Path to a JSON file with one or more QAPair entries.")
    ap.add_argument("--weak-only", action="store_true")
    ap.add_argument("--strong-only", action="store_true")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--models-yaml", default="conf/models/default.yaml")
    args = ap.parse_args()

    if args.weak_only and args.strong_only:
        raise SystemExit("--weak-only and --strong-only are mutually exclusive")

    clients = build_role_clients_from_yaml(args.models_yaml)
    judge = clients["judge"]

    qa_records = json.loads(Path(args.eval_input).read_text())
    if isinstance(qa_records, dict):
        qa_records = [qa_records]

    results: list[dict] = []
    for rec in qa_records:
        qa = QAPair.model_validate(rec)
        if not args.strong_only:
            weak_samples = score_solver(clients["weak_solver"], judge, qa, args.n_samples)
        else:
            weak_samples = []
        if not args.weak_only:
            strong_samples = score_solver(clients["strong_solver"], judge, qa, args.n_samples)
        else:
            strong_samples = []
        results.append({
            "qa": qa.model_dump(),
            "weak_samples": [s.model_dump() for s in weak_samples],
            "strong_samples": [s.model_dump() for s in strong_samples],
            "weak_avg": (sum(s.weighted_score for s in weak_samples) / len(weak_samples)) if weak_samples else None,
            "strong_avg": (sum(s.weighted_score for s in strong_samples) / len(strong_samples)) if strong_samples else None,
        })

    out_path = Path(args.eval_input).with_suffix(".scores.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["score_solver", "weighted_average"]
