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
from ..llm import prompts as _prompts_module  # late-bound
from ..llm.client import extract_json
from ..llm.errors import OutputTruncatedError

logger = logging.getLogger(__name__)


def rubric_score(per_criterion: Sequence[float], weights: Sequence[int]) -> float:
    """Signed-weight rubric score in [0, 1].

    Positive items add ``w * s``, negative items (``w < 0``) subtract when the
    answer makes the error. The sum is divided by the positive-weight total
    and clipped at 0: the clipped PRBench form the paper reports (Table 4).
    """
    if len(per_criterion) != len(weights):
        raise ValueError(f"{len(per_criterion)} scores for {len(weights)} rubric items")
    max_score = sum(w for w in weights if w > 0)
    if max_score == 0:
        return 0.0
    raw = sum(s * w for s, w in zip(per_criterion, weights, strict=True))
    return max(0.0, raw / max_score)


def _ask_solver(solver: LLMClient, qa: QAPair) -> tuple[str, bool]:
    """Return ``(answer, truncated)``. A reply cut at the token budget is the
    model's own failure (paper App A): keep whatever answer it reached."""
    try:
        return solver.chat([
            {"role": "system", "content": _prompts_module.SOLVER_SYSTEM},
            {"role": "user", "content": _prompts_module.solver_user(qa.context, qa.question)},
        ]), False
    except OutputTruncatedError as e:
        return e.partial, True


_JUDGE_SCORE_TOLERANCE = 0.25  # scores this far outside [0, 1] are unusable


def _normalise_judge_scores(raw_scores: list[float], expected_n: int) -> list[float]:
    """Check the judge returned one usable score per rubric item; clamp to [0, 1].

    Raises ValueError on a wrong count, a non-number, or a score grossly
    outside [0, 1]: such a judge is confused, and guessing (padding with
    zeros, truncating, zeroing) would mis-score the answer.
    """
    if len(raw_scores) != expected_n:
        raise ValueError(f"got {len(raw_scores)} scores for {expected_n} rubric items")
    floats: list[float] = []
    for s in raw_scores:
        try:
            floats.append(float(s))
        except (TypeError, ValueError) as e:
            raise ValueError(f"score {s!r} is not a number") from e
    for s in floats:
        if s < (0.0 - _JUDGE_SCORE_TOLERANCE) or s > (1.0 + _JUDGE_SCORE_TOLERANCE):
            raise ValueError(f"score {s} is outside [0, 1]")
    return [max(0.0, min(1.0, s)) for s in floats]


class JudgeOutputError(ValueError):
    """The judge never returned a usable per-criterion score list."""


def _parse_judge(raw: str, n_crit: int) -> tuple[list[float], str]:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    raw_scores = obj.get("per_criterion")
    if not isinstance(raw_scores, list):
        raise ValueError(f"per_criterion is not a list: {type(raw_scores).__name__}")
    return _normalise_judge_scores(raw_scores, n_crit), str(obj.get("feedback", ""))


def judge_answer(judge: LLMClient, qa: QAPair, candidate: str,
                 retries: int = 0) -> tuple[list[float], str]:
    """Grade one answer; re-ask up to ``retries`` times, naming the problem.

    The judge never sees the reference answer (paper Sec 3.1). A reply cut
    at ``max_tokens`` is not re-asked: the same prompt hits the same cap.
    """
    n_crit = len(qa.rubric.criteria)
    base = [
        {"role": "system", "content": _prompts_module.JUDGE_SYSTEM},
        {"role": "user", "content": _prompts_module.judge_user(
            qa.context, qa.question, candidate, qa.rubric.to_prompt_json())},
    ]
    messages = base
    for attempt in range(retries + 1):
        try:
            raw = judge.chat(messages)
        except OutputTruncatedError as e:
            raise JudgeOutputError(str(e)) from e
        try:
            return _parse_judge(raw, n_crit)
        except ValueError as e:
            logger.warning("judge reply unusable (attempt %d): %s -- raw=%r", attempt, e, raw[:200])
            last = e
            messages = base + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Your reply could not be used ({e}). Reply ONLY with the JSON object; "
                    f"per_criterion must hold exactly {n_crit} numbers in [0, 1], in rubric order."
                )},
            ]
    raise JudgeOutputError(f"no usable judge reply after {retries + 1} attempts: {last}")


_SOLVER_ERROR = "solver_error"   # judge_feedback prefix of a sample worth re-running


def _infra_failure(answer: str, n_crit: int, reason: str) -> SolverScore:
    return SolverScore(answer=answer, per_criterion=[0.0] * n_crit, weighted_score=0.0,
                       judge_feedback=reason, infra_error=True)


def _score_one(solver: LLMClient, judge: LLMClient, qa: QAPair,
               weights: list[int], idx: int, retries: int) -> SolverScore:
    # Fig 7: SOLVER_ERROR or an empty response is an infrastructure failure,
    # flagged so the loop never reads it as a low score.
    try:
        ans, truncated = _ask_solver(solver, qa)
    except Exception as e:
        logger.warning("solver call failed sample %d: %s", idx, e)
        return _infra_failure(f"ERROR: {e}", len(weights), _SOLVER_ERROR)
    if not ans.strip():
        if truncated:
            return SolverScore(answer="", per_criterion=[0.0] * len(weights), weighted_score=0.0,
                               judge_feedback="no answer: the token budget ran out first",
                               truncated=True)
        return _infra_failure(ans, len(weights), f"{_SOLVER_ERROR}: empty response")
    try:
        per, fb = judge_answer(judge, qa, ans, retries)
    except JudgeOutputError as e:
        return _infra_failure(ans, len(weights), f"PARSE_FAIL: {e}")
    return SolverScore(
        answer=ans,
        per_criterion=per,
        weighted_score=rubric_score(per, weights),
        judge_feedback=fb,
        truncated=truncated,
    )


def _run_samples(solver: LLMClient, judge: LLMClient, qa: QAPair, idxs: list[int],
                 parallel: bool, retries: int) -> list[SolverScore]:
    weights = qa.rubric.weights
    if not parallel or len(idxs) <= 1:
        return [_score_one(solver, judge, qa, weights, i, retries) for i in idxs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(idxs)) as pool:
        futures = [pool.submit(_score_one, solver, judge, qa, weights, i, retries) for i in idxs]
        return [f.result() for f in futures]


def score_solver(
    solver: LLMClient,
    judge: LLMClient,
    qa: QAPair,
    n_samples: int = 3,
    parallel: bool = True,
    retries: int = 0,
) -> list[SolverScore]:
    """Run `n_samples` solver attempts and grade each with the judge.

    Samples whose solver call failed are run again, up to ``retries`` times,
    on the same question (Fig 7: retry the eval, do not refine the question).
    A judge failure re-asks only the judge, up to ``retries`` times, and keeps
    the answer: re-running the solver would buy a new answer for nothing.

    When ``parallel`` is True the samples are issued concurrently via a
    ThreadPoolExecutor: the remote backends are HTTP clients, and the local
    HF backend serializes generation internally.
    """
    scores = _run_samples(solver, judge, qa, list(range(n_samples)), parallel, retries)
    for _ in range(retries):
        failed = [i for i, s in enumerate(scores)
                  if s.infra_error and s.judge_feedback.startswith(_SOLVER_ERROR)]
        if not failed:
            break
        for i, s in zip(failed, _run_samples(solver, judge, qa, failed, parallel, retries),
                        strict=True):
            scores[i] = s
    return scores


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
    ap.add_argument("--config-yaml", default="conf/config.yaml")
    args = ap.parse_args()

    if args.weak_only and args.strong_only:
        raise SystemExit("--weak-only and --strong-only are mutually exclusive")

    clients = build_role_clients_from_yaml(args.models_yaml, args.config_yaml)
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


__all__ = ["JudgeOutputError", "judge_answer", "rubric_score", "score_solver"]
