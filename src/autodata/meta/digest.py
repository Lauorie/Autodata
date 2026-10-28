"""Trajectories as text for the analyzer (arXiv 2606.25996, Sec 4 step 3).

The analyzer "reads the full solver exchanges": for every round, the
package, the verifier's verdict, each solver answer with its judge feedback,
and the per-criterion scores. Answers are clipped at ``answer_chars``; the
rest is shown whole. Rounds lost to infrastructure are marked, not shown.
"""

from __future__ import annotations

from typing import Any

from ..pipeline.inner_loop import STOP_OUTCOMES, RoundRecord
from .evaluation import EvalResult, Session


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else f"{text[:limit]} [... {len(text) - limit} more chars]"


def _per_sample(samples: list[dict[str, Any]], k: int) -> str:
    return " ".join(f"{s['per_criterion'][k]:.1f}" for s in samples) or "-"


def _rubric(rec: RoundRecord) -> list[str]:
    criteria = rec.qa["rubric"]["criteria"] if rec.qa else []
    lines = ["Rubric (weight | weak score per answer | strong score per answer | criterion):"]
    for k, c in enumerate(criteria):
        lines.append(f"  [{k + 1}] {c['weight']:+d} | weak {_per_sample(rec.weak_samples, k)}"
                     f" | strong {_per_sample(rec.strong_samples, k)} | {c['criterion']}")
    return lines


def _answers(label: str, samples: list[dict[str, Any]], answer_chars: int) -> list[str]:
    if not samples:
        return []
    lines = [f"{label} answers:"]
    for i, s in enumerate(samples, 1):
        cut = " (cut at the token budget)" if s.get("truncated") else ""
        lines.append(f"  #{i} score {s['weighted_score']:.2f}{cut}: "
                     f"{_clip(s['answer'], answer_chars)}")
        lines.append(f"     judge: {s.get('judge_feedback', '')}")
    return lines


def _verdict(rec: RoundRecord) -> str:
    v = rec.qv_verdict
    if v is None:
        return "Verifier: not reached"
    state = "PASS" if v["passed"] else f"FAIL ({', '.join(v['issues'])})"
    return f"Verifier: {state}. {v['feedback'].strip()}"


def _scores(rec: RoundRecord) -> str:
    a = rec.acceptance or {}
    nums = " ".join(f"{k}={a[k]:.2f}" for k in ("weak_avg", "weak_max", "strong_avg", "gap")
                    if a.get(k) is not None)
    return f"Scores: {nums or '-'}; {'; '.join(a.get('reasons') or []) or 'all checks pass'}"


def render_round(rec: RoundRecord, answer_chars: int) -> str:
    head = f"--- Round {rec.iteration} | {rec.outcome} ---"
    if rec.outcome in STOP_OUTCOMES:
        return f"--- Round {rec.iteration} | stopped by infrastructure (left out) ---"
    if rec.qa is None:   # malformed: the reply never became a package
        return f"{head}\n{rec.error or ''}"
    qa = rec.qa
    lines = [
        head,
        f"Type: {qa['question_type']} | tags: {', '.join(qa['skill_tags'])}",
        f"Context: {qa['context']}",
        f"Question: {qa['question']}",
        f"Reference answer: {qa['reference_answer']}",
        _verdict(rec),
    ]
    if rec.weak_samples:
        lines += [_scores(rec), *_rubric(rec)]
    lines += _answers("Weak", rec.weak_samples, answer_chars)
    lines += _answers("Strong", rec.strong_samples, answer_chars)
    return "\n".join(lines)


def _session_head(n: int, s: Session) -> str:
    state = ("PASSED" if s.passed else
             f"STOPPED ({s.stopped_by}, left out)" if s.stopped_by else "FAILED")
    return f"=== Session {n} | paper {s.paper_id} \"{s.paper_title}\" | {state} ==="


def render_sessions(result: EvalResult, answer_chars: int) -> str:
    blocks = []
    for n, s in enumerate(result.sessions, 1):
        rounds = [render_round(r, answer_chars) for r in s.rounds]
        blocks.append("\n".join([_session_head(n, s), *rounds]))
    return "\n\n".join(blocks)


__all__ = ["render_round", "render_sessions"]
