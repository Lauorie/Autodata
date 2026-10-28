"""Scoring a prompt set: sessions, passes, and what counts against the prompts.

Sec 4 scores a candidate by its validation pass rate. A session stopped by
infrastructure says nothing about the prompts and leaves the denominator;
a malformed challenger reply is the prompts' failure and stays in.
"""

from __future__ import annotations

from typing import Any

from autodata.llm.prompt_set import PromptSet
from autodata.meta.digest import render_sessions
from autodata.meta.evaluation import EvalResult, Session, evaluate
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.final_filter import RubricShape
from autodata.pipeline.inner_loop import InnerLoopConfig
from tests.fakes import (
    STRONG_PER,
    WEAK_PER,
    ScriptedBackend,
    challenger_payload,
    judge_by_solver,
    make_clients,
    make_paper,
    qv_payload,
)

ONE_ROUND = InnerLoopConfig(max_iterations=1, num_solver_samples=2, parallel_solvers=False)


def _backend(**handlers: Any) -> ScriptedBackend:
    base: dict[str, Any] = {
        "challenger": lambda m: challenger_payload(),
        "quality_verifier": lambda m: qv_payload(),
        "weak_solver": lambda m: "WEAK-ANSWER",
        "strong_solver": lambda m: "STRONG-ANSWER",
        "judge": judge_by_solver(WEAK_PER, STRONG_PER),
    }
    base.update(handlers)
    return ScriptedBackend(handlers=base)


def _run(backend: ScriptedBackend, n_papers: int = 1, samples: int = 1,
         prompts: PromptSet | None = None, shape: RubricShape | None = None) -> EvalResult:
    papers = [make_paper(paper_id=f"p{i}") for i in range(n_papers)]
    return evaluate(prompts or PromptSet.seed(), papers, make_clients(backend),
                    AcceptanceCriteria(), ONE_ROUND, shape or RubricShape(),
                    samples_per_paper=samples, workers=1)


def _session(passed: bool, stopped_by: str | None = None) -> Session:
    return Session(paper_id="p", paper_title="P", rounds=(), passed=passed,
                   stopped_by=stopped_by)


# -- the score -----------------------------------------------------------------

def test_infra_stops_leave_the_denominator() -> None:
    result = EvalResult((_session(True), _session(False), _session(False, "infra_error")))
    assert (result.passed, result.scored) == (1, 2)
    assert result.score == 0.5


def test_too_many_infra_stops_void_the_evaluation() -> None:
    result = EvalResult((_session(True), _session(False), _session(False, "error")))
    assert not result.is_valid(min_scored_fraction=0.8)
    assert result.is_valid(min_scored_fraction=0.6)


def test_nothing_scored_is_never_valid() -> None:
    result = EvalResult((_session(False, "error"),))
    assert result.score is None
    assert not result.is_valid(min_scored_fraction=0.0)


# -- sessions ------------------------------------------------------------------

def test_accepted_question_passes_the_session() -> None:
    result = _run(_backend())
    assert result.score == 1.0
    assert result.sessions[0].stopped_by is None


# -- the fixed end-of-loop filter -----------------------------------------------
# The verifier inside the loop is one of the prompts being optimized. The
# filter that decides what reaches the dataset is not: an accepted question
# that it drops is a failed session, or the score rewards questions the
# dataset never keeps.

def test_accepted_question_must_pass_the_final_filter() -> None:
    result = _run(_backend(quality_verifier=[qv_payload(), qv_payload("FAIL")]))
    session = result.sessions[0]
    assert [r.outcome for r in session.rounds] == ["accepted"]
    assert session.final_verdict is not None and not session.final_verdict.passed
    assert (session.passed, session.stopped_by, result.score) == (False, None, 0.0)


def test_final_filter_reads_the_seed_verifier_prompt() -> None:
    seed = PromptSet.seed()
    backend = _backend()
    _run(backend, prompts=seed.evolve(qv_system=seed.qv_system + "\nVARIANT-QV"))
    systems = [m[0]["content"] for r, m in backend.calls if r == "quality_verifier"]
    assert ["VARIANT-QV" in s for s in systems] == [True, False]


def test_final_filter_applies_the_configured_rubric_shape() -> None:
    backend = _backend()
    result = _run(backend, shape=RubricShape(min_negative=4))   # the fake rubric has 3
    assert not result.sessions[0].passed
    assert sum(r == "quality_verifier" for r, _ in backend.calls) == 1


def test_final_filter_outage_leaves_the_session_out() -> None:
    def down(_m: Any) -> str:
        raise RuntimeError("gateway down")

    replies = iter([lambda m: qv_payload(), down])
    result = _run(_backend(quality_verifier=lambda m: next(replies)(m)))
    assert result.sessions[0].stopped_by == "final_filter_error"
    assert result.score is None


def test_gateway_failure_is_an_infra_stop() -> None:
    def down(_m: Any) -> str:
        raise RuntimeError("gateway down")

    result = _run(_backend(challenger=down))
    assert result.sessions[0].stopped_by == "error"
    assert result.score is None


def test_malformed_reply_counts_against_the_prompts() -> None:
    result = _run(_backend(challenger=lambda m: "not json at all"))
    assert result.sessions[0].stopped_by is None
    assert result.score == 0.0


def test_each_paper_runs_several_sessions() -> None:
    result = _run(_backend(quality_verifier=lambda m: qv_payload("FAIL")), n_papers=2, samples=3)
    assert [s.paper_id for s in result.sessions] == ["p0"] * 3 + ["p1"] * 3


def test_sessions_use_the_variant_prompts() -> None:
    backend = _backend(quality_verifier=lambda m: qv_payload("FAIL"))
    _run(backend, prompts=PromptSet.seed().evolve(challenger_system="VARIANT-SYSTEM"))
    assert "VARIANT-SYSTEM" in backend.text_seen_by("challenger")


# -- what the analyzer reads ---------------------------------------------------

def test_digest_shows_answers_scores_and_feedback() -> None:
    text = render_sessions(_run(_backend()), answer_chars=500)
    assert "WEAK-ANSWER" in text and "STRONG-ANSWER" in text
    assert "identifies the variance source in the estimator" in text
    assert "0.5 0.5" in text and "1.0 1.0" in text     # per-criterion, per sample
    assert "graded" in text                            # judge feedback
    assert "accepted" in text


def test_digest_shows_the_final_filter_verdict() -> None:
    fail = qv_payload("FAIL", feedback="FILTER-SAYS-WHY")
    text = render_sessions(_run(_backend(quality_verifier=[qv_payload(), fail])),
                           answer_chars=300)
    assert "final filter" in text.lower() and "FILTER-SAYS-WHY" in text


def test_digest_clips_long_answers() -> None:
    long_answer = "z" * 5000
    text = render_sessions(_run(_backend(weak_solver=lambda m: long_answer)), answer_chars=300)
    assert "z" * 300 in text and "z" * 301 not in text
    assert "4700 more chars" in text


def test_digest_marks_infra_rounds_without_detail() -> None:
    # An outage says nothing about the prompts: the analyzer sees that a
    # round was lost, not the stack trace.
    def down(_m: Any) -> str:
        raise RuntimeError("gateway down")

    text = render_sessions(_run(_backend(challenger=down)), answer_chars=300)
    assert "infrastructure" in text and "gateway down" not in text
