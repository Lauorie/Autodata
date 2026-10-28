"""End-to-end inner-loop tests with scripted LLM backends (no network).

Pins the paper's data contract (arXiv 2606.25996, Sec 3.1 and Fig 7-9):
- the challenger and quality verifier read the whole paper;
- the solvers see only the challenger-written context and the question;
- the judge grades without the reference answer;
- malformed challenger output costs one round, not the whole paper.
"""

from __future__ import annotations

from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_paper
from tests.fakes import (
    RUBRIC,
    STRONG_LOW_PER,
    STRONG_PER,
    WEAK_PER,
    ScriptedBackend,
    challenger_payload,
    judge_by_solver,
    make_clients,
    make_paper,
    qv_payload,
)


def _cfg(max_iterations: int = 3) -> InnerLoopConfig:
    return InnerLoopConfig(max_iterations=max_iterations, num_solver_samples=2,
                           parallel_solvers=False)


def _backend(**handlers) -> ScriptedBackend:
    base = {
        "challenger": lambda m: challenger_payload(),
        "quality_verifier": lambda m: qv_payload(),
        "weak_solver": lambda m: "WEAK-ANSWER",
        "strong_solver": lambda m: "STRONG-ANSWER",
        "judge": judge_by_solver(WEAK_PER, STRONG_PER),
    }
    base.update(handlers)
    return ScriptedBackend(handlers=base)


def test_accepts_discriminative_question() -> None:
    backend = _backend()
    qa, rounds = run_inner_loop_for_paper(make_paper(), make_clients(backend),
                                          AcceptanceCriteria(), _cfg())
    assert qa is not None
    assert qa.paper_id == "papX"
    assert qa.question_type == "failure mode prediction"
    assert qa.skill_tags == ["causal_reasoning", "counterfactual"]
    assert len(rounds) == 1 and rounds[0].accepted
    assert rounds[0].acceptance["weak_avg"] == 0.5
    assert abs(rounds[0].acceptance["strong_avg"] - 16 / 18) < 1e-9


def test_qv_failure_iterates() -> None:
    backend = _backend(quality_verifier=[
        qv_payload("FAIL", check_1_verdict="LEAKS_ANSWER", feedback="context gives it away"),
        qv_payload(),
    ])
    qa, rounds = run_inner_loop_for_paper(make_paper(), make_clients(backend),
                                          AcceptanceCriteria(), _cfg())
    assert qa is not None
    assert [r.accepted for r in rounds] == [False, True]
    assert rounds[0].qv_verdict["passed"] is False


def test_exhausts_iterations() -> None:
    backend = _backend(judge=judge_by_solver(WEAK_PER, STRONG_LOW_PER))
    qa, rounds = run_inner_loop_for_paper(make_paper(), make_clients(backend),
                                          AcceptanceCriteria(), _cfg(max_iterations=2))
    assert qa is None
    assert len(rounds) == 2
    assert not any(r.accepted for r in rounds)


def test_challenger_and_verifier_read_the_whole_paper() -> None:
    tail = "TAIL-OF-PAPER-SENTINEL"
    paper = make_paper("# Long paper\n\n" + "filler sentence. " * 4000 + tail)
    backend = _backend()
    run_inner_loop_for_paper(paper, make_clients(backend), AcceptanceCriteria(), _cfg())
    assert tail in backend.text_seen_by("challenger")
    assert tail in backend.text_seen_by("quality_verifier")


def test_solvers_see_context_not_paper() -> None:
    paper_marker = "PAPER-BODY-SENTINEL"
    context = "CHALLENGER-CONTEXT-SENTINEL: a team distils a large model."
    paper = make_paper(f"# Paper X\n\n{paper_marker} " + "method details. " * 200)
    backend = _backend(challenger=lambda m: challenger_payload(context=context))
    run_inner_loop_for_paper(paper, make_clients(backend), AcceptanceCriteria(), _cfg())
    for solver in ("weak_solver", "strong_solver"):
        seen = backend.text_seen_by(solver)
        assert context in seen
        assert "Which training change would remove the mismatch" in seen
        assert paper_marker not in seen


def test_judge_never_sees_reference_answer() -> None:
    reference = "REFERENCE-ANSWER-SENTINEL"
    backend = _backend(challenger=lambda m: challenger_payload(reference_answer=reference))
    run_inner_loop_for_paper(make_paper(), make_clients(backend), AcceptanceCriteria(), _cfg())
    judged = backend.text_seen_by("judge")
    assert "STRONG-ANSWER" in judged
    assert reference not in judged


def test_malformed_rubric_costs_one_round() -> None:
    bad_rubric = [dict(RUBRIC[0], weight=0)] + RUBRIC[1:]
    backend = _backend(challenger=[challenger_payload(rubric=bad_rubric), challenger_payload()])
    qa, rounds = run_inner_loop_for_paper(make_paper(), make_clients(backend),
                                          AcceptanceCriteria(), _cfg())
    assert qa is not None
    assert len(rounds) == 2
    assert rounds[0].error is not None and not rounds[0].accepted
    # The second challenger call carries the validation error back as feedback.
    challenger_calls = [msgs for role, msgs in backend.calls if role == "challenger"]
    assert "weight" in challenger_calls[1][-1]["content"]
