"""End-to-end inner-loop tests with scripted LLM backends (no network).

Pins the paper's data contract (arXiv 2606.25996, Sec 3.1 and Fig 7-9):
- the challenger and quality verifier read the whole paper;
- the solvers see only the challenger-written context and the question;
- the judge grades without the reference answer;
- malformed challenger output costs one round, not the whole paper.

and the Fig 7 loop control:
- the strong solver runs only after the weak solver fails the question;
- refinement rounds list every earlier failed question, grouped by failure
  mode, and ask for an entirely new question from a different angle;
- solver infrastructure errors never enter that history.

and the Fig 7 error policy:
- an infrastructure failure re-runs the evaluation of the same question;
  when retries run out, the paper stops (no new question is asked for);
- a paper too long for the model stops at once;
- an answer cut at the token budget is the solver's failure, and the
  history says so.
"""

from __future__ import annotations

from autodata.llm.errors import ContextLengthError, OutputTruncatedError
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_paper
from tests.fakes import (
    RUBRIC,
    STRONG_LOW_PER,
    STRONG_PER,
    WEAK_HIGH_PER,
    WEAK_PER,
    WEAK_ZERO_PER,
    ScriptedBackend,
    challenger_payload,
    judge_by_question,
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
    assert rounds[0].outcome == "accepted"
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
    assert [r.outcome for r in rounds] == ["failed_on_strong"] * 2


def _challenger_prompts(backend: ScriptedBackend) -> list[str]:
    return [msgs[-1]["content"] for role, msgs in backend.calls if role == "challenger"]


def _run(backend: ScriptedBackend, max_iterations: int = 3):
    return run_inner_loop_for_paper(make_paper(), make_clients(backend),
                                    AcceptanceCriteria(), _cfg(max_iterations))


def test_first_round_asks_for_a_question_without_history() -> None:
    backend = _backend()
    _run(backend)
    first = _challenger_prompts(backend)[0]
    assert "Generate a challenging research question-answer pair with grading rubrics" in first
    assert "PREVIOUSLY FAILED" not in first


def test_too_easy_question_skips_strong_solver() -> None:
    backend = _backend(judge=judge_by_solver(WEAK_HIGH_PER, STRONG_PER))
    qa, rounds = _run(backend, max_iterations=1)
    assert qa is None
    assert rounds[0].outcome == "too_easy"
    assert backend.text_seen_by("strong_solver") == ""
    assert rounds[0].strong_samples == []
    assert rounds[0].acceptance["gap"] is None


def test_refinement_groups_failed_questions_by_mode() -> None:
    questions = ["Q-leaky", "Q-easy", "Q-hard", "Q-good"]
    backend = _backend(
        challenger=[challenger_payload(question=q) for q in questions],
        quality_verifier=[
            qv_payload("FAIL", check_1_verdict="LEAKS_ANSWER", feedback="context gives it away"),
            qv_payload(), qv_payload(), qv_payload(),
        ],
        judge=judge_by_question({
            "Q-easy": (WEAK_HIGH_PER, STRONG_PER),
            "Q-hard": (WEAK_PER, STRONG_LOW_PER),
            "Q-good": (WEAK_PER, STRONG_PER),
        }),
    )
    qa, rounds = _run(backend, max_iterations=4)
    assert qa is not None and qa.question == "Q-good"
    assert [r.outcome for r in rounds] == ["failed_qv", "too_easy", "failed_on_strong", "accepted"]
    # Strong ran on Q-hard and Q-good only: 2 rounds x 2 samples.
    assert sum(role == "strong_solver" for role, _ in backend.calls) == 4

    last = _challenger_prompts(backend)[-1]
    order = ["TOO EASY", "Q-easy", "weak_avg=1.00",
             "FAILED ON STRONG", "Q-hard", "gap=0.00",
             "FAILED QV", "Q-leaky", "LEAKS_ANSWER"]
    positions = [last.find(token) for token in order]
    assert -1 not in positions, dict(zip(order, positions, strict=True))
    assert positions == sorted(positions)
    assert "ENTIRELY NEW question from a DIFFERENT angle" in last
    assert "Adjust" not in last


def test_weak_zero_is_not_reported_as_too_easy() -> None:
    backend = _backend(
        challenger=[challenger_payload(question=q) for q in ("Q-broken", "Q-good")],
        judge=judge_by_question({
            "Q-broken": (WEAK_ZERO_PER, STRONG_PER),
            "Q-good": (WEAK_PER, STRONG_PER),
        }),
    )
    qa, rounds = _run(backend)
    assert qa is not None
    assert rounds[0].outcome == "weak_zero"
    second = _challenger_prompts(backend)[1]
    assert "WEAK SCORED ZERO" in second and "Q-broken" in second
    assert "TOO EASY" not in second


def test_malformed_round_keeps_failed_history() -> None:
    bad_rubric = [dict(RUBRIC[0], weight=0)] + RUBRIC[1:]
    backend = _backend(
        challenger=[challenger_payload(question="Q-easy"),
                    challenger_payload(rubric=bad_rubric),
                    challenger_payload(question="Q-good")],
        judge=judge_by_question({
            "Q-easy": (WEAK_HIGH_PER, STRONG_PER),
            "Q-good": (WEAK_PER, STRONG_PER),
        }),
    )
    qa, rounds = _run(backend)
    assert qa is not None
    assert [r.outcome for r in rounds] == ["too_easy", "malformed", "accepted"]
    third = _challenger_prompts(backend)[2]
    assert "malformed" in third
    assert "TOO EASY" in third and "Q-easy" in third


def test_transient_solver_failure_rescores_the_same_question() -> None:
    calls = {"n": 0}

    def flaky_weak(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset")
        return "WEAK-ANSWER"

    backend = _backend(weak_solver=flaky_weak)
    qa, rounds = _run(backend)
    assert qa is not None
    assert [r.outcome for r in rounds] == ["accepted"]
    assert len(_challenger_prompts(backend)) == 1


def test_persistent_solver_failure_stops_the_paper() -> None:
    def broken_weak(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("weak solver crashed")

    backend = _backend(weak_solver=broken_weak)
    qa, rounds = _run(backend)
    assert qa is None
    assert [r.outcome for r in rounds] == ["infra_error"]
    assert len(_challenger_prompts(backend)) == 1     # no new question asked for
    assert backend.text_seen_by("strong_solver") == ""


def test_paper_too_long_stops_at_once() -> None:
    def too_long(messages: list[dict[str, str]]) -> str:
        raise ContextLengthError("maximum context length is 262144 tokens")

    backend = _backend(challenger=too_long)
    qa, rounds = _run(backend)
    assert qa is None
    assert [r.outcome for r in rounds] == ["paper_too_long"]


def test_challenger_cut_at_token_limit_costs_one_round() -> None:
    def cut(messages: list[dict[str, str]]) -> str:
        raise OutputTruncatedError("challenger", 32768, '{"question_type": "fail')

    calls = {"n": 0}

    def first_cut(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        return cut(messages) if calls["n"] == 1 else challenger_payload()

    backend = _backend(challenger=first_cut)
    qa, rounds = _run(backend)
    assert qa is not None
    assert [r.outcome for r in rounds] == ["malformed", "accepted"]
    assert "max_tokens=32768" in _challenger_prompts(backend)[1]


def test_truncated_weak_attempts_are_named_in_history() -> None:
    def weak(messages: list[dict[str, str]]) -> str:
        if "Q-long" in messages[-1]["content"]:
            raise OutputTruncatedError("weak_solver", 65536, "")
        return "WEAK-ANSWER"

    backend = _backend(
        challenger=[challenger_payload(question=q) for q in ("Q-long", "Q-good")],
        weak_solver=weak,
        judge=judge_by_question({"Q-good": (WEAK_PER, STRONG_PER)}),
    )
    qa, rounds = _run(backend)
    assert qa is not None
    assert [r.outcome for r in rounds] == ["weak_zero", "accepted"]
    assert all(s["truncated"] for s in rounds[0].weak_samples)
    assert "2/2 weak attempts hit the token budget" in _challenger_prompts(backend)[1]


def test_unreadable_verifier_reply_is_reasked() -> None:
    backend = _backend(quality_verifier=["I think it is fine.", qv_payload()])
    qa, rounds = _run(backend)
    assert qa is not None
    assert [r.outcome for r in rounds] == ["accepted"]
    qv_calls = [msgs for role, msgs in backend.calls if role == "quality_verifier"]
    assert qv_calls[1][-2] == {"role": "assistant", "content": "I think it is fine."}


def test_persistently_unreadable_verifier_stops_the_paper() -> None:
    backend = _backend(quality_verifier=lambda m: "I think it is fine.")
    qa, rounds = _run(backend)
    assert qa is None
    assert [r.outcome for r in rounds] == ["infra_error"]
    assert len(_challenger_prompts(backend)) == 1


def test_verifier_cut_at_token_limit_stops_the_paper() -> None:
    def cut(messages: list[dict[str, str]]) -> str:
        raise OutputTruncatedError("quality_verifier", 4096, '{"check_1_verdict": "NO_')

    backend = _backend(quality_verifier=cut)
    qa, rounds = _run(backend)
    assert [r.outcome for r in rounds] == ["infra_error"]
    assert "max_tokens=4096" in rounds[0].error
    assert sum(role == "quality_verifier" for role, _ in backend.calls) == 1
