"""Tests for JSON extraction from model output and solver scoring."""

from autodata.llm import SolverScore
from autodata.llm.client import extract_json
from autodata.llm.errors import OutputTruncatedError
from autodata.llm.prompt_set import PromptSet
from autodata.pipeline.evaluate_rubric import score_solver
from tests.fakes import WEAK_PER, Handler, ScriptedBackend, judge_payload, make_clients, make_qa


def test_extract_json_bare() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced() -> None:
    out = extract_json('here is the answer:\n```json\n{"x": [1, 2]}\n```\nthat is it')
    assert out == {"x": [1, 2]}


def test_extract_json_trailing_comma_tolerated() -> None:
    out = extract_json('{"a": 1, "b": [2, 3,], }')
    assert out == {"a": 1, "b": [2, 3]}


def test_extract_json_array() -> None:
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_skips_unbalanced_first_brace() -> None:
    """When the first `{` is unbalanced (e.g. inside prose), keep scanning."""
    text = 'noise { incomplete... real answer follows: {"x": 1}'
    assert extract_json(text) == {"x": 1}


def test_extract_json_rejects_mismatched_close() -> None:
    """`{...]` (object closed by bracket) is not a valid balanced JSON span;
    the scanner should keep looking for a real one."""
    text = '{ "x": 1 ] then valid: {"y": 2}'
    assert extract_json(text) == {"y": 2}


def test_extract_json_braces_inside_strings() -> None:
    """A `}` inside a string literal must not close the outer object."""
    assert extract_json('{"q": "what is { in regex"}') == {"q": "what is { in regex"}


def test_extract_json_prose_with_braces() -> None:
    """The greedy `(\\{.*\\})` regex used to grab from prose's `{` all the way
    to a far-distant `}`. The balanced scanner must not."""
    text = 'I think {something} but here it is: ```json\n{"answer": 42}\n```'
    assert extract_json(text) == {"answer": 42}


# ---------------------------------------------------------------------------
# Fig 7: SOLVER_ERROR or an empty response is an infrastructure failure, not a
# score. Flag it so the loop does not read it as "the weak solver failed".
# ---------------------------------------------------------------------------

def _score(solver: Handler, judge: Handler | list[str], n_samples: int = 1,
           retries: int = 0) -> tuple[list[SolverScore], ScriptedBackend]:
    backend = ScriptedBackend(handlers={"weak_solver": solver, "judge": judge})
    clients = make_clients(backend)
    scores = score_solver(clients["weak_solver"], clients["judge"], make_qa(),
                          n_samples=n_samples, parallel=False, retries=retries,
                          prompts=PromptSet.seed())
    return scores, backend


def _score_once(solver: Handler, judge: Handler) -> SolverScore:
    return _score(solver, judge)[0][0]


def _calls(backend: ScriptedBackend, role: str) -> list[list[dict[str, str]]]:
    return [msgs for r, msgs in backend.calls if r == role]


def _boom(messages: list[dict[str, str]]) -> str:
    raise RuntimeError("proxy down")


def test_graded_answer_is_not_an_infra_error() -> None:
    s = _score_once(lambda m: "WEAK-ANSWER", lambda m: judge_payload(WEAK_PER))
    assert not s.infra_error
    assert s.weighted_score == 0.5


def test_solver_exception_is_an_infra_error() -> None:
    s = _score_once(_boom, lambda m: judge_payload(WEAK_PER))
    assert s.infra_error


def test_empty_solver_answer_is_an_infra_error() -> None:
    s = _score_once(lambda m: "   ", _boom)   # judge must not be asked
    assert s.infra_error


def test_unparseable_judge_output_is_an_infra_error() -> None:
    s = _score_once(lambda m: "WEAK-ANSWER", lambda m: "the answer looks fine")
    assert s.infra_error


# ---------------------------------------------------------------------------
# Unusable judge output is re-asked with the problem named; a truncated solver
# answer is the model's failure (paper App A), scored, not retried.
# ---------------------------------------------------------------------------

def test_out_of_range_judge_reply_is_reasked() -> None:
    bad = judge_payload([2.5] * 10)
    (score,), backend = _score(lambda m: "WEAK-ANSWER", [bad, judge_payload(WEAK_PER)], retries=1)
    assert not score.infra_error
    assert score.weighted_score == 0.5
    second = _calls(backend, "judge")[1]
    assert second[-2] == {"role": "assistant", "content": bad}
    assert "exactly 10" in second[-1]["content"]


def test_unusable_judge_reply_becomes_infra_after_retries() -> None:
    (score,), backend = _score(lambda m: "WEAK-ANSWER", lambda m: "no json here", retries=2)
    assert score.infra_error
    assert len(_calls(backend, "judge")) == 3


def test_truncated_judge_reply_is_not_reasked() -> None:
    def cut(messages: list[dict[str, str]]) -> str:
        raise OutputTruncatedError("judge", 4096, '{"per_criterion": [1.0, 0.5')

    (score,), backend = _score(lambda m: "WEAK-ANSWER", cut, retries=2)
    assert score.infra_error
    assert len(_calls(backend, "judge")) == 1


def test_answer_lost_to_thinking_budget_scores_zero_unjudged() -> None:
    def thought_too_long(messages: list[dict[str, str]]) -> str:
        raise OutputTruncatedError("weak_solver", 65536, "")

    (score,), backend = _score(thought_too_long, _boom)
    assert score.truncated and not score.infra_error
    assert score.weighted_score == 0.0
    assert _calls(backend, "judge") == []


def test_partial_answer_is_judged_and_flagged() -> None:
    def cut(messages: list[dict[str, str]]) -> str:
        raise OutputTruncatedError("weak_solver", 65536, "the first half of an answer")

    (score,), backend = _score(cut, lambda m: judge_payload(WEAK_PER))
    assert score.truncated and not score.infra_error
    assert score.weighted_score == 0.5
    assert "the first half of an answer" in _calls(backend, "judge")[0][-1]["content"]


def test_only_failed_samples_are_rescored() -> None:
    calls = {"n": 0}

    def flaky(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset")
        return "WEAK-ANSWER"

    scores, _ = _score(flaky, lambda m: judge_payload(WEAK_PER), n_samples=2, retries=1)
    assert [s.infra_error for s in scores] == [False, False]
    assert calls["n"] == 3
