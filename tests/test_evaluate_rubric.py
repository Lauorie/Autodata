"""Tests for JSON extraction from model output and solver scoring."""

from autodata.llm import SolverScore
from autodata.llm.client import extract_json
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

def _score_once(solver: Handler, judge: Handler) -> SolverScore:
    backend = ScriptedBackend(handlers={"weak_solver": solver, "judge": judge})
    clients = make_clients(backend)
    return score_solver(clients["weak_solver"], clients["judge"], make_qa(),
                        n_samples=1, parallel=False)[0]


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
