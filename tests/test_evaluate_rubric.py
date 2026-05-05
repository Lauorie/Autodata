"""Tests for rubric scoring + JSON extraction helpers."""

from autodata.llm.client import extract_json
from autodata.pipeline.evaluate_rubric import weighted_average


def test_weighted_average_uniform() -> None:
    assert weighted_average([1.0, 0.0], [1, 1]) == 0.5


def test_weighted_average_skewed() -> None:
    # Weights 1, 4 → 0.0*1 + 1.0*4 = 4 / 5 = 0.8
    assert abs(weighted_average([0.0, 1.0], [1, 4]) - 0.8) < 1e-9


def test_weighted_average_pad_with_zeros() -> None:
    # Judge returned 1 score, rubric has 2 criteria → second treated as 0.
    out = weighted_average([1.0], [1, 1])
    assert abs(out - 0.5) < 1e-9


def test_weighted_average_empty() -> None:
    assert weighted_average([], []) == 0.0


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
