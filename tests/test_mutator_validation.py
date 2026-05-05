"""Tests for the meta-optimizer's mutated-source validator."""

from autodata.meta.mutator import summarize_failure_patterns, validate_mutated_source

GOOD_SRC = """
CHALLENGER_SYSTEM = "x"
QUALITY_VERIFIER_SYSTEM = "x"
SOLVER_SYSTEM = "x"
JUDGE_SYSTEM = "x"

def challenger_user(paper_title, paper_id, chunk_text, feedback=""): return ""
def quality_verifier_user(paper_title, chunk_text, question, answer, rubric_json): return ""
def solver_user(chunk_text, question): return ""
def judge_user(question, reference_answer, candidate_answer, rubric_json): return ""
"""


def test_validates_good_source() -> None:
    ok, msg = validate_mutated_source(GOOD_SRC)
    assert ok, msg


def test_catches_missing_symbol() -> None:
    bad = GOOD_SRC.replace("def judge_user", "def grader_user")
    ok, msg = validate_mutated_source(bad)
    assert not ok
    assert "judge_user" in msg


def test_catches_syntax_error() -> None:
    bad = GOOD_SRC + "\ndef oops(:\n"
    ok, msg = validate_mutated_source(bad)
    assert not ok
    assert "syntax" in msg


def test_failure_summary_aggregates() -> None:
    rejs = [
        ["weak_avg 0.78 > 0.65", "gap 0.10 < 0.20"],
        ["weak_avg 0.70 > 0.65"],
        ["strong_avg 0.50 < 0.60"],
    ]
    fbs = ["judge said too vague", "judge said off-topic"]
    summary = summarize_failure_patterns(rejs, fbs)
    assert "weak_avg_too_high" in summary
    assert "gap_too_small" in summary
    assert "strong_avg_too_low" in summary
