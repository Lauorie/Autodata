"""Prompts as data: a PromptSet is plain text, rendered without exec or format()."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodata.llm.prompt_set import MUTABLE_FIELDS, PromptSet
from autodata.pipeline.quality_verifier import _QV_PASSING


def test_seed_set_is_clean() -> None:
    assert PromptSet.seed().problems() == []


def test_first_turn_carries_paper_and_task() -> None:
    p = PromptSet.seed()
    text = p.render_challenger("Title T", "id7", "PAPER BODY", feedback="")
    assert "PAPER BODY" in text and "id=id7" in text
    assert p.challenger_first in text
    assert "ENTIRELY NEW" not in text


def test_refine_turn_carries_history() -> None:
    p = PromptSet.seed()
    text = p.render_challenger("T", "id7", "BODY", feedback="PREVIOUSLY FAILED: q1")
    assert "PREVIOUSLY FAILED: q1" in text and "ENTIRELY NEW" in text
    assert p.challenger_first not in text


def test_values_with_dollars_and_braces_pass_through() -> None:
    body = "loss $\\mathcal{L}$ costs $5 and {braces} stay"
    text = PromptSet.seed().render_challenger("T", "i", body)
    assert body in text


def test_system_prompts_are_not_formatted() -> None:
    p = PromptSet.seed().evolve(challenger_system='Reply {"a": 1} and $raw')
    assert p.challenger_system == 'Reply {"a": 1} and $raw'


def test_judge_message_keeps_question_line() -> None:
    text = PromptSet.seed().render_judge("CTX", "Q?", "ANS", "[]")
    assert "QUESTION: Q?\n" in text and "CANDIDATE_ANSWER:\nANS" in text


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    p = PromptSet.seed().evolve(qv_system=PromptSet.seed().qv_system + "\nExtra rule.")
    p.save(tmp_path)
    assert PromptSet.load(tmp_path) == p


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    PromptSet.seed().save(tmp_path)
    (tmp_path / "judge_system.txt").unlink()
    with pytest.raises(FileNotFoundError):
        PromptSet.load(tmp_path)


def test_load_rejects_a_broken_set(tmp_path: Path) -> None:
    PromptSet.seed().evolve(challenger_refine="no history here").save(tmp_path)
    with pytest.raises(ValueError, match="challenger_refine"):
        PromptSet.load(tmp_path)


def test_refine_without_feedback_slot_is_a_problem() -> None:
    p = PromptSet.seed().evolve(challenger_refine="Write a new question.")
    assert any("challenger_refine" in s and "feedback" in s for s in p.problems())


def test_unknown_placeholder_is_a_problem() -> None:
    p = PromptSet.seed().evolve(challenger_refine="$feedback then $paper_text")
    assert any("paper_text" in s for s in p.problems())


def test_dropping_a_parser_key_is_a_problem() -> None:
    p = PromptSet.seed()
    p = p.evolve(qv_system=p.qv_system.replace("check_2_verdict", "check_two"))
    assert any("check_2_verdict" in s for s in p.problems())


def test_empty_prompt_is_a_problem() -> None:
    p = PromptSet.seed().evolve(challenger_first="  ")
    assert any("challenger_first" in s for s in p.problems())


@pytest.mark.parametrize("field", ["solver_system", "judge_system", "judge_user",
                                   "solver_user", "challenger_user", "qv_user"])
def test_the_yardstick_and_data_contract_are_frozen(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        PromptSet.seed().evolve(**{field: "changed"})


def test_mutable_fields_are_the_optimized_agent() -> None:
    assert set(MUTABLE_FIELDS) == {"challenger_system", "challenger_first",
                                   "challenger_refine", "qv_system"}


def test_qv_contract_matches_the_parser() -> None:
    p = PromptSet.seed()
    for key, passing in _QV_PASSING.items():
        broken = p.evolve(qv_system=p.qv_system.replace(passing, "X"))
        assert any(passing in s for s in broken.problems()), key
