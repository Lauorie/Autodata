"""Regression tests for bugs flagged by the codex review.

Each test pins a single behaviour the corresponding fix introduces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from autodata.data_module.chunker import Chunk
from autodata.llm import LLMClient
from autodata.llm.client import LLMRoleConfig
from autodata.llm.schemas import QAPair, RubricCriterion, RubricSpec
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.evaluate_rubric import _normalise_judge_scores
from autodata.pipeline.inner_loop import (
    InnerLoopConfig,
    _qv_step,
    run_inner_loop_for_chunk,
)


# ---------------------------------------------------------------------------
# Codex MAJOR #4: Verifier strict-bool coercion
# ---------------------------------------------------------------------------

@dataclass
class _CannedBackend:
    response: str

    def chat(self, role_cfg, messages):
        return self.response


def _qa_for_qv() -> QAPair:
    return QAPair(
        question="q?", context="ctx", reference_answer="a",
        rubric=RubricSpec(criteria=[RubricCriterion(description="answer mentions key concept", weight=1)]),
        paper_id="p", chunk_id="p::0::0",
    )


@pytest.mark.parametrize("payload, expected", [
    ('{"passed": true}', True),
    ('{"passed": false}', False),
    ('{"passed": "false"}', False),       # bool("false") would be True; must NOT be
    ('{"passed": "no"}', False),
    ('{"passed": "true"}', True),
    ('{"passed": 0}', False),
    ('{"passed": 1}', True),
    ('{}', False),                         # missing -> False
    # Fail-closed cases that must NOT be coerced to True:
    ('{"passed": "ok"}', False),
    ('{"passed": "maybe"}', False),
    ('{"passed": 2}', False),               # nonzero non-1 number
    ('{"passed": -1}', False),
    ('{"passed": null}', False),
    ('{"passed": ["yes"]}', False),
])
def test_qv_strict_bool(payload: str, expected: bool) -> None:
    cli = LLMClient(
        LLMRoleConfig(role="quality_verifier", endpoint="proxy", model="x"),
        _CannedBackend(response=payload),
    )
    chunk = Chunk(paper_id="p", title="t", section="s",
                  text="x" * 800, chunk_id="p::0::0")
    verdict = _qv_step(cli, chunk, _qa_for_qv())
    assert verdict.passed is expected


# ---------------------------------------------------------------------------
# Codex MAJOR #5: Judge scores must be clamped + length-aligned BEFORE weighted_average
# ---------------------------------------------------------------------------

def test_normalise_clamps_minor_overshoot() -> None:
    # within tolerance -> clamped (judge rounded a bit)
    out = _normalise_judge_scores([1.1, 0.5, 1.05], expected_n=3)
    assert out == [1.0, 0.5, 1.0]


def test_normalise_clamps_minor_undershoot() -> None:
    out = _normalise_judge_scores([-0.1, 0.0, 0.5], expected_n=3)
    assert out == [0.0, 0.0, 0.5]


def test_normalise_fail_closed_on_gross_overshoot() -> None:
    # > 1.25 -> judge confused, treat the whole vector as zero (fail closed)
    out = _normalise_judge_scores([2.0, 0.5, 0.6], expected_n=3)
    assert out == [0.0, 0.0, 0.0]


def test_normalise_fail_closed_on_gross_undershoot() -> None:
    out = _normalise_judge_scores([-0.5, 0.5, 0.5], expected_n=3)
    assert out == [0.0, 0.0, 0.0]


def test_normalise_fail_closed_on_garbage() -> None:
    out = _normalise_judge_scores(["nope", 0.5], expected_n=2)
    assert out == [0.0, 0.0]


def test_normalise_pads_short() -> None:
    out = _normalise_judge_scores([0.7], expected_n=3)
    assert out == [0.7, 0.0, 0.0]


def test_normalise_truncates_long() -> None:
    out = _normalise_judge_scores([0.1, 0.2, 0.3, 0.4, 0.5], expected_n=2)
    assert out == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Codex MAJOR #1: Inner loop must dereference prompts via the prompts module
# (so a hot-swap in meta-optimization actually takes effect).
# ---------------------------------------------------------------------------

def test_inner_loop_uses_late_bound_prompts(monkeypatch) -> None:
    """When `autodata.llm.prompts.CHALLENGER_SYSTEM` is monkeypatched at
    runtime, the inner loop should pick up the new value rather than a
    snapshot taken at import time. This is the silent-meta bug from codex.
    """
    from autodata.llm import prompts as base_prompts

    captured: list[str] = []

    @dataclass
    class _CaptureBackend:
        def chat(self, role_cfg, messages):
            for m in messages:
                if m["role"] == "system":
                    captured.append(m["content"])
            # Return a minimal valid challenger payload so the loop progresses.
            if role_cfg.role == "challenger":
                return json.dumps({
                    "question": "q?", "reference_answer": "a",
                    "rubric": {"criteria": [{"description": "d", "weight": 1}]},
                })
            if role_cfg.role == "quality_verifier":
                return json.dumps({"passed": False, "feedback": "stop", "issues": []})
            return ""

    cli = lambda role: LLMClient(
        LLMRoleConfig(role=role, endpoint="proxy", model="x"), _CaptureBackend(),
    )
    clients = {r: cli(r) for r in
               ["challenger", "quality_verifier", "weak_solver", "strong_solver", "judge"]}

    monkeypatch.setattr(base_prompts, "CHALLENGER_SYSTEM", "MUTATED-CHALLENGER-SYSTEM")
    chunk = Chunk(paper_id="p", title="t", section="s",
                  text="x" * 800, chunk_id="p::0::0")
    run_inner_loop_for_chunk(
        chunk, clients, AcceptanceCriteria(),
        InnerLoopConfig(max_iterations=1, num_solver_samples=1,
                        qa_per_chunk=1, parallel_solvers=False),
    )
    # The challenger must have been called with the *new* system prompt.
    assert any("MUTATED-CHALLENGER-SYSTEM" in c for c in captured), captured
