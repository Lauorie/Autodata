"""The analyzer and the implementer of the outer loop (arXiv 2606.25996, Sec 4).

The analyzer reads trajectories and writes a root-cause analysis; the
implementer rewrites the mutable prompts. Neither is told what the paper's
optimizer found: those findings are the result to reproduce, not a hint.
"""

from __future__ import annotations

from typing import Any

import pytest

from autodata.llm import LLMClient, LLMRoleConfig
from autodata.llm.prompt_set import PromptSet
from autodata.meta.analyzer import analyze, analyzer_system
from autodata.meta.implementer import ImplementerError, implementer_system, propose
from autodata.pipeline.acceptance import AcceptanceCriteria
from tests.fakes import ScriptedBackend

SEED = PromptSet.seed()


def _meta(*replies: str) -> tuple[LLMClient, ScriptedBackend]:
    backend = ScriptedBackend(handlers={"meta_optimizer": list(replies)})
    role = LLMRoleConfig(role="meta_optimizer", endpoint="proxy", model="fake")
    return LLMClient(role, backend), backend


def _reply(**fields: str) -> str:
    blocks = [f"<<<{k}>>>\n{v}\n<<<end>>>" for k, v in fields.items()]
    return "\n".join(blocks)


def _propose(client: LLMClient, max_chars: int = 50_000, retries: int = 1) -> Any:
    return propose(client, SEED, analysis="ANALYSIS-TEXT", history="LOG-TEXT",
                   criteria=AcceptanceCriteria(), max_prompt_chars=max_chars, retries=retries)


# -- no hints ------------------------------------------------------------------

# Sec 4's four discoveries, in the words a hint would use.
_FINDINGS = ("paper-specific", "generic", "leak", "positive-only", "capped", "cap ",
             "self-test", "parsing error", "string weight")


@pytest.mark.parametrize("system", [analyzer_system(AcceptanceCriteria()),
                                    implementer_system(AcceptanceCriteria(), 20_000)])
def test_meta_prompts_do_not_hint_at_the_papers_findings(system: str) -> None:
    # NO_LEAKAGE is a verdict value the parser reads (Fig 9 check 1), not advice.
    lowered = system.lower().replace("no_leakage", "")
    assert [w for w in _FINDINGS if w in lowered] == []


def test_meta_prompts_state_the_configured_criteria() -> None:
    criteria = AcceptanceCriteria(weak_avg_max=0.55, gap_min=0.3)
    for system in (analyzer_system(criteria), implementer_system(criteria, 20_000)):
        assert "0.55" in system and "0.3" in system


# -- analyzer ------------------------------------------------------------------

def test_analyzer_reads_trajectories_and_the_log() -> None:
    client, backend = _meta("ROOT CAUSES")
    out = analyze(client, sessions="TRAJECTORY-DUMP", history="LOG-TEXT",
                  criteria=AcceptanceCriteria())
    assert out == "ROOT CAUSES"
    seen = backend.text_seen_by("meta_optimizer")
    assert "TRAJECTORY-DUMP" in seen and "LOG-TEXT" in seen


# -- implementer ---------------------------------------------------------------

def test_implementer_rewrites_only_the_fields_it_returns() -> None:
    new = SEED.challenger_system + "\nOne more rule."
    client, backend = _meta(_reply(rationale="why", challenger_system=new))
    proposal = _propose(client)
    assert proposal.prompts.challenger_system == new
    assert proposal.prompts.qv_system == SEED.qv_system
    assert proposal.changed == ("challenger_system",)
    assert proposal.rationale == "why"
    seen = backend.text_seen_by("meta_optimizer")
    assert "ANALYSIS-TEXT" in seen and "LOG-TEXT" in seen and SEED.qv_system in seen


def test_implementer_never_sees_the_frozen_prompts() -> None:
    client, backend = _meta(_reply(rationale="r", challenger_first="New first turn."))
    _propose(client)
    seen = backend.text_seen_by("meta_optimizer")
    assert SEED.judge_system not in seen and SEED.solver_system not in seen


def test_frozen_field_is_refused_then_reasked() -> None:
    good = _reply(rationale="r", challenger_first="New first turn.")
    client, backend = _meta(_reply(rationale="r", judge_system="Grade kindly."), good)
    proposal = _propose(client)
    assert proposal.changed == ("challenger_first",)
    assert "judge_system" in backend.calls[-1][1][-1]["content"]


def test_broken_template_is_refused_then_reasked() -> None:
    good = _reply(rationale="r", challenger_refine="$feedback\n\nTry a new angle.")
    client, backend = _meta(_reply(rationale="r", challenger_refine="Try a new angle."), good)
    assert _propose(client).prompts.challenger_refine.startswith("$feedback")
    assert "feedback" in backend.calls[-1][1][-1]["content"]


def test_dropped_parser_key_is_refused() -> None:
    broken = SEED.qv_system.replace("check_2_verdict", "check_two")
    client, _ = _meta(_reply(rationale="r", qv_system=broken))
    with pytest.raises(ImplementerError, match="check_2_verdict"):
        _propose(client, retries=0)


def test_reply_without_a_change_is_refused() -> None:
    client, _ = _meta(_reply(rationale="r", challenger_system=SEED.challenger_system))
    with pytest.raises(ImplementerError, match="no field changed"):
        _propose(client, retries=0)


def test_trailing_blank_lines_are_not_a_change() -> None:
    client, _ = _meta(_reply(rationale="r", challenger_system=SEED.challenger_system + "\n\n"))
    with pytest.raises(ImplementerError, match="no field changed"):
        _propose(client, retries=0)


def test_oversized_prompts_are_refused() -> None:
    client, _ = _meta(_reply(rationale="r", challenger_system=SEED.challenger_system + "x" * 900))
    with pytest.raises(ImplementerError, match="characters"):
        _propose(client, max_chars=len(SEED.challenger_system) + 500, retries=0)


def test_gives_up_after_the_retries() -> None:
    client, backend = _meta("no blocks here", "still none")
    with pytest.raises(ImplementerError):
        _propose(client, retries=1)
    assert len(backend.calls) == 2
