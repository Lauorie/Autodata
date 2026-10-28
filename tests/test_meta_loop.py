"""The outer loop of arXiv 2606.25996, Sec 4, with evaluation faked.

Each iteration: Boltzmann-pick a parent, run it on a train minibatch, analyze,
propose a mutant, re-evaluate parent and mutant on the validation papers,
and keep the mutant only if it beats the parent's averaged score.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from autodata.llm import LLMClient, LLMRoleConfig
from autodata.llm.prompt_set import PromptSet
from autodata.meta import outer_loop
from autodata.meta.evaluation import EvalResult, Session
from autodata.meta.outer_loop import MetaConfig, run_meta_optimization, split_papers
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig
from tests.fakes import ScriptedBackend, make_paper

SEED = PromptSet.seed()
TRAIN = [make_paper(paper_id=f"t{i}") for i in range(6)]
VAL = [make_paper(paper_id=f"v{i}") for i in range(3)]


class FakeEval:
    """Pass rate by a marker in the challenger prompt; records every call."""

    def __init__(self, rates: dict[str, float]) -> None:
        self.rates = rates
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompts: PromptSet, papers: list[Any], clients: Any, criteria: Any,
                 inner_cfg: InnerLoopConfig, samples_per_paper: int = 1,
                 workers: int = 1) -> EvalResult:
        marker = next((m for m in self.rates if m in prompts.challenger_system), "SEED")
        self.calls.append({"marker": marker, "papers": [p.paper_id for p in papers],
                           "samples": samples_per_paper, "rounds": inner_cfg.max_iterations})
        n = len(papers) * samples_per_paper
        rate = self.rates.get(marker, 0.0)
        if rate < 0:   # every session lost to infrastructure
            return EvalResult(tuple(Session("p", "P", (), False, "infra_error") for _ in range(n)))
        k = round(rate * n)
        return EvalResult(tuple(Session("p", "P", (), i < k, None) for i in range(n)))


def _mutant(marker: str) -> str:
    return (f"<<<rationale>>>\ntry {marker}\n<<<end>>>\n"
            f"<<<challenger_system>>>\n{SEED.challenger_system}\n{marker}\n<<<end>>>")


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rates: dict[str, float],
         replies: list[str], iterations: int = 1) -> tuple[list[Any], FakeEval, ScriptedBackend]:
    fake = FakeEval({"SEED": 0.5, **rates})
    monkeypatch.setattr(outer_loop, "evaluate", fake)
    backend = ScriptedBackend(handlers={"meta_optimizer": replies})
    meta = LLMClient(LLMRoleConfig(role="meta_optimizer", endpoint="proxy", model="fake"),
                     backend)
    cfg = MetaConfig(iterations=iterations, train_minibatch=2, val_samples_per_paper=2,
                     session_rounds=2, edit_retries=0, workers=1)
    population = run_meta_optimization(
        SEED, TRAIN, VAL, {"meta_optimizer": meta}, AcceptanceCriteria(),
        InnerLoopConfig(max_iterations=15), cfg, tmp_path, random.Random(0))
    return population, fake, backend


def _history(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]


def test_better_mutant_joins_the_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pop, _, _ = _run(tmp_path, monkeypatch, {"BETTER": 5 / 6}, ["analysis", _mutant("BETTER")])
    assert [v.variant_id for v in pop] == ["seed", "it001"]
    assert pop[0].scores == (0.5,) and pop[1].parent_id == "seed"
    assert _history(tmp_path)[0]["status"] == "accepted"
    assert json.loads((tmp_path / "population.json").read_text())[1]["variant_id"] == "it001"


def test_worse_mutant_is_rejected_but_kept_on_disk(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    pop, _, _ = _run(tmp_path, monkeypatch, {"WORSE": 1 / 6}, ["analysis", _mutant("WORSE")])
    assert [v.variant_id for v in pop] == ["seed"]
    assert _history(tmp_path)[0]["status"] == "rejected"
    assert "WORSE" in (tmp_path / "variants" / "it001" / "diff.patch").read_text()
    assert PromptSet.load(tmp_path / "variants" / "it001" / "prompts").challenger_system.endswith(
        "WORSE")


def test_a_tie_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pop, _, _ = _run(tmp_path, monkeypatch, {"SAME": 0.5}, ["analysis", _mutant("SAME")])
    assert len(pop) == 1


def test_parent_scores_accumulate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = ["a1", _mutant("WORSE"), "a2", _mutant("WORSE")]
    pop, _, _ = _run(tmp_path, monkeypatch, {"WORSE": 0.0}, replies, iterations=2)
    assert pop[0].scores == (0.5, 0.5)


def test_void_evaluation_neither_accepts_nor_scores(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    pop, _, _ = _run(tmp_path, monkeypatch, {"LOST": -1.0}, ["analysis", _mutant("LOST")])
    assert [v.variant_id for v in pop] == ["seed"] and pop[0].scores == ()
    assert _history(tmp_path)[0]["status"] == "invalid_eval"


def test_train_minibatch_then_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake, _ = _run(tmp_path, monkeypatch, {"BETTER": 1.0}, ["analysis", _mutant("BETTER")])
    train, val_parent, val_child = fake.calls
    assert len(train["papers"]) == 2 and set(train["papers"]) <= {p.paper_id for p in TRAIN}
    assert train["samples"] == 1
    for call in (val_parent, val_child):
        assert call["papers"] == [p.paper_id for p in VAL] and call["samples"] == 2
    assert [val_parent["marker"], val_child["marker"]] == ["SEED", "BETTER"]
    assert {c["rounds"] for c in fake.calls} == {2}        # session_rounds, not the run budget


def test_unusable_proposal_skips_validation(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake, _ = _run(tmp_path, monkeypatch, {}, ["analysis", "no blocks"])
    assert len(fake.calls) == 1
    assert _history(tmp_path)[0]["status"] == "no_proposal"


def test_the_log_reaches_the_next_analyzer(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    replies = ["a1", _mutant("WORSE"), "a2", _mutant("WORSE")]
    _, _, backend = _run(tmp_path, monkeypatch, {"WORSE": 0.0}, replies, iterations=2)
    second_analysis = backend.calls[2][1][-1]["content"]
    assert "it001" in second_analysis and "rejected" in second_analysis


# -- the paper split -----------------------------------------------------------

def test_split_is_disjoint_and_deterministic() -> None:
    papers = [make_paper(paper_id=f"x{i}") for i in range(10)]
    train, val = split_papers(papers, n_train=6, n_val=3, seed=7)
    assert len(train) == 6 and len(val) == 3
    assert not {p.paper_id for p in train} & {p.paper_id for p in val}
    again = split_papers(papers, n_train=6, n_val=3, seed=7)
    assert [p.paper_id for p in again[0]] == [p.paper_id for p in train]


def test_split_refuses_too_few_papers() -> None:
    with pytest.raises(ValueError, match="papers"):
        split_papers([make_paper(paper_id="a")], n_train=1, n_val=1, seed=0)


# -- end to end, nothing faked but the models -------------------------------------

def test_one_real_iteration_end_to_end(tmp_path: Path) -> None:
    from tests.fakes import STRONG_PER, WEAK_PER, challenger_payload, judge_by_solver, qv_payload

    backend = ScriptedBackend(handlers={
        "challenger": lambda m: challenger_payload(),
        "quality_verifier": lambda m: qv_payload(),
        "weak_solver": lambda m: "WEAK-ANSWER",
        "strong_solver": lambda m: "STRONG-ANSWER",
        "judge": judge_by_solver(WEAK_PER, STRONG_PER),
        "meta_optimizer": ["ROOT CAUSES", _mutant("TIGHTER")],
    })
    clients = {r: LLMClient(LLMRoleConfig(role=r, endpoint="proxy", model="fake"), backend)
               for r in ("challenger", "quality_verifier", "weak_solver", "strong_solver",
                         "judge", "meta_optimizer")}
    cfg = MetaConfig(iterations=1, train_minibatch=1, val_samples_per_paper=1,
                     session_rounds=1, edit_retries=0, workers=1)
    inner = InnerLoopConfig(max_iterations=15, num_solver_samples=2, parallel_solvers=False)
    run_meta_optimization(SEED, TRAIN[:2], VAL[:2], clients, AcceptanceCriteria(), inner, cfg,
                          tmp_path, random.Random(0))

    analyzer_input = backend.calls[[r for r, _ in backend.calls].index("meta_optimizer")][1]
    assert "WEAK-ANSWER" in analyzer_input[-1]["content"]
    assert "TIGHTER" in backend.text_seen_by("challenger")     # the mutant was validated
    record = _history(tmp_path)[0]
    assert record["status"] == "rejected"                      # same scores: a tie
    assert (record["parent_score"], record["child_score"]) == (1.0, 1.0)
    assert (tmp_path / "iterations" / "001" / "analysis.md").read_text().startswith("ROOT")
