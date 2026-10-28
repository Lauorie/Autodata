"""The outer loop: meta-optimize the challenger and verifier prompts.

arXiv 2606.25996, Sec 4. Each iteration:
  1. Pick a parent from the population by Boltzmann sampling (T = 0.1).
  2. Run the parent on a minibatch of train papers; keep the trajectories.
  3. The analyzer reads them and writes a root-cause analysis.
  4. The implementer reads the analysis, the log and the parent's prompts,
     and proposes a mutant.
  5. Re-evaluate parent and mutant on the held-out validation papers. A
     session counts only if its question also passes the fixed end-of-loop
     filter (evaluation.py).
  6. Keep the mutant only if its score strictly beats the parent's mean
     over all its evaluations.
  7. Append the outcome to the log the next iteration reads.

An evaluation with too many sessions lost to infrastructure decides
nothing: the iteration is logged as invalid_eval and no score is kept.
Iterations run one after another; the paper runs several at once.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..data_module import Paper
from ..llm import LLMClient
from ..llm.errors import LLMError
from ..llm.prompt_set import PromptSet
from ..pipeline.acceptance import AcceptanceCriteria
from ..pipeline.final_filter import RubricShape
from ..pipeline.inner_loop import InnerLoopConfig
from .analyzer import analyze
from .digest import render_sessions
from .evaluation import EvalResult, evaluate
from .history import IterationRecord, append_record, render_log
from .implementer import ImplementerError, propose
from .population import Variant, save_population, save_variant, select_parent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaConfig:
    iterations: int = 4
    temperature: float = 0.1
    train_minibatch: int = 5
    val_samples_per_paper: int = 4
    session_rounds: int = 1
    answer_chars: int = 1500
    min_scored_fraction: float = 0.8
    max_prompt_chars: int = 20000
    edit_retries: int = 2
    workers: int = 4


def split_papers(papers: Sequence[Paper], n_train: int, n_val: int,
                 seed: int) -> tuple[list[Paper], list[Paper]]:
    """Disjoint train and validation papers, the same for the same seed."""
    if len(papers) < n_train + n_val:
        raise ValueError(f"need {n_train + n_val} papers, have {len(papers)}")
    shuffled = list(papers)
    random.Random(seed).shuffle(shuffled)
    return shuffled[n_val:n_val + n_train], shuffled[:n_val]


def _save_sessions(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in result.sessions:
            row = {"paper_id": s.paper_id, "passed": s.passed, "stopped_by": s.stopped_by,
                   "rounds": [asdict(r) for r in s.rounds],
                   "final_filter": s.final_verdict.model_dump() if s.final_verdict else None}
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


@dataclass(frozen=True)
class _Run:
    """What every iteration shares."""

    seed: PromptSet
    train: Sequence[Paper]
    val: Sequence[Paper]
    clients: dict[str, LLMClient]
    criteria: AcceptanceCriteria
    session_cfg: InnerLoopConfig
    shape: RubricShape
    cfg: MetaConfig
    out_dir: Path

    def evaluate(self, prompts: PromptSet, papers: Sequence[Paper], samples: int) -> EvalResult:
        return evaluate(prompts, papers, self.clients, self.criteria, self.session_cfg,
                        self.shape, samples, self.cfg.workers)


def _iterate(it: int, run: _Run, population: list[Variant], history: list[IterationRecord],
             rng: random.Random) -> IterationRecord:
    cfg, d = run.cfg, run.out_dir / "iterations" / f"{it:03d}"
    idx = select_parent(population, cfg.temperature, rng)
    parent = population[idx]
    log = render_log(history)

    batch = rng.sample(list(run.train), min(cfg.train_minibatch, len(run.train)))
    train = run.evaluate(parent.prompts, batch, 1)
    _save_sessions(d / "train.jsonl", train)
    meta = run.clients["meta_optimizer"]
    try:
        analysis = analyze(meta, render_sessions(train, cfg.answer_chars), log, run.criteria)
    except LLMError as e:
        return IterationRecord(it, parent.variant_id, "analysis_failed", note=str(e))
    (d / "analysis.md").write_text(analysis + "\n", encoding="utf-8")
    try:
        proposal = propose(meta, parent.prompts, analysis, log, run.criteria,
                           cfg.max_prompt_chars, cfg.edit_retries)
    except (ImplementerError, LLMError) as e:
        return IterationRecord(it, parent.variant_id, "no_proposal", note=str(e))

    child = Variant(f"it{it:03d}", parent.variant_id, it, proposal.prompts)
    parent_val = run.evaluate(parent.prompts, run.val, cfg.val_samples_per_paper)
    child_val = run.evaluate(child.prompts, run.val, cfg.val_samples_per_paper)
    _save_sessions(d / "val_parent.jsonl", parent_val)
    _save_sessions(d / "val_child.jsonl", child_val)
    common = {"child_id": child.variant_id, "changed": proposal.changed,
              "rationale": proposal.rationale}
    if not (parent_val.is_valid(cfg.min_scored_fraction)
            and child_val.is_valid(cfg.min_scored_fraction)):
        save_variant(child, run.out_dir, run.seed)
        note = (f"scored {parent_val.scored}/{len(parent_val.sessions)} parent and "
                f"{child_val.scored}/{len(child_val.sessions)} child sessions")
        return IterationRecord(it, parent.variant_id, "invalid_eval", note=note, **common)

    parent = parent.with_score(parent_val.score)
    population[idx] = parent
    child = child.with_score(child_val.score)
    accepted = child_val.score > parent.mean
    if accepted:
        population.append(child)
    save_variant(parent, run.out_dir, run.seed)
    save_variant(child, run.out_dir, run.seed)
    return IterationRecord(it, parent.variant_id, "accepted" if accepted else "rejected",
                           parent_score=parent.mean, child_score=child_val.score, **common)


def run_meta_optimization(seed: PromptSet, train_papers: Sequence[Paper],
                          val_papers: Sequence[Paper], clients: dict[str, LLMClient],
                          criteria: AcceptanceCriteria, inner_cfg: InnerLoopConfig,
                          shape: RubricShape, cfg: MetaConfig, out_dir: str | Path,
                          rng: random.Random) -> list[Variant]:
    """Run ``cfg.iterations`` iterations; return the final population."""
    out = Path(out_dir)
    run = _Run(seed, train_papers, val_papers, clients, criteria,
               replace(inner_cfg, max_iterations=cfg.session_rounds), shape, cfg, out)
    population = [Variant("seed", None, 0, seed)]
    save_variant(population[0], out, seed)
    history: list[IterationRecord] = []
    for it in range(1, cfg.iterations + 1):
        record = _iterate(it, run, population, history, rng)
        history.append(record)
        append_record(out / "history.jsonl", record)
        save_population(population, out)
        logger.info("[meta %d/%d] %s", it, cfg.iterations, render_log([record]))
    return population


__all__ = ["MetaConfig", "run_meta_optimization", "split_papers"]
