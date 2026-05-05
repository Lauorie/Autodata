"""Outer evolution loop over the prompts module.

A `Variant` is a snapshot of `prompts.py` source. Each variant has a fitness
score = fraction of validation chunks for which the inner loop produced an
ACCEPTED QA pair (the "validation pass rate" reported in the AutoData blog).

The driver:
  1. Seed population with the original (unmutated) prompts -> evaluate fitness.
  2. For G generations:
       a. Boltzmann-sample a parent.
       b. Ask the meta-optimizer LLM for a mutated prompts source.
       c. Validate syntactically; load it as a runtime module.
       d. Evaluate fitness on the same validation set.
       e. Accept iff fitness > parent's fitness.

We isolate per-variant prompts via Python module hot-swap rather than
in-place mutation of the installed package — this matches the blog's
"never mutate in place" guidance.
"""

from __future__ import annotations

import importlib.util
import logging
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..data_module import Chunk
from ..llm import LLMClient
from ..pipeline.acceptance import AcceptanceCriteria
from ..pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_chunk
from .boltzmann import boltzmann_sample
from .mutator import propose_mutation, summarize_failure_patterns, validate_mutated_source

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    variant_id: str
    parent_id: str | None
    generation: int
    prompts_source: str
    score: float = 0.0
    n_accepted: int = 0
    n_chunks: int = 0
    rejection_reasons: list[list[str]] = field(default_factory=list)
    judge_feedbacks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetaConfig:
    n_generations: int = 4
    temperature: float = 0.1
    validation_chunks: int = 8


# ---------------------------------------------------------------------------

def _load_prompts_module(src: str, name: str) -> Any:
    """Materialise a Python module object from raw source string."""
    mod_name = f"autodata._mutated_prompts_{name}"
    spec = importlib.util.spec_from_loader(mod_name, loader=None)
    if spec is None:  # pragma: no cover
        raise RuntimeError("could not create spec")
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, f"<{mod_name}>", "exec"), module.__dict__)
    sys.modules[mod_name] = module
    return module


def _evaluate_variant(
    variant: Variant,
    chunks: list[Chunk],
    clients: dict[str, LLMClient],
    criteria: AcceptanceCriteria,
    inner_cfg: InnerLoopConfig,
) -> Variant:
    """Run the inner loop on `chunks` using this variant's prompts source.

    Hot-swaps `autodata.llm.prompts` for the duration of the eval so the
    inner-loop code transparently picks up the mutated prompts.
    """
    from autodata.llm import prompts as base_prompts

    mutated = _load_prompts_module(variant.prompts_source, variant.variant_id)
    saved = {k: getattr(base_prompts, k, None) for k in dir(mutated) if not k.startswith("_")}
    for k in saved:
        if hasattr(mutated, k):
            setattr(base_prompts, k, getattr(mutated, k))
    try:
        n_acc = 0
        rejs: list[list[str]] = []
        fbs: list[str] = []
        for chunk in chunks:
            qa, rounds = run_inner_loop_for_chunk(chunk, clients, criteria, inner_cfg)
            if qa is not None:
                n_acc += 1
            for r in rounds:
                if r.acceptance and not r.accepted:
                    rejs.append(list(r.acceptance.get("reasons", [])))
                for s in r.weak_samples + r.strong_samples:
                    fb = s.get("judge_feedback") if isinstance(s, dict) else None
                    if fb:
                        fbs.append(fb)
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(base_prompts, k):
                    delattr(base_prompts, k)
            else:
                setattr(base_prompts, k, v)

    variant.n_accepted = n_acc
    variant.n_chunks = len(chunks)
    variant.score = n_acc / max(len(chunks), 1)
    variant.rejection_reasons = rejs
    variant.judge_feedbacks = fbs[:30]
    logger.info("variant %s gen=%d -> score=%.3f (%d/%d)",
                variant.variant_id, variant.generation,
                variant.score, n_acc, len(chunks))
    return variant


def run_meta_optimization(
    seed_prompts_path: str | Path,
    val_chunks: list[Chunk],
    clients: dict[str, LLMClient],
    criteria: AcceptanceCriteria,
    inner_cfg: InnerLoopConfig,
    out_dir: str | Path,
    cfg: MetaConfig,
    rng: random.Random | None = None,
) -> list[Variant]:
    rng = rng or random.Random(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_src = Path(seed_prompts_path).read_text()
    seed = Variant(variant_id="gen0_seed", parent_id=None,
                   generation=0, prompts_source=seed_src)
    _evaluate_variant(seed, val_chunks, clients, criteria, inner_cfg)
    population: list[Variant] = [seed]
    _persist_variant(seed, out_dir)

    for g in range(1, cfg.n_generations + 1):
        scores = [v.score for v in population]
        parent_idx = boltzmann_sample(scores, cfg.temperature, rng)
        parent = population[parent_idx]
        logger.info("[gen %d] parent=%s score=%.3f", g, parent.variant_id, parent.score)

        try:
            new_src = propose_mutation(
                clients["meta_optimizer"],
                parent.prompts_source,
                summarize_failure_patterns(parent.rejection_reasons, parent.judge_feedbacks),
            )
            ok, msg = validate_mutated_source(new_src)
            if not ok:
                logger.warning("[gen %d] mutation invalid: %s", g, msg)
                continue
        except Exception as e:
            logger.exception("[gen %d] mutation proposal failed: %s", g, e)
            continue

        child = Variant(
            variant_id=f"gen{g}_{uuid.uuid4().hex[:6]}",
            parent_id=parent.variant_id,
            generation=g,
            prompts_source=new_src,
        )
        _evaluate_variant(child, val_chunks, clients, criteria, inner_cfg)
        if child.score > parent.score:
            population.append(child)
            logger.info("[gen %d] ACCEPT child %s (%.3f > %.3f)",
                        g, child.variant_id, child.score, parent.score)
        else:
            logger.info("[gen %d] REJECT child %s (%.3f ≤ %.3f)",
                        g, child.variant_id, child.score, parent.score)
        _persist_variant(child, out_dir)

    _persist_population(population, out_dir)
    return population


def _persist_variant(v: Variant, out_dir: Path) -> None:
    d = out_dir / v.variant_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompts.py").write_text(v.prompts_source)
    meta = {k: val for k, val in asdict(v).items() if k != "prompts_source"}
    import json
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _persist_population(population: list[Variant], out_dir: Path) -> None:
    import json
    rows = [{
        "variant_id": v.variant_id,
        "parent_id": v.parent_id,
        "generation": v.generation,
        "score": v.score,
        "n_accepted": v.n_accepted,
        "n_chunks": v.n_chunks,
    } for v in population]
    (out_dir / "population.json").write_text(json.dumps(rows, indent=2))


__all__ = ["MetaConfig", "Variant", "run_meta_optimization"]
