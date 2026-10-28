"""Candidate prompt sets and how they are kept (arXiv 2606.25996, Sec 4).

A variant is a prompt set with the validation scores it has collected.
Single scores are noisy (solvers sample at temperature 1.0), so a variant
is re-evaluated each time it is picked as a parent and ranked by its mean.
Every variant, kept or rejected, is saved with a diff against the seed.
"""

from __future__ import annotations

import difflib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..llm.prompt_set import MUTABLE_FIELDS, PromptSet
from .boltzmann import boltzmann_sample


@dataclass(frozen=True)
class Variant:
    variant_id: str
    parent_id: str | None
    iteration: int
    prompts: PromptSet
    scores: tuple[float, ...] = ()

    @property
    def mean(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    def with_score(self, score: float) -> Variant:
        return replace(self, scores=(*self.scores, score))


def select_parent(population: Sequence[Variant], temperature: float,
                  rng: random.Random) -> int:
    """Boltzmann pick, P(c) ∝ exp(mean_c / T). Only the unscored seed can
    lack a mean, and only while it is alone."""
    means = [v.mean if v.mean is not None else 0.0 for v in population]
    return boltzmann_sample(means, temperature, rng)


def diff_against(seed: PromptSet, prompts: PromptSet) -> str:
    parts: list[str] = []
    for f in MUTABLE_FIELDS:
        a, b = getattr(seed, f), getattr(prompts, f)
        if a != b:
            parts.extend(difflib.unified_diff(a.splitlines(keepends=True),
                                              b.splitlines(keepends=True),
                                              f"seed/{f}.txt", f"variant/{f}.txt"))
    return "".join(parts)


def save_variant(v: Variant, out_dir: Path, seed: PromptSet) -> None:
    d = out_dir / "variants" / v.variant_id
    v.prompts.save(d / "prompts")
    (d / "diff.patch").write_text(diff_against(seed, v.prompts), encoding="utf-8")
    meta = {"variant_id": v.variant_id, "parent_id": v.parent_id, "iteration": v.iteration,
            "scores": list(v.scores), "mean": v.mean}
    (d / "variant.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_population(population: Sequence[Variant], out_dir: Path) -> None:
    rows = [{"variant_id": v.variant_id, "parent_id": v.parent_id, "iteration": v.iteration,
             "scores": list(v.scores), "mean": v.mean} for v in population]
    (out_dir / "population.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


__all__ = ["Variant", "diff_against", "save_population", "save_variant", "select_parent"]
