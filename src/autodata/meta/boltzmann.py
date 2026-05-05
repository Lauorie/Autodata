"""Boltzmann sampling for parent selection.

Per AutoData blog: P(parent_c) ∝ exp(s_c / T), with T=0.1.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def boltzmann_sample(
    scores: Sequence[float],
    temperature: float = 0.1,
    rng: random.Random | None = None,
) -> int:
    """Return the index of the parent sampled from the Boltzmann distribution.

    Numerically stable: subtracts max before exp.
    """
    if not scores:
        raise ValueError("empty score list")
    if temperature <= 0:
        # degenerate: take the argmax (with ties broken by RNG)
        rng = rng or random
        best = max(scores)
        idxs = [i for i, s in enumerate(scores) if s == best]
        return rng.choice(idxs)
    rng = rng or random
    max_s = max(scores)
    weights = [math.exp((s - max_s) / temperature) for s in scores]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


__all__ = ["boltzmann_sample"]
