"""Tests for Boltzmann parent selection."""

import random

from autodata.meta.boltzmann import boltzmann_sample


def test_argmax_at_zero_temp() -> None:
    rng = random.Random(0)
    idx = boltzmann_sample([0.1, 0.5, 0.3], temperature=0.0, rng=rng)
    assert idx == 1


def test_uniform_at_high_temp() -> None:
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(2000):
        counts[boltzmann_sample([0.1, 0.5, 0.3], temperature=100.0, rng=rng)] += 1
    # All buckets should be nontrivially populated.
    assert all(c > 400 for c in counts), counts


def test_concentrates_at_low_temp() -> None:
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(2000):
        counts[boltzmann_sample([0.1, 0.5, 0.3], temperature=0.05, rng=rng)] += 1
    # Index 1 (highest) should dominate.
    assert counts[1] > counts[0] + counts[2]


def test_empty_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        boltzmann_sample([], temperature=0.1)
