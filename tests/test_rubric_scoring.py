"""Rubric scoring with signed weights (Fig 8 baseline: positives and negatives).

Score = clip(sum(w_i * s_i) / sum(positive w_i), 0, 1), the clipped PRBench
form the paper reports in Table 4. Expected values below are hand-derived.
"""

import pytest

from autodata.pipeline.evaluate_rubric import rubric_score


@pytest.mark.parametrize("per, weights, want", [
    ([1.0, 0.5, 0.0], [4, 2, 2], 0.625),        # (4 + 1 + 0) / 8
    ([1.0, 1.0, 0.0], [5, 3, -4], 1.0),         # negative not triggered: 8 / 8
    ([1.0, 1.0, 1.0], [5, 3, -4], 0.5),         # negative triggered: (8 - 4) / 8
    ([1.0, 0.0, 0.5], [5, 3, -4], 0.375),       # (5 - 2) / 8
    ([0.0, 0.0, 1.0], [5, 3, -4], 0.0),         # -4 / 8 clips to 0
])
def test_rubric_score(per: list[float], weights: list[int], want: float) -> None:
    assert rubric_score(per, weights) == pytest.approx(want)


def test_rubric_score_without_positive_weights_is_zero() -> None:
    assert rubric_score([1.0, 1.0], [-2, -3]) == 0.0


def test_rubric_score_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        rubric_score([1.0], [2, 3])
