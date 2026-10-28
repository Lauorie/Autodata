"""Rubric item validation: the format half of Fig 8's rubric contract.

Counts (>= 4 positive, >= 3 negative, 10-20 total) belong to the quality
verifier prompt, which the meta-optimizer may rewrite, so the schema does
not enforce them.
"""

import pytest
from pydantic import ValidationError

from autodata.llm.schemas import RubricCriterion


@pytest.mark.parametrize("item", [
    {"criterion": "states the trade-off", "weight": 5, "category": "positive"},
    {"criterion": "confuses forward and reverse KL", "weight": -10, "category": "negative"},
    {"criterion": "  padded text  ", "weight": 1, "category": "positive"},
])
def test_accepts_well_formed_item(item: dict) -> None:
    RubricCriterion.model_validate(item)


@pytest.mark.parametrize("item", [
    {"criterion": "zero weight", "weight": 0, "category": "positive"},
    {"criterion": "zero weight", "weight": 0, "category": "negative"},
    {"criterion": "too heavy", "weight": 11, "category": "positive"},
    {"criterion": "too negative", "weight": -11, "category": "negative"},
    {"criterion": "sign says error", "weight": -3, "category": "positive"},
    {"criterion": "sign says insight", "weight": 3, "category": "negative"},
    {"criterion": "unknown category", "weight": 3, "category": "neutral"},
    {"criterion": "missing category", "weight": 3},
])
def test_rejects_malformed_item(item: dict) -> None:
    with pytest.raises(ValidationError):
        RubricCriterion.model_validate(item)
