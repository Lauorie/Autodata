"""LLM client layer: proxy (OpenAI-compat) + local HF inference."""

from .client import LLMClient, LLMRoleConfig, build_client_pool
from .schemas import QAPair, RubricCriterion, RubricSpec, SolverScore, VerifierVerdict

__all__ = [
    "LLMClient",
    "LLMRoleConfig",
    "QAPair",
    "RubricCriterion",
    "RubricSpec",
    "SolverScore",
    "VerifierVerdict",
    "build_client_pool",
]
