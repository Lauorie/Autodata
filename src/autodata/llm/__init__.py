"""LLM client layer: proxy and vLLM (OpenAI-compatible) + local HF inference."""

from .client import LLMClient, LLMRoleConfig, build_client_pool
from .errors import (
    ContextLengthError,
    LLMError,
    OutputTruncatedError,
    PermanentLLMError,
    TransientLLMError,
)
from .schemas import QAPair, RubricCriterion, RubricSpec, SolverScore, VerifierVerdict

__all__ = [
    "ContextLengthError",
    "LLMClient",
    "LLMError",
    "LLMRoleConfig",
    "OutputTruncatedError",
    "PermanentLLMError",
    "QAPair",
    "RubricCriterion",
    "RubricSpec",
    "SolverScore",
    "TransientLLMError",
    "VerifierVerdict",
    "build_client_pool",
]
