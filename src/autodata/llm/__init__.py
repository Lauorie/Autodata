"""LLM client layer: proxy and vLLM (OpenAI-compatible) + local HF inference."""

from .client import LLMClient, LLMRoleConfig, build_client_pool
from .errors import (
    ContextLengthError,
    LLMError,
    OutputTruncatedError,
    PermanentLLMError,
    TransientLLMError,
)
from .prompt_set import PromptSet
from .schemas import QAPair, RubricCriterion, RubricSpec, SolverScore, VerifierVerdict
from .usage import UsageMeter

__all__ = [
    "ContextLengthError",
    "LLMClient",
    "LLMError",
    "LLMRoleConfig",
    "OutputTruncatedError",
    "PermanentLLMError",
    "PromptSet",
    "QAPair",
    "RubricCriterion",
    "RubricSpec",
    "SolverScore",
    "TransientLLMError",
    "UsageMeter",
    "VerifierVerdict",
    "build_client_pool",
]
