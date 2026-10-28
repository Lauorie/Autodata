"""LLM call failures, sorted by what the caller should do about them.

- TransientLLMError: network, timeout, 429, 5xx. Retry with backoff.
- PermanentLLMError: bad request, auth, unknown model. Retrying repeats it.
- ContextLengthError: the prompt does not fit the model. A permanent error.
- OutputTruncatedError: the reply hit ``max_tokens``. Not retried; the caller
  decides whether the partial text counts (a solver's does: the paper scores
  a truncated answer as the model's failure, App A).
"""

from __future__ import annotations

import re

import openai


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    pass


class PermanentLLMError(LLMError):
    pass


class ContextLengthError(PermanentLLMError):
    pass


class OutputTruncatedError(LLMError):
    def __init__(self, role: str, max_tokens: int, partial: str) -> None:
        super().__init__(f"{role} reply hit max_tokens={max_tokens}")
        self.role = role
        self.max_tokens = max_tokens
        self.partial = partial


_CONTEXT_LENGTH = re.compile(
    r"context.length|maximum model length|prompt is too long|too many tokens|"
    r"context_length_exceeded|input is too long",
    re.IGNORECASE,
)
_TRANSIENT_STATUS = {408, 409, 429}
# A gateway (PaperBypass) reports an upstream failure as its own 502 whose body
# reads "upstream error 400"; the upstream status decides what to do.
_UPSTREAM_STATUS = re.compile(r"upstream error (\d{3})")


def _effective_status(e: openai.APIStatusError) -> int:
    m = _UPSTREAM_STATUS.search(str(e))
    return int(m.group(1)) if e.status_code >= 500 and m else e.status_code


def classify_api_error(e: openai.APIError) -> LLMError:
    """Map an OpenAI SDK error to the taxonomy above."""
    if isinstance(e, openai.APIConnectionError):    # includes APITimeoutError
        return TransientLLMError(str(e))
    if isinstance(e, openai.APIStatusError):
        status = _effective_status(e)
        if status in _TRANSIENT_STATUS or status >= 500:
            return TransientLLMError(str(e))
        if status == 413 or (status == 400 and _CONTEXT_LENGTH.search(str(e))):
            return ContextLengthError(str(e))
        return PermanentLLMError(str(e))
    return TransientLLMError(str(e))                # e.g. a malformed response body


__all__ = [
    "ContextLengthError",
    "LLMError",
    "OutputTruncatedError",
    "PermanentLLMError",
    "TransientLLMError",
    "classify_api_error",
]
