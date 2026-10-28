"""OpenAI-compatible backend: error taxonomy, truncation, thinking controls.

The SDK client is injected, so no network is touched. Only transient errors
are retried; a context-length error, a bad request, or an output that hit
``max_tokens`` fails at once.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from autodata.llm.client import LLMRoleConfig, _OpenAIBackend
from autodata.llm.errors import (
    ContextLengthError,
    OutputTruncatedError,
    PermanentLLMError,
    TransientLLMError,
)
from autodata.llm.thinking import split_thinking

_REQ = httpx.Request("POST", "http://test/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], status: int, msg: str) -> openai.APIStatusError:
    return cls(msg, response=httpx.Response(status, request=_REQ), body=None)


def _reply(content: str | None, finish: str = "stop", **extra: Any) -> SimpleNamespace:
    message = SimpleNamespace(content=content, **extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])


class _FakeSDK:
    """Stands in for ``openai.OpenAI``: replays outcomes, records kwargs."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_OpenAIBackend.chat.retry, "sleep", lambda _seconds: None)


def _role(**overrides: Any) -> LLMRoleConfig:
    base: dict[str, Any] = {"role": "judge", "endpoint": "proxy", "model": "m", "max_tokens": 100}
    base.update(overrides)
    return LLMRoleConfig(**base)


def _chat(sdk: _FakeSDK, role: LLMRoleConfig, flavor: str = "openrouter") -> str:
    return _OpenAIBackend(sdk, flavor=flavor, label="test").chat(role, [{"role": "user", "content": "q"}])


# --- error taxonomy -----------------------------------------------------------

def test_rate_limit_is_retried() -> None:
    sdk = _FakeSDK(_status_error(openai.RateLimitError, 429, "slow down"), _reply("ok"))
    assert _chat(sdk, _role()) == "ok"
    assert len(sdk.requests) == 2


def test_connection_error_is_retried_until_attempts_run_out() -> None:
    sdk = _FakeSDK(*[openai.APIConnectionError(request=_REQ) for _ in range(4)])
    with pytest.raises(TransientLLMError):
        _chat(sdk, _role())
    assert len(sdk.requests) == 4


def test_context_length_is_not_retried() -> None:
    msg = "This model's maximum context length is 262144 tokens. However, you requested 300000"
    sdk = _FakeSDK(_status_error(openai.BadRequestError, 400, msg))
    with pytest.raises(ContextLengthError):
        _chat(sdk, _role())
    assert len(sdk.requests) == 1


def test_other_bad_request_is_permanent() -> None:
    sdk = _FakeSDK(_status_error(openai.BadRequestError, 400, "unknown parameter: foo"))
    with pytest.raises(PermanentLLMError) as info:
        _chat(sdk, _role())
    assert not isinstance(info.value, ContextLengthError)
    assert len(sdk.requests) == 1


def test_auth_error_is_permanent() -> None:
    sdk = _FakeSDK(_status_error(openai.AuthenticationError, 401, "bad key"))
    with pytest.raises(PermanentLLMError):
        _chat(sdk, _role())
    assert len(sdk.requests) == 1


# The PaperBypass gateway wraps an upstream status in its own 502:
# {'success': False, 'error': {'message': 'upstream error 400', 'reason': 'upstream_error'}}
def _gateway_502(upstream: int) -> openai.APIStatusError:
    body = (f"{{'success': False, 'error': {{'code': 800000047, "
            f"'message': 'upstream error {upstream}', 'reason': 'upstream_error'}}}}")
    return _status_error(openai.InternalServerError, 502, body)


def test_gateway_wrapped_bad_request_is_permanent() -> None:
    sdk = _FakeSDK(_gateway_502(400))
    with pytest.raises(PermanentLLMError):
        _chat(sdk, _role())
    assert len(sdk.requests) == 1


@pytest.mark.parametrize("upstream", [429, 503])
def test_gateway_wrapped_transient_is_retried(upstream: int) -> None:
    sdk = _FakeSDK(_gateway_502(upstream), _reply("ok"))
    assert _chat(sdk, _role()) == "ok"
    assert len(sdk.requests) == 2


def test_non_api_exception_propagates_unretried() -> None:
    sdk = _FakeSDK(ValueError("bug in our code"))
    with pytest.raises(ValueError):
        _chat(sdk, _role())
    assert len(sdk.requests) == 1


def test_finish_length_raises_truncated_with_partial_answer() -> None:
    sdk = _FakeSDK(_reply("half an ans", finish="length"))
    with pytest.raises(OutputTruncatedError) as info:
        _chat(sdk, _role())
    assert info.value.partial == "half an ans"
    assert len(sdk.requests) == 1


def test_empty_content_on_stop_is_retried() -> None:
    sdk = _FakeSDK(_reply(""), _reply("ok"))
    assert _chat(sdk, _role()) == "ok"
    assert len(sdk.requests) == 2


# --- thinking controls ---------------------------------------------------------

def test_openrouter_thinking_off_sends_effort_none() -> None:
    sdk = _FakeSDK(_reply("ok"))
    _chat(sdk, _role(thinking=False))
    assert sdk.requests[0]["extra_body"]["reasoning"] == {"effort": "none"}


def test_openrouter_thinking_on_keeps_model_default() -> None:
    sdk = _FakeSDK(_reply("ok", reasoning="thought"))
    _chat(sdk, _role(thinking=True))
    assert "reasoning" not in (sdk.requests[0].get("extra_body") or {})


def test_vllm_sends_enable_thinking_and_sampling() -> None:
    sdk = _FakeSDK(_reply("ok", reasoning_content="thought"))
    role = _role(endpoint="vllm", thinking=True, temperature=1.0, top_p=0.95, top_k=20,
                 presence_penalty=1.5)
    _chat(sdk, role, flavor="vllm")
    req = sdk.requests[0]
    assert req["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert req["extra_body"]["top_k"] == 20
    assert req["top_p"] == 0.95 and req["presence_penalty"] == 1.5 and req["temperature"] == 1.0


def test_thinking_block_in_content_is_stripped() -> None:
    sdk = _FakeSDK(_reply("<think>\nlong thought\n</think>\n\nthe answer"))
    assert _chat(sdk, _role(endpoint="vllm", thinking=True), flavor="vllm") == "the answer"


def test_unterminated_thinking_is_truncation_with_empty_answer() -> None:
    sdk = _FakeSDK(_reply("<think>\nstill thinking", finish="length"))
    with pytest.raises(OutputTruncatedError) as info:
        _chat(sdk, _role(endpoint="vllm", thinking=True), flavor="vllm")
    assert info.value.partial == ""


def test_thinking_with_effort_none_is_rejected() -> None:
    with pytest.raises(ValueError):
        _role(thinking=True, reasoning_effort="none")


# --- split_thinking ------------------------------------------------------------

@pytest.mark.parametrize("text, prompt_opened, want", [
    ("<think>\nplan\n</think>\n\nanswer", False, ("answer", True)),
    ("plan\n</think>\n\nanswer", True, ("answer", True)),      # template opened <think>
    ("<think>\nplan only", False, ("", False)),
    ("plan only", True, ("", False)),
    ("plain answer", False, ("plain answer", True)),
])
def test_split_thinking(text: str, prompt_opened: bool, want: tuple[str, bool]) -> None:
    assert split_thinking(text, prompt_opened=prompt_opened) == want
