"""Token usage metering: every completed call is counted, per role and per call."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from autodata.llm.client import LLMRoleConfig, _OpenAIBackend, build_client_pool
from autodata.llm.errors import OutputTruncatedError, PermanentLLMError
from autodata.llm.usage import UsageMeter


def _usage(prompt: int, completion: int, reasoning: int = 0, cached: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _reply(content: str, finish: str = "stop", usage: Any = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning="thought")
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=usage)


class _FakeSDK:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any) -> Any:
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_OpenAIBackend.chat.retry, "sleep", lambda _seconds: None)


def _role(role: str = "judge", model: str = "ds-flash") -> LLMRoleConfig:
    return LLMRoleConfig(role=role, endpoint="proxy", model=model, max_tokens=100, thinking=True)


def _chat(sdk: _FakeSDK, meter: UsageMeter, role: LLMRoleConfig | None = None) -> str:
    backend = _OpenAIBackend(sdk, flavor="openrouter", label="test", meter=meter)
    return backend.chat(role or _role(), [{"role": "user", "content": "q"}])


def test_completed_call_is_counted_under_its_role() -> None:
    meter = UsageMeter()
    _chat(_FakeSDK(_reply("ok", usage=_usage(44, 21, reasoning=18, cached=4))), meter)
    t = meter.totals()["judge"]
    assert t["model"] == "ds-flash"
    assert (t["calls"], t["prompt_tokens"], t["completion_tokens"]) == (1, 44, 21)
    assert (t["reasoning_tokens"], t["cached_tokens"]) == (18, 4)


def test_truncated_call_still_spent_tokens() -> None:
    meter = UsageMeter()
    with pytest.raises(OutputTruncatedError):
        _chat(_FakeSDK(_reply("half", finish="length", usage=_usage(10, 100))), meter)
    assert meter.totals()["judge"]["completion_tokens"] == 100


def test_failed_request_is_not_counted() -> None:
    req = httpx.Request("POST", "http://test/v1/chat/completions")
    err = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    meter = UsageMeter()
    with pytest.raises(PermanentLLMError):
        _chat(_FakeSDK(err), meter)
    assert meter.totals() == {}


def test_retried_calls_each_count() -> None:
    meter = UsageMeter()
    # an empty answer is retried; both replies cost tokens
    _chat(_FakeSDK(_reply("", usage=_usage(5, 7)), _reply("ok", usage=_usage(5, 3))), meter)
    t = meter.totals()["judge"]
    assert (t["calls"], t["completion_tokens"]) == (2, 10)


def test_missing_usage_counts_the_call_with_zero_tokens() -> None:
    meter = UsageMeter()
    _chat(_FakeSDK(_reply("ok", usage=None)), meter)
    t = meter.totals()["judge"]
    assert (t["calls"], t["prompt_tokens"], t["usage_missing"]) == (1, 0, 1)


def test_each_call_is_appended_to_the_jsonl_log(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    meter = UsageMeter(path)
    _chat(_FakeSDK(_reply("ok", usage=_usage(44, 21, reasoning=18))), meter)
    _chat(_FakeSDK(_reply("ok", usage=_usage(9, 2))), meter, _role("strong_solver", "qw-flash"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["role"] for r in rows] == ["judge", "strong_solver"]
    assert rows[0]["model"] == "ds-flash" and rows[0]["reasoning_tokens"] == 18
    assert rows[1]["finish_reason"] == "stop" and rows[1]["latency_s"] >= 0


def test_concurrent_records_sum_exactly() -> None:
    meter = UsageMeter()

    def work() -> None:
        for _ in range(200):
            meter.record("weak_solver", "q4b", _usage(1, 2), latency_s=0.0, finish_reason="stop")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t = meter.totals()["weak_solver"]
    assert (t["calls"], t["prompt_tokens"], t["completion_tokens"]) == (1600, 1600, 3200)


def test_pool_shares_one_meter_across_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "http://test/v1")
    monkeypatch.setenv("PROXY_API_KEY", "k")
    monkeypatch.setenv("WEAK_SOLVER_BASE_URL", "http://weak/v1")
    meter = UsageMeter()
    roles = {
        "judge": _role(),
        "weak_solver": LLMRoleConfig(role="weak_solver", endpoint="vllm", model="q4b",
                                     max_tokens=100, base_url_env="WEAK_SOLVER_BASE_URL",
                                     api_key_env="WEAK_SOLVER_API_KEY"),
    }
    pool = build_client_pool(roles, meter=meter)
    assert pool["judge"].backend._meter is meter
    assert pool["weak_solver"].backend._meter is meter
