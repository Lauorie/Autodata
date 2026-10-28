"""LLM client abstraction.

Three endpoints supported:
- ``proxy``: OpenAI Chat Completions API at the wismodel proxy (which fronts OpenRouter).
- ``vllm``: an OpenAI-compatible vLLM server (the 4B weak solver; see
  scripts/serve_weak_solver.sh).
- ``local_hf``: in-process HuggingFace transformers inference. One sample at a
  time: fine for smoke tests, too slow for a thinking budget.

A pool of ``LLMClient``s is shared across the inner loop so we don't
re-instantiate the HF model for every call.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import OutputTruncatedError, TransientLLMError, classify_api_error
from .thinking import split_thinking
from .usage import UsageMeter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRoleConfig:
    """Configuration for one role (challenger, judge, weak solver, ...)."""

    role: str
    endpoint: str            # 'proxy' | 'vllm' | 'local_hf'
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000   # caps thinking + answer together
    thinking: bool = False
    # With thinking on, the effort the proxy asks for; 'auto' keeps the
    # model's default. Thinking off always sends 'none'.
    reasoning_effort: str = "auto"
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    # Env vars holding the server address and key ('vllm' and 'proxy').
    base_url_env: str = "PROXY_BASE_URL"
    api_key_env: str = "PROXY_API_KEY"

    def __post_init__(self) -> None:
        if self.thinking and self.reasoning_effort == "none":
            raise ValueError(f"{self.role}: thinking=true contradicts reasoning_effort=none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _balanced_extract(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` substring in ``text``.

    Walks the string, tracks a stack of opener types so close brackets must
    match (``{...]`` is rejected), and skips characters inside string
    literals. If the first opener never balances, we keep scanning for the
    next one rather than giving up immediately.
    """
    pairs = {"{": "}", "[": "]"}
    closers = set(pairs.values())
    n = len(text)
    start = 0
    while start < n:
        if text[start] not in pairs:
            start += 1
            continue
        stack: list[str] = []
        in_str = False
        escape = False
        i = start
        while i < n:
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
            elif ch in pairs:
                stack.append(pairs[ch])
            elif ch in closers:
                if not stack or stack[-1] != ch:
                    break  # unbalanced — try the next opener
                stack.pop()
                if not stack:
                    return text[start:i + 1]
            i += 1
        start += 1  # this opener didn't balance; try the next candidate
    return None


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array from a model's text output.

    Tries (in order):
      1. JSON fenced in a ```json ... ``` block
      2. The first balanced ``{...}`` or ``[...]`` substring
      3. The whole stripped text
    Falls back to repairing trailing commas before giving up.
    """
    if text is None:
        raise ValueError("empty model output")
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        candidate = m.group(1)
    else:
        bal = _balanced_extract(text)
        candidate = bal if bal is not None else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
        return json.loads(cleaned)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (proxy / vLLM)
# ---------------------------------------------------------------------------

class _OpenAIBackend:
    """Chat Completions over the OpenAI SDK.

    ``flavor`` picks how thinking is switched: 'openrouter' (the proxy) uses
    ``reasoning.effort``; 'vllm' uses the chat template's ``enable_thinking``.
    """

    def __init__(self, client: Any, flavor: str, label: str,
                 meter: UsageMeter | None = None) -> None:
        if flavor not in ("openrouter", "vllm"):
            raise ValueError(f"unknown flavor {flavor!r}")
        self._client = client
        self._flavor = flavor
        self._label = label
        self._warned_no_reasoning: set[str] = set()
        self._meter = meter

    @classmethod
    def from_env(cls, role_cfg: LLMRoleConfig, meter: UsageMeter | None = None) -> _OpenAIBackend:
        from openai import OpenAI

        base_url = os.environ.get(role_cfg.base_url_env)
        api_key = os.environ.get(role_cfg.api_key_env)
        if role_cfg.endpoint == "vllm":
            api_key = api_key or "EMPTY"            # vLLM ignores the key
        if not api_key or not base_url:
            raise RuntimeError(
                f"{role_cfg.base_url_env} and {role_cfg.api_key_env} env vars must be set "
                "(copy .env.example to .secrets/remote.env)"
            )
        channel = os.environ.get("PROXY_HEADER_X_CHANNEL") if role_cfg.endpoint == "proxy" else None
        # max_retries=0: `chat` retries transient errors itself; SDK retries on
        # top would multiply every attempt and retry errors the taxonomy stops.
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=1800.0, max_retries=0,
                        default_headers={"X-Channel": channel} if channel else None)
        flavor = "openrouter" if role_cfg.endpoint == "proxy" else "vllm"
        return cls(client, flavor=flavor, label=role_cfg.base_url_env, meter=meter)

    def _request(self, role_cfg: LLMRoleConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
        extra_body: dict[str, Any] = {}
        if self._flavor == "vllm":
            extra_body["chat_template_kwargs"] = {"enable_thinking": role_cfg.thinking}
        elif not role_cfg.thinking:
            extra_body["reasoning"] = {"effort": "none"}
        elif role_cfg.reasoning_effort != "auto":
            extra_body["reasoning"] = {"effort": role_cfg.reasoning_effort}
        if role_cfg.top_k is not None:
            extra_body["top_k"] = role_cfg.top_k
        req: dict[str, Any] = {
            "model": role_cfg.model,
            "messages": messages,
            "temperature": role_cfg.temperature,
            "max_tokens": role_cfg.max_tokens,
            "extra_body": extra_body or None,
        }
        if role_cfg.top_p is not None:
            req["top_p"] = role_cfg.top_p
        if role_cfg.presence_penalty is not None:
            req["presence_penalty"] = role_cfg.presence_penalty
        return req

    def _check_reasoning(self, role_cfg: LLMRoleConfig, message: Any, raw: str) -> None:
        """Log once per role when thinking was asked for and none came back."""
        if not role_cfg.thinking or role_cfg.role in self._warned_no_reasoning:
            return
        if getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None):
            return
        if "</think>" in raw:
            return
        self._warned_no_reasoning.add(role_cfg.role)
        logger.warning("[%s] thinking=true but %s returned no reasoning", role_cfg.role,
                       role_cfg.model)

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def chat(self, role_cfg: LLMRoleConfig, messages: list[dict[str, str]]) -> str:
        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**self._request(role_cfg, messages))
        except openai.APIError as e:
            err = classify_api_error(e)
            logger.warning("[%s] %s from %s: %s", role_cfg.role, type(err).__name__,
                           self._label, e)
            raise err from e
        choice = resp.choices[0]
        if self._meter is not None:
            self._meter.record(role_cfg.role, role_cfg.model, getattr(resp, "usage", None),
                               time.monotonic() - start, choice.finish_reason)
        raw = choice.message.content or ""
        answer, finished = split_thinking(raw, prompt_opened=False)
        self._check_reasoning(role_cfg, choice.message, raw)
        if choice.finish_reason == "length" or not finished:
            raise OutputTruncatedError(role_cfg.role, role_cfg.max_tokens, answer)
        if not answer:
            logger.warning("[%s] empty content (finish=%s), retrying", role_cfg.role,
                           choice.finish_reason)
            raise TransientLLMError("empty content")
        return answer


# ---------------------------------------------------------------------------
# Local HF backend (single-process, used for the 4B weak solver)
# ---------------------------------------------------------------------------

class _LocalHFBackend:
    """Single-instance HF causal-LM inference for the weak solver.

    Models are cached by path; loading is thread-safe.
    """

    _instances: dict[str, _LocalHFBackend] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_path: str) -> _LocalHFBackend:
        with cls._lock:
            inst = cls._instances.get(model_path)
            if inst is None:
                inst = cls(model_path)
                cls._instances[model_path] = inst
            return inst

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading local HF model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
            device_map=device,
        )
        self._model.eval()
        self._device = device
        self._gen_lock = threading.Lock()
        logger.info("model loaded on %s, params=%.2fB",
                    device, sum(p.numel() for p in self._model.parameters()) / 1e9)

    def chat(self, role_cfg: LLMRoleConfig, messages: list[dict[str, str]]) -> str:
        import torch

        # Qwen3.5 thinks unless told otherwise; pass the switch explicitly.
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=role_cfg.thinking,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        do_sample = role_cfg.temperature > 0
        sampling: dict[str, Any] = {}
        if do_sample:
            sampling = {"temperature": role_cfg.temperature, "top_p": role_cfg.top_p or 1.0}
            if role_cfg.top_k is not None:
                sampling["top_k"] = role_cfg.top_k
        with self._gen_lock, torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=role_cfg.max_tokens,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                **sampling,
            )
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        hit_budget = (len(gen_tokens) >= role_cfg.max_tokens
                      and int(gen_tokens[-1]) != self._tokenizer.eos_token_id)
        # Think tags are ordinary tokens, so skipping special tokens keeps them.
        raw = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        answer, finished = split_thinking(raw, prompt_opened=role_cfg.thinking)
        if hit_budget or not finished:
            raise OutputTruncatedError(role_cfg.role, role_cfg.max_tokens, answer)
        return answer


# ---------------------------------------------------------------------------
# Public client + factory
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """Front-facing LLM caller, parameterised by role.

    Holds a reference to a backend object, which may be shared across roles
    (e.g. all proxy roles share one OpenAI client).
    """

    role_cfg: LLMRoleConfig
    backend: _OpenAIBackend | _LocalHFBackend

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.backend.chat(self.role_cfg, messages)

    def chat_json(self, messages: list[dict[str, str]], retries: int = 2) -> Any:
        """Call the model and parse a JSON object/array from its response.

        Re-prompts on parse failure.
        """
        last_err: Exception | None = None
        msgs = list(messages)
        for attempt in range(retries + 1):
            text = self.chat(msgs)
            try:
                return extract_json(text)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                logger.warning("[%s] JSON parse fail attempt %d: %s", self.role_cfg.role, attempt, e)
                msgs = list(messages) + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"Your previous response could not be parsed as JSON ({e}). "
                                   f"Reply ONLY with valid JSON, no prose, no markdown fences.",
                    },
                ]
        raise ValueError(f"could not parse JSON after {retries + 1} attempts: {last_err}")


@dataclass
class _Pool:
    """Shared backends so we don't create N OpenAI clients or N model copies."""

    remote: dict[tuple[str, str], _OpenAIBackend] = field(default_factory=dict)
    local_hf: dict[str, _LocalHFBackend] = field(default_factory=dict)


def build_client_pool(role_configs: dict[str, LLMRoleConfig],
                      meter: UsageMeter | None = None) -> dict[str, LLMClient]:
    """Build one ``LLMClient`` per role, sharing backends underneath.

    ``meter`` counts the tokens of every remote call (local HF is not metered).
    """
    pool = _Pool()
    out: dict[str, LLMClient] = {}
    for role, cfg in role_configs.items():
        if cfg.endpoint in ("proxy", "vllm"):
            key = (cfg.endpoint, cfg.base_url_env)
            if key not in pool.remote:
                pool.remote[key] = _OpenAIBackend.from_env(cfg, meter)
            out[role] = LLMClient(cfg, pool.remote[key])
        elif cfg.endpoint == "local_hf":
            backend = pool.local_hf.get(cfg.model)
            if backend is None:
                backend = _LocalHFBackend.get(cfg.model)
                pool.local_hf[cfg.model] = backend
            out[role] = LLMClient(cfg, backend)
        else:
            raise ValueError(f"unknown endpoint {cfg.endpoint!r} for role {role}")
    return out


__all__ = [
    "LLMClient",
    "LLMRoleConfig",
    "TransientLLMError",
    "build_client_pool",
    "extract_json",
]
