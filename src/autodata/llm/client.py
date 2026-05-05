"""LLM client abstraction.

Two endpoints supported:
- ``proxy``: OpenAI Chat Completions API at the wismodel proxy (which fronts OpenRouter).
- ``local_hf``: in-process HuggingFace transformers inference (used for the local 4B weak solver).

A pool of ``LLMClient``s is shared across the inner loop so we don't
re-instantiate the HF model for every call.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRoleConfig:
    """Configuration for one role (challenger, judge, weak solver, ...)."""

    role: str
    endpoint: str            # 'proxy' | 'local_hf'
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    reasoning: bool = False  # whether the model emits a `reasoning` field that should be ignored
    # 'none' tells the proxy/OpenRouter to skip thinking on reasoning-capable models
    # (huge latency saving on Kimi-K2.6 / Qwen3.5-397B-A17B). 'auto' leaves the
    # model's default behaviour intact.
    reasoning_effort: str = "auto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TransientLLMError(RuntimeError):
    pass


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
# Proxy (OpenAI-compatible) backend
# ---------------------------------------------------------------------------

class _ProxyBackend:
    """Wraps the OpenAI SDK pointed at the wismodel proxy."""

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("PROXY_API_KEY")
        base_url = os.environ.get("PROXY_BASE_URL")
        channel = os.environ.get("PROXY_HEADER_X_CHANNEL")
        if not api_key or not base_url:
            raise RuntimeError(
                "PROXY_API_KEY and PROXY_BASE_URL env vars must be set "
                "(see .secrets/remote.env.example)"
            )
        default_headers = {"X-Channel": channel} if channel else None
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            timeout=600.0,
        )

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def chat(self, role_cfg: LLMRoleConfig, messages: list[dict[str, str]]) -> str:
        extra_body: dict[str, Any] = {}
        if role_cfg.reasoning_effort != "auto":
            extra_body["reasoning"] = {"effort": role_cfg.reasoning_effort}
        try:
            resp = self._client.chat.completions.create(
                model=role_cfg.model,
                messages=messages,
                temperature=role_cfg.temperature,
                max_tokens=role_cfg.max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as e:  # network / 5xx / rate-limit
            logger.warning("proxy error for %s: %s", role_cfg.model, e)
            raise TransientLLMError(str(e)) from e
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            # Reasoning-only response (model spent its budget thinking) -> retry with a nudge
            logger.warning("empty content from %s (finish=%s) — retrying", role_cfg.model, choice.finish_reason)
            raise TransientLLMError("empty content from reasoning model")
        return content


# ---------------------------------------------------------------------------
# Local HF backend (single-process, used for the 4B weak solver)
# ---------------------------------------------------------------------------

class _LocalHFBackend:
    """Single-instance HF causal-LM inference for the weak solver.

    Models are cached by path; loading is thread-safe.
    """

    _instances: dict[str, "_LocalHFBackend"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_path: str) -> "_LocalHFBackend":
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

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        do_sample = role_cfg.temperature > 0
        with self._gen_lock, torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=role_cfg.max_tokens,
                do_sample=do_sample,
                temperature=role_cfg.temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


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
    backend: _ProxyBackend | _LocalHFBackend

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

    proxy: _ProxyBackend | None = None
    local_hf: dict[str, _LocalHFBackend] = field(default_factory=dict)


def build_client_pool(role_configs: dict[str, LLMRoleConfig]) -> dict[str, LLMClient]:
    """Build one ``LLMClient`` per role, sharing backends underneath."""
    pool = _Pool()
    out: dict[str, LLMClient] = {}
    for role, cfg in role_configs.items():
        if cfg.endpoint == "proxy":
            if pool.proxy is None:
                pool.proxy = _ProxyBackend()
            out[role] = LLMClient(cfg, pool.proxy)
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
