"""Separate a reasoning model's thinking from its answer.

Qwen3.5 thinks by default: its chat template ends the generation prompt with
``<think>\\n``, so raw output reads ``...thought...</think>\\n\\nanswer``. The
think tags are ordinary tokens (``special: false``), so decoding keeps them.
A server with a reasoning parser (vLLM ``--reasoning-parser qwen3``) moves the
thought to ``reasoning_content`` and leaves only the answer.
"""

from __future__ import annotations

_OPEN, _CLOSE = "<think>", "</think>"


def split_thinking(text: str, prompt_opened: bool) -> tuple[str, bool]:
    """Return ``(answer, finished)``.

    ``prompt_opened`` is True when the prompt itself ended with ``<think>``
    (local generation with thinking on). An unterminated thought means the
    model never reached its answer: ``("", False)``.
    """
    if _CLOSE in text:
        return text.rsplit(_CLOSE, 1)[1].strip(), True
    if prompt_opened or text.lstrip().startswith(_OPEN):
        return "", False
    return text.strip(), True


__all__ = ["split_thinking"]
