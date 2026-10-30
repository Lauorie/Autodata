"""Token usage of every completed LLM call, per call and per role.

A call counts once the server answered, whatever came next: a truncated
reply or an empty one that gets retried still spent its tokens. A request
that failed outright returned no usage and is not counted.

``UsageMeter(path)`` appends one JSON line per call to ``path``; ``totals()``
sums them per role. ``sum_usage_logs`` sums the logs of several processes
the same way. Price is left out: it lives with the gateway, not here.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")


def _tokens(usage: Any) -> dict[str, int]:
    """Read the four counts from an SDK usage object; absent ones are 0."""
    details_out = getattr(usage, "completion_tokens_details", None)
    details_in = getattr(usage, "prompt_tokens_details", None)
    raw = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details_out, "reasoning_tokens", None),
        "cached_tokens": getattr(details_in, "cached_tokens", None),
    }
    return {k: int(v or 0) for k, v in raw.items()}


def _add(totals: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    """Fold one logged call into per-role totals."""
    t = totals.setdefault(row["role"], {"model": row["model"], "calls": 0, "usage_missing": 0,
                                        "latency_s": 0.0, **dict.fromkeys(_FIELDS, 0)})
    t["calls"] += 1
    t["usage_missing"] += bool(row["usage_missing"])
    t["latency_s"] += row["latency_s"]
    for k in _FIELDS:
        t[k] += row[k]


class UsageMeter:
    """Thread-safe call log and per-role totals."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._totals: dict[str, dict[str, Any]] = {}

    def record(self, role: str, model: str, usage: Any, latency_s: float,
               finish_reason: str | None) -> None:
        row = {"role": role, "model": model, **_tokens(usage), "usage_missing": usage is None,
               "latency_s": round(latency_s, 3), "finish_reason": finish_reason}
        with self._lock:
            _add(self._totals, row)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    def totals(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {role: dict(t) for role, t in self._totals.items()}


def sum_usage_logs(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Per-role totals over call logs, e.g. one per training process."""
    totals: dict[str, dict[str, Any]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    _add(totals, json.loads(line))
    return totals


__all__ = ["UsageMeter", "sum_usage_logs"]
