"""The iteration log (arXiv 2606.25996, Sec 4 step 7).

One record per iteration, appended to ``history.jsonl`` and rendered for
the next analyzer and implementer so they do not repeat rejected changes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# accepted | rejected | invalid_eval | no_proposal | analysis_failed
STATUSES = frozenset({"accepted", "rejected", "invalid_eval", "no_proposal", "analysis_failed"})


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    parent_id: str
    status: str
    child_id: str | None = None
    parent_score: float | None = None   # parent's mean after this iteration's re-evaluation
    child_score: float | None = None
    changed: tuple[str, ...] = ()
    rationale: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")


def _score(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_log(records: Sequence[IterationRecord], limit: int = 30) -> str:
    """The last ``limit`` records, oldest first."""
    if not records:
        return "(no earlier iterations)"
    lines = []
    for r in records[-limit:]:
        head = (f"iteration {r.iteration}: parent {r.parent_id} ({_score(r.parent_score)}) -> "
                f"{r.status}")
        if r.child_id:
            head += f", child {r.child_id} ({_score(r.child_score)})"
        if r.changed:
            head += f", changed {', '.join(r.changed)}"
        lines.append(head)
        for label, text in (("rationale", r.rationale), ("note", r.note)):
            if text:
                lines.append(f"  {label}: {' '.join(text.split())}")
    return "\n".join(lines)


def append_record(path: Path, record: IterationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


__all__ = ["STATUSES", "IterationRecord", "append_record", "render_log"]
