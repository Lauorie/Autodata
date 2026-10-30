"""Refinement feedback for the challenger (arXiv 2606.25996, Sec 3.1, Fig 7).

Each refinement round lists every earlier failed question of the paper,
grouped by failure mode: too easy (with weak solver scores), failed on the
strong solver (with gap information), rejected by the quality verifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Failure modes in the order the challenger reads them. The first three are
# Fig 7's groups. WEAK SCORED ZERO keeps a no-zeros failure from posing as
# "too easy" when the weak scores say the opposite.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("too_easy", "TOO EASY (the weak solver scored too high)"),
    ("failed_on_strong", "FAILED ON STRONG (the strong solver did not clear the bar)"),
    ("failed_qv", "FAILED QV (rejected by the quality verifier)"),
    ("weak_zero", "WEAK SCORED ZERO (a weak attempt scored 0: the question may be "
                  "unanswerable from the context)"),
)
FAILURE_MODES = frozenset(mode for mode, _ in _GROUPS)

SNIPPET_CHARS = 300


@dataclass(frozen=True)
class FailedAttempt:
    mode: str        # one of FAILURE_MODES
    question: str
    detail: str      # scores, gap, or verifier issues

    def __post_init__(self) -> None:
        if self.mode not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {self.mode!r}")


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and cap length, for judge and verifier feedback."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def render_history(history: Sequence[FailedAttempt]) -> str:
    """Group failed questions by mode. Questions stay whole so the challenger
    can see exactly what not to repeat."""
    if not history:
        return ""
    blocks = ["PREVIOUSLY FAILED QUESTIONS, grouped by failure mode:"]
    for mode, heading in _GROUPS:
        items = [a for a in history if a.mode == mode]
        if not items:
            continue
        lines = [f"{heading}:"]
        for a in items:
            lines += [f"- Q: {a.question}", f"  {a.detail}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["FAILURE_MODES", "FailedAttempt", "render_history", "snippet"]
