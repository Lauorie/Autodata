"""Code-edit agent: ask the meta-optimizer LLM to propose a mutated prompts file.

The blog mutates the *agent harness* (prompts + scoring code). For tractability
we mutate only the prompts module — that's where most of the leverage lives,
and it avoids breaking the inner-loop control flow with hallucinated diffs.

The mutator is fed:
- the parent prompts source verbatim
- a list of failure patterns observed when running the parent (top recurring
  rejection reasons + sample judge feedback)

It is asked to return the FULL revised prompts source (rewrite, not diff)
so we don't have to parse and apply patches.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from typing import Any

from ..llm import LLMClient

logger = logging.getLogger(__name__)


META_SYSTEM = """\
You are a meta-optimizer improving the prompt scaffolds of a synthetic-data
generation pipeline (Agentic Self-Instruct). The pipeline uses these prompts
to drive a Challenger LLM, a Quality Verifier LLM, a Solver LLM, and a Judge LLM
in a loop until each generated QA pair satisfies acceptance criteria.

You will be given:
  1. The current Python source of `prompts.py` (verbatim)
  2. A list of recurring failure modes observed in the most recent run

Your job: produce a revised `prompts.py` source that should improve the
fraction of accepted QA pairs without sacrificing quality.

Constraints (hard):
- Reply with ONLY the new full Python source. No prose, no markdown fences.
- All public symbols MUST be preserved: CHALLENGER_SYSTEM, QUALITY_VERIFIER_SYSTEM,
  SOLVER_SYSTEM, JUDGE_SYSTEM, challenger_user(...), quality_verifier_user(...),
  solver_user(...), judge_user(...). The `__all__` list must remain.
- Function signatures must be unchanged (same parameter names + order).
- Maintain the JSON output contracts described in the existing prompts so
  downstream parsers keep working.
- Stay under 8 KB total.

Permitted edits (examples):
- Tighten / elaborate the Challenger system prompt to push paper-specific reasoning
- Add explicit context-leak rules to the Quality Verifier
- Constrain rubric weight format more strictly
- Add positive-only criterion enforcement
- Adjust JSON schema instructions for parsing reliability
"""


def summarize_failure_patterns(rejection_reasons: Iterable[Iterable[str]],
                               judge_feedbacks: Iterable[str],
                               top_k: int = 6) -> str:
    """Aggregate the top-K recurring failure patterns from a run."""
    flat = Counter()
    for reasons in rejection_reasons:
        for r in reasons:
            # bucket "weak_avg 0.78 > 0.65" -> "weak_avg_too_high"
            tag = r.split()[0] if r else "unknown"
            if "weak_avg" in r:
                tag = "weak_avg_too_high"
            elif "weak_max" in r:
                tag = "weak_max_too_high"
            elif "strong_avg" in r and "<" in r:
                tag = "strong_avg_too_low"
            elif "strong_avg" in r and "≥" in r or "saturated" in r:
                tag = "strong_avg_saturated"
            elif "gap" in r:
                tag = "gap_too_small"
            elif "weak_too_strong" in r:
                tag = "weak_too_strong_short_circuit"
            flat[tag] += 1
    top_reasons = "\n".join(f"  - {tag}: {count}" for tag, count in flat.most_common(top_k))
    sample_fb = "\n".join(f"  - {fb[:200]}" for fb in list(judge_feedbacks)[:5])
    return f"TOP RECURRING REJECTION REASONS:\n{top_reasons}\n\nSAMPLE JUDGE FEEDBACK:\n{sample_fb}"


def propose_mutation(
    meta_optimizer: LLMClient,
    parent_prompts_source: str,
    failure_summary: str,
) -> str:
    """Return revised `prompts.py` source from the meta-optimizer LLM."""
    user = (
        f"=== CURRENT PROMPTS SOURCE ===\n{parent_prompts_source}\n\n"
        f"=== FAILURE SUMMARY ===\n{failure_summary}\n\n"
        "Reply with the new full prompts.py source ONLY."
    )
    raw = meta_optimizer.chat([
        {"role": "system", "content": META_SYSTEM},
        {"role": "user", "content": user},
    ])
    # If the model wrapped the source in code fences, strip them.
    src = raw.strip()
    if src.startswith("```"):
        src = src.split("\n", 1)[1] if "\n" in src else src
        if src.endswith("```"):
            src = src.rsplit("```", 1)[0]
        # also handle ```python on the fence
        src = src.removeprefix("python\n")
    return src.strip() + "\n"


def validate_mutated_source(src: str) -> tuple[bool, str]:
    """Quick syntactic + symbol-presence check before importing."""
    required = [
        "CHALLENGER_SYSTEM",
        "QUALITY_VERIFIER_SYSTEM",
        "SOLVER_SYSTEM",
        "JUDGE_SYSTEM",
        "def challenger_user",
        "def quality_verifier_user",
        "def solver_user",
        "def judge_user",
    ]
    missing = [r for r in required if r not in src]
    if missing:
        return False, f"missing required symbols: {missing}"
    try:
        compile(src, "<mutated prompts>", "exec")
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    return True, "ok"


__all__: list[Any] = [
    "META_SYSTEM",
    "propose_mutation",
    "summarize_failure_patterns",
    "validate_mutated_source",
]
