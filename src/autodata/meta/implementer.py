"""The implementer of the outer loop (arXiv 2606.25996, Sec 4 step 4).

It reads the analysis, the iteration log and the current mutable prompts,
and returns new text for the fields it changes, one delimited block each:

    <<<rationale>>>
    ...
    <<<end>>>
    <<<challenger_system>>>
    full new text
    <<<end>>>

Blocks, not JSON: a prompt full of quotes and braces is easy to break in a
JSON string. A reply that touches a frozen field, breaks a template slot or
a parser key, changes nothing, or grows past the size cap is sent back with
the problems named, up to ``retries`` times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm import LLMClient
from ..llm.prompt_set import MUTABLE_FIELDS, PromptSet
from ..pipeline.acceptance import AcceptanceCriteria
from .analyzer import pipeline_brief

_BLOCK = re.compile(r"^<<<(\w+)>>>\n(.*?)\n<<<end>>>", re.S | re.M)


class ImplementerError(ValueError):
    """No usable proposal after the re-asks."""


@dataclass(frozen=True)
class Proposal:
    prompts: PromptSet
    rationale: str
    changed: tuple[str, ...]


def implementer_system(c: AcceptanceCriteria, max_prompt_chars: int) -> str:
    return f"""\
{pipeline_brief(c)}

You edit the challenger's and the verifier's instructions so that more packages are \
accepted, without making the questions worse as research questions. You receive the \
current instructions, a root-cause analysis of a recent run, and the log of earlier \
iterations.

Fields you may rewrite:
- challenger_system: the challenger's system prompt.
- challenger_first: the task text of the challenger's first turn on a paper.
- challenger_refine: the task text of a later turn. $feedback stands for the paper's \
earlier failed questions; it must stay, and no other $name may appear.
- qv_system: the verifier's system prompt.
The solver and judge prompts and the acceptance criteria are fixed.

Contracts the code depends on:
- The challenger replies with one JSON object with the keys question_type, skill_tags, \
context, question, reference_answer and rubric. rubric is a flat array of items with the \
keys criterion (text), weight (a nonzero integer from -10 to 10) and category ("positive" \
when weight > 0, "negative" when weight < 0).
- The verifier replies with one JSON object with the keys check_1_verdict, \
check_2_verdict, check_3_verdict, check_4_verdict, overall and feedback. A package passes \
only when the checks read NO_LEAKAGE, GOOD, PASS and CONSISTENT and overall reads PASS.
- The four fields together stay under {max_prompt_chars} characters.

Use the log: do not repeat a rejected change unless the analysis gives a new reason.

Reply in this format and nothing else. Include only the fields you change, each in full:
<<<rationale>>>
What you changed and why, in a few sentences.
<<<end>>>
<<<challenger_system>>>
the full new text
<<<end>>>"""


def _current(prompts: PromptSet) -> str:
    return "\n".join(f"<<<{f}>>>\n{getattr(prompts, f)}\n<<<end>>>" for f in MUTABLE_FIELDS)


def _parse(raw: str, parent: PromptSet,
           max_prompt_chars: int) -> tuple[Proposal | None, list[str]]:
    # Strip trailing newlines as PromptSet.load does, so an unchanged field
    # with a blank line after it is no change.
    blocks = {name: text.rstrip("\n") for name, text in _BLOCK.findall(raw)}
    rationale = blocks.pop("rationale", "").strip()
    problems = [f"{name} is not a field you may change" for name in blocks
                if name not in MUTABLE_FIELDS]
    changes = {k: v for k, v in blocks.items()
               if k in MUTABLE_FIELDS and v != getattr(parent, k)}
    if not changes:
        problems.append("no field changed: return at least one field with new text")
    if problems:
        return None, problems
    child = parent.evolve(**changes)
    problems = child.problems()
    size = sum(len(getattr(child, f)) for f in MUTABLE_FIELDS)
    if size > max_prompt_chars:
        problems.append(f"the four fields hold {size} characters, over {max_prompt_chars}")
    if problems:
        return None, problems
    changed = tuple(f for f in MUTABLE_FIELDS if f in changes)
    return Proposal(child, rationale, changed), []


def propose(client: LLMClient, parent: PromptSet, analysis: str, history: str,
            criteria: AcceptanceCriteria, max_prompt_chars: int, retries: int) -> Proposal:
    user = (f"=== CURRENT INSTRUCTIONS ===\n{_current(parent)}\n\n"
            f"=== ANALYSIS ===\n{analysis}\n\n=== ITERATION LOG ===\n{history}")
    base = [{"role": "system", "content": implementer_system(criteria, max_prompt_chars)},
            {"role": "user", "content": user}]
    messages = base
    problems: list[str] = []
    for _ in range(retries + 1):
        raw = client.chat(messages)
        proposal, problems = _parse(raw, parent, max_prompt_chars)
        if proposal is not None:
            return proposal
        messages = base + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your reply could not be used: " + "; ".join(problems)
                                        + ". Reply again in the format, each changed field "
                                          "in full."},
        ]
    raise ImplementerError("; ".join(problems))


__all__ = ["ImplementerError", "Proposal", "implementer_system", "propose"]
