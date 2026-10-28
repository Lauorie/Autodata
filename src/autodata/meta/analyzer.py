"""The analyzer of the outer loop (arXiv 2606.25996, Sec 4 step 3).

It reads the trajectories of the parent's train sessions and writes a
root-cause analysis of systematic failures. Its prompt describes the
pipeline and the acceptance criteria, nothing more: what the paper's
optimizer found is the result to reproduce, so it is not a hint here.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..pipeline.acceptance import AcceptanceCriteria


def describe_criteria(c: AcceptanceCriteria) -> str:
    zeros = "; no weak answer scores 0" if c.no_zeros else ""
    return (f"the verifier passes it; the mean weak score is at most {c.weak_avg_max} and "
            f"the best weak score at most {c.weak_max_max}{zeros}; the mean strong score is "
            f"at least {c.strong_avg_min} and below {c.strong_avg_max}; and the mean strong "
            f"score exceeds the mean weak score by at least {c.gap_min}")


def pipeline_brief(c: AcceptanceCriteria) -> str:
    return f"""\
The pipeline turns computer-science papers into research questions. For each paper:
1. A challenger model reads the whole paper and writes a package: a question type, skill \
tags, a context, a question, a reference answer and a weighted rubric.
2. A quality verifier reads the paper and the package, and passes or rejects it.
3. A weak solver answers several times. It sees only the context and the question.
4. A judge scores each answer on each rubric criterion in [0, 1], without the reference \
answer. Positive criteria add their weight times the score; negative criteria subtract \
theirs. The sum is divided by the total positive weight and floored at 0.
5. If the weak scores pass, a strong solver answers and is scored the same way.

A package is accepted when {describe_criteria(c)}. When a package fails, the challenger \
gets the paper's earlier failed questions, grouped by failure mode, and writes a new one, \
up to a round budget.

Round outcomes: accepted; failed_qv (the verifier rejected it); too_easy (weak scores \
above the ceiling); weak_zero (a weak answer scored 0); failed_on_strong (strong scores \
too low, too high, or too close to the weak ones); malformed (the challenger's reply \
could not be parsed). Rounds lost to infrastructure failures are left out."""


def analyzer_system(c: AcceptanceCriteria) -> str:
    return f"""\
{pipeline_brief(c)}

You are the analyzer of a loop that rewrites the challenger's and the verifier's \
instructions. You receive the trajectories of a run of the current instructions: every \
round of every session, with the package, the verifier's verdict, the solver answers, the \
per-criterion scores and the judge's feedback. You also receive the log of earlier \
iterations: what was changed and whether it helped.

Write a root-cause analysis of the failures. For each recurring pattern give what happens, \
how often, the evidence (session and round numbers, quoted text, scores), and the cause in \
the challenger's or the verifier's behavior. Read the answers and the per-criterion scores, \
not only the outcome labels. Say which causes a change of instructions could remove and \
which it could not. Do not write new instructions: another agent does that.

Plain text, under 1000 words."""


def analyze(client: LLMClient, sessions: str, history: str, criteria: AcceptanceCriteria) -> str:
    user = f"=== TRAJECTORIES ===\n{sessions}\n\n=== ITERATION LOG ===\n{history}"
    return client.chat([
        {"role": "system", "content": analyzer_system(criteria)},
        {"role": "user", "content": user},
    ]).strip()


__all__ = ["analyze", "analyzer_system", "describe_criteria", "pipeline_brief"]
