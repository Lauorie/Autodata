"""Prompts as data (arXiv 2606.25996, Appendix C.1; Sec 4).

Every prompt is a text file in a prompt directory, one file per field. The
seed set ``prompt_text/cs`` holds the baseline CS prompts (Fig 8 challenger,
Fig 9 quality verifier), adapted from file-reading agents to direct calls:
the paper text arrives in the user message instead of ``./paper.txt``.

User messages are ``string.Template`` files filled with ``safe_substitute``:
no ``exec``, no ``str.format``, so braces in prompts and dollars in papers
pass through untouched.

The meta-optimizer may rewrite only ``MUTABLE_FIELDS``, the instructions of
the agent being optimized (challenger and quality verifier; Sec 4 Setup).
The rest is fixed:

- The solver and judge prompts are the yardstick. A variant that rewrote
  them could raise its own score without making better data.
- The user-message frames are the data contract: the challenger and
  verifier read the whole paper, the solvers read only the context and the
  question, and the judge has no reference answer (Sec 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path
from string import Template

SEED_DIR = Path(__file__).parent / "prompt_text" / "cs"

MUTABLE_FIELDS = ("challenger_system", "challenger_first", "challenger_refine", "qv_system")

# Placeholders each template uses, no more and no fewer. Fields not listed
# here are sent as they are.
_TEMPLATE_VARS: dict[str, frozenset[str]] = {
    "challenger_user": frozenset({"paper_title", "paper_id", "paper_text", "task"}),
    "challenger_refine": frozenset({"feedback"}),
    "qv_user": frozenset({"paper_title", "paper_text", "question_type", "context",
                          "question", "rubric_json"}),
    "solver_user": frozenset({"context", "question"}),
    "judge_user": frozenset({"context", "question", "candidate_answer", "rubric_json"}),
}

# Strings the output parsers depend on. A prompt that drops one asks for a
# reply the pipeline cannot read.
_PARSER_KEYS: dict[str, tuple[str, ...]] = {
    "challenger_system": ("question_type", "skill_tags", "context", "question",
                          "reference_answer", "rubric", "criterion", "weight", "category"),
    "qv_system": ("check_1_verdict", "check_2_verdict", "check_3_verdict", "check_4_verdict",
                  "overall", "feedback", "NO_LEAKAGE", "GOOD", "PASS", "CONSISTENT"),
    "judge_system": ("per_criterion", "feedback"),
}


@dataclass(frozen=True)
class PromptSet:
    challenger_system: str
    challenger_user: str
    challenger_first: str
    challenger_refine: str
    qv_system: str
    qv_user: str
    solver_system: str
    solver_user: str
    judge_system: str
    judge_user: str

    # -- files ---------------------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path) -> PromptSet:
        """Read ``<field>.txt`` for every field; raise ValueError on problems."""
        d = Path(directory)
        texts = {f.name: (d / f"{f.name}.txt").read_text(encoding="utf-8").rstrip("\n")
                 for f in fields(cls)}
        prompts = cls(**texts)
        if issues := prompts.problems():
            raise ValueError(f"prompt set {d}: " + "; ".join(issues))
        return prompts

    @classmethod
    def seed(cls) -> PromptSet:
        return _seed()

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            (d / f"{f.name}.txt").write_text(getattr(self, f.name) + "\n", encoding="utf-8")

    # -- change and check ----------------------------------------------------

    def evolve(self, **changes: str) -> PromptSet:
        """Copy with some mutable fields replaced. Does not validate."""
        frozen = sorted(set(changes) - set(MUTABLE_FIELDS))
        if frozen:
            raise ValueError(f"not mutable: {', '.join(frozen)}")
        return replace(self, **changes)

    def problems(self) -> list[str]:
        """Everything that would stop the pipeline from using this set."""
        issues: list[str] = []
        for f in fields(self):
            text = getattr(self, f.name)
            if not text.strip():
                issues.append(f"{f.name}: empty")
                continue
            expected = _TEMPLATE_VARS.get(f.name)
            if expected is not None:
                used = set(Template(text).get_identifiers())
                if missing := sorted(expected - used):
                    issues.append(f"{f.name}: missing ${', $'.join(missing)}")
                if unknown := sorted(used - expected):
                    issues.append(f"{f.name}: unknown ${', $'.join(unknown)}")
            if absent := [k for k in _PARSER_KEYS.get(f.name, ()) if k not in text]:
                issues.append(f"{f.name}: no longer names {', '.join(absent)}")
        return issues

    # -- rendering -----------------------------------------------------------

    def render_challenger(self, paper_title: str, paper_id: str, paper_text: str,
                          feedback: str = "") -> str:
        # Refinement round (Fig 7): earlier failures arrive grouped by mode.
        task = (Template(self.challenger_refine).safe_substitute(feedback=feedback)
                if feedback else self.challenger_first)
        return Template(self.challenger_user).safe_substitute(
            paper_title=paper_title, paper_id=paper_id, paper_text=paper_text, task=task)

    def render_qv(self, paper_title: str, paper_text: str, question_type: str,
                  context: str, question: str, rubric_json: str) -> str:
        return Template(self.qv_user).safe_substitute(
            paper_title=paper_title, paper_text=paper_text, question_type=question_type,
            context=context, question=question, rubric_json=rubric_json)

    def render_solver(self, context: str, question: str) -> str:
        return Template(self.solver_user).safe_substitute(context=context, question=question)

    def render_judge(self, context: str, question: str, candidate_answer: str,
                     rubric_json: str) -> str:
        return Template(self.judge_user).safe_substitute(
            context=context, question=question, candidate_answer=candidate_answer,
            rubric_json=rubric_json)


@cache
def _seed() -> PromptSet:
    return PromptSet.load(SEED_DIR)


__all__ = ["MUTABLE_FIELDS", "SEED_DIR", "PromptSet"]
