"""Scripted LLM backends and payload builders shared by the pipeline tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from autodata.data_module import Paper
from autodata.llm import LLMClient, LLMRoleConfig

ROLES = ("challenger", "quality_verifier", "weak_solver", "strong_solver", "judge")

# 7 positive (sum 18) + 3 negative criteria: the Fig 8 baseline shape.
RUBRIC = [
    {"criterion": "identifies the variance source in the estimator", "weight": 3, "category": "positive"},
    {"criterion": "links the bias term to the sampling distribution", "weight": 3, "category": "positive"},
    {"criterion": "predicts the failure under distribution shift", "weight": 3, "category": "positive"},
    {"criterion": "states the trade-off between the two objectives", "weight": 3, "category": "positive"},
    {"criterion": "names the condition under which the bound holds", "weight": 2, "category": "positive"},
    {"criterion": "explains the role of the temperature parameter", "weight": 2, "category": "positive"},
    {"criterion": "derives the direction of the gradient change", "weight": 2, "category": "positive"},
    {"criterion": "claims the method removes all bias", "weight": -3, "category": "negative"},
    {"criterion": "confuses forward and reverse KL", "weight": -2, "category": "negative"},
    {"criterion": "assumes the teacher is always correct", "weight": -2, "category": "negative"},
]


def challenger_payload(**overrides: object) -> str:
    body: dict[str, object] = {
        "question_type": "failure mode prediction",
        "skill_tags": ["causal_reasoning", "counterfactual"],
        "context": "A team distils a large model into a small one and sees a train/inference mismatch.",
        "question": "Which training change would remove the mismatch, and what does it cost?",
        "reference_answer": "Train on the student's own samples; it costs extra generation compute.",
        "rubric": RUBRIC,
    }
    body.update(overrides)
    return json.dumps(body)


def qv_payload(overall: str = "PASS", **overrides: object) -> str:
    body: dict[str, object] = {
        "check_1_verdict": "NO_LEAKAGE",
        "check_2_verdict": "GOOD",
        "check_3_verdict": "PASS",
        "check_3_counts": "Positive: 7, Negative: 3, Total: 10",
        "check_3_issues": [],
        "check_4_verdict": "CONSISTENT",
        "overall": overall,
        "feedback": "ok",
    }
    body.update(overrides)
    return json.dumps(body)


def judge_payload(per: list[float]) -> str:
    return json.dumps({"per_criterion": per, "feedback": "graded"})


# Hand-derived rubric scores for RUBRIC (positive weight total = 18):
WEAK_PER = [0.5] * 7 + [0.0] * 3          # 9 / 18  = 0.500
STRONG_PER = [1.0] * 6 + [0.0] * 4        # 16 / 18 = 0.889
STRONG_LOW_PER = [0.5] * 7 + [0.0] * 3    # 9 / 18  = 0.500


Handler = Callable[[list[dict[str, str]]], str]


@dataclass
class ScriptedBackend:
    """Answers each role with a handler (a callable or a queue of strings).

    Records every call as ``(role, messages)`` so tests can assert on the
    exact prompts each agent received.
    """

    handlers: dict[str, Handler | list[str]]
    calls: list[tuple[str, list[dict[str, str]]]] = field(default_factory=list)

    def chat(self, role_cfg: LLMRoleConfig, messages: list[dict[str, str]]) -> str:
        self.calls.append((role_cfg.role, messages))
        h = self.handlers[role_cfg.role]
        if callable(h):
            return h(messages)
        if not h:
            raise RuntimeError(f"no scripted response left for role {role_cfg.role}")
        return h.pop(0)

    def text_seen_by(self, role: str) -> str:
        return "\n".join(m["content"] for r, msgs in self.calls if r == role for m in msgs)


def make_clients(backend: ScriptedBackend) -> dict[str, LLMClient]:
    return {
        r: LLMClient(LLMRoleConfig(role=r, endpoint="proxy", model="fake"), backend)
        for r in ROLES
    }


def judge_by_solver(weak_per: list[float], strong_per: list[float]) -> Handler:
    """Judge handler that grades by which solver wrote the candidate answer."""

    def handle(messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        return judge_payload(strong_per if "STRONG-ANSWER" in user else weak_per)

    return handle


def make_paper(text: str = "", paper_id: str = "papX") -> Paper:
    body = text or ("# Paper X\n\n" + "Section 3 assumes A. Therefore X implies Y. " * 30)
    return Paper(paper_id=paper_id, title="Paper X", text=body, source_path=Path("/dev/null"))
