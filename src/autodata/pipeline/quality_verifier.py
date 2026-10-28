"""Quality verifier step (arXiv 2606.25996, Fig 9): four checks, fail closed."""

from __future__ import annotations

from typing import Any

from ..data_module import Paper
from ..llm import LLMClient, QAPair, VerifierVerdict
from ..llm import prompts as _prompts_module  # late-bound on every call
from ..llm.client import extract_json


class VerifierOutputError(ValueError):
    """The verifier replied with something that is not a JSON object."""


# Fig 9 verdict fields and the value each must take for the package to pass.
_QV_PASSING = {
    "check_1_verdict": "NO_LEAKAGE",
    "check_2_verdict": "GOOD",
    "check_3_verdict": "PASS",
    "check_4_verdict": "CONSISTENT",
}


def _verdict_token(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def verify_quality(qv: LLMClient, paper: Paper, qa: QAPair) -> VerifierVerdict:
    raw = qv.chat([
        {"role": "system", "content": _prompts_module.QUALITY_VERIFIER_SYSTEM},
        {"role": "user", "content": _prompts_module.quality_verifier_user(
            paper.title, paper.text, qa.question_type, qa.context, qa.question,
            qa.rubric.to_prompt_json())},
    ])
    try:
        obj = extract_json(raw)
    except ValueError as e:
        raise VerifierOutputError(str(e)) from e
    if not isinstance(obj, dict):
        raise VerifierOutputError(f"expected a JSON object, got {type(obj).__name__}")
    # Fail closed: the package passes only when OVERALL is PASS *and* every
    # check holds its passing value. A malformed or self-contradicting verdict
    # must never let a bad QA pair through the gate.
    checks = {k: _verdict_token(obj.get(k)) for k in _QV_PASSING}
    issues = [f"{k}={v or 'MISSING'}" for k, v in checks.items() if v != _QV_PASSING[k]]
    overall = _verdict_token(obj.get("overall"))
    if overall != "PASS" and not issues:
        issues.append(f"overall={overall or 'MISSING'}")
    feedback = str(obj.get("feedback", ""))
    rubric_issues = obj.get("check_3_issues") or []
    if isinstance(rubric_issues, list) and rubric_issues:
        feedback += "\nRubric issues: " + "; ".join(str(x) for x in rubric_issues)
    if "check_3_counts" in obj:
        checks["check_3_counts"] = str(obj["check_3_counts"])
    return VerifierVerdict(
        passed=overall == "PASS" and not issues,
        feedback=feedback,
        issues=issues,
        checks=checks,
    )


__all__ = ["VerifierOutputError", "verify_quality"]
