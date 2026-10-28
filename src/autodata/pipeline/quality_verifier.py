"""Quality verifier step (arXiv 2606.25996, Fig 9): four checks, fail closed."""

from __future__ import annotations

from typing import Any

from ..data_module import Paper
from ..llm import LLMClient, QAPair, VerifierVerdict
from ..llm.client import extract_json
from ..llm.errors import OutputTruncatedError
from ..llm.prompt_set import PromptSet


class VerifierOutputError(ValueError):
    """The verifier never returned a JSON verdict object."""


# Fig 9 verdict fields and the value each must take for the package to pass.
_QV_PASSING = {
    "check_1_verdict": "NO_LEAKAGE",
    "check_2_verdict": "GOOD",
    "check_3_verdict": "PASS",
    "check_4_verdict": "CONSISTENT",
}


def _verdict_token(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _ask_verdict(qv: LLMClient, paper: Paper, qa: QAPair, retries: int,
                 prompts: PromptSet) -> dict[str, Any]:
    """Get a JSON verdict object, re-asking up to ``retries`` times with the
    problem named. A reply cut at ``max_tokens`` is not re-asked: the same
    prompt hits the same cap."""
    base = [
        {"role": "system", "content": prompts.qv_system},
        {"role": "user", "content": prompts.render_qv(
            paper.title, paper.text, qa.question_type, qa.context, qa.question,
            qa.rubric.to_prompt_json())},
    ]
    messages = base
    for _ in range(retries + 1):
        try:
            raw = qv.chat(messages)
        except OutputTruncatedError as e:
            raise VerifierOutputError(str(e)) from e
        try:
            obj = extract_json(raw)
        except ValueError as e:
            problem = f"not JSON: {e}"
        else:
            if isinstance(obj, dict):
                return obj
            problem = f"expected a JSON object, got {type(obj).__name__}"
        messages = base + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your reply could not be used ({problem}). "
                                        "Reply ONLY with the JSON verdict object."},
        ]
    raise VerifierOutputError(f"no usable verdict after {retries + 1} attempts: {problem}")


def verify_quality(qv: LLMClient, paper: Paper, qa: QAPair, retries: int = 0, *,
                   prompts: PromptSet) -> VerifierVerdict:
    obj = _ask_verdict(qv, paper, qa, retries, prompts)
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
