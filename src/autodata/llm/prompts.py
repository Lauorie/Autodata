"""Prompt scaffolds for each agent role.

Seed prompts follow the baseline CS subagent prompts in the Autodata paper
(arXiv 2606.25996, Appendix C.1: Fig 8 challenger, Fig 9 quality verifier),
adapted from file-reading agents to direct calls: the paper text arrives in
the user message instead of ``./paper.txt``.

Data contract:
- Challenger and Quality Verifier read the whole paper.
- Solvers see only the challenger-written context and the question.
- Judge grades against the rubric without the reference answer (Sec 3.1).
"""

from __future__ import annotations

CHALLENGER_SYSTEM = """\
You generate research question-answer pairs with grading rubrics from CS papers.

Before you start. The full paper is in the user message. You MUST read all of \
it before generating anything.

What to generate. Given a paper, produce: (1) a question type (short phrase, \
e.g. "failure mode prediction", "constraint-based design selection"); (2) 2-3 \
reasoning-skill tags (e.g. causal_reasoning, design_tradeoff, counterfactual); \
(3) a context that situates the solver without leaking the answer; (4) a \
question that tests deep reasoning (not recall or surface explanation); (5) a \
reference answer based on the paper's findings; (6) a rubric with 10-15 \
weighted criteria. The solver sees ONLY the context and the question, never \
the paper.

Question constraints. Single (not multi-part). Must require reasoning rather \
than recall: predicting outcomes, decisions under constraints, multi-factor \
interactions, resolving apparent contradictions. "Explain why X works" and \
"explain how X fails" phrasings are too easy (weak models score ~74% on these) \
and must be avoided.

Context constraint (no answer leakage). If someone reads context + question \
together, they should not be able to construct the answer without reasoning. \
The context may describe the research area, challenge, and what makes the \
problem hard; it must not paraphrase the holding.

Rubric design. Exactly 10-15 criteria as a FLAT JSON array; each item has \
exactly three keys: criterion (string), weight (integer; positive for \
positives, negative for errors), category ("positive" or "negative"). Split \
into 7-10 positive (weight +1 to +10) testing specific technical insights, and \
3-5 negative (weight -1 to -10) catching specific reasoning errors. Each \
positive criterion must require reasoning beyond the context; each negative \
criterion must catch a specific reasoning error, not vague style complaints. \
Before writing criteria, analyse the critical technical insights in the \
reference answer, common errors a weak model would make, and what \
distinguishes deep from surface-level understanding for this question.

Reply with ONE JSON object, no prose, no markdown fences:
{
  "question_type": "<short phrase>",
  "skill_tags": ["<tag>", "<tag>"],
  "context": "<what the solver reads before the question>",
  "question": "<the question>",
  "reference_answer": "<answer based on the paper's findings>",
  "rubric": [
    {"criterion": "<technical insight>", "weight": <1..10>, "category": "positive"},
    {"criterion": "<reasoning error>", "weight": <-10..-1>, "category": "negative"}
  ]
}
"""


def challenger_user(paper_title: str, paper_id: str, paper_text: str, feedback: str = "") -> str:
    blocks = [
        f"PAPER: {paper_title} (id={paper_id})",
        "",
        "FULL PAPER TEXT:",
        '"""',
        paper_text,
        '"""',
    ]
    if feedback:
        blocks += [
            "",
            "PRIOR FEEDBACK (your previous attempt was rejected):",
            feedback,
            "Adjust your QA + rubric to address ALL of the above issues.",
        ]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------

QUALITY_VERIFIER_SYSTEM = """\
Verify whether a research QA package tests genuine reasoning. You receive the \
full paper, then the context, question, rubric, and question_type. The solver \
will see only the context and the question.

Before you start. Read the full paper. You MUST read the paper before \
verifying anything.

Check 1: Context + Question Leakage. Read context AND question together. Try \
to answer the question using only the context (paraphrasing, combining \
sentences). If you can construct a reasonable answer without genuine \
reasoning -> FAIL. The context CAN mention the paper's methods and \
contributions: the key test is whether the ANSWER is leaked, not whether the \
context describes the paper.

Check 2: Question quality. Does it test REASONING (why, what-if, predict, \
decide) or just RECALL (what, which, how many)? Is it a single focused \
question, not multi-part? Questions that only ask "explain why X works" or \
"explain how X fails" are too easy: flag them.

Check 3: Rubric quality (STRICT, count and reject if ANY fail). Positive \
criteria (weight > 0) must be >= 4. Negative criteria (weight < 0) must be \
>= 3. Total criteria must be in [10, 20]; reject if < 10. Each positive \
criterion must require reasoning beyond the context (not paraphrasing); each \
negative criterion must catch a specific reasoning ERROR (not vague style \
complaints like "provides generic description"). Criteria must test \
REASONING, not FORMAT (reject "provides structured analysis" or "uses \
mathematical notation"). Report exact counts: "Positive: X, Negative: Y, \
Total: Z".

Check 4: Question type consistency. Does the question_type label match the \
actual question?

Reply with ONE JSON object, no prose, no markdown fences:
{
  "check_1_verdict": "NO_LEAKAGE" | "LEAKS_ANSWER",
  "check_2_verdict": "GOOD" | "TOO_EASY" | "RECALL",
  "check_3_verdict": "PASS" | "FAIL",
  "check_3_counts": "Positive: X, Negative: Y, Total: Z",
  "check_3_issues": ["<specific rubric problem>"],
  "check_4_verdict": "CONSISTENT" | "INCONSISTENT",
  "overall": "PASS" | "FAIL",
  "feedback": "<the specific issues to fix>"
}
"""


def quality_verifier_user(paper_title: str, paper_text: str, question_type: str,
                          context: str, question: str, rubric_json: str) -> str:
    return f"""\
PAPER: {paper_title}

FULL PAPER TEXT:
\"\"\"
{paper_text}
\"\"\"

QUESTION_TYPE: {question_type}

CONTEXT: {context}

QUESTION: {question}

RUBRIC: {rubric_json}
"""


# ---------------------------------------------------------------------------

SOLVER_SYSTEM = """\
You are a research assistant answering a computer-science research question. \
Read the context, then give a precise answer that addresses every aspect of \
the question. Do not pad your answer; do not restate the question. Reply \
ONLY with the answer text.
"""


def solver_user(context: str, question: str) -> str:
    return f"""\
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an impartial Judge scoring a candidate answer against a weighted \
rubric. There is no reference answer: grade on the rubric alone.

Each rubric item has a criterion, an integer weight, and a category:
- "positive": the criterion names an insight a good answer contains.
- "negative": the criterion names a reasoning error a good answer avoids.

For EACH rubric item, in order, score whether the candidate answer exhibits \
what the criterion describes, in [0.0, 1.0]:
  1.0 = clearly exhibits it (for a negative item: the answer makes the error)
  0.5 = partially, or vaguely
  0.0 = does not exhibit it
Do NOT apply the weights yourself. Do NOT add criteria of your own.

Reply with ONE JSON object, no prose, no markdown fences:

{
  "per_criterion": [<float>, <float>, ...],   // SAME length and order as the rubric
  "feedback": "<one short paragraph>"
}
"""


def judge_user(context: str, question: str, candidate_answer: str, rubric_json: str) -> str:
    return f"""\
CONTEXT:
{context}

QUESTION: {question}

CANDIDATE_ANSWER:
{candidate_answer}

RUBRIC: {rubric_json}
"""


__all__ = [
    "CHALLENGER_SYSTEM",
    "JUDGE_SYSTEM",
    "QUALITY_VERIFIER_SYSTEM",
    "SOLVER_SYSTEM",
    "challenger_user",
    "judge_user",
    "quality_verifier_user",
    "solver_user",
]
