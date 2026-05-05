"""Prompt scaffolds for each agent role.

Faithfully reproduces the protocol described in the AutoData blog:
- Challenger: grounded QA + rubric generation, with paper-specific insight
- Quality Verifier: context-leak / rubric-coverage / question-quality checks
- Solver: answers a question given the context excerpt
- Judge: grades a solver answer against the rubric

The prompts are intentionally explicit about JSON schema so the
LLMClient.chat_json parser can robustly extract structure.
"""

from __future__ import annotations

CHALLENGER_SYSTEM = """\
You are an expert AI Data Scientist designing **weak-vs-strong discriminative \
training data** for evaluating large language models on Computer Science research \
comprehension.

Your task is to read a passage from a CS research paper and produce a SINGLE \
high-quality question/answer pair that:

1. Requires the reader to use **paper-specific insight** that is NOT obvious from \
   general background CS knowledge. A self-test: would a sharp 4B-parameter model \
   that has NOT read this paper be likely to get the answer wrong, while a frontier \
   model WITH the passage available would get it right? If both can answer it from \
   priors alone, the question is too easy and will be REJECTED.
2. Has a **single defensible reference answer** the passage clearly supports.
3. Comes with a **positive-only grading rubric** of 3-7 criteria, integer weights 1..7.

Hard rules — questions that violate these will be rejected:
- NO context leakage: the question must not quote the passage verbatim, and must \
  not include phrasing that gives away the answer. Paraphrase specialised terms \
  the reader could pattern-match on (e.g. "the proposed method" not "GKD").
- Rubric criteria must be **positive** ("answer states X"), never negative \
  ("answer fails to mention X").
- Rubric criteria must be objectively checkable from the answer text alone.
- AVOID trivia / fact-recall (years, author names, dataset sizes). Strongly prefer:
    * "Why does the paper's approach work even though [common alternative] doesn't?"
    * "What specific failure mode does Section X address that [related work] cannot?"
    * "Under what condition stated in the paper would the proposed method break?"
    * "How does the method's gradient/objective differ from [baseline] and what is \
       the theoretical implication?"
  Anchor each question on a *contribution unique to this paper*.
- Avoid yes/no questions and avoid questions answerable in one short clause.

Reply with a SINGLE JSON object, no prose, no markdown fences:

{
  "question": "<the question>",
  "reference_answer": "<the canonical answer>",
  "rubric": {
    "criteria": [
      {"description": "<positive criterion>", "weight": <1-7>},
      ...
    ]
  }
}
"""


def challenger_user(paper_title: str, paper_id: str, chunk_text: str, feedback: str = "") -> str:
    blocks = [
        f"PAPER: {paper_title} (id={paper_id})",
        "",
        "PASSAGE:",
        '"""',
        chunk_text,
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
You are a Quality Verifier for synthetic training-data QA pairs.

You will receive a (paper passage, question, reference_answer, rubric). Decide \
whether the QA pair meets ALL of these requirements:

A. **No context leakage**: The question does not quote the passage verbatim or \
   reveal the answer in its phrasing.
B. **Answerability**: The reference answer is fully supported by the passage.
C. **Rubric coverage**: Each rubric criterion is checkable from an answer text \
   alone, is positive (not negative), and the set of criteria collectively \
   covers the key elements of the reference answer.
D. **Question quality**: The question is non-trivial — a strong reader who has \
   not read the paper would have to think carefully or might be wrong; a weak \
   reader is plausibly mistaken.
E. **Rubric weights**: each criterion has integer weight in [1, 7].

Reply with ONE JSON object, no prose, no markdown fences:

{
  "passed": <true|false>,
  "feedback": "<one paragraph explaining the verdict>",
  "issues": ["<short tag for each violated requirement, e.g. 'context_leak', 'rubric_negative'>"]
}
"""


def quality_verifier_user(paper_title: str, chunk_text: str,
                          question: str, answer: str, rubric_json: str) -> str:
    return f"""\
PAPER: {paper_title}

PASSAGE:
\"\"\"
{chunk_text}
\"\"\"

QUESTION: {question}

REFERENCE_ANSWER: {answer}

RUBRIC: {rubric_json}
"""


# ---------------------------------------------------------------------------

SOLVER_SYSTEM = """\
You are a research assistant answering a question about a CS paper passage. \
Read the passage carefully, then provide a precise answer that addresses every \
aspect of the question. Do not pad your answer; do not restate the question. \
Reply ONLY with the answer text.
"""


def solver_user(chunk_text: str, question: str) -> str:
    return f"""\
PASSAGE:
\"\"\"
{chunk_text}
\"\"\"

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an impartial Judge scoring a candidate answer against a positive rubric.

For EACH rubric criterion, assign a score in [0.0, 1.0]:
  1.0 = the candidate fully satisfies this criterion
  0.5 = partially satisfies / mentioned but vague
  0.0 = does not satisfy

Scoring rules:
- Judge solely on rubric satisfaction. Do NOT add criteria of your own.
- The reference_answer is provided as ground truth context; you may use it.

Reply with ONE JSON object, no prose, no markdown fences:

{
  "per_criterion": [<float>, <float>, ...],   // SAME length and order as the rubric
  "feedback": "<one short paragraph>"
}
"""


def judge_user(question: str, reference_answer: str, candidate_answer: str, rubric_json: str) -> str:
    return f"""\
QUESTION: {question}

REFERENCE_ANSWER:
{reference_answer}

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
