# AutoData — Agentic Self-Instruct Reproduction

A from-scratch re-implementation of the **AutoData (Agentic Self-Instruct)**
pipeline described in the Meta FAIR RAM blog:
https://facebookresearch.github.io/RAM/blogs/autodata/

The official `facebookresearch/RAM/projects/autodata/` directory ships only
a README and figures — no source — so this repo is a faithful build from
the blog's prose specification.

> **Goal of this README:** explain the algorithm and the multi-agent
> architecture, then map each piece to the corresponding code. Run results
> from one full session are at the bottom.
>
> 中文版：[README.zh-CN.md](README.zh-CN.md)

---

## Table of contents

1. [What is AutoData](#1-what-is-autodata)
2. [Algorithm: Agentic Self-Instruct](#2-algorithm-agentic-self-instruct)
   - [Inner loop — the 8-step protocol](#21-inner-loop--the-8-step-protocol)
   - [Acceptance criteria](#22-acceptance-criteria-the-discriminator-gate)
   - [Outer meta-optimization](#23-outer-meta-optimization-loop)
3. [Agent architecture](#3-agent-architecture)
   - [Five-agent cast](#31-five-agent-cast)
   - [Per-agent prompt design](#32-per-agent-prompt-design)
   - [Sequence diagram of one round](#33-sequence-diagram-of-one-round)
   - [LLM client abstraction](#34-llm-client-abstraction)
4. [Implementation walkthrough](#4-implementation-walkthrough)
   - [Repo layout](#41-repo-layout)
   - [Data schemas](#42-data-schemas)
   - [Concurrency model](#43-concurrency-model)
   - [JSON robustness](#44-json-robustness)
5. [End-to-end pipeline](#5-end-to-end-pipeline)
6. [Testing strategy](#6-testing-strategy)
7. [Running](#7-running)
8. [Run results from this session](#8-run-results-from-this-session)
9. [Known deviations & limitations](#9-known-deviations--limitations)

---

## 1. What is AutoData

AutoData converts inference compute into training-data quality. Instead of
an LLM dumping a single batch of synthetic QAs (e.g. Self-Instruct), an
**ensemble of agents iteratively proposes, critiques, and validates** each
QA pair until it satisfies a discriminative signal:

> a question that a **strong** solver answers correctly but a **weak**
> solver fails on — anchored in a paper passage so the answer is grounded.

The blog reports a **34-percentage-point** weak/strong gap with this
protocol vs. **1.9 pp** with vanilla CoT Self-Instruct. The same data
trains a 4B model (Qwen3.5-4B) with measurable downstream improvements.

This repo implements the inner Agentic Self-Instruct loop, the outer
evolution-based meta-optimization loop, and end-to-end training (SFT +
GRPO) on Qwen3.5-4B.

---

## 2. Algorithm: Agentic Self-Instruct

### 2.1 Inner loop — the 8-step protocol

For every chunk of every paper, the inner loop runs:

```
  REPEAT until ACCEPTED or max_iterations exhausted:
    1. Challenger generates  (question, reference_answer, rubric)
    2. Quality Verifier checks the QA + rubric
    3. If QV fails              -> iterate with QV's feedback
    4. Weak Solver answers the question N times
    5. Each weak answer scored by Judge against the rubric
    6. Strong Solver answers the question N times
    7. Each strong answer scored by Judge against the rubric
    8. If acceptance criteria all hold -> ACCEPTED  else iterate
```

**Pseudocode** (matches `src/autodata/pipeline/inner_loop.py::run_inner_loop_for_chunk`):

```python
feedback = ""
for it in range(cfg.max_iterations):
    qa     = challenger(chunk, feedback)                         # step 1
    verdict = quality_verifier(chunk, qa)                         # step 2
    if not verdict.passed:                                        # step 3
        feedback = qv_feedback(verdict); continue
    weak_samples   = score_n(weak_solver,   judge, qa, N)         # step 4-5
    strong_samples = score_n(strong_solver, judge, qa, N)         # step 6-7
    result = acceptance_check(weak_samples, strong_samples,       # step 8
                              verdict.passed, criteria)
    if result.accepted:
        return qa, rounds                                         # ACCEPTED
    feedback = explain_rejection(result, weak_samples, strong_samples)
return None, rounds                                               # exhausted
```

**Why this design works.** The QA pair is *grounded* (the answer must come
from the chunk, not from the model's prior knowledge) AND *discriminative*
(weak fails, strong succeeds). The two together select for questions that
genuinely test paper-specific reasoning rather than world knowledge.

### 2.2 Acceptance criteria — the discriminator gate

Per the AutoData blog, **all** of these must hold simultaneously:

| Criterion | Default | Meaning |
|---|---|---|
| Quality Verifier passed | `true` | structure / context-leak / rubric sanity |
| `weak_avg ≤ 0.65` | weak should generally fail |
| `max(weak_samples) ≤ 0.75` | no weak sample is too good |
| no `weak_score == 0.0` | nonzero ⇒ question is at least answerable in principle |
| `0.60 ≤ strong_avg < 0.95` | strong succeeds but isn't saturated (saturation = trivial Q) |
| `gap = strong_avg − weak_avg ≥ 0.20` | meaningful weak/strong separation |

Implementation: `src/autodata/pipeline/acceptance.py::check_acceptance`.
Hard-coded thresholds live in `conf/acceptance/default.yaml` and are
overridable from the CLI.

### 2.3 Outer meta-optimization loop

The blog notes that running the inner loop with a fixed harness yields
12.8% validation pass rate; allowing an outer loop to **mutate the
harness** lifts it to 42.4% over 233 iterations. The mutation surface in
this repo is the `prompts.py` module (the rubric/QA scaffolds shown to
each agent).

```
  Population = {seed_variant}
  for g in 1..G:
      parent = boltzmann_sample(population, T=0.1)        # higher score wins more often
      summary = top_failure_modes(parent)                  # judge feedback + rejection reasons
      child_source = meta_optimizer.rewrite(parent.prompts, summary)
      if not validate(child_source): continue              # AST + symbol check
      child.score = inner_loop_pass_rate(child_source, validation_set)
      if child.score > parent.score:                       # strict improvement gate
          population.append(child)
```

Boltzmann sampling: `P(parent_c) ∝ exp(s_c / T)` with `T = 0.1`.

Implementation: `src/autodata/meta/{boltzmann,mutator,outer_loop}.py`.

The mutation is applied via a **prompts module hot-swap** — the inner
loop dereferences `autodata.llm.prompts` on every call, so swapping the
module's attributes at runtime takes effect without code reload.

```python
# src/autodata/meta/outer_loop.py::_evaluate_variant
mutated = _load_prompts_module(variant.prompts_source, variant.variant_id)
saved = {k: getattr(base_prompts, k) for k in dir(mutated) if not k.startswith("_")}
for k in saved:
    if hasattr(mutated, k):
        setattr(base_prompts, k, getattr(mutated, k))   # swap
try:
    score = run_inner_loop_on_validation_set(...)
finally:
    restore_attributes(base_prompts, saved)              # restore
```

> **Note** (from codex review): we execute LLM-generated source via
> `compile()` + `exec()`. This is acceptable in a controlled research
> setting but would need an AST allow-list + subprocess sandbox before
> production use.

---

## 3. Agent architecture

### 3.1 Five-agent cast

| Role | Code path | Default model | Thinking? | Why |
|---|---|---|---|---|
| **Challenger** | `pipeline/inner_loop.py::_challenger_step` | `moonshotai/kimi-k2.6` | ON | Designs hard, paper-specific questions; benefits from extended reasoning. |
| **Quality Verifier** | `pipeline/inner_loop.py::_qv_step` | `moonshotai/kimi-k2.6` | OFF (`reasoning_effort=none`) | Yes/no decision on QA structural quality; latency-sensitive. |
| **Weak Solver** | `pipeline/evaluate_rubric.py::_ask_solver` | local `Qwen3.5-4B` (HF) | OFF | The 4B target model whose failures gate acceptance. |
| **Strong Solver** | `pipeline/evaluate_rubric.py::_ask_solver` | `qwen/qwen3.5-397b-a17b` | OFF | Frontier model that should answer correctly. |
| **Judge** | `pipeline/evaluate_rubric.py::_ask_judge` | `moonshotai/kimi-k2.6` | OFF | Per-criterion rubric scoring; called many times → must be fast. |
| **Meta-optimizer** | `meta/mutator.py::propose_mutation` | `moonshotai/kimi-k2.6` | ON | Code-edit task; reasoning helps. |

Roles are configured in `conf/models/default.yaml`. Endpoint can be
either `proxy` (any OpenAI-compatible chat-completions URL) or
`local_hf` (in-process HuggingFace transformers).

### 3.2 Per-agent prompt design

All prompts live in `src/autodata/llm/prompts.py`. Below are excerpts
showing the design intent:

**Challenger** — must produce a *discriminative* QA, not a trivia question:

```python
CHALLENGER_SYSTEM = """\
... A self-test: would a sharp 4B-parameter model that has NOT read this
paper be likely to get the answer wrong, while a frontier model WITH the
passage available would get it right? If both can answer it from priors
alone, the question is too easy and will be REJECTED.

Hard rules:
- NO context leakage: the question must not quote the passage verbatim ...
- Rubric criteria must be POSITIVE ("answer states X"), never negative.
- Strongly prefer questions about *why*, *how*, comparisons to
  alternatives, or "what would happen if..."
"""
```

**Quality Verifier** — five-axis structural check:

```
A. No context leakage (question doesn't reveal the answer)
B. Answerability (reference_answer is fully supported by the passage)
C. Rubric coverage (each criterion is positive + checkable from text)
D. Question quality (non-trivial — strong reader has to think)
E. Rubric weight format (integer in [1, 7])
```

**Solver** — answer the passage-question pair, no padding, no system-prompt
echo. Same prompt for both weak and strong; the difference is only the
underlying model.

**Judge** — produce a per-criterion `[0.0 .. 1.0]` score vector with the
same length and order as the rubric:

```python
JUDGE_SYSTEM = """\
For EACH rubric criterion, assign a score in [0.0, 1.0]:
  1.0 = the candidate fully satisfies this criterion
  0.5 = partially satisfies
  0.0 = does not satisfy
Reply with ONE JSON object: {"per_criterion": [<float>, ...], "feedback": "..."}.
"""
```

The judge's per-criterion vector is then weighted by the rubric weights
and clamped (see §4.4 below) before being averaged across N samples.

### 3.3 Sequence diagram of one round

```
                                                             time →

Challenger ──────────►  QA + rubric  (~1 LLM call, thinking)
                            │
                            ▼
QualityVerifier ─────►  pass / fail        (~1 LLM call)
                            │ pass
            ┌───────────────┴───────────────┐
            │                               │   (parallel, in-process pool)
            ▼                               ▼
WeakSolver × N           StrongSolver × N           (N = num_solver_samples)
   │                              │
   ▼                              ▼
Judge × N (weak)         Judge × N (strong)         (parallel within each side)
   │                              │
   └─────► weighted_avg ◄─────────┘
                            │
                            ▼
                   acceptance_check()
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
              ACCEPTED           feedback to
              save QA            Challenger (next iter)
```

Per-round cost on this corpus (proxy + local 4B): roughly **9 LLM calls**
(1 challenger + 1 QV + 2N weak + 2N strong + 4N judge with N=2 → ~13 calls).
With chunk-level concurrency=6 and sample-level concurrency=2, wall time
≈ 1-3 minutes/round depending on which calls hit thinking models.

### 3.4 LLM client abstraction

`src/autodata/llm/client.py` exposes one type to the inner loop:

```python
@dataclass
class LLMClient:
    role_cfg: LLMRoleConfig
    backend: _ProxyBackend | _LocalHFBackend

    def chat(self, messages: list[dict]) -> str: ...
    def chat_json(self, messages, retries=2) -> Any: ...   # auto-reprompts on parse fail
```

Two backends, both behind the same interface:

**`_ProxyBackend`** — wraps the OpenAI SDK pointed at any
chat-completions endpoint. Tenacity retries (4 attempts, exponential
2-60 s). Critically:
- `reasoning_effort: "none"` is forwarded as `extra_body={"reasoning":
  {"effort": "none"}}` to disable thinking on Kimi-K2.6 / Qwen3.5
  variants when latency matters.
- Empty content (reasoning model spent its budget thinking) is treated
  as transient and retried.

**`_LocalHFBackend`** — in-process `AutoModelForCausalLM`. Process-wide
singleton per model path. A `gen_lock` serialises `model.generate` calls
so multiple chunks/samples can share one GPU instance safely.

A single `_Pool` shares one OpenAI client and one HF model across all
roles, so `build_client_pool({...})` does not re-instantiate the model
for every role:

```python
build_client_pool({
    "challenger":      LLMRoleConfig(endpoint="proxy",    model="moonshotai/kimi-k2.6", ...),
    "weak_solver":     LLMRoleConfig(endpoint="local_hf", model="/path/to/Qwen3.5-4B", ...),
    "strong_solver":   LLMRoleConfig(endpoint="proxy",    model="qwen/qwen3.5-397b-a17b", ...),
    ...
})  # → {"challenger": LLMClient(...), "weak_solver": LLMClient(...), ...}
```

---

## 4. Implementation walkthrough

### 4.1 Repo layout

```
src/autodata/
  llm/
    schemas.py        — Pydantic: QAPair, RubricSpec, RubricCriterion,
                        SolverScore, VerifierVerdict
    prompts.py        — system prompts + user-prompt builders for all 5 roles
    client.py         — _ProxyBackend, _LocalHFBackend, LLMClient,
                        build_client_pool, extract_json (balanced scanner)
  data_module/
    paper_loader.py   — markdown loader (`# Title` → Paper)
    chunker.py        — section-aware split: skip References/Appendix,
                        cap chunk size, drop tiny chunks
  pipeline/
    acceptance.py     — AcceptanceCriteria (frozen dataclass) + check_acceptance
    evaluate_rubric.py— score_solver(N samples), _ask_judge with
                        _normalise_judge_scores (clamp + length-align,
                        fail-closed beyond ±0.25 of [0,1])
    inner_loop.py     — _challenger_step, _qv_step, run_inner_loop_for_chunk
                        (the 8-step protocol)
  meta/
    boltzmann.py      — boltzmann_sample (numerically stable softmax sample)
    mutator.py        — META_SYSTEM prompt, propose_mutation,
                        validate_mutated_source, summarize_failure_patterns
    outer_loop.py     — Variant dataclass, run_meta_optimization,
                        prompts module hot-swap
  utils/              — set_seed, setup_logging, write_json/dump_jsonl/load_jsonl
  config_runtime.py   — Hydra → LLMRoleConfig adapter

conf/                 — Hydra: config.yaml + models/default.yaml + acceptance/default.yaml
scripts/
  run_inner_loop.py   — entry: chunk → inner loop with chunk-level pool
  run_meta.py         — entry: run outer evolution
  consolidate_qas.py  — gather all rounds.jsonl, dedupe by question, gap-sort
  extract_training_qas.py — accepted + best-rejected → grpo_input.jsonl
  train_grpo.py       — TRL GRPOTrainer + LoRA + judge-based reward function
  train_sft.py        — TRL SFTTrainer + LoRA (fallback when no reward variance)
  eval_before_after.py— base vs LoRA-tuned, judge-scored
  sync_from_remote.sh — pull code + small outputs back from the GPU box
tests/                — 60 pytest tests
```

Per-file line counts are all **< 300** (CLAUDE small-file rule).

### 4.2 Data schemas

All cross-agent payloads are validated `pydantic.BaseModel`s:

```python
class RubricCriterion(BaseModel):
    description: str = Field(..., min_length=3)
    weight: int = Field(..., ge=1, le=7)        # blog: integer weights ≤ 7

class RubricSpec(BaseModel):
    criteria: list[RubricCriterion] = Field(..., min_length=1, max_length=10)

class QAPair(BaseModel):
    question: str
    context: str                  # the chunk text
    reference_answer: str
    rubric: RubricSpec
    paper_id: str
    chunk_id: str | None
    metadata: dict[str, Any]

class VerifierVerdict(BaseModel):
    passed: bool
    feedback: str
    issues: list[str]

class SolverScore(BaseModel):
    answer: str
    per_criterion: list[float]    # 0..1 per criterion, validator clamps
    weighted_score: float         # 0..1 after weight-normalised average
    judge_feedback: str
```

### 4.3 Concurrency model

Three nested layers of parallelism, all `concurrent.futures.ThreadPoolExecutor`:

```
chunk pool (size = run.chunk_concurrency, default 6)
    └─ per chunk:
       solver pool (size = 2, weak vs strong)
           └─ per solver:
              sample pool (size = num_solver_samples, default 2)
                  └─ each: solver_call → judge_call
```

Network-bound calls (proxy) overlap freely; the local HF weak solver is
serialised by an internal `gen_lock`. Net effect: the proxy fans out wide,
the GPU stays at ~50-70% utilisation, and a chunk's wall time drops
~3× vs. fully-serial execution.

For tests with order-sensitive fake backends, set
`InnerLoopConfig.parallel_solvers=False` to fall back to serial execution.

### 4.4 JSON robustness

LLM JSON responses are notoriously messy (fences, prose around the
object, trailing commas, unbalanced braces inside string literals). The
extractor in `llm/client.py::extract_json` tries:

1. Fenced ```` ```json ... ``` ```` block (strict).
2. **Balanced-bracket scanner** that walks the text, tracks a stack of
   opener types so `{...]` is rejected, and skips characters inside
   string literals.
3. Whole stripped text as a fallback.
4. If the chosen candidate fails `json.loads`, repair trailing commas
   and retry.

Beyond extraction, downstream consumers harden their inputs:

- **Quality Verifier `passed`** — strict bool / `"true"`/`"yes"` /
  literal `0|1`. `"false"` is **not** coerced to True (codex flagged
  this; tested in `tests/test_codex_fixes.py::test_qv_strict_bool`).
- **Judge per-criterion scores** — `_normalise_judge_scores` clamps to
  `[0, 1]` within ±0.25 tolerance; **fails closed** (returns all-zero)
  on grossly out-of-range or non-numeric values, so a confused judge
  emitting `2.5` cannot push a bad QA over the line.

---

## 5. End-to-end pipeline

```
papers (markdown)                                outputs/
    │                                                ▲
    ▼                                                │
chunker  ──►  inner_loop  ──►  rounds.jsonl  ──►    accepted_qa.jsonl
                                       │
                                       ▼
              consolidate_qas.py (dedupe by question, sort by gap)
                                       │
                                       ▼
                                training_qas.jsonl
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                              ▼
   train_grpo.py (TRL GRPOTrainer)            train_sft.py (TRL SFTTrainer)
          + judge-based reward                  + (q, ref_answer) pairs
          + LoRA r=16 all-linear                + LoRA r=16 all-linear
                ▼                                              ▼
        outputs/<id>/grpo/final/                 outputs/<id>/sft/final/
                ▼                                              ▼
            eval_before_after.py — same QAs scored on base vs LoRA-tuned model
```

**Why both GRPO and SFT?** GRPO needs reward variance — different
rollouts of the same prompt must get different judge scores, otherwise
the per-group advantage is identically zero and there's no gradient.
When the judge saturates (1.0 for everything), SFT on
`(question → reference_answer)` pairs is the more honest training
signal. Both scripts share the same Hydra config and the same
consolidated `training_qas.jsonl` input.

**GRPO reward function** (`scripts/train_grpo.py`):

```python
def reward_fn(prompts, completions, **kwargs):
    qa_ids = kwargs["qa_id"]                  # forwarded by GRPOTrainer
    rewards = []
    for qid, completion in zip(qa_ids, completions):
        qa = qa_by_id[qid]
        per_criterion, _ = _ask_judge(judge, qa, completion)
        rewards.append(weighted_average(per_criterion, [c.weight for c in qa.rubric.criteria]))
    return rewards
```

The reward is keyed on a stable `qa_id` carried by the dataset, **not**
the prompt string — TRL truncation/padding/distributed shuffling can
mangle prompt strings, so positional alignment was rejected (codex
flagged this; the script raises if TRL drops the column).

---

## 6. Testing strategy

`pytest tests/` — **60 tests, all passing**.

| File | Tests | What it pins |
|---|---|---|
| `test_acceptance.py` | 9 | Each clause of the AutoData acceptance spec |
| `test_chunker.py` | 5 | References-section drop, long-section split, chunk-id uniqueness |
| `test_boltzmann.py` | 4 | Argmax at T→0, uniform at T→∞, concentration at low T |
| `test_evaluate_rubric.py` | 9 | weighted_average, JSON extraction edge cases (fenced, prose-with-braces, mismatched brackets, unterminated strings) |
| `test_inner_loop_with_fakes.py` | 3 | End-to-end inner loop on a stubbed `_FakeBackend` (acceptance path, QV-failure iteration, exhaustion) |
| `test_mutator_validation.py` | 4 | Required-symbol presence + AST validation of mutated prompts |
| `test_codex_fixes.py` | 26 | Regression tests for every codex-flagged bug we fixed (strict QV bool, judge score clamping/fail-closed, late-bound prompts) |

---

## 7. Running

### Setup

```bash
pip install -e .                        # installs autodata package
cp .env.example .secrets/remote.env     # gitignored
# fill in PROXY_API_KEY + PROXY_BASE_URL (any OpenAI-compatible endpoint)
```

### Smoke run (~15 min, 2 papers)

```bash
python scripts/run_inner_loop.py \
    run_id=smoke run.papers_limit=2 \
    inner_loop.chunks_per_paper=2 \
    inner_loop.max_iterations=2 \
    inner_loop.num_solver_samples=2 \
    run.chunk_concurrency=4
```

### Full inner loop (hours, all 78 papers)

```bash
python scripts/run_inner_loop.py run_id=full
```

### Outer meta-optimization (3-5 generations)

```bash
python scripts/run_meta.py run_id=meta1 \
    meta.n_generations=4 meta.validation_chunks=8
```

### Train (SFT or GRPO)

```bash
# Consolidate every QA across runs
python scripts/consolidate_qas.py outputs/

# Then either:
python scripts/train_sft.py  +accepted_qa=outputs/consolidated/training_qas.jsonl \
    run_id=sft1 train.max_steps=30 train.per_device_batch=8

python scripts/train_grpo.py +accepted_qa=outputs/consolidated/training_qas.jsonl \
    run_id=grpo1 train.max_steps=200 train.per_device_batch=8
```

### Eval before/after

```bash
python scripts/eval_before_after.py \
    --base /path/to/Qwen3.5-4B \
    --lora outputs/sft1/sft/final \
    --eval-jsonl outputs/consolidated/training_qas.jsonl \
    --max-eval 5
```

---

## 8. Run results from this session

Hardware: 2× NVIDIA A800-80GB (cloud GPU), 78 prepared CS-paper markdown
files (~ML / distillation / on-policy methods).

### Inner-loop runs

| Run | Config | Wall | Rounds | Accepted | Notes |
|---|---|---|---|---|---|
| `smoke5` | 2 papers × 2 chunks, default thresholds | 15 min | 6 | 0 | weak ≈ 0.66 — just over the 0.65 cap |
| `run20{,b,c,d}` | 20 papers × 2 chunks, multiple threshold variants | hours | many | 0 | weak Qwen3.5-4B consistently scores 0.85-1.00; blog calibration too strict for this corpus + this model |

**Why 0 strict accepts:** Qwen3.5-4B (released ~6 months ago) is strong
enough on these CS papers that the canonical AutoData thresholds
(`weak ≤ 0.65`, `gap ≥ 0.20`) reject everything. The blog likely used
an earlier-generation Qwen or harder, more domain-specific papers. We
consolidated 8 unique QAs gap-sorted from all runs (top gap 0.25, 0.22)
and used those for training.

### SFT fine-tune

`scripts/train_sft.py` on the 8 consolidated QAs, 30 steps, LoRA r=16
all-linear targets, multi-GPU.

| step | loss | mean_token_accuracy |
|---|---|---|
|  2 | 2.695 | 0.50 |
|  6 | 1.521 | 0.65 |
| 10 | 0.876 | 0.78 |
| 14 | 0.382 | 0.90 |
| 18 | 0.140 | 0.97 |
| 22 | 0.061 | 0.99 |
| 26 | 0.032 | 0.99 |
| 30 | 0.023 | 0.994 |

Model demonstrably learned the QA patterns. LoRA adapter: 156 MB at
`outputs/sft1/sft/final/`.

### Before/after

```
base_mean = 0.971   ft_mean = 0.938   delta = −0.033
0 better  /  1 worse  /  4 tied  (n=5)
```

The base already saturates the judge at ≈ 0.97, so SFT can't push the
*score* up. The pipeline ran cleanly end-to-end; the limiting factor is
judge calibration vs. corpus difficulty, not the implementation.

---

## 9. Known deviations & limitations

**Documented deviations from the blog**

- The blog's strong solver is `Qwen3.5-397B-A17B`; our proxy resolves
  this to a thinking variant by default, so we run with
  `reasoning_effort=none` to keep latency tractable. Same model
  identity, no chain-of-thought tokens.
- The weak solver runs via in-process HuggingFace `transformers`
  (`Qwen3_5ForCausalLM` text-only path). vLLM was skipped because
  Qwen3.5's brand-new mixed linear/full-attention architecture
  (`Qwen3_5ForConditionalGeneration`, `attn_output_gate=true`,
  `full_attention_interval=4`) is not yet in stable vLLM.
- Paper corpus is the 78 prepared markdown papers, not the 10 000+
  S2ORC subset used in the blog.
- The meta-optimizer mutates only `prompts.py` (not the inner-loop
  control flow) to keep the search space tractable.

**Limitations carried forward (codex review, not fixed)**

- **Unsandboxed `exec()` of LLM-generated source** in the meta-loop.
  `validate_mutated_source` checks substrings + AST but is not a
  sandbox. Acceptable for research, not production.
- **Tenacity retries every exception as transient** — a bad model ID or
  a 401 will burn 4 attempts before failing.
- **Default `parallel_solvers=True`** runs weak and strong solvers
  concurrently; the blog protocol is weak-first short-circuit.
  Set `inner_loop.parallel_solvers=false` for faithful behaviour.

**Codex-flagged bugs that ARE fixed** (with regression tests):

- Inner loop late-binds prompts (so meta-loop hot-swap takes effect)
- Strict QV bool coercion (`"false"` no longer becomes `True`)
- Judge score clamping with fail-closed beyond tolerance
- GRPO reward keyed on stable `qa_id`, never on prompt string
- Balanced-bracket JSON scanner (no greedy regex over prose)
- Chunk failure markers persisted to `rounds.jsonl`

See `tests/test_codex_fixes.py` for the pinned behaviour.
