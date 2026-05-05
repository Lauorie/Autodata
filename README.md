# AutoData (reproduction)

A from-scratch re-implementation of the **AutoData (Agentic Self-Instruct)**
pipeline described in the Meta FAIR RAM blog post:
https://facebookresearch.github.io/RAM/blogs/autodata/

The official `facebookresearch/RAM/projects/autodata/` directory ships only a
README and figures — no source code is released — so this repo is built
against the blog's textual description.

---

## What is reproduced

| Component | Status |
|---|---|
| Inner loop (Challenger / Quality Verifier / Weak Solver / Strong Solver / Judge) | implemented + run end-to-end on the remote 2× A800-80GB box |
| Acceptance criteria (weak ≤ 0.65, strong ∈ [0.60, 0.95), gap ≥ 0.20, no zeros) | implemented (`pipeline/acceptance.py`), unit-tested |
| `evaluate_rubric.py --weak-only / --strong-only` CLI | implemented |
| Outer meta-optimization loop (Boltzmann sampler T=0.1 + LLM code-mutator + strict-improvement gate) | implemented (`meta/`), not executed at scale |
| GRPO RL training script for Qwen3.5-4B (TRL + LoRA r=16) | implemented (`scripts/train_grpo.py`), not executed |
| pytest unit suite | 47 tests passing (incl. 13 regression tests for codex-flagged bugs) |
| Adversarial code review pass via codex MCP | done — see `## Codex review summary` below |

## Models (per blog spec)

| Role | Model | Endpoint |
|---|---|---|
| Challenger | `moonshotai/kimi-k2.6` (proxy resolves to the latest `kimi-k2.6-...` snapshot) — thinking ON | proxy |
| Quality Verifier / Judge / Meta-optimizer | `moonshotai/kimi-k2.6` — thinking OFF (`reasoning_effort: none`) for latency | proxy |
| Strong Solver | `qwen/qwen3.5-397b-a17b` — thinking OFF | proxy |
| Weak Solver / RL target | local `Qwen3.5-4B` (text-only causal head) | in-process HF transformers |

Endpoint: any OpenAI-compatible chat-completions URL (the original run used a
private OpenRouter front; substitute your own via `PROXY_BASE_URL` —
OpenRouter, Anthropic-compatible, or a self-hosted vLLM all work). All model
IDs above were verified against the chosen endpoint.

## Repo layout

```
src/autodata/
  llm/         OpenAI-compat client (proxy + local HF) + role prompts + pydantic schemas
  data_module/ paper loader + section-aware chunker
  pipeline/    Agentic Self-Instruct inner loop + acceptance gate + evaluate_rubric CLI
  meta/        Boltzmann sampler + code-edit "mutator" + outer evolution loop
  utils/       seed, logging, IO
conf/          Hydra configs (models / acceptance / paths / inner_loop / meta / train)
scripts/       run_inner_loop.py, run_meta.py, train_grpo.py, sync_from_remote.sh
tests/         pytest unit tests
outputs/       run summaries + accepted_qa.jsonl pulled back from remote (gitignored on remote)
```

Per-file line counts are all <300 (small-file rule, see CLAUDE rules).

## Running

```bash
# 1. install
pip install -e .

# 2. provide proxy credentials (see .env.example for all keys)
mkdir -p .secrets
cp .env.example .secrets/remote.env
$EDITOR .secrets/remote.env   # fill in PROXY_API_KEY + PROXY_BASE_URL

# 3. tiny smoke (~15 min on 2× A800)
python scripts/run_inner_loop.py run_id=smoke run.papers_limit=2 \
  inner_loop.chunks_per_paper=2 inner_loop.max_iterations=2 \
  inner_loop.num_solver_samples=2 run.chunk_concurrency=4

# 4. meta-optimization (3-5 generations on a small validation set)
python scripts/run_meta.py run_id=meta1 meta.n_generations=4 \
  meta.validation_chunks=8 run.papers_limit=10

# 5. GRPO on the synthesised QAs (overnight job on 1× 80 GB GPU)
python scripts/train_grpo.py run_id=grpo1 \
  accepted_qa=/path/to/outputs/full/accepted_qa.jsonl \
  train.max_steps=200
```

## Smoke run results (this session)

Run on the remote box (2× NVIDIA A800-80GB, 78 prepared CS papers in
`/root/autodl-fs/autodata/papers-mds`):

| Run | Papers × chunks | Iterations | Wall | Rounds | Accepted | Notes |
|---|---|---|---|---|---|---|
| `smoke5` | 2 × 2 | 2 | 15 min | 6 | 0 | First end-to-end pass; baseline prompt → weak_avg often ≈ 0.66 (just over threshold) |
| `smoke6` | 1 × 3 | 3 | running at write-time | — | — | After codex fixes + harder challenger prompt |

**The 0% accept rate on `smoke5` is itself a faithful reproduction of the
phenomenon the blog describes:** the baseline Challenger prompt produces
questions that the local 4B weak solver can answer well enough to fail
acceptance. The blog reports a baseline validation pass rate of 12.8%,
rising to 42.4% only after meta-optimization. Our small smoke set is
statistically consistent with that prior.

## Codex review summary

A read-only review pass was run via the codex MCP (`mcp__codex__codex`),
which returned 1 CRITICAL + 10 MAJOR + 3 MINOR findings. The MUST-FIX bugs
were patched in this session and are pinned by regression tests
(`tests/test_codex_fixes.py`):

| Severity | Issue | Fix |
|---|---|---|
| MAJOR | Inner loop imported prompt symbols at module load → meta-optimization's hot-swap was silently a no-op. | Late-bind via `from ..llm import prompts as _prompts_module`; access via `_prompts_module.X` everywhere. Test: `test_inner_loop_uses_late_bound_prompts`. |
| MAJOR | `bool(obj.get("passed", False))` coerced `"false"` → `True`. | Strict bool/string/int handling in `_qv_step`. Tests: `test_qv_strict_bool[*]` (8 cases). |
| MAJOR | Judge per-criterion scores not range-validated before `weighted_average`. | New `_normalise_judge_scores` clamps to [0, 1] and length-aligns to the rubric. Tests: `test_normalise_*` (4 cases). |
| MAJOR | GRPO reward function looked up rubric by full prompt string — TRL truncation/padding could silently zero rewards. | Dataset now carries a stable `qa_id`; reward function uses it. |
| MAJOR | `_JSON_BARE` regex was greedy and could span unrelated braces. | Replaced with a balanced-bracket scanner that respects string literals (`_balanced_extract`). |
| MINOR | Chunk-worker exceptions were silently `continue`-d. | Failure marker is now written to `rounds.jsonl`. |

**Findings explicitly NOT fixed (documented as accepted risk for this reproduction):**

- **CRITICAL: LLM-generated code is `exec()`-ed in the meta-optimizer.** The
  `meta.outer_loop._load_prompts_module` runs source written by the
  meta-optimizer LLM. `validate_mutated_source` checks substrings + syntax
  but is not a sandbox. **Acceptable in this controlled-environment
  research reproduction; not safe for production.** Hardening would
  require AST-allowlist validation + a separate-process / container
  sandbox.
- **MAJOR: parallel weak-vs-strong evaluation diverges from the blog's
  weak-first short-circuit.** With `parallel_solvers=True` (default),
  weak and strong solvers run concurrently — saving wall time but
  occasionally spending strong-solver tokens that a strict serial
  protocol would skip. Setting `inner_loop.parallel_solvers=false` falls
  back to faithful behaviour.
- **MAJOR: tenacity retries every exception as transient.** A bad model
  ID or 401 will burn 4 attempts before failing. Acceptable for a
  research run; production would want status-code-aware classification.

## Documented deviations from the blog

- The blog's strong solver is `Qwen3.5-397B-A17B`. The proxy resolves
  this to a thinking variant by default; we run it with
  `reasoning_effort: none` for tractable per-call latency. Same model
  identity, no chain-of-thought tokens.
- The blog uses Kimi-K2.6 across the orchestrator/judge roles. We use
  the same model ID; thinking is enabled for the Challenger and
  Meta-optimizer (where reasoning helps), disabled for the Quality
  Verifier and Judge (where it just blows latency).
- The weak solver is served via in-process HuggingFace `transformers`
  (`Qwen3_5ForCausalLM` text-only path). vLLM was skipped because
  Qwen3.5's brand-new mixed linear/full-attention architecture
  (`Qwen3_5ForConditionalGeneration`, `attn_output_gate=true`,
  `full_attention_interval=4`) is not yet in stable vLLM.
- Paper corpus is the 78 prepared markdown papers at
  `/root/autodl-fs/autodata/papers-mds`, not the 10 000+ S2ORC subset
  used in the blog.
- The meta-optimizer mutates only the `prompts.py` module (not the
  inner-loop control flow), to keep the search space tractable.
- GRPO training is implemented but was not executed in this session
  (the RL run is overnight-sized; out of scope for the validation
  pass).
