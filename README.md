# AutoData: Agentic Self-Instruct, reproduced

This repo reproduces the CS-paper setting of *Autodata: An agentic data
scientist to create high quality synthetic data* (Kulikov et al.,
arXiv:2606.25996). An agent reads a paper and writes a research question,
a context and a weighted rubric. A weak solver and a strong solver answer
it. The question is kept only if the weak solver fails and the strong one
succeeds. The kept questions train the weak solver with GRPO. An outer loop
rewrites the agent's own prompts.

[中文版](README.zh-CN.md)

## Status

Everything below runs and is tested (241 tests). What has run live, and
what has not:

| Stage | Live run | Result |
|---|---|---|
| Inner loop + final filter | gate, 5 SciArena papers, flash models | 5/5 accepted, 5/5 kept; 18 rounds; 35 min; $0.43 |
| Inner loop, default models | smoke, 1 paper, 2 rounds | both rounds `failed_on_strong` |
| Held-out eval | 2 gate questions | base mean@3 0.279, best@3 0.309 (a plumbing check, not a result) |
| Meta-optimization | trial, 1 iteration, 10 validation sessions per side | seed 2/10, mutant 5/10 at the fixed filter (see below) |
| GRPO training | never run | settings checked against the TRL 1.16 source only |

Accepted gate questions: weak 0.20–0.49, strong 0.60–0.85, gap 0.33–0.42.

The meta trial is a plumbing check. Ten sessions per side and one seed
prove nothing about prompts. The analyzer, given no hints, found two of the
causes the paper's Sec 4 reports: high-weight rubric items that need facts
found only in the paper, and negative items that punish a defensible
alternative answer. The mutant passed 8/10 under the loop's own verifier.
The fixed end-of-loop filter then dropped 3 of the 8: one context gave the
answer away, two rubrics rewarded restating the context. The meta score now
counts a session only if its question passes that filter.

Earlier versions of this README said the 4B solver was "too strong" for
the task. That finding came from a bug: the solvers read the whole paper.
Those numbers are gone.

## How it works

### Roles

| Role | Default model | `models=flash` |
|---|---|---|
| challenger, quality verifier, judge, meta-optimizer | Kimi-K3 | DeepSeek-V4.1-Flash |
| strong solver | Qwen3.8-max | Qwen3.8-Flash |
| weak solver | Qwen3.5-4B on vLLM | Qwen3.5-4B on vLLM |

The paper used Kimi-K2.6, Qwen3.5-397B-A17B and Qwen3.5-4B. All remote
roles go through one OpenAI-compatible gateway (`PROXY_BASE_URL`).

### The inner loop (Sec 3.1, Fig 7)

For each paper, up to 15 rounds:

1. The challenger reads the whole paper. It writes a question type, skill
   tags, a context, a question, a reference answer and a rubric of 10–15
   weighted criteria.
2. The quality verifier reads the paper and the package. It checks for an
   answer leaked into the context, recall questions, rubric counts, and a
   question type that does not match (Fig 9).
3. The weak solver answers 3 times. It sees the context and the question,
   never the paper.
4. The judge scores each answer per criterion, without the reference answer.
5. If the weak scores pass, the strong solver answers 3 times and is judged
   the same way.
6. Accept when weak_avg ≤ 0.65, weak_max ≤ 0.75, no weak score is 0,
   0.60 ≤ strong_avg < 0.95, and the gap is ≥ 0.20 (Fig 7).
7. Otherwise the challenger gets the paper's failed questions, grouped by
   failure mode, and tries a new angle.

Both solvers think, with one budget of 65,536 tokens, temperature 1.0. The
judge grades only the text after `</think>`. A gateway failure re-runs the
same question; it is never scored as the model's failure.

An accepted question then meets the final filter: the seed verifier prompt
reads it once more, and fixed rubric counts apply (≥ 4 positive, ≥ 3
negative, 10–20 total). Only questions that pass reach `final_qa.jsonl`.

### Prompts as data

The ten prompts live in `src/autodata/llm/prompt_text/cs/` as text files,
loaded into a frozen `PromptSet`. Four are mutable: the challenger's system
prompt, first-round and refine instructions, and the verifier's system
prompt. The solver and judge prompts are the yardstick and stay fixed. So do
the user-message frames. `prompts_dir` points a run at another set.

### Meta-optimization (Sec 4)

Each iteration follows the paper's seven steps:

1. Pick a parent by Boltzmann sampling, T = 0.1.
2. Run it on a minibatch of train papers.
3. An analyzer reads the full trajectories and writes a root-cause analysis.
4. An implementer rewrites the mutable prompts in delimited blocks. Nothing
   it writes is executed.
5. Run parent and mutant on the validation papers.
6. Keep the mutant only if its score strictly beats the parent's mean over
   all its evaluations.
7. Log the outcome for the next analyzer.

The score is the session pass rate. A session passes only if its question
is accepted and passes the fixed final filter. Sessions lost to the gateway
leave the denominator. Below 80% scored, the evaluation is void.

### Data, evaluation, training (Sec 3.1.2)

- `build_dataset.py` reads `final_qa.jsonl` only, dedupes, and holds out
  100 questions, split by paper so no test paper appears in training.
- `eval_solver.py` answers each test question 3 times with the weak
  solver's settings and reports mean@3 and best@3. A question with a sample
  lost to the gateway is left out and counted.
- `train_grpo.py` runs GRPO through TRL with the same chat prompt, thinking
  switch, budget and sampling as the loop's weak solver. The reward is the
  loop's judge on the answer. A judge failure returns no reward, not zero.
  16 prompts per step, lr 1e-6, 200 steps, KL 0.04.

## Running

```bash
pip install -e ".[dev]"                 # pipeline and tests
pip install -e ".[serve,train]"         # on the GPU box
cp .env.example .secrets/remote.env     # PROXY_*, WEAK_SOLVER_BASE_URL
export AUTODATA_DATA_ROOT=/data/autodata   # papers-sciarena/, outputs/, models/Qwen3.5-4B/
```

Serve the weak solver, then run the stages. Every script is a Hydra app;
any config key can be set on the command line.

```bash
MODEL_DIR=$AUTODATA_DATA_ROOT/models/Qwen3.5-4B bash scripts/serve_weak_solver.sh

python scripts/run_inner_loop.py run_id=gate run.papers_limit=5 models=flash
python scripts/build_dataset.py run_id=data1 'data.runs=[<outputs_root>/gate]'
python scripts/eval_solver.py run_id=data1 eval.test_set=<outputs_root>/data1/data/test.jsonl
python scripts/run_meta.py run_id=meta1 models=flash meta.iterations=2
python scripts/train_grpo.py run_id=grpo1 train.data=<outputs_root>/data1/data/train.jsonl
```

Each run writes to `<outputs_root>/<run_id>/`, with its Hydra config and
run log. Runs that call models also write `usage.jsonl`, tokens per call.

To work from a laptop against a GPU box, set the `REMOTE_*` keys in
`.secrets/remote.env`. `scripts/sync_to_remote.sh` pushes code (`-n` for a
dry run). `scripts/sync_from_remote.sh` pulls results back.

Tests: `pytest`. Lint: `ruff check src tests scripts`.

## Cost and time

With `models=flash`, at the gateway's list prices:

- Inner loop: the 5-paper gate took 35 minutes and $0.43, 4 papers at a
  time.
- Meta-optimization: the trial (2 train sessions, 2 × 10 validation
  sessions) took 96 minutes and $0.54 at `meta.workers=10`. The default
  config (5 train papers, 25 validation papers × 4 samples, parent and
  mutant) should cost about $5 and 7 hours per iteration at
  `meta.workers=4`; that is an estimate. The strong solver takes most of
  the time.

## Where this differs from the paper

- **Corpus.** 2,543 SciArena papers in Markdown, not 10k S2ORC papers.
  Three files are broken: two hold another document, one is empty (skipped).
- **Acceptance.** Fig 7's thresholds. The Sec 3.1 prose gives others
  (strong ≥ 0.65, weak < 0.5).
- **Round budget.** 15. The paper reports 6.59 rounds per accepted item and
  gives no budget.
- **Meta sessions.** One round per session (`meta.session_rounds=1`).
  Whole 15-round runs pass almost always (gate: 5/5) and stop telling
  prompts apart. The paper ran 6-hour sessions. At one round the refine
  prompt is never exercised.
- **Meta scale.** 4 iterations, run one after another. The paper ran 233,
  several at once.
- **Final filter.** The paper does not print its filter prompt. We reuse
  the seed verifier and Fig 9's counts. The counts demand 3 negative
  criteria, so a positive-only rubric, one of the paper's Sec 4 findings,
  would fail here (`final_filter.min_negative`).
- **Training.** Full fine-tuning by default, to match lr 1e-6. 8
  completions per prompt is our choice. On one 85 GB card, a 65,536-token
  rollout with a 248k vocabulary, full fine-tuning and a colocated vLLM do
  not fit together. LoRA, a shorter budget, or a second card is needed.
- **Not built.** The CoT Self-Instruct baseline, its test set, and the
  legal-reasoning setting.

## Layout

```
conf/                    Hydra config: config.yaml, models/, acceptance/
scripts/                 run_inner_loop, run_meta, build_dataset, eval_solver,
                         train_grpo, serve_weak_solver, sync_*
src/autodata/
  llm/                   client, schemas, PromptSet, prompt_text/cs/*.txt, usage meter
  pipeline/              inner_loop, quality_verifier, evaluate_rubric,
                         acceptance, final_filter, at_k
  meta/                  outer_loop, evaluation, digest, analyzer, implementer,
                         population, history
  data_module/           paper_loader, qa_dataset, arxiv_html
  training/              grpo (rollout settings and reward)
tests/                   fakes.py drives every agent without a network
```
