#!/usr/bin/env bash
# End-to-end: extract training QAs from a finished inner-loop run, then GRPO.
# Usage: bash scripts/run_full_pipeline.sh <run_id> [min_accepted] [include_best_rejected]
set -euo pipefail

RUN_ID="${1:-run20b}"
MIN_ACCEPTED="${2:-10}"
INCLUDE_BEST_REJECTED="${3:-30}"
ROOT="/root/autodata"
OUT="/root/autodl-fs/autodata/outputs/${RUN_ID}"

cd "$ROOT"
set -a; . .secrets/remote.env; set +a
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-fs/hf

echo "=== extract training QAs ==="
python scripts/extract_training_qas.py \
  --run-dir "$OUT" \
  --out "$OUT/grpo_input.jsonl" \
  --min-accepted "$MIN_ACCEPTED" \
  --include-best-rejected "$INCLUDE_BEST_REJECTED"

N=$(wc -l < "$OUT/grpo_input.jsonl")
echo "Training set has $N rows"
if [ "$N" -lt 4 ]; then
  echo "Too few rows for meaningful GRPO. Aborting." >&2
  exit 1
fi

echo "=== GRPO training (demo: max_steps from conf/config.yaml) ==="
mkdir -p "$OUT/grpo"
python -u scripts/train_grpo.py \
  run_id="$RUN_ID" \
  accepted_qa="$OUT/grpo_input.jsonl" \
  hydra.job.chdir=false \
  2>&1 | tee "$OUT/grpo/train.log"

echo "=== before/after eval on first 5 ==="
python -u scripts/eval_before_after.py \
  --base /root/autodl-fs/autodata/Qwen3.5-4B \
  --lora "$OUT/grpo/final" \
  --eval-jsonl "$OUT/grpo_input.jsonl" \
  --max-eval 5 \
  --models-yaml conf/models/default.yaml \
  2>&1 | tee "$OUT/grpo/eval.log"

echo "=== DONE. Results in $OUT/grpo/ ==="
