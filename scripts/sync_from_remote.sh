#!/usr/bin/env bash
# Pull run results from the run box: summaries, questions, filter verdicts,
# meta history and populations, usage and logs. Code goes the other way
# (sync_to_remote.sh); pulling it back would overwrite local work.
# Usage:  bash scripts/sync_from_remote.sh [-n]  (-n: dry run)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_remote.sh"

mkdir -p "$ROOT/outputs"
remote_rsync -az --prune-empty-dirs "$@" \
  --include '*/' --include 'summary.json' --include 'accepted_qa.jsonl' \
  --include 'final_qa.jsonl' --include 'final_filter.jsonl' \
  --include 'history.jsonl' --include 'population.json' --include 'split.json' \
  --include 'usage.jsonl' --include '*.log' --exclude '*' \
  "$REMOTE:$REMOTE_OUTPUTS_DIR/" "$ROOT/outputs/"
