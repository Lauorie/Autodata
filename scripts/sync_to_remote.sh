#!/usr/bin/env bash
# Push the working tree to the run box. A file deleted here is deleted there,
# so no stale module outlives its removal. Secrets, outputs, the git history
# and local notes never leave this machine, and the box's own copies of them
# are left alone.
# Usage:  bash scripts/sync_to_remote.sh [-n]    (-n: dry run, list changes)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_remote.sh"

remote_rsync -az --delete --itemize-changes "$@" \
  --exclude '.git/' --exclude '.claude/' --exclude 'Autodata.md' \
  --exclude '.env' --exclude '.secrets/' --exclude 'outputs/' --exclude '.hydra/' \
  --exclude 'multirun/' --exclude '.venv/' --exclude '__pycache__/' \
  --exclude '.pytest_cache/' --exclude '.ruff_cache/' --exclude '.mypy_cache/' \
  --exclude '*.egg-info/' \
  "$ROOT/" "$REMOTE:$REMOTE_REPO_DIR/"
