#!/usr/bin/env bash
# Pull the (modified) code + key outputs back from the remote host to local.
# Usage:  bash scripts/sync_from_remote.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$ROOT/.secrets/remote.env" ]; then
  echo "missing .secrets/remote.env" >&2; exit 1
fi
set -a; . "$ROOT/.secrets/remote.env"; set +a

echo "Pulling source from remote..."
sshpass -p "$REMOTE_PASSWORD" rsync -az --delete -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new" \
  --exclude '.secrets/' --exclude '__pycache__/' --exclude 'outputs/' --exclude '.hydra/' --exclude '.venv/' \
  "$REMOTE_USER@$REMOTE_HOST:/root/autodata/" "$ROOT/"

echo "Pulling outputs (summaries + accepted_qa.jsonl + meta population only)..."
mkdir -p "$ROOT/outputs"
sshpass -p "$REMOTE_PASSWORD" rsync -az -e "ssh -p $REMOTE_PORT" \
  --include '*/' --include 'summary.json' --include 'accepted_qa.jsonl' \
  --include 'population.json' --include '*.log' --exclude '*' \
  "$REMOTE_USER@$REMOTE_HOST:/root/autodl-fs/autodata/outputs/" "$ROOT/outputs/"

echo "Done."
