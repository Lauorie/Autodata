#!/usr/bin/env bash
# Serve the weak solver (Qwen3.5-4B) as an OpenAI-compatible endpoint.
#
#   MODEL_DIR=<path to Qwen3.5-4B weights> bash scripts/serve_weak_solver.sh
#
# Then add to .secrets/remote.env, which the run scripts load:
#   WEAK_SOLVER_BASE_URL=http://<HOST>:<PORT>/v1
#
# --reasoning-parser qwen3 moves the thinking into reasoning_content, so the
# judge grades only the answer. --max-model-len covers the 65,536-token solver
# budget (solver_generation.max_tokens in conf/config.yaml) plus the prompt;
# raise both together. --language-model-only skips the vision encoder.
# Extra arguments pass through to vllm serve (e.g. --tensor-parallel-size 2).
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.5-4B weights directory}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen3.5-4b \
  --host "$HOST" --port "$PORT" \
  --max-model-len 81920 \
  --reasoning-parser qwen3 \
  --language-model-only \
  "$@"
