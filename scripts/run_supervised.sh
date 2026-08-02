#!/bin/bash
# Self-healing benchmark runner for local (ollama) models.
#
# The ollama Metal backend can die under sustained load (GPU command-buffer
# faults), after which it answers 200s with empty bodies. chessbench detects
# that (health probe + empty-response circuit breaker) and exits nonzero with
# all in-flight games left incomplete. This wrapper restarts ollama and
# resumes, up to MAX_RESTARTS times. Safe by construction: completed games
# are never replayed, incomplete ones always are, and no game starts until
# the health probe passes.
#
# Usage: scripts/run_supervised.sh <out_dir> [chessbench args...]
set -u
cd "$(dirname "$0")/.."
OUT="${1:?usage: run_supervised.sh <out_dir> [chessbench args...]}"
shift
MAX_RESTARTS=8

for i in $(seq 0 "$MAX_RESTARTS"); do
  if [ "$i" -gt 0 ]; then
    echo "[supervisor] cycle $i/$MAX_RESTARTS: restarting ollama and resuming"
    osascript -e 'quit app "Ollama"' 2>/dev/null
    sleep 5
    open -a Ollama
    sleep 20
  fi
  uv run chessbench --out "$OUT" "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] run completed cleanly"
    exit 0
  fi
  echo "[supervisor] chessbench exited rc=$rc"
done
echo "[supervisor] giving up after $MAX_RESTARTS restarts — needs human attention"
exit 1
