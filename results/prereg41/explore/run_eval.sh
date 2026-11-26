#!/usr/bin/env bash
# prereg41 judgement (CPU, PLAN.md 5), run once, detached; launch on a subst drive, hidden, BelowNormal.
# Usage: run_eval.sh [--smoke]
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg41
LOG=data/prereg41/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) args $*" >> "$LOG"
"$PY" results/prereg41/explore/p41_eval.py eval "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
