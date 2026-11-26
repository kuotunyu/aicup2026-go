#!/usr/bin/env bash
# prereg26 judgement (CPU, PLAN.md 8 step 5), run once, detached from Claude.
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
LOG=data/prereg26/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
"$PY" results/prereg26/explore/p26_eval.py eval >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
