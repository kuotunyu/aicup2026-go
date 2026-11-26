#!/usr/bin/env bash
# prereg40 stage 0b and judgement (CPU, PLAN.md 4-5), each run once, detached from Claude; scheduling session's OK first.
# Usage: run_eval.sh stage0b   (commit stage0b.json, then)   run_eval.sh eval      [--smoke for either]
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg40
LOG=data/prereg40/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) args $*" >> "$LOG"
"$PY" results/prereg40/explore/p40_eval.py "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
