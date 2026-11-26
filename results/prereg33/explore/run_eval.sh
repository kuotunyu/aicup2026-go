#!/usr/bin/env bash
# prereg33 judgement (CPU, PLAN.md 2 / 3 step 2), run once, detached from Claude. Only after train_record.json is
# committed and the shared-PC scheduler has agreed.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg33/explore/run_eval.sh" -WindowStyle Hidden
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg33
LOG=data/prereg33/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
"$PY" results/prereg33/explore/p33_eval.py eval >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
