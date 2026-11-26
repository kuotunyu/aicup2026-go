#!/usr/bin/env bash
# prereg31 judgement (CPU, PLAN.md 4 / 6 step 7), run once, detached from Claude. Only after stage0b.json says GO.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg31/explore/run_eval.sh" -WindowStyle Hidden
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg31
LOG=data/prereg31/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
"$PY" results/prereg31/explore/p31_eval.py eval >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
