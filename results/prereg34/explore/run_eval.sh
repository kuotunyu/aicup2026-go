#!/usr/bin/env bash
# prereg34 judgement (CPU, PLAN.md 6 / 7 step 4), run once, detached from Claude. Only after stage0b.json says GO
# (p34_eval.py refuses otherwise) and align_record.json, train_record.json, stage0b.json are committed.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg34/explore/run_eval.sh" -WindowStyle Hidden
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg34
LOG=data/prereg34/eval_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
"$PY" results/prereg34/explore/p34_eval.py eval >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
