#!/usr/bin/env bash
# prereg29 S29 training (CPU, PLAN.md 1 / 6 step 4), detached from Claude.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg29/explore/run_train.sh" -WindowStyle Hidden
# After an interruption: run_train.sh --resume (same commit and code).
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg29
LOG=data/prereg29/train_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) args $*" >> "$LOG"
"$PY" results/prereg29/explore/p29_train.py train "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
