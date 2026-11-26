#!/usr/bin/env bash
# prereg37: launch the 160,000-step training; output appended to data/prereg37/train_log.txt.
#   bash results/prereg37/explore/run_train.sh --workers 4 [--resume]
# Stop cleanly: create data/prereg37/STOP (checked every 100 steps), remove it before --resume.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${C10_GPU_PY:-/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$REPO" || exit 2
mkdir -p data/prereg37
LOG=data/prereg37/train_log.txt
echo "=== $(date '+%F %T') run_train.sh $* (git $(git rev-parse --short HEAD))" >> "$LOG"
"$PY" -u results/prereg37/explore/p37_run.py train "$@" >> "$LOG" 2>&1
rc=$?
echo "=== $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
