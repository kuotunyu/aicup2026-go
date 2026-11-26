#!/usr/bin/env bash
# prereg36: launch the CNN training with the GPU python; output appended to data/prereg36/train_log.txt.
#   bash results/prereg36/explore/run_train.sh [--workers N] [--resume] [--smoke --steps 200] ...
# Stop cleanly: create data/prereg36/STOP (checked every 100 steps), remove it before --resume.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${C10_GPU_PY:-/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$REPO" || exit 2
mkdir -p data/prereg36
LOG=data/prereg36/train_log.txt
echo "=== $(date '+%F %T') run_train.sh $* (git $(git rev-parse --short HEAD))" >> "$LOG"
"$PY" -u results/prereg36/explore/p36_train.py "$@" >> "$LOG" 2>&1
rc=$?
echo "=== $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
