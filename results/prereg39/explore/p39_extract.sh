#!/usr/bin/env bash
# prereg39: launch the P1 extraction (4090); output appended to data/prereg39/p1/extract_log.txt.
#   bash results/prereg39/explore/p39_extract.sh [--sets L,D,pub2,priv2,E,E2] [--workers 8]
# Resumable: finished chunks are skipped. Yield the GPU: create data/prereg39/p1/STOP; delete it before resuming.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${C10_GPU_PY:-/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$REPO" || exit 2
mkdir -p data/prereg39/p1
LOG=data/prereg39/p1/extract_log.txt
if [ -e data/prereg39/p1/STOP ]; then echo "STOP file present: delete it first"; exit 3; fi
echo "=== $(date '+%F %T') p39_extract.sh $* (git $(git rev-parse --short HEAD))" >> "$LOG"
"$PY" -u results/prereg39/explore/p39_extract.py run "$@" >> "$LOG" 2>&1
rc=$?
echo "=== $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
