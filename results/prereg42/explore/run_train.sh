#!/usr/bin/env bash
# prereg42: train one variant on the 4090 with the GPU python; output appended to data/prereg42/<variant>/train_log.txt.
#   bash results/prereg42/explore/run_train.sh --variant V2 [--workers 4] [--resume]
# A fresh run (no checkpoint yet) first runs the 300-step timing (PLAN.md 3): projected 40,000 steps > 4.5 h -> exit 5,
# nothing trained. Stop cleanly: create data/prereg42/<variant>/STOP (checked every 100 steps); remove it before --resume.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${C10_GPU_PY:-/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$REPO" || exit 2
VAR=""
WORKERS=4
prev=""
for x in "$@"; do
  [ "$prev" = "--variant" ] && VAR="$x"
  [ "$prev" = "--workers" ] && WORKERS="$x"
  prev="$x"
done
[ -n "$VAR" ] || { echo "--variant V1|V2|V3|V3b|V4 is required"; exit 2; }
mkdir -p "data/prereg42/$VAR"
LOG="data/prereg42/$VAR/train_log.txt"
echo "=== $(date '+%F %T') run_train.sh $* (git $(git rev-parse --short HEAD))" >> "$LOG"
if ! ls "data/prereg42/$VAR/ckpt/"step_*.pt >/dev/null 2>&1; then
  "$PY" -u results/prereg42/explore/p42_train.py --variant "$VAR" --workers "$WORKERS" --bench 300 \
    > "data/prereg42/$VAR/bench.json" 2>> "$LOG"
  rc=$?
  cat "data/prereg42/$VAR/bench.json" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "=== $(date '+%F %T') bench exit $rc (5 = over the 4.5 h budget): not training" >> "$LOG"
    exit $rc
  fi
fi
"$PY" -u results/prereg42/explore/p42_train.py "$@" >> "$LOG" 2>&1
rc=$?
echo "=== $(date '+%F %T') exit $rc" >> "$LOG"
exit $rc
