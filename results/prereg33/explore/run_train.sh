#!/usr/bin/env bash
# prereg33 S33 training (CPU, PLAN.md 1 / 3 step 2: seam R33, then S33 seeds 0-2), detached from Claude.
# Only after the code is committed and the shared-PC scheduler has agreed.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg33/explore/run_train.sh" -WindowStyle Hidden
# After an interruption or a STOP (same commit and code): delete data/prereg33/STOP, then
#   -ArgumentList "/q/results/prereg33/explore/run_train.sh","--resume"
# A RAM watch (prereg31's ram_watch.py, unchanged) runs next to the training: available RAM < 4 GB -> it creates
# data/prereg33/STOP; the training stops at the next 500-step mark or before the next seed.
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg33
LOG=data/prereg33/train_log.txt
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) run $RUN args $*" >> "$LOG"
"$PY" -I results/prereg31/explore/ram_watch.py --log "$LOG" --stop data/prereg33/STOP \
  --out data/prereg33/ram_watch.jsonl --run-id "$RUN" >> data/prereg33/ram_watch_log.txt 2>&1 &
"$PY" results/prereg33/explore/p33_train.py train "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ) run $RUN" >> "$LOG"
wait
