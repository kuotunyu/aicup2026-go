#!/usr/bin/env bash
# prereg34 S34 training (CPU, PLAN.md 4 / 7 step 4: seam R34, then S34 seeds 0-2), detached from Claude.
# Only after the code and align_record.json are committed and the shared-PC scheduler has agreed.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg34/explore/run_train.sh" -WindowStyle Hidden
# After an interruption or a STOP (same commit and code): delete data/prereg34/STOP, then
#   -ArgumentList "/q/results/prereg34/explore/run_train.sh","--resume"
# A RAM watch (prereg31's ram_watch.py, unchanged) runs next to the training: available RAM < 4 GB -> it creates
# data/prereg34/STOP; the training stops at the next held-out evaluation (500 steps) or before the next seed.
# After the training: commit train_record.json, then p34_train.py stage0b (seconds), commit stage0b.json.
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg34
LOG=data/prereg34/train_log.txt
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) run $RUN args $*" >> "$LOG"
"$PY" -I results/prereg31/explore/ram_watch.py --log "$LOG" --stop data/prereg34/STOP \
  --out data/prereg34/ram_watch.jsonl --run-id "$RUN" >> data/prereg34/ram_watch_log.txt 2>&1 &
"$PY" results/prereg34/explore/p34_train.py train "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ) run $RUN" >> "$LOG"
wait
