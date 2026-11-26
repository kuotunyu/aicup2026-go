#!/usr/bin/env bash
# prereg31 S31 training (CPU, PLAN.md 3 / 6 step 5: seam R with HD off, then S31 seeds 0-9), detached from Claude.
# Launch (PowerShell): Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg31/explore/run_train.sh" -WindowStyle Hidden
# Shared PC (BELOW_NORMAL priority, 8 threads unchanged): -ArgumentList "/q/results/prereg31/explore/run_train.sh","--low-priority"
# After an interruption or a STOP (same commit and code): delete data/prereg31/STOP, then
#   -ArgumentList "/q/results/prereg31/explore/run_train.sh","--resume"   (add "--low-priority" if wanted)
# A RAM watch runs next to the training: available RAM < 4 GB -> it creates data/prereg31/STOP (PLAN.md revision 1).
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg31
LOG=data/prereg31/train_log.txt
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) run $RUN args $*" >> "$LOG"
"$PY" -I results/prereg31/explore/ram_watch.py --log "$LOG" --stop data/prereg31/STOP \
  --out data/prereg31/ram_watch.jsonl --run-id "$RUN" >> data/prereg31/ram_watch_log.txt 2>&1 &
"$PY" results/prereg31/explore/p31_train.py train "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ) run $RUN" >> "$LOG"
wait
