#!/usr/bin/env bash
# prereg32 pilot (CPU, 4 torch threads, PLAN.md 3): control E0, then E1..E6, seeds 0 and 1; detached from Claude.
# Launch once (PowerShell), after everything in this folder is committed:
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" `
#     -ArgumentList "/q/results/prereg32/explore/run_pilot.sh" -WindowStyle Hidden
# Only after an interruption or a STOP-file stop (same commit and code): delete data/prereg32/STOP by hand, then
#   -ArgumentList "/q/results/prereg32/explore/run_pilot.sh","--resume"
# A RAM watch runs next to it (the parameterised one of prereg31): available RAM < 4 GB -> it creates
# data/prereg32/STOP, and the pilot stops at the next held-out evaluation (that run is discarded) or before the
# next run. The time limit 2026-11-11T08:00 (local) only decides whether a NEXT run is started: a run still going
# at that time has to be stopped by creating data/prereg32/STOP by hand.
cd /q/ || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg32
LOG=data/prereg32/pilot_log.txt
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) run $RUN args $*" >> "$LOG"
"$PY" -I results/prereg31/explore/ram_watch.py --log "$LOG" --stop data/prereg32/STOP \
  --out data/prereg32/ram_watch.jsonl --run-id "$RUN" >> data/prereg32/ram_watch_log.txt 2>&1 &
"$PY" results/prereg32/explore/p32_pilot.py run --deadline 2026-11-11T08:00 --low-priority "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ) run $RUN" >> "$LOG"
wait
