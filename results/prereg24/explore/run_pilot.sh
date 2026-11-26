#!/usr/bin/env bash
# prereg24 pilot (CPU, PLAN.md 5 step 3): one queue of arm:seed jobs, detached from Claude.
# Launch (PowerShell), two queues in parallel (8 torch threads each):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg24/explore/run_pilot.sh","qA" -WindowStyle Hidden
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg24/explore/run_pilot.sh","qB" -WindowStyle Hidden
# A finished arm:seed is never retrained (p24_pilot.py refuses an existing model file).
Q=${1:?queue qA or qB}
case "$Q" in
  qA) JOBS="R100:0,P25:0,P25:1,P25:2,R33:0,R33:1,R33:2" ;;
  qB) JOBS="P50:0,P50:1,P50:2,R56:0,R56:1,R56:2" ;;
  *) echo "unknown queue $Q"; exit 2 ;;
esac
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg24/pilot
LOG=data/prereg24/pilot/train_$Q.log
echo "start $Q $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
"$PY" results/prereg24/explore/p24_pilot.py train --jobs "$JOBS" >> "$LOG" 2>&1
echo "end $Q rc=$? $(date -u +%FT%TZ)" >> "$LOG"
