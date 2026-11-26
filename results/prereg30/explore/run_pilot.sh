#!/usr/bin/env bash
# prereg30 pilot extraction on the 4090 (PLAN.md 2), detached from Claude, resumable chunk by chunk.
# Launch (PowerShell); the script path must use forward slashes (Git Bash dirname):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "<worktree>/results/prereg30/explore/run_pilot.sh","pilot" -WindowStyle Hidden
#   optional 2nd argument: minutes after which no new chunk is started (e.g. "pilot 60"); rerun the same line to resume.
# Yield the GPU: create data/prereg30/pilot/STOP (the run ends after the chunk in progress). Resume: delete STOP, launch again.
# Before anything: driver must be 616.92 and nvidia-smi must list no other compute (Type C) process
# (p1_extract checks the latter itself and refuses to start).
MODE=${1:?pilot}
STOP_MIN=${2:-0}
cd "$(dirname "$0")/../../.." || exit 1
[ -f results/prereg30/explore/p30_extract.py ] && [ -f data/prereg30/pilot/sets.json ] || { echo "wrong cwd $(pwd)"; exit 4; }
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe
case "$MODE" in
  pilot) D=data/prereg30/pilot ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
mkdir -p "$D"
LOG=$D/run.log
echo "launcher start $MODE $(date -u +%FT%TZ) head $(git rev-parse HEAD) stop_after_min $STOP_MIN" >> "$LOG"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | tr -d '\r ')
echo "driver $DRV" >> "$LOG"
if [ "$DRV" != "616.92" ]; then echo "driver is not 616.92: not starting" >> "$LOG"; exit 3; fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG" 2>&1
"$PY" results/prereg30/explore/p30_extract.py run --set probe --sets "$D/sets.json" --out-dir "$D" \
    --workers 8 --stop-after-min "$STOP_MIN" >> "$LOG" 2>&1
echo "$MODE rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "launcher end $(date -u +%FT%TZ)" >> "$LOG"
