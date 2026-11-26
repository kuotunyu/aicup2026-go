#!/usr/bin/env bash
# prereg25 P1 extraction on the 4090 (PLAN.md 2.3), detached from Claude, resumable chunk by chunk.
# Launch (PowerShell; the worktree is on the subst drive Q:):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg25/explore/run_v.sh","seam" -WindowStyle Hidden
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg25/explore/run_v.sh","v" -WindowStyle Hidden
# seam: the first 400 games (chunks 0, 1) of the P3 probe again -> data/prereg25/seam (seam G).
# v:    the V games of data/prereg25/v/sets.json -> data/prereg25/v.
# Yield the GPU: create <out dir>/STOP (the run ends after the chunk in progress). Resume: delete STOP, launch again.
MODE=${1:?seam or v}
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe
case "$MODE" in
  seam) D=data/prereg25/seam; ARGS="--sets data/prereg09/p3/sets.json --max-games 400" ;;
  v)    D=data/prereg25/v;    ARGS="--sets data/prereg25/v/sets.json" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
mkdir -p "$D"
LOG=$D/run.log
echo "launcher start $MODE $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | tr -d '\r ')
echo "driver $DRV" >> "$LOG"
if [ "$DRV" != "616.92" ]; then echo "driver is not 616.92: not starting" >> "$LOG"; exit 3; fi
"$PY" results/prereg25/explore/p25_extract.py run --set probe $ARGS --out-dir "$D" --workers 12 >> "$LOG" 2>&1
echo "$MODE rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "launcher end $(date -u +%FT%TZ)" >> "$LOG"
