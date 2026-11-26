#!/usr/bin/env bash
# prereg34 P1 extraction on the 4090 (PLAN.md 3), detached from Claude, resumable chunk by chunk.
# Launch (PowerShell; the worktree is on the subst drive Q:; only after the war room said yes):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg34/explore/run_p34.sh","seam" -WindowStyle Hidden
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/q/results/prereg34/explore/run_p34.sh","ogs" -WindowStyle Hidden
# seam: the first 400 games (chunks 0, 1) of the P3 probe again -> data/prereg34/seam (seam G34; then
#       .venv p34_extract.py seam -> seam_g34.json; the OGS run only after it passed).
# ogs:  the OGS games of data/prereg34/p1/sets.json -> data/prereg34/p1/probe.
# Yield the GPU: create <out dir>/STOP (the run ends after the chunk in progress). Resume: delete STOP, launch again.
MODE=${1:?seam or ogs}
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv-gpu/Scripts/python.exe
case "$MODE" in
  seam) D=data/prereg34/seam; ARGS="--sets data/prereg09/p3/sets.json --max-games 400" ;;
  ogs)  D=data/prereg34/p1;   ARGS="--sets data/prereg34/p1/sets.json"; mkdir -p "$D"
        if [ ! -f results/prereg34/explore/seam_g34.json ] || ! grep -q '"passed": true' results/prereg34/explore/seam_g34.json; then
          echo "seam G34 has not passed: not starting" >> "$D/run.log"; exit 3
        fi ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
mkdir -p "$D"
LOG=$D/run.log
echo "launcher start $MODE $(date -u +%FT%TZ) head $(git rev-parse HEAD)" >> "$LOG"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | tr -d '\r ')
echo "driver $DRV" >> "$LOG"
if [ "$DRV" != "616.92" ]; then echo "driver is not 616.92: not starting" >> "$LOG"; exit 3; fi
"$PY" results/prereg34/explore/p34_extract.py run --set probe $ARGS --out-dir "$D" --workers 12 >> "$LOG" 2>&1
echo "$MODE rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "launcher end $(date -u +%FT%TZ)" >> "$LOG"
