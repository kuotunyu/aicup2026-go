#!/usr/bin/env bash
# prereg40 S40 training (CPU, PLAN.md 7), detached from Claude. Needs the scheduling session's OK first.
# Launch from PowerShell on the subst drive (as prereg23 / prereg25 did for worktrees under AppData\Local), e.g.
#   subst P: <worktree>; Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/p/results/prereg40/explore/run_train.sh" -WindowStyle Hidden
#   then set the python process to BelowNormal: (Get-Process python | ? Path -like "*aicup2026-go*") | % { $_.PriorityClass = "BelowNormal" }
# Smoke: run_train.sh --smoke. After an interruption: run_train.sh --resume (same commit and code). Stop: create data/prereg40/STOP.
cd "$(dirname "$0")/../../.." || exit 1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe
mkdir -p data/prereg40
LOG=data/prereg40/train_log.txt
echo "start $(date -u +%FT%TZ) head $(git rev-parse HEAD) args $*" >> "$LOG"
"$PY" results/prereg40/explore/p40_train.py train "$@" >> "$LOG" 2>&1
echo "end rc=$? $(date -u +%FT%TZ)" >> "$LOG"
