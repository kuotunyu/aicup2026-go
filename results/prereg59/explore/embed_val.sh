#!/usr/bin/env bash
# prereg59: S0 PLAYER-side vectors of every training game of the validation players (GPU, --workers 2).
# Run only when the 4090 is free. Copy of record: results/prereg59/explore/embed_val.sh; run data/prereg59/embed_val.sh
# Start (hidden): powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -WorkingDirectory 'D:\AI-Portfolio\competitions\aicup2026-go' -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'data/prereg59/embed_val.sh'"
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
LOG=$MAIN/data/prereg59/embed_val_log.txt
PY=$MAIN/.venv-gpu/Scripts/python.exe
OUT=$MAIN/data/prereg59/emb_own_val/V2_step_320000_raw_tta1/rows_val.npz
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
cd $MAIN || exit 3
if [ -f "$OUT" ]; then say "exists: $OUT (skip)"; say "all done"; exit 0; fi
if [ ! -f data/prereg59/val_rows.npy ]; then say "data/prereg59/val_rows.npy is missing (p59_eval.py rows)"; exit 5; fi
until [ "$(avail)" -ge 7000 ]; do sleep 30; done
say "start validation players, player side"
"$PY" -u results/prereg59/explore/p59_rows.py embed --workers 2 >> $LOG 2>&1
rc=$?; say "end rc=$rc"; [ $rc -ne 0 ] && exit $rc
if [ ! -f "$OUT" ]; then say "no output file"; exit 6; fi
say "all done"
