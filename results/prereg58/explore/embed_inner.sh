#!/usr/bin/env bash
# prereg58: S0 opponent-side vectors of the inner games used by the inner questions (GPU, --workers 2, hidden).
# Waits for the launch chain (data/prereg58/embed_s0opp_log.txt: "all done") and for available RAM >= 7000 MB.
# Start: powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'results/prereg58/explore/embed_inner.sh'"
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
LOG=$MAIN/data/prereg58/embed_inner_log.txt
CHAIN=$MAIN/data/prereg58/embed_s0opp_log.txt
PY=$MAIN/.venv-gpu/Scripts/python.exe
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
cd $MAIN || exit 3
say "waiting for the launch chain"
until grep -q "all done" $CHAIN; do
  if grep -q "end rc=[1-9]" $CHAIN; then say "the launch chain failed: not starting"; exit 4; fi
  sleep 20
done
until [ "$(avail)" -ge 7000 ]; do sleep 30; done
say "start inner opponent side"
"$PY" -u results/prereg58/explore/p58_rows.py embed --cache-dir data/prereg44/cache/inner \
  --rows data/prereg58/inner_rows.npy --side opp --name inner \
  --ckpt data/prereg43/S0/ckpt/step_320000.pt --weights raw --tta 1 --workers 2 \
  --emb-root data/prereg58/emb_opp_inner >> $LOG 2>&1
rc=$?; say "end rc=$rc"; [ $rc -ne 0 ] && exit $rc
say "all done"
