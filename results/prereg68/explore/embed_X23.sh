#!/usr/bin/env bash
# prereg68: X2 / X3 vectors on the 4090 (E store + both official pools), like data/prereg62/embed_X.sh.
# An arm whose step_400000 checkpoint does not exist yet is skipped; a job whose output npz AND provenance sidecar
# exist is skipped (p68_embed.py refuses an npz without a valid sidecar). Every job goes through p68_embed.py
# (results/prereg68/explore/NOTES.md 9): the frozen embedder plus a per-game manifest / checkpoint binding.
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
LOG=$MAIN/data/prereg68/embed_log.txt
PY=$MAIN/.venv-gpu/Scripts/python.exe
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
cd $MAIN || exit 3
mkdir -p data/prereg68
for arm in X2 X3; do
  CK=data/prereg62/$arm/ckpt/step_400000.pt
  if [ ! -f "$CK" ]; then say "skip $arm: $CK missing"; continue; fi
  EMB=data/prereg62/$arm/emb/p60_step_400000_raw_tta1
  for job in "store_E|--source store --players E" \
             "pool_public|--source pool --src data/test_v2_download/x/tests --pool public" \
             "pool_private|--source pool --src data/test_v2_download/x/tests --pool private"; do
    out=${job%%|*}; args=${job#*|}
    if [ -f "$EMB/$out.npz" ] && [ -f "$EMB/$out.provenance.json" ]; then say "skip $arm $out: exists"; continue; fi
    until [ "$(avail)" -ge 7000 ]; do sleep 30; done
    say "start $arm $args"
    "$PY" -u results/prereg68/explore/p68_embed.py --arm $arm --ckpt $CK --workers 2 $args >> $LOG 2>&1
    rc=$?; say "end rc=$rc $arm $args"; [ $rc -ne 0 ] && exit $rc
  done
done
say "all done"
