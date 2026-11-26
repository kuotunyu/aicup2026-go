#!/usr/bin/env bash
# prereg64: c49 vectors needed before the judgement (GPU 4090, --workers 2, sequential). Not started by the agent.
# Copy of record: results/prereg64/explore/embed_gpu.sh; run copy: data/prereg64/embed_gpu.sh (identical).
#  job 1: c49 OPPONENT-side vectors of the inner question games (data/prereg58/inner_rows.npy, 73,169 rows of
#         data/prereg44/cache/inner) -> data/prereg64/emb_opp_inner/V2_step_320000_raw_tta1/rows_inner.npz
#  job 2: c49 PLAYER-side vectors of the prototype owners' games (data/prereg64/proto_rows.npy, 18,892 rows of
#         data/prereg49/cache, written by p64_rows.py rows) -> data/prereg64/emb_proto/V2_step_320000_raw_tta1/rows_proto.npz
# Each job: wait until available RAM >= 7000 MB; p58_rows.py skips a job whose output exists; a failed job stops
# the script (no "all done"). Log: data/prereg64/embed_log.txt.
# Start (hidden): powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -WorkingDirectory 'D:\AI-Portfolio\competitions\aicup2026-go' -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'data/prereg64/embed_gpu.sh'"
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
LOG=$MAIN/data/prereg64/embed_log.txt
PY=$MAIN/.venv-gpu/Scripts/python.exe
CK=data/prereg49/S0/ckpt/step_320000.pt
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
cd $MAIN || exit 3
for f in data/prereg58/inner_rows.npy data/prereg64/proto_rows.npy $CK; do
  if [ ! -f "$f" ]; then say "missing $f: not starting"; exit 5; fi
done
run() { name=$1; shift
  until [ "$(avail)" -ge 7000 ]; do sleep 30; done
  say "start $name"
  "$PY" -u results/prereg58/explore/p58_rows.py embed "$@" --ckpt $CK --weights raw --tta 1 --workers 2 >> $LOG 2>&1
  rc=$?; say "end rc=$rc $name"; [ $rc -ne 0 ] && exit $rc; return 0; }
run "job 1 inner opponent side" --cache-dir data/prereg44/cache/inner --rows data/prereg58/inner_rows.npy \
  --side opp --name inner --emb-root data/prereg64/emb_opp_inner
run "job 2 prototype owners player side" --cache-dir data/prereg49/cache --rows data/prereg64/proto_rows.npy \
  --side own --name proto --emb-root data/prereg64/emb_proto
say "all done"
