#!/usr/bin/env bash
# prereg45: wait for the war room's GO (its stop_now file absent), then run the P1 extraction (4090).
# Log: data/prereg45/p1/extract_log.txt. Yield: the war room's stop_now file -> STOP (stops before the next chunk).
# Resume: delete data/prereg45/p1/STOP and run this again (finished chunks are skipped).
set -u
REPO=/d/AI-Portfolio/competitions/aicup2026-go
PY=$REPO/.venv-gpu/Scripts/python.exe
STOPNOW=/c/AI-Portfolio-cache/war_room/4090_c10_stop_now.txt
OUT=$REPO/data/prereg45/p1
LOG=$OUT/extract_log.txt
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p $OUT
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
cd $REPO || exit 2
[ -e $OUT/STOP ] && { say "STOP present: delete it first"; exit 3; }
say "armed (git $(git rev-parse --short HEAD)); waiting for GO (stop_now absent), RAM >= 8 GB, driver 616.92"
until [ ! -e "$STOPNOW" ] && [ "$(avail)" -ge 8192 ]; do sleep 30; done
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' \r')
[ "$drv" = "616.92" ] || { say "driver $drv != 616.92: not starting"; exit 4; }
say "GO: start (avail $(avail) MB, GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader))"
C10_P45_SHARE_GPU=1 "$PY" -u results/prereg45/explore/p45_extract.py run --sets M1,M2 --workers 2 >> $LOG 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if [ -e "$STOPNOW" ] && [ ! -e $OUT/STOP ]; then say "stop_now seen: STOP (stops before next chunk)"; touch $OUT/STOP; fi
  if [ "$(avail)" -lt 6144 ] && [ ! -e $OUT/STOP ]; then say "RAM < 6 GB: STOP"; touch $OUT/STOP; fi
  sleep 15
done
wait $pid; rc=$?; say "exit $rc"; exit $rc
