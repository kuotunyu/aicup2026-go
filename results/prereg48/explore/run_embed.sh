#!/usr/bin/env bash
# prereg48: TTA-8 vectors on the 4090. Usage: run_embed.sh D | E | pools
# Waits for the war room's GO (stop_now absent) and RAM >= 8 GB; stops at stop_now / RAM < 6 GB.
set -u
WHAT=${1:?D|E|pools}
REPO=/d/AI-Portfolio/competitions/aicup2026-go
PY=$REPO/.venv-gpu/Scripts/python.exe
STOPNOW=/c/AI-Portfolio-cache/war_room/4090_c10_stop_now.txt
LOG=$REPO/data/prereg48/embed_log.txt
CK=data/prereg43/S0/ckpt/step_320000.pt
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p $REPO/data/prereg48
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
cd $REPO || exit 2
say "$WHAT armed (git $(git rev-parse --short HEAD)); waiting for GO and RAM >= 8 GB"
until [ ! -e "$STOPNOW" ] && [ "$(avail)" -ge 8192 ]; do sleep 30; done
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' \r')
[ "$drv" = "616.92" ] || { say "driver $drv != 616.92"; exit 4; }
run() {  # args to p43_run.py embed
  [ -e "$STOPNOW" ] && { say "stop_now: not starting $*"; exit 6; }
  say "start $* (avail $(avail) MB, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
  "$PY" -u results/prereg43/explore/p43_run.py embed --ckpt $CK --weights raw --workers 0 --emb-root data/prereg48/emb_S0 "$@" >> $LOG 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(avail)" -lt 6144 ]; then say "RAM < 6 GB: killing"; taskkill //F //T //PID $(cat /proc/$pid/winpid) >/dev/null 2>&1; exit 7; fi
    if [ -e "$STOPNOW" ]; then say "stop_now: killing (rerun this step later)"; taskkill //F //T //PID $(cat /proc/$pid/winpid) >/dev/null 2>&1; exit 6; fi
    sleep 10
  done
  wait $pid; rc=$?; say "end rc=$rc"; [ $rc -eq 0 ] || exit $rc
}
case $WHAT in
  D) run --tta 1 --source store --players D --limit 200 --compare data/prereg43/emb_S0/V2_step_320000_raw_tta1/store_D.npz \
       --emb-root data/prereg48/seam_tta1
     run --tta 8 --source store --players D ;;
  E) run --tta 8 --source store --players E ;;
  pools) for p in public private; do run --tta 8 --source pool --src data/test_v2_download/x/tests --pool $p; done ;;
esac
say "$WHAT all done"
