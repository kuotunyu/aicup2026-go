#!/usr/bin/env bash
# prereg51 arm L on the 4090 (waits for the war room GO file 4090_c10_p52_go.txt): train, then embed E and the two official pools
# with the last checkpoint, then write the done file.
# Gate: no 4090_c10_stop_now.txt, VRAM in use < 6 GB, available RAM >= 9 GB for 10 minutes in a row, driver 616.92.
# Pauses (STOP -> save within 100 steps -> exit 4) on stop_now or available RAM < 5.5 GB, then waits for the gate
# again (RAM recovery alone resumes it). Ends only at DONE or on an error.
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
OUT=$MAIN/data/prereg51/L
CL=$MAIN/data/prereg51/chain_L_log.txt
WR=/c/AI-Portfolio-cache/war_room
STOPNOW=$WR/4090_c10_stop_now.txt
PAUSED=$WR/4090_c10_paused.txt
DONE=$WR/4090_c10_p52_done.txt
PROG=$WR/4090_c10_progress.txt
REC=$MAIN/results/prereg51/explore/train_record_L.json
PY=$MAIN/.venv-gpu/Scripts/python.exe
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p $OUT
say() { echo "$(date '+%F %T') $*" >> $CL; }
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
finished() { grep -q '"finished": true' $REC 2>/dev/null; }
progress() {
  last=$(ls $OUT/ckpt 2>/dev/null | sort | tail -1)
  step=$(grep -o '"step": [0-9]*' $OUT/train_log.jsonl 2>/dev/null | tail -1 | grep -o '[0-9]*$')
  echo "$(date '+%F %T') C10 prereg51-L: $1; step ${step:-0}/480000; last checkpoint ${last:-none}" >> $PROG
}
REPORTED=""
daily() { d=$(date +%F); if [ "$(date +%H)" = "08" ] && [ "$REPORTED" != "$d" ]; then progress "08:00 report"; REPORTED=$d; fi; }
gate() {
  ok=0
  while :; do
    daily
    if [ -f "$WR/4090_c10_p52_go.txt" ] && [ ! -f "$STOPNOW" ]; then
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
      a=$(avail)
      if [ -n "$used" ] && [ "$used" -lt 6144 ] && [ "$a" -ge 8000 ]; then
        ok=$((ok + 1)); [ $ok -ge 10 ] && return 0
      else
        [ $ok -gt 0 ] && say "gate reset: VRAM ${used} MiB / RAM ${a} MB"
        ok=0
      fi
    else
      ok=0
    fi
    sleep 60
  done
}
cd $MAIN || exit 3
say "chain armed (git $(git rev-parse --short HEAD))"
while ! finished; do
  gate
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' \r')
  if [ "$drv" != "616.92" ]; then say "ERROR driver $drv != 616.92"; progress "ERROR driver $drv"; exit 2; fi
  rm -f $OUT/STOP $PAUSED
  say "train start/resume (avail $(avail) MB)"; progress "training started/resumed"
  "$PY" -u results/prereg51/explore/p51_run.py train --arm L --workers 4 --auto-resume --log-every 1000 >> $OUT/train_log.txt 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ ! -f $OUT/STOP ]; then
      if [ -f "$STOPNOW" ]; then touch $OUT/STOP; say "stop_now seen: STOP set"; fi
      a=$(avail); if [ "$a" -lt 4608 ]; then touch $OUT/STOP; say "RAM $a MB < 4.5 GB: STOP set"; fi
    fi
    daily
    sleep 15
  done
  wait $pid; rc=$?
  last=$(ls $OUT/ckpt 2>/dev/null | sort | tail -1)
  say "train process ended rc=$rc; last checkpoint $last"
  if ! finished; then
    if [ -f $OUT/STOP ]; then
      echo "$(date '+%F %T') C10 prereg51-L paused at $last; 4090 free" > $PAUSED
      progress "paused"
      say "paused; waiting for the gate again"
    else
      say "ERROR ended without STOP and not finished (check train_log)"; progress "ERROR training exited rc=$rc"; exit 5
    fi
  fi
done
rm -f $OUT/STOP
CK=data/prereg51/L/ckpt/step_480000.pt
for job in "--source store --players E" "--source pool --src data/test_v2_download/x/tests --pool public" \
           "--source pool --src data/test_v2_download/x/tests --pool private"; do
  gate
  say "embed start: $job"
  "$PY" -u results/prereg51/explore/p51_run.py embed --arm L --ckpt $CK --weights raw --tta 1 --workers 4 --emb-root data/prereg51/emb_L $job >> $MAIN/data/prereg51/embed_L_log.txt 2>&1
  rc=$?; say "embed end rc=$rc: $job"
  if [ $rc -ne 0 ]; then progress "ERROR embed rc=$rc"; exit 6; fi
done
echo "$(date '+%F %T') C10 prereg51-L training + E / pool vectors done; 4090 released" > $DONE
say "all done; wrote $DONE"; progress "DONE (training + vectors)"
