#!/usr/bin/env bash
# prereg45: restart run_extract.sh after a RAM-guard STOP (not after the war room's stop_now). Loops until M1 and M2
# are complete or 2026-11-15 22:00 (PLAN 1). Restarts only when stop_now is absent and available RAM >= 9 GB.
set -u
REPO=/d/AI-Portfolio/competitions/aicup2026-go
OUT=$REPO/data/prereg45/p1
LOG=$OUT/extract_log.txt
STOPNOW=/c/AI-Portfolio-cache/war_room/4090_c10_stop_now.txt
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') supervisor: $*" >> $LOG; }
running() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*p45_extract.py*' }).Count" | tr -d '\r '; }
done_all() { [ "$(ls $OUT/M1 2>/dev/null | grep -c '^chunk_.*[0-9]\.npz$')" -ge 358 ] && [ "$(ls $OUT/M2 2>/dev/null | grep -c '^chunk_.*[0-9]\.npz$')" -ge 335 ]; }
m1_done() { [ "$(ls $OUT/M1 2>/dev/null | grep -c '^chunk_.*[0-9]\.npz$')" -ge 358 ]; }
DEADLINE=$(date -d '2026-11-15 22:00' +%s)
say "start"
while [ $(date +%s) -lt $DEADLINE ]; do
  if m1_done; then touch $OUT/STOP; say "M1 complete: STOP (PLAN revision 1: M2 not extracted)"; exit 0; fi
  if [ "$(running)" = "0" ] && [ -e $OUT/STOP ] && [ ! -e "$STOPNOW" ] && [ "$(avail)" -ge 9216 ]; then
    rm -f $OUT/STOP; say "restart after STOP (avail $(avail) MB)"
    bash $REPO/results/prereg45/explore/run_extract.sh &
  fi
  sleep 30
done
say "deadline reached"
