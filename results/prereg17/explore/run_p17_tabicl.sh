#!/usr/bin/env bash
# prereg17 exploration: detached launcher of p17_tabicl.py (PLAN.md 2.2). Git Bash, from anywhere:
#   bash results/prereg17/explore/run_p17_tabicl.sh <repo root> smoke|real
# Started from PowerShell with Start-Process so it does not depend on the conversation. Stop between chunks:
# create <out dir>/STOP. While the GPU has another python / katago compute process (or > 4 GB used) the step exits 4
# and this launcher waits 5 minutes and retries (at most 48 times); a different driver (exit 6) or an error stops it;
# a finished run exits 0. A STOP file left from an earlier stop must be deleted before a restart.
set -u
ROOT="$1"
MODE="$2"
PY="C:/Users/USER/venvs/c10-tabicl/Scripts/python.exe"
if [ "$MODE" = "smoke" ]; then
  FEAT="$ROOT/data/prereg17/smoke/feat"; OUT="$ROOT/data/prereg17/smoke/tabicl"
elif [ "$MODE" = "real" ]; then
  FEAT="$ROOT/data/prereg17/feat"; OUT="$ROOT/data/prereg17/tabicl"
else
  echo "mode is smoke or real"; exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/launcher.log"
export PYTHONDONTWRITEBYTECODE=1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
echo "$(date -Iseconds) start $MODE (pid $$)" >> "$LOG"
for i in $(seq 1 48); do
  if [ -e "$OUT/STOP" ]; then echo "$(date -Iseconds) STOP present" >> "$LOG"; exit 3; fi
  nvidia-smi >> "$LOG" 2>&1
  "$PY" -I "$ROOT/results/prereg17/explore/p17_tabicl.py" run --feat-dir "$FEAT" --out-dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Iseconds) attempt $i exit $rc" >> "$LOG"
  if [ $rc -eq 0 ]; then echo "$(date -Iseconds) DONE" >> "$LOG"; exit 0; fi
  if [ $rc -eq 4 ]; then sleep 300; continue; fi
  if [ $rc -eq 3 ] && [ ! -e "$OUT/STOP" ]; then sleep 30; continue; fi
  echo "$(date -Iseconds) FAILED" >> "$LOG"; exit $rc
done
echo "$(date -Iseconds) gave up after 48 attempts" >> "$LOG"
exit 5
