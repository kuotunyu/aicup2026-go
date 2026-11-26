#!/usr/bin/env bash
# prereg59: the unattended CPU chain after embed_val.sh: kernel -> commit, seam -> commit, judge once -> commit,
# RESULTS.md -> commit. Stops at the first failure; every step is skipped when its committed output exists, so the
# chain can be restarted. Copy of record: results/prereg59/explore/after_embed.sh; run data/prereg59/after_embed.sh
# Start (hidden): powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -WorkingDirectory 'D:\AI-Portfolio\competitions\aicup2026-go' -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'data/prereg59/after_embed.sh'"
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
S=results/prereg59/explore
LOG=$MAIN/data/prereg59/after_log.txt
EMB_LOG=$MAIN/data/prereg59/embed_val_log.txt
PY=$MAIN/.venv/Scripts/python.exe
STOP_AT="2026-11-23T12:00:00"          # PLAN.md hard stop
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
wait_ram() { until [ "$(avail)" -ge 7000 ]; do sleep 30; done; }
past_stop() { [ "$(date '+%FT%T')" \> "$STOP_AT" ]; }
clean() { [ -n "$(git ls-files -- "$1")" ] && [ -z "$(git status --porcelain -- "$1")" ]; }   # committed, unchanged
commit() {   # commit <file> <message>: retries while another session holds the index
  local f=$1 msg=$2 i
  for i in $(seq 1 40); do
    git add -- "$f" >> $LOG 2>&1 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com \
      commit -q -m "$msg" -- results/prereg59 >> $LOG 2>&1
    if clean "$f"; then say "committed $f: $(git log --oneline -1 | cut -c1-120)"; return 0; fi
    sleep 15
  done
  say "could not commit $f"; return 1
}
run() {   # run <name> <p59_eval.py arguments...>
  local name=$1; shift
  wait_ram
  say "start $name"
  "$PY" -u $S/p59_eval.py "$@" >> $LOG 2>&1
  local rc=$?; say "end $name rc=$rc"; return $rc
}
cd $MAIN || exit 3
say "chain started; waiting for 'all done' in data/prereg59/embed_val_log.txt"
until [ -f "$EMB_LOG" ] && [ "$(tail -n 1 "$EMB_LOG" | grep -c 'all done')" -eq 1 ]; do
  if past_stop; then say "past the hard stop $STOP_AT while waiting for the vectors: nothing run"; exit 9; fi
  sleep 60
done
# (i) kernel
if clean $S/kernel.json; then say "kernel.json is committed: skip"
else
  if [ ! -f $S/kernel.json ]; then run kernel kernel || exit 11; fi
  commit $S/kernel.json "prereg59 kernel frozen (unattended chain): $("$PY" $S/p59_eval.py summary kernel 2>/dev/null | tail -n 1) - before any inner number of this version" || exit 12
fi
# (ii) seam
if clean $S/seam.json; then say "seam.json is committed: skip"
else
  if [ ! -f $S/seam.json ]; then run seam seam || exit 21; fi
  commit $S/seam.json "prereg59 seam (unattended chain): $("$PY" $S/p59_eval.py summary seam 2>/dev/null | tail -n 1)" || exit 22
fi
"$PY" $S/p59_eval.py summary seam >> $LOG 2>&1 || { say "seam.json is not ok: stop before the judgement"; exit 23; }
# (iii) the judgement, once
if clean $S/eval.json; then say "eval.json is committed: skip"
else
  if [ ! -f $S/eval.json ]; then
    if [ -f data/prereg59/eval_started.json ]; then
      say "eval_started.json exists without eval.json: an earlier judgement run died; judged once - stop (a person decides)"; exit 31
    fi
    if past_stop; then say "past the hard stop $STOP_AT: no judgement"; exit 32; fi
    run eval eval || exit 33
  fi
  commit $S/eval.json "prereg59 judged once (unattended chain): $("$PY" $S/p59_eval.py summary eval 2>/dev/null | tail -n 1)" || exit 34
fi
# (iv) RESULTS.md, written mechanically from eval.json / kernel.json
if clean $S/RESULTS.md; then say "RESULTS.md is committed: skip"
else
  if [ ! -f $S/RESULTS.md ]; then run results results || exit 41; fi
  commit $S/RESULTS.md "prereg59 RESULTS (written mechanically from eval.json / kernel.json by the unattended chain)" || exit 42
fi
say "all done"
