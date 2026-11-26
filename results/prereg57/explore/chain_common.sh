# prereg57: shared by run_dev.sh / run_judge.sh / run_apply.sh (sourced; the caller sets LOG_NAME first).
# Conventions of data/prereg51/chain_p51L_b.sh and data/prereg55/embed_S.sh:
#   * log to data/prereg57/<LOG_NAME>; GPU python = .venv-gpu; no thread environment variables;
#   * before every step: data/prereg57/STOP -> exit 4; wait until available RAM >= 7000 MB;
#   * a step whose output exists is skipped, so starting the chain again resumes it (adapt: --auto-resume from its
#     highest step number); the chain stops on the first error;
#   * adapt exit code 4 = it saved and stopped (STOP file, or free RAM < 4 GB): with a STOP file the chain exits 4,
#     otherwise it waits for RAM and resumes;
#   * adapt refuses to start (exit 1 -> the chain stops) while another python compute process is on the GPU.
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
D=$MAIN/data/prereg57
LOG=$D/$LOG_NAME
STOP=$D/STOP
PY=$MAIN/.venv-gpu/Scripts/python.exe
FT=results/prereg57/explore/p57_ft.py
WORKERS=${P57_WORKERS:-4}
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p $D
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
check_stop() { if [ -f "$STOP" ]; then say "STOP file $STOP: exiting"; exit 4; fi; }
wait_ram() {
  while :; do
    check_stop
    a=$(avail)
    if [ -n "$a" ] && [ "$a" -ge 7000 ]; then return 0; fi
    sleep 30
  done
}
ckpt_name() { printf 'step_%06d.pt' "$1"; }

# adapt_run <base> <pool kind> <pool> <share> <lr> <steps> <ckpt-at> <out dir>
adapt_run() {
  local base=$1 kind=$2 pool=$3 share=$4 lr=$5 steps=$6 at=$7 out=$8
  local last=$out/ckpt/$(ckpt_name $steps)
  while [ ! -f "$last" ]; do
    wait_ram
    if [ -f "$out/STOP" ]; then say "STOP file $out/STOP: exiting"; exit 4; fi
    say "adapt start $base $kind $pool share $share lr $lr -> step $steps (avail $(avail) MB, git $(git rev-parse --short HEAD))"
    "$PY" -u $FT adapt --base $base --pool-kind $kind --pool $pool --share $share --lr $lr --steps $steps \
      --ckpt-at $at --out $out --workers $WORKERS --auto-resume >> $LOG 2>&1
    rc=$?
    say "adapt end rc=$rc $out"
    if [ $rc -eq 4 ]; then
      check_stop
      say "paused without a STOP file (low RAM): waiting to resume"
      sleep 60
      continue
    fi
    if [ $rc -ne 0 ]; then exit $rc; fi
    if [ ! -f "$last" ]; then say "ERROR adapt ended rc=0 without $last"; exit 5; fi
  done
}

# embed_run <base> <pool kind> <pool> <out dir> <step>     (EMA weights, PLAN.md Method 3)
embed_run() {
  local base=$1 kind=$2 pool=$3 out=$4 n=$5
  local npz=$out/emb/step_${n}_ema.npz
  if [ -f "$npz" ]; then return 0; fi
  wait_ram
  say "embed start $out step $n (avail $(avail) MB)"
  "$PY" -u $FT embed --base $base --pool-kind $kind --pool $pool --ckpt $out/ckpt/$(ckpt_name $n) --weights ema \
    --out $npz --workers $WORKERS >> $LOG 2>&1
  rc=$?
  say "embed end rc=$rc $npz"
  if [ $rc -ne 0 ]; then exit $rc; fi
  if [ ! -f "$npz" ]; then say "ERROR embed ended rc=0 without $npz"; exit 5; fi
}
