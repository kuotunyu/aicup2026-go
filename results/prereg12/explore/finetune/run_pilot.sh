#!/usr/bin/env bash
# prereg12 HumanSL fine-tuning pilot, detached from Claude (PLAN.md section 4). Called by data/prereg12/run_pilot.sh.
# Launch (PowerShell):
#   Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/data/prereg12/run_pilot.sh" -WindowStyle Hidden
#   CPU stages only (sets + caches, no GPU):  ... -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/data/prereg12/run_pilot.sh cpu-only"
# Order: sets -> cache train / holdout / bank / mono (CPU) -> wait for a free GPU -> bench -> train -> extract bank,
# mono -> ft_eval prep-inner + eval (CPU; writes results/prereg12/explore/finetune/eval_report.json, the gate).
# The GPU is free only when BOTH checks are empty: hs_lib.other_compute_processes (nvidia-smi rows of Type 'C') and
# nvidia-smi --query-compute-apps rows naming python. Checked every 5 min; every GPU step checks again before CUDA.
# Yield the GPU / stop: create data/prereg12/STOP (steps stop after the current chunk / at the next step; the
# launcher exits). Resume: delete STOP and launch again (finished chunks, checkpoints and reports are reused).
# Smoke (CPU, never a result): FT_ROOT=data/prereg12/smoke/<x> DEVICE=cpu GPU_PY=.venv/Scripts/python.exe
#   SETS_FLAGS="--train-per-level 20 --holdout-per-level 6" INNER_FLAGS="--max-games 6 --chunk 3"
#   STEP_FLAGS="--tiny --allow-partial-cache" TRAIN_FLAGS="--max-steps 20 --eval-every 10" EVAL_FLAGS="--fake-ft"
#   (FT_ROOT other than data/prereg12 is required for any of these; DEVICE=cpu skips the GPU wait).
cd /d/AI-Portfolio/competitions/aicup2026-go || exit 1
S=results/prereg12/explore/finetune
D=${FT_ROOT:-data/prereg12}
LOG=$D/run_pilot.log
CPU=.venv/Scripts/python.exe
GPU=${GPU_PY:-.venv-gpu/Scripts/python.exe}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-16}
ROOT=()
if [ "$D" != "data/prereg12" ]; then ROOT=(--out-root "$D"); fi
if [ "$D" = "data/prereg12" ] && [ -n "$SETS_FLAGS$INNER_FLAGS$STEP_FLAGS$TRAIN_FLAGS$EVAL_FLAGS" -o "$DEVICE" != "cuda" ]; then
  echo "smoke variables need FT_ROOT other than data/prereg12" >&2; exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
mkdir -p "$D"
say() { echo "$* $(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$LOG"; }
stopped() { [ -e "$D/STOP" ]; }
say "launcher start (mode ${1:-all})"

# the pilot runs only from committed, unmodified code and plan (PLAN.md 7)
if ! git ls-files --error-unmatch "$S/PLAN.md" > /dev/null 2>&1 || ! git diff --quiet HEAD -- "$S"; then
  say "REFUSED: $S is not committed / has local changes"; exit 2
fi

cpu_step() {   # cpu_step <args...>: run until done; exit on error
  if stopped; then say "STOP present, launcher exits"; exit 0; fi
  "$CPU" "$S/ft_pilot.py" "$@" "${ROOT[@]}" >> "$LOG" 2>&1; rc=$?
  say "cpu $* rc=$rc"
  if [ $rc -ne 0 ]; then say "CPU step failed, launcher exits"; exit 2; fi
}

[ -e "$D/sets.json" ] || cpu_step sets $SETS_FLAGS
for s in train holdout; do cpu_step cache --set $s --workers "$WORKERS"; done
for s in bank mono; do cpu_step cache --set $s --workers "$WORKERS" $INNER_FLAGS; done
if stopped; then say "STOP present, launcher exits"; exit 0; fi
if [ "${1:-all}" = "cpu-only" ]; then say "cpu-only: done"; exit 0; fi

wait_gpu() {
  if [ "$DEVICE" = "cpu" ]; then return 0; fi
  while true; do
    if stopped; then say "STOP present, launcher exits"; exit 0; fi
    "$CPU" "$S/ft_pilot.py" gpu-free >> "$LOG" 2>&1; rc=$?
    py=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -ci "python")
    if [ $rc -eq 0 ] && [ "${py:-1}" = "0" ]; then return 0; fi
    say "GPU busy (type-C check rc=$rc, python compute apps=$py), waiting 5 min"
    sleep 300
  done
}

gpu_step() {   # gpu_step <done-test command> <args...>
  local test="$1"; shift
  local fails=0
  while true; do
    if eval "$test"; then return 0; fi
    wait_gpu
    "$GPU" "$S/ft_pilot.py" "$@" --device "$DEVICE" "${ROOT[@]}" $STEP_FLAGS >> "$LOG" 2>&1; rc=$?
    say "gpu $* rc=$rc"
    if stopped; then say "STOP present, launcher exits"; exit 0; fi
    if [ $rc -eq 3 ]; then continue; fi          # another job took the GPU first: wait again
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1))
      if [ $fails -ge 3 ]; then say "GPU step failed 3 times, launcher exits"; exit 2; fi
      sleep 300                                  # e.g. CUDA OOM caused by another job: resume from the checkpoint
    fi
  done
}

ft_done() {   # ft_done <set>: every cache chunk has its FT chunk
  local n_c n_f
  n_c=$(ls "$D/cache/$1"/chunk_*.npz 2>/dev/null | grep -vc tmp)
  n_f=$(ls "$D/ft/$1"/chunk_*.npz 2>/dev/null | grep -vc tmp)
  [ "$n_c" -gt 0 ] && [ "$n_c" = "$n_f" ]
}

gpu_step "[ -e $D/train/bench.json ]" bench
gpu_step "[ -e $D/train/train_done.json ]" train $TRAIN_FLAGS
gpu_step "ft_done bank" extract --set bank
gpu_step "ft_done mono" extract --set mono
say "GPU stages done"

[ -e "data/prereg12/q09_inner/prep_record.json" ] || { "$CPU" "$S/ft_eval.py" prep-inner >> "$LOG" 2>&1 || { say "prep-inner failed"; exit 2; }; }
if [ "$D" = "data/prereg12" ] && [ -e "$S/eval_report.json" ]; then say "eval_report.json exists (judged once), not rerun"; exit 0; fi
"$CPU" "$S/ft_eval.py" eval --out-root "$D" $EVAL_FLAGS >> "$LOG" 2>&1; rc=$?
say "eval rc=$rc (real run: results/prereg12/explore/finetune/eval_report.json; commit it)"
say "launcher end"
