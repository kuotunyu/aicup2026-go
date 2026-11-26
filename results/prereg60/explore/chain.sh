#!/usr/bin/env bash
# prereg60 unattended chain on the 4090: seam (a) in fp16 (if seam.json is not ok yet) -> train (30,000 steps) ->
# commit the training record -> vectors of E and of the two official pools -> the one E judgement (CPU) -> commit
# eval.json.
# (data/prereg60/chain.sh and results/prereg60/explore/chain.sh are the same file; start the one under data/.)
#
#   start (hidden, detached):  see the launch line in results/prereg60/explore/NOTES.md
#   allow it to take the GPU:  touch data/prereg60/GO
#   stop:                      touch data/prereg60/STOP   (training saves within 100 steps; remove STOP to go on)
#   log:                       data/prereg60/chain_log.txt (+ train_log.txt, embed_log.txt, judge_log.txt)
#
# Gate before every GPU step: data/prereg60/GO exists and nvidia-smi lists no python compute process for 2 minutes
# in a row. Driver must be 616.92. Before every step: no STOP file, available RAM >= 7000 MB.
# Training exit code 4 = saved and stopped: with a STOP file the chain exits 4, otherwise (low RAM) it waits for
# RAM >= 7000 MB and resumes from the highest step number. Any other error stops the chain.
# Starting it again resumes: finished steps are skipped (training record, vector files, eval.json).
# The judgement is made once: p60_eval.py refuses if eval.json exists; this chain only commits an existing one.
set -u
MAIN=/d/AI-Portfolio/competitions/aicup2026-go
D=$MAIN/data/prereg60
X=results/prereg60/explore
LOG=$D/chain_log.txt
STOP=$D/STOP
GO=$D/GO
GPUPY=$MAIN/.venv-gpu/Scripts/python.exe
PY=$MAIN/.venv/Scripts/python.exe
WORKERS=${P60_WORKERS:-4}
STEPS=030000
CK=data/prereg60/S0/ckpt/step_$STEPS.pt
VEC=data/prereg60/emb/p60_step_${STEPS}_raw_tta1
SRC=data/test_v2_download/x/tests
REC=$X/train_record.json
EVAL=$X/eval.json
GIT_ID=(-c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com)
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS PYTHONUTF8
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p $D
say() { echo "=== $(date '+%F %T') $*" >> $LOG; }
avail() { powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue" | tr -d '\r '; }
check_stop() { if [ -f "$STOP" ]; then say "STOP file $STOP: exiting"; exit 4; fi; }
wait_ram() {
  local told=0
  while :; do
    check_stop
    a=$(avail)
    if [ -n "$a" ] && [ "$a" -ge 7000 ]; then return 0; fi
    if [ $told -eq 0 ]; then say "waiting for RAM (available ${a:-?} MB < 7000)"; told=1; fi
    sleep 30
  done
}
# number of python compute processes on the GPU; "x" when nvidia-smi fails
gpu_pythons() {
  local out
  out=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null) || { echo x; return; }
  echo "$out" | tr -d '\r' | awk -F, '{n=$2; gsub(/\\/,"/",n); sub(/.*\//,"",n); if (tolower(n) ~ /^python/) c++} END {print c+0}'
}
# GO file present and no python compute process on the GPU for 2 minutes in a row (8 checks, 15 s apart, + 1)
gate() {
  local ok=0 told=0
  while :; do
    check_stop
    n=$(gpu_pythons)
    if [ -f "$GO" ] && [ "$n" = "0" ]; then
      ok=$((ok + 1)); [ $ok -ge 9 ] && return 0
    else
      if [ $told -eq 0 ] || [ $ok -gt 0 ]; then say "gate waiting: GO $([ -f "$GO" ] && echo yes || echo no), python on GPU: $n"; told=1; fi
      ok=0
    fi
    sleep 15
  done
}
driver() {
  local drv
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' \r')
  if [ "$drv" != "616.92" ]; then say "ERROR driver $drv != 616.92"; exit 2; fi
}
finished() { grep -q '"finished": true' $REC 2>/dev/null && [ -f $CK ]; }
# commit_file <path> <message>: add + commit that one path (retry while the index is locked); no-op when unchanged
commit_file() {
  local f=$1 msg=$2 i
  if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 && [ -z "$(git status --porcelain -- "$f")" ]; then return 0; fi
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if git add -- "$f" >> $LOG 2>&1 && git "${GIT_ID[@]}" commit -m "$msg" -- "$f" >> $LOG 2>&1; then
      say "committed $f ($(git rev-parse --short HEAD))"; return 0
    fi
    say "commit of $f failed (try $i); retrying in 15 s"
    sleep 15
  done
  say "ERROR could not commit $f"; exit 7
}

cd $MAIN || exit 3
say "chain armed (git $(git rev-parse --short HEAD), workers $WORKERS)"
check_stop

# ---------------------------------------------------------------- 0. seam (a) on the 4090 (PLAN: before training)
# The committed CPU fp32 run of seam (a) is 0.0696 against the stored fp16 vectors (> 0.05; NOTES.md 11), so the seam
# is repeated here in the stored vectors' own precision: 200 E games through the widened c49 on the GPU (seconds),
# then p60_eval.py seam --device gpu (CPU, ~2.5 min) and a commit of seam.json. Not ok -> the chain stops here.
SEAMJ=$X/seam.json
seam_ok() { grep -q '^ "ok": true' $SEAMJ 2>/dev/null; }
if ! finished && ! seam_ok; then
  wait_ram
  gate
  driver
  rm -rf $D/seam_gpu
  say "seam (a) on the GPU: embed 200 E games with the widened c49"
  "$GPUPY" -u $X/p60_run.py embed --ckpt data/prereg49/S0/ckpt/step_320000.pt --source store --players E \
    --limit 200 --no-early --workers $WORKERS \
    --compare data/prereg49/emb/V2_step_320000_raw_tta1/store_E.npz >> $D/seam_log.txt 2>&1
  rc=$?
  say "seam embed end rc=$rc"
  if [ $rc -ne 0 ]; then say "ERROR seam embed rc=$rc (see seam_log.txt)"; exit 9; fi
  wait_ram
  "$PY" -u $X/p60_eval.py seam --device gpu >> $D/seam_log.txt 2>&1
  rc=$?
  say "seam end rc=$rc: $(grep -o '"max_abs_diff": [0-9.e-]*' $SEAMJ | head -1)"
  commit_file $SEAMJ "prereg60 seam on the 4090 (fp16): widened c49 vs c49's stored vectors, v2-7 E bit for bit (c49 only)"
  if [ $rc -ne 0 ] || ! seam_ok; then say "ERROR the seam is not ok: not training (see seam_log.txt, seam.json)"; exit 9; fi
fi

# ---------------------------------------------------------------- 1. train
while ! finished; do
  wait_ram
  gate
  driver
  say "train start/resume (avail $(avail) MB, git $(git rev-parse --short HEAD))"
  "$GPUPY" -u $X/p60_run.py train --workers $WORKERS --auto-resume >> $D/train_log.txt 2>&1
  rc=$?
  say "train ended rc=$rc; last checkpoint $(ls $D/S0/ckpt 2>/dev/null | sort | tail -1)"
  if [ $rc -eq 4 ]; then
    check_stop
    if [ -f "$D/S0/STOP" ]; then say "STOP file $D/S0/STOP: exiting"; exit 4; fi
    say "paused without a STOP file (low RAM): waiting to resume"
    sleep 60
    continue
  fi
  if [ $rc -ne 0 ]; then say "ERROR train rc=$rc (see train_log.txt)"; exit 5; fi
  if ! finished; then say "ERROR train ended rc=0 without a finished record / $CK"; exit 5; fi
done
commit_file $REC "prereg60 train record: 30,000 steps from c49 (18 planes, all moves) - no score"

# ---------------------------------------------------------------- 2. vectors: E, public, private
embed() {   # embed <output file> <p60_run embed arguments...>
  local out=$1; shift
  if [ -f "$out" ]; then return 0; fi
  wait_ram
  gate
  driver
  say "embed start: $* (avail $(avail) MB)"
  "$GPUPY" -u $X/p60_run.py embed --ckpt $CK --workers $WORKERS "$@" >> $D/embed_log.txt 2>&1
  rc=$?
  say "embed end rc=$rc: $out"
  if [ $rc -ne 0 ]; then say "ERROR embed rc=$rc (see embed_log.txt)"; exit 6; fi
  if [ ! -f "$out" ]; then say "ERROR embed ended rc=0 without $out"; exit 6; fi
}
embed $VEC/store_E.npz --source store --players E
embed $VEC/pool_public.npz --source pool --src $SRC --pool public
embed $VEC/pool_private.npz --source pool --src $SRC --pool private
say "GPU work done: the 4090 is free"

# ---------------------------------------------------------------- 3. the judgement (CPU, once)
if [ ! -f $EVAL ]; then
  wait_ram
  N=$("$PY" $X/p60_eval.py e-use 2>> $LOG | tr -d '\r' | tail -1)
  case "$N" in ''|*[!0-9]*) say "ERROR e-use gave '$N'"; exit 8;; esac
  say "judge start: --e-use $N (avail $(avail) MB)"
  "$PY" -u $X/p60_eval.py judge --e-use $N >> $D/judge_log.txt 2>&1
  rc=$?
  say "judge end rc=$rc"
  if [ $rc -ne 0 ]; then say "ERROR judge rc=$rc (see judge_log.txt)"; exit 8; fi
  if [ ! -f $EVAL ]; then say "ERROR judge ended rc=0 without $EVAL"; exit 8; fi
fi
commit_file $EVAL "prereg60 judgement (E, once): eval.json"
say "all done: $(grep -o '"adopt": [a-z]*' $EVAL | tail -1); next (only if adopted): p60_eval.py apply --pool public|private, assemble"
