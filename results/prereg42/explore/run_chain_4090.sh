#!/usr/bin/env bash
# prereg42: the 4090 queue of PLAN.md 3 as one independent process: train the given variants one after another, then
# the D vectors of each finished one, then TTA-8 of the base checkpoint on D.
#   bash results/prereg42/explore/run_chain_4090.sh V2 V4 [V3]
# Stops at the first non-zero exit (STOP file -> 4; over budget -> 5). Re-running continues: a variant with
# checkpoints is resumed (--resume), finished vectors are skipped. Progress: data/prereg42/chain_log.txt.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO" || exit 2
LOG=data/prereg42/chain_log.txt
mkdir -p data/prereg42
say() { echo "=== $(date '+%F %T') $*" >> "$LOG"; }
say "chain start: $* (git $(git rev-parse --short HEAD))"
for v in "$@"; do
  if [ -f "results/prereg42/explore/train_record_$v.json" ] && grep -q '"finished": true' "results/prereg42/explore/train_record_$v.json"; then
    say "$v already finished"
  else
    extra=""
    ls "data/prereg42/$v/ckpt/"step_*.pt >/dev/null 2>&1 && extra="--resume"
    say "train $v $extra"
    bash "$HERE/run_train.sh" --variant "$v" --workers 4 $extra; rc=$?
    say "train $v rc=$rc"
    [ $rc -eq 0 ] || exit $rc
  fi
  say "embed D $v"
  bash "$HERE/run_embed_D.sh" "$v"; rc=$?
  say "embed D $v rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
say "embed D base TTA-8"
bash "$HERE/run_embed_D.sh" base step_40000 --tta 8; rc=$?
say "chain end rc=$rc"
exit $rc
