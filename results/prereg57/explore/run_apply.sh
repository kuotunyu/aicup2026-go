#!/usr/bin/env bash
# prereg57 apply chain on the 4090 (PLAN.md Apply): base c49, the two official new pools separately - adapt to that
# pool's candidates only (chosen share / lr, to step n), then the EMA vectors of that pool's games.
#     bash results/prereg57/explore/run_apply.sh <share> <lr> <n>     # dev.json's chosen values
# Needs the committed dev.json with a passed gate (p57_ft.py checks it). Log: data/prereg57/apply_log.txt; stop:
# create data/prereg57/STOP.
# Afterwards (CPU, needs an adopting eval.json):
#     .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py apply --pool public
#     .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py apply --pool private
#     .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py assemble
set -u
if [ $# -ne 3 ]; then echo "usage: run_apply.sh <share> <lr> <n>   (dev.json's chosen values)"; exit 2; fi
share=$1
lr=$2
n=$3
LOG_NAME=apply_log.txt
. "$(cd "$(dirname "$0")" && pwd)/chain_common.sh"
cd $MAIN || exit 3
say "apply chain start: share $share lr $lr n $n"
for pool in public private; do
  out=data/prereg57/apply/$pool
  if [ -f $out/emb/step_${n}_ema.npz ]; then continue; fi
  adapt_run c49 official $pool $share $lr $n $n $out
  embed_run c49 official $pool $out $n
done
say "apply chain done; next: p57_eval.py apply --pool public / private, then assemble"
