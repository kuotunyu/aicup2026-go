#!/usr/bin/env bash
# prereg57 judgement chain on the 4090 (PLAN.md Judgement): base c49, E pools 0..9, each with its own adapted copy
# (the chosen share / lr, adapted only to step n), then the EMA vectors of that pool's games at step n.
#     bash results/prereg57/explore/run_judge.sh <share> <lr> <n>     # exactly dev.json's "next" line
# Needs the committed dev.json with a passed gate: p57_ft.py refuses E otherwise, or when share / lr / n differ from
# its choice. Log: data/prereg57/judge_log.txt; stop: create data/prereg57/STOP.
# Afterwards (CPU):  .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py judge --e-use N   -> eval.json.
set -u
if [ $# -ne 3 ]; then echo "usage: run_judge.sh <share> <lr> <n>   (dev.json's chosen values)"; exit 2; fi
share=$1
lr=$2
n=$3
LOG_NAME=judge_log.txt
. "$(cd "$(dirname "$0")" && pwd)/chain_common.sh"
cd $MAIN || exit 3
say "judge chain start: share $share lr $lr n $n"
for p in 0 1 2 3 4 5 6 7 8 9; do
  out=data/prereg57/judge/E$p
  if [ -f $out/emb/step_${n}_ema.npz ]; then continue; fi
  adapt_run c49 E $p $share $lr $n $n $out
  embed_run c49 E $p $out $n
done
say "judge chain done; next: .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py judge --e-use N"
