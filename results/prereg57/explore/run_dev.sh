#!/usr/bin/env bash
# prereg57 dev chain on the 4090 (PLAN.md Development): base S0, D pools 0 and 1, the 3 (share, lr) runs; every run is
# adapted to 6,000 steps with checkpoints at 1,000 / 2,000 / 4,000 / 6,000, then the EMA vectors of the pool's games
# at each of the 4 steps (6 adaptations + 24 short embeddings; about 2.3 h alone on the 4090).
#     bash results/prereg57/explore/run_dev.sh         # log: data/prereg57/dev_log.txt; stop: create data/prereg57/STOP
# Afterwards (CPU):  .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py dev   -> dev.json (commit it).
# See chain_common.sh for the conventions (RAM wait, STOP, skip finished steps, stop on error).
set -u
LOG_NAME=dev_log.txt
. "$(cd "$(dirname "$0")" && pwd)/chain_common.sh"
cd $MAIN || exit 3
say "dev chain start"
for cfg in "0.25 2e-4" "0.5 2e-4" "0.5 5e-4"; do
  set -- $cfg
  share=$1
  lr=$2
  for p in 0 1; do
    out=data/prereg57/dev/sh${share}_lr${lr}/D$p
    need=0
    for n in 1000 2000 4000 6000; do
      if [ ! -f $out/emb/step_${n}_ema.npz ]; then need=1; fi
    done
    if [ $need -eq 0 ]; then continue; fi
    adapt_run S0 D $p $share $lr 6000 1000,2000,4000,6000 $out
    for n in 1000 2000 4000 6000; do
      embed_run S0 D $p $out $n
    done
  done
done
say "dev chain done; next: .venv/Scripts/python.exe results/prereg57/explore/p57_eval.py dev"
