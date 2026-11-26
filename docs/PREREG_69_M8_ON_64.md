# PREREG_69: prereg65's safe same-player merge M8 on top of 64 (competition 2)

- Plan `results/prereg69/explore/PLAN.md` (a65b7f8, before any code / number); readings `NOTES.md` + `frozen.json` (b31bcb5).
- Gate (no inner judgement possible): prereg65's M8 simulation (E n H, 46 players, 100 pools per mode, seeds 9000 + sd)
  with 64's fusion form as the base (NOTES 2: w_A log_softmax(lp_S25) + w_O OPP3 with S0 in place of c49, 64's frozen
  weights; c49 opponent vectors of E do not exist, no GPU used).
- Seam S: prereg65's own base on the same loop reproduces every number of `results/prereg65/explore/merge_sim.json` exactly.
- Result (`results/prereg69/explore/merge_sim.json`): 0% repeats 0 changes in 100/100 pools; 35% mean gain +0.000389
  (35 pools changed); 100% +0.00535 (record) -> **gate passed**.
- Test day: `tools/t69_rank.py apply` (steps `docs/TESTDAY_69.md`). Data: no inner, no data/local, no tools/score.py.
