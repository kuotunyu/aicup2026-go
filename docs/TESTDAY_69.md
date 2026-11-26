# TESTDAY_69 (competition 2: 64 + M8)

1. `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe tools/t69_rank.py apply` (CPU, about a minute; git clean; needs
   the gate passed in `results/prereg69/explore/merge_sim.json`). Seams: A (Q25 = test25), O / M (t64), T (merge off =
   `results/test64/rank` byte for byte).
2. Output `results/test69/rank/submission_rank.csv`; changed questions with old / new class and M8 group in
   `decide_summary.json` -> `prereg69_record`.
3. `tools/check_submission.py rank <file> --questions data/test_v2_download/rank/rank_prediction_test_public.csv
   data/test_v2_download/rank/rank_prediction_test_private.csv --official`.
4. Upload (competition-2 #13) is the main session's decision; record by hand (official_log.py does not know 69).

2026-11-22 run: sha256 c7595f2eb43c...; changed vs 64: public 0, private 3 (priv_q_0043 C->D, priv_q_0133 4D->3D,
priv_q_0392 C->B); check_submission --official OK.
