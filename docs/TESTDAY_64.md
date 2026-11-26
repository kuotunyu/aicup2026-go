# TESTDAY_64: competition 2, Q64 (PREREG_64)

Git Bash, repository root, CPU only, one python process, available RAM >= 7000 MB.

1. `git status` clean (tracked files); results/test25 and results/test59 committed.
2. Rehearsal (optional): `PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/t64_rank.py apply --out <dir outside the repository>`
3. Real: `PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/t64_rank.py apply`
   -> results/test64/rank/{rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv,
   t64_apply_report.json}. Exit 0 only when seams A, O, M, B and check_submission --official pass; exit 4 = stop.
4. Read decide_summary.json "prereg64_record": changes vs 59 and vs 25 per pool, merge groups (record only, the
   merge is not applied).
5. Commit results/test64. Upload (competition 2 #12) and the official_log.py record are the main session's decision;
   the tool does neither.
