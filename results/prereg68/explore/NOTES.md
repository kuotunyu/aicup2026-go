# prereg68 notes (narrow readings of PLAN.md; written with the code, before any number of X2 / X3)

2026-11-22. Code `p68_eval.py` imports prereg67's `p67_eval.py` (v2-9) unchanged; every recipe number not
listed here is prereg67's (`results/prereg67/explore/frozen.json`, NOTES.md 1-4, 8, 12).

1. **Members and weights.** Order c49, S, B, p61, S1, S2, X, X2, X3; weights 1, 1, 1, 2, 2, 2, 2, 2, 2; w_E 14,
   cfg_ens = prereg67's (lam 0.85, centre_colour True). `E_ens = sum_m (w_m / W) Z_m`, W = sum of the weights of the
   members in the version, members with weight 0 skipped and never loaded (prereg67 `ensemble` arithmetic).
   The v2-9 members' files must have the sha256 recorded in `results/prereg67/explore/eval.json` member_files;
   prereg67 eval.json must still be the adopted 7-member version with E 0.7379199600122529 (else stop).
2. **New members' files.** `data/prereg62/X2|X3/emb/p60_step_400000_raw_tta1/{store_E,pool_public,pool_private}.npz`
   (made on the 4090 by `data/prereg68/embed_X23.sh`; copy `embed_X23.sh` here).
3. **check-member** = prereg67 NOTES.md 7 verbatim (prereg67 `check_files` with checkpoint step 400,000, then the
   member alone in the v2-8 back end on the std10 pseudo pools, pass iff std10 mean >= 0.6406 and no game without a
   sample in any of the three files). Once per name; written to `members.json` here. No E number of a member.
4. **Version (mechanical).** v2-9's 7 members + each of X2 / X3 that has a passing members.json entry, unchanged
   files (sha256 as recorded) and file mtimes <= 2026-11-24 12:00:00 (Taiwan time; PLAN "latest 11/24 12:00").
   Both = "9"; one = "8-X2" / "8-X3" (PLAN: "If only one of X2 / X3 finishes by 11/24 12:00, the 8-member version").
   Judge refuses: (a) a member's files exist before the cutoff but check-member was not run; (b) a member is neither
   checked nor present and the cutoff has not passed (PLAN "after both members exist"): wait; a member that FAILED
   check-member is dropped at once, no waiting; (c) neither new member is in: nothing to judge, v2-9 stays.
5. **Seams** (`seam.json`, committed before any check-member / judge; X2 = X3 = weight 0, never loaded):
   (a) E: the 7 v2-9 members through prereg68's ensemble function -> per pool equal to prereg67 eval.json
   per_pool.ens bit for bit, mean 0.7379199600122529, no question without a vector.
   (b) pseudo (std10): prereg67 has no recorded 7-member pseudo-pool score; the only recorded ensemble pseudo score is
   its seam's float64 5-member recipe (`seam.json` pseudo.recipe_5m_tilt2_085cc_wE14.float64_per_pool, mean
   0.658554576215823). Gating: the 5 fixed members (c49, S, B, S1, S2; p61, X, X2, X3 weight 0) through prereg68's
   pseudo function equal it per pool bit for bit. Recorded, not gating: the 7-member v2-9 std10 mean (reference for
   reading X2 / X3; nothing is chosen from it).
6. **Judgement** (E 20 pools, once): new version vs v2-9 rebuilt in the same run (must equal prereg67 eval.json bit
   for bit, else stop). Adopt iff mean >= +0.005, pool bootstrap 10,000 seed 43 (`Q42.bootstrap`) lower bound > 0,
   >= 16 of 20 pools higher (strictly). E use = largest committed `e_pools_use_number` + 1 (prereg68's own eval.json
   excluded); expected 12. Needs committed PLAN, NOTES, p68_eval.py, seam.json (ok), members.json; refuses if
   eval.json exists. No single-model E score and no other E number of an 8- / 9-member ensemble is computed.
7. **Apply** (pub2 / priv2): A66's non-CNN terms as prereg67 apply; v2-9 rebuilt with the same per-member Z must
   equal `results/prereg67/apply/<pool>/top5.csv` byte for byte after CRLF -> LF; then the judged version. eval.json
   must be committed and adopt. **Assemble**: prereg35 checks (ids, prefixes, 5 distinct candidates of the pool),
   1,600 rows, header and the original 800 rows identical to `results/prereg67/apply/submission_player.csv`.
8. Resources: as prereg67 (BELOW_NORMAL, 4 logical processors, 1 BLAS thread, waits for >= 7,000 MB available,
   RSS watchdog 3 GB). Member vectors loaded one at a time.

## Flags

F1. prereg66 F1 / prereg67 F1-F3 hold unchanged.
F2. X2 / X3 train on the same player list as X (prereg62 recipe, E excluded) - from the PLANs, not re-checked.

## Audit 2026-11-22 (`data/advice/codex_1122b.md`): fixes made before any X2 / X3 file exists and before the judgement

Written before any X2 / X3 vector, any check-member and any E number of an 8- / 9-member ensemble. PLAN.md,
weights, formula, gates and the v2-9 seam are unchanged (the audit confirmed them); points 2-4, 6 and 7 above are
narrowed as follows (where they differ, 9-12 rule).

9. **Member identity and provenance (audit 2).** Directories are fixed (`check-member --dir` removed).
   (a) Arm X2 / X3 = `data/prereg62/<arm>/train_record_<arm>.json`: finished, 400,000 steps, not smoke, variant /
   spec / every prereg60 config key equal to prereg62 X's committed record (`results/prereg62/explore/
   train_record_X.json`, incl. T player-list hash, cache hash, base checkpoint) except the seed, which must be 1 / 2;
   the physical `ckpt/step_400000.pt` must hash to the record's `checkpoints["step_400000.pt"]`.
   (b) All three npz carry that checkpoint sha256; it differs from every v2-9 member's (their store_E), from X's
   record and from the other new arm's record. store_E: store_row (prereg67 check) and player-set hash equal c49's.
   Every file: finite values, count 1-D of the vectors' length, every count > 0.
   (c) Official pools: `p68_embed.py` (called by `embed_X23.sh` instead of `p68_colab_run.py embed`) wraps
   `p36_lib.pool_items` for its process - the function `p36_embed.pool_games` calls to build the list the frozen
   embedder consumes - and builds a per-game manifest (pool, cand / q, candidate or question id, game index within
   it, colour, sha256 of the raw SGF, target moves) from the same (cands, qs) arguments, asserting its (sgf,
   colour) sequence equals the returned item list. The manifest sha and checkpoint sha are bound in
   `<out>.pending.json` before any part is computed (a parts folder without that binding, or with another one, is
   refused). Cold start only (re-audit `codex_1122d.md` item 2): the frozen embedder keeps parts with emb / count
   only and reuses any part file present, so an existing `<out>.parts` folder or pending record (store_E and pools)
   is refused before the embedder runs - delete both and embed from scratch; the sidecar records cold_start true.
   Acceptance (re-audit `codex_1122e.md`): check-member and apply accept an X2 / X3 vector file (store_E,
   pool_public, pool_private) only with a provenance record of format v2 written by p68_embed.py with cold_start
   exactly true, the checkpoint sha (and for pools the manifest sha) and the npz sha matching; a missing record,
   cold_start false or absent, or an npz changed after its record -> invalid with that reason (delete and embed
   again from scratch; a record is never completed afterwards). The embed entry refuses such existing files too.
   After the frozen embedder replaced the npz: checkpoint hashed again and equal to the npz's, per-game
   counts equal the manifest's target moves (prereg60 early=True samples every non-pass target move; checked on X's
   existing pool files: 7,100 / 7,100 equal in both pools), then `<name>.provenance.json` is written atomically with
   the manifest, its sha, the npz sha256 and the CSV sha256. store_E gets a sidecar too (npz and checkpoint sha).
   check-member, judge (via members.json) and apply rebuild the manifest from the official CSVs with the embedder's
   own readers and require equality; apply also requires its own item list (prereg43 official_rec) to match it.
   What this proves: the vectors were computed in that process, by that checkpoint, over games in exactly that order
   with exactly those SGF texts (a swapped pool, re-ordered / swapped same-colour same-length games, a re-made npz,
   another checkpoint, a file from outside the wrapper are rejected). What it cannot prove: that the frozen embedder
   wrote row i from item i (that is its code, unchanged since the v2-9 members, whose apply seam reproduced
   prereg67's top5 byte for byte); the sidecar is written next to the npz, not inside it (the embedder is frozen),
   so it binds by sha256, and a person who regenerates both on purpose is not stopped. The hook was exercised with a
   stand-in embedder in test_p68.py and an import-only check in .venv-gpu (p36_embed.L is the hooked module).
10. **One judgement (audit 1).** judge has no `--out`; eval.json and `judge_started.json` are fixed. judge stops if
   either exists in the working tree or was ever committed (git log). After all preflight checks and the v2-9 seam
   (rebuilt E 20 pools bit for bit) and before the one `e_scores` call of the new version, `judge_started.json` is
   created with O_CREAT | O_EXCL (version, members, member file sha, members.json sha, git HEAD, E use, pid) and
   committed at once (only that path); a second process loses at the exclusive create. If that commit fails (or the
   file is not tracked and identical to HEAD afterwards) judge stops before the new version is scored; the start
   record stays and blocks reruns (re-audit item 1). A failing git query (log / ls-tree / show) is an error, never
   read as "not committed". A run that fails after the
   start record leaves it (and `judge_failed.json`) and is never judged again automatically: report it. eval.json is
   written tmp + fsync + replace. E use = largest `e_pools_use_number` of `results/prereg*/explore/eval*.json` read
   from git HEAD (git show), prereg68's own excluded; an uncommitted file or edit does not move it (expected 12).
11. **Version states (audit 3).** Cutoff = 2026-11-24 12:00:00 Asia/Taipei (UTC+8, explicit; was the machine's
   clock) and is the member files' deadline (npz + sidecar mtimes). Per new member: no entry and nothing on disk =
   wait before the cutoff, out-timeout after; files made after the cutoff = out-timeout (check-member refuses them,
   writes nothing); present before the cutoff and not checked = problem (stop). check-member: invalid files (9)
   -> status "invalid", no std10, may be checked again after a repair (attempts kept); while invalid: wait before
   the cutoff; repaired but not re-checked = problem; after the cutoff a member still recorded invalid is a named
   problem (judge refuses; the earlier reasons are quoted) whether its files are unchanged, deleted or re-made late
   (re-audit item 3) - only a member that never had files before the cutoff (out-timeout) or a valid member below
   0.6406 (out-below-gate) leads to the 8-member version. Valid files -> std10 once -> status
   "scored" (final): below 0.6406 = out-below-gate (the other member may be judged at once); passed = in, but if
   any of its six files (3 npz + 3 sidecars) is missing or its sha256 differs from members.json -> problem: restore
   them; never an automatic 8-member downgrade. **Registered reading (written 2026-11-22, before any E number of an
   8- / 9-member ensemble):** PLAN "latest 11/24 12:00" is the member files' deadline only; the judgement itself
   has no latest start time beyond it (it runs as soon as the version is determined, before or after 12:00).
12. **Apply / assemble (audit 4).** apply has no `--eval`: only this round's eval.json and judge_started.json, both
   committed, eval format prereg68, adopt true, members / version / E use equal to the start record, weights and w_E
   prereg68's, members.json and the start record unchanged since. X2 / X3 pool files: provenance (9 c) against
   `--src`. top5.csv written tmp + replace, an older apply_report renamed `.stale` first, then the report (format
   v2: pool, version, members, eval sha, E use, top5 sha, CSV sha) atomically. assemble renames an existing final
   CSV / report `.stale`, requires both pools' reports to match the adopted eval and the top5.csv now on disk,
   assembles in `_assemble_stage/`, runs the prereg35 checks and the v2-9 header / original-800 / 1,600 checks
   there, and only then writes the report and replaces `submission_player.csv` (atomic). Any failure leaves no
   final CSV.
