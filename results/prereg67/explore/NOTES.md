# prereg67 notes (narrow readings of PLAN.md; written before any code number of this registration)

2026-11-21 20:25. Every recipe number is in `frozen.json` with a pointer into Gemini Hydrogen's files
(`data/advice/gemini_1121h/`, not in git; sha256 in `frozen.json`). As in prereg63 / prereg66: port what Hydrogen's
CODE does (`03_beyond_ensemble_k12.py`, experiment `5m_tilt2_085cc_wE14`, lines 205-313 and 349), the REPORT
section 4 gives the members / weights / versions; the seams are the check. The back end (everything but the CNN term)
is prereg66's A66, imported unchanged from `results/prereg66/explore/p66_eval.py`.

1. **Member term.** For member m: `E_m, ok_m = BK.e_term(pool, emb_m, cfg_ens)`, `Z_m = st.zscore_rows(E_m, ok_m)`.
   `cfg_ens` = v2-6's cfg (prereg66 `E45.v26()["cfg"]`) with `centre_colour = True` and `whiten = {lam 0.85, lda None}`
   (gnorm True, pool_w mean, sim proto unchanged) = Hydrogen 03 line 207 (`cfg_085_cc`); e_term reads only these keys.
2. **Ensemble.** `E_ens = sum_m (w_m / W) Z_m`, W = sum of the weights of the members in the version, members added
   in the fixed order c49, S, B, p61, S1, S2, X, members with weight 0 skipped (so the c49-only seam is
   `E_ens = 1.0 * Z_c49` exactly). Float64 throughout. Hydrogen stored Z_m, G and the z terms as float32 and used
   `np.tensordot` (03 lines 210-224, 280); that is a development-cache artefact and is not copied (recorded check 9c
   shows its size). `E_ens` is not row-z-scored again (REPORT 4.2.1, re_z = 0).
3. **ok.** `ok_ens = AND_m ok_m` (REPORT 4.2.1). It has no consumer in the formula (no re-z), and Hydrogen asserted
   all-True on its pools (01 line 350). Narrow reading: judge / apply stop if any question of an E pool or official
   pool has `ok_ens` False (count recorded); nothing is filled.
4. **Score.** `S = csls_20( G + 14.0 E_ens + (5.5 z_p1_orig + 3 z_multi + 3 z_sp128) + 1.8 z_rk_tri ) + w_post(k) z_rk_tri`
   with `G = st.fuse([T_n], w_G)` (v2-6's G weights, `EV.g_combo()`), i.e. prereg66 `combine()` with
   `F = G + E_ens * w_E` in place of `fuse_cfg(..., E)` (same additions, same order: `fuse_cfg` is
   `st.fuse(T + [E], w + [wE])`, which accumulates left to right). z_p1_orig, z_multi, z_sp128, z_rk_tri are A66's:
   on E read from `data/prereg66/z_E.npz` (+ prereg49 `e_context` for z_p1_orig) checked against prereg66
   prep_e.json's sha256; on the official pools recomputed exactly as prereg66 apply; the rank term always uses the
   c49 vectors and `c49_head.npz` (not the ensemble) as in v2-8 and in Hydrogen (01 lines 173, 262-264, 297-301).
5. **Weights.** c49 1, S 1, B 1, p61 2, S1 2, S2 2, X 2.0 (PLAN: "clearly" above S1 is undefined -> 2.0; the 2.5 is
   never used). w_E 14.0.
6. **Version (mechanical, PLAN).** Fixed members c49, S, B, S1, S2 (file sha256 frozen in `frozen.json`). p61 and X
   enter iff `members.json` has a passing `check-member` entry for them, their three files still have the recorded
   sha256, and the three files' modification times are <= 2026-11-22 02:00:00 (Taiwan time; PLAN hard time rule).
   A failing / missing member is dropped alone: {p61, X} both in = 7 members; p61 only = A (6); neither = B (5);
   X only (p61 failed or missing) = c49, S, B, S1, S2, X (6) - "drop the failing / missing member(s)" read literally.
   Judge refuses to run when a candidate member's three files exist with mtime <= the cutoff but `members.json` has
   no entry for it (run check-member first), and when `members.json` exists but is not committed.
7. **check-member (PLAN single-model check).** Member m alone in the v2-8 back end = A66 with m's vectors for the CNN
   term (cfg78: lam 0.78, centre_colour False, w_E 8), on Hydrogen's std10 pseudo pools (point 8). Pass iff
   (a) std10 mean >= 0.6406 (PLAN literal; note c49 itself is 0.639759 on std10 and 0.6406 on all20 - the threshold
   is read as written, on std10), (b) no game without a sample (count == 0) in any of the member's three files
   (store_E, pool_public, pool_private: every game the ensemble would use; stricter than the pseudo-pool rows only),
   (c) file checks: store_row equal to c49's store_E, 7,100 pool rows, embed tta 1 / raw weights / games equal to
   c49's, checkpoint step 200,000 (p61) / 400,000 (X). No E number of the member is computed.
8. **std10 pseudo pools** = Hydrogen 01 lines 195-311 for slots 0-9: candidates2 files only (`p52_pseudo.read_candidates`),
   public then private (`PS.POOLS` order), seeds 52000-52004 then 52100-52104, `PS.split`, G terms on each pool
   kind's own PoolStore, P1 / z_multi / sp128 / rank as Hydrogen (single-game rank log-posteriors over all 12,000
   rows, 8,192 / 4,096-row chunks, cnt argument all ones as line 298). Rows of each member's pool files: first 6,000.
9. **Seams** (`seam.json`, committed before any check-member / judge):
   (a) pseudo: c49 alone, cfg78, w_E 8 -> per-pool equal to `01_prep_summary.json` models.c49.v28.per_pool[0:10]
   bit for bit, mean 0.6397593752945936 (printed 0.639759);
   (b) E: c49 alone, cfg78, w_E 8 through the ensemble function -> per pool equal to prereg66 eval.json
   per_pool.A66 bit for bit, mean 0.7183609308316172;
   (c) recorded, gating: Hydrogen's `5m_tilt2_085cc_wE14` std10 mean 0.658555 (03 results json, 6 d.p.) rebuilt with
   its float32 caches emulated, |diff| <= 1e-6 (checks the port of the recipe, pseudo pools only, no E); the float64
   value is recorded beside it. Also recorded (not gating): S1 and S2 alone through the check-member function
   (Hydrogen 01: std10 0.6438 / 0.6444 with float32 caches).
   Any gating failure: stop, do not judge.
10. **Judgement** (E 20 pools, once): ens vs v2-8 (A66 rebuilt in the judge bit for bit against prereg66 eval.json,
   else stop): mean >= +0.01, pool bootstrap 10,000 seed 43 (`Q42.bootstrap`) lower bound > 0, >= 18/20 pools
   higher. Recorded: by k, version, members and weights, ok counts. No single-model E score is computed.
   E use number = (largest `e_pools_use_number` in committed `results/prereg*/explore/eval*.json`) + 1; expected 11.
11. **Apply** (pub2 / priv2): first A66 rebuilt through the ensemble function with c49 alone must equal
   `results/prereg66/apply/<pool>/top5.csv` byte for byte (LF), then the version from eval.json. **Assemble**: 1,600
   rows (prereg35 `cmd_assemble` checks: ids, prefixes, 5 distinct candidates of the right pool), original 800 rows
   identical to v2-8's `results/prereg66/apply/submission_player.csv`.
12. Resources: one process, BELOW_NORMAL on 4 logical processors with 1 BLAS thread (`BK.polite`), waits for
   >= 7,000 MB available, RSS watchdog 3 GB (`P63.rss_watchdog`). Member vectors are loaded one at a time.

## Flags

F1. prereg66 F1 holds unchanged (the rank term saw E players' labels; it is the same in the base and the ensemble,
    so it cancels in the comparison).
F2. All members' training sets exclude E (c49 / S / B: prereg49 / 55 / 51; S1 / S2 / p61 / X continue from c49 /
    p60 code with the same player list) - taken from their PLANs, not re-checked here.
F3. Hydrogen's development used only candidate pseudo pools (its FORBIDDEN list, 01 lines 25-28); its choice of
    w_E 14 / lam 0.85 / cc / tilt weights was made on all20 pseudo pools, which include std10; check-member's
    threshold is on std10 too.
