# prereg66 notes (narrow readings of PLAN.md; written before any code or number of this registration)

2026-11-21 19:00. Every recipe number is in `frozen.json` with a pointer to the line of Gemini Carbon's files
(`data/advice/gemini_1121b/`, not in git; sha256 in `frozen.json`). Lesson of prereg63 kept: port what Gemini's CODE
does (`04_final_registerable_recipe.py`, function `evaluate_suite`, system `04_R_final_PRIMARY`, lines 341-355),
not the REPORT prose; the seams are the check.

1. **A66 (per pool)** = `csls_20( F_078 + (5.5 z_p1_orig + 3.0 z_multi + 3.0 z_sp128) + 1.8 z_rk_tri ) + w_post(k) z_rk_tri`
   with the addition order of lines 341-355. `F_078 = BK.fuse_cfg(pool, BK.e_term(pool, c49, cfg78), cfg78)`,
   cfg78 = v2-6/v2-7 backend cfg with `whiten = {"lam": 0.78, "lda": None}` (lines 303-304; v2-6's whiten is
   already `{"lam": 0.7, "lda": null}`, so only lam changes). `w_G` of FROZEN_RECIPE is a comment in the code: the
   G weights come from `fuse_cfg` unchanged (no parameter is read from `w_G`).
2. **z_p1_orig** = v2-7's joint / b4 P1 term (L-LDA, `P39E.p1_terms(...)["joint"]`, line 317). On E the v2-7 judgement
   path (`p49_eval.e_context` -> zs) computes it; Gemini's call uses the same function with the usable mask of
   `p63_eval.load_p1_full`. prep-e checks that this mask and the side-0 means equal `P39E.load_set`'s for E and E2
   (then the two calls are the same operations on the same values); if not, stop and report.
3. **z_multi** = `p63_eval.z_multi_of` with `data/prereg63/p1multi.npz` (L+M1 fit, prereg63 fit.json sha256).
4. **z_sp128** = `pool_whiten_proto_z` (lines 143-175) on `Z = ((c13_feature_385(means) - mu) / sd) @ W`
   (lines 131-140, 290-291), lam 0.50. (mu, sd, W) = Gemini's cache `c13_breg_models.npz` keys `sp128_*`, fit by
   `02_task2_systematic_experiments.py` `fit_or_load_breg_models` lines 236-278 on L+M1 side-0 P1 (Fisher LDA 128,
   reg 1e-2; float32 features at fit time, float64 at projection - as Gemini ran it). The three arrays are copied
   to `data/prereg66/sp128.npz` (sha256 in seam.json) and the copy is checked against the source arrays.
   Not refit (RAM); the D / L_c49 seams check the arrays reproduce Gemini's numbers.
5. **z_rk_tri** = `tri_rank_compat_z` (lines 212-268) with `precompute_single_game_rank_logprobs` (lines 178-209):
   S25 set model (`data/prereg25/models`, Q09 map standardisation) weight 1.0; mean of the Q09 map's 5 members with
   `not uses_ex` weight 0.6; class-count-balanced CNN head (tau 35) weight 0.5; Gaussian kernel sigma 0.85, eps 1e-6;
   row z over questions with an S25 posterior. Head: c49 vectors -> `results/prereg64/rank/c49_head.npz` (E, official
   pools, L_c49); S0 vectors -> Gemini's `s0_head.npz` (D seam only).
   **Memory**: Gemini precomputes the single-game log-posteriors over every row of the suite (`x_raw` is float64
   [N, 1540]: > 1 GB on E+E2). Here they are computed per pool on the rows the pool uses only (the functions are
   row-wise; Gemini's 8,192-row chunks vs one pool chunk can change BLAS rounding at ~1e-15). The D seam uses the
   same per-pool path, so it checks this.
6. **w_post(k)**: k = number of games of the question (`pool["k"]` on D / E / L_c49; `len(q_rows[i])` on the
   official pools, which must be in {1, 2, 3, 5}).
7. **"New parts off"** (PLAN seam): lam 0.70, w_p1_orig 8, the other weights 0. Terms with weight 0 are not added
   (so a NaN can not leak in through 0 * x); then the expression is `csls(F_070 + 8 z_p1_orig, 20)` = v2-7.
8. **Seams before the judgement** (`seam.json`, committed before prep-e / judge): (a) D 10 pools with S0 vectors /
   s0_head: base = 0.7467766213119217 bit for bit (Gemini line 610) and PRIMARY mean within 1e-6 of 0.768315
   (json rounded to 6 d.p.; per-pool within 1e-6 of the json's rounded values); (b) L_c49 12 pools rebuilt with
   Gemini's `01b.make_light_pool` and its caches `bundle_L_c49.npz` / `pstore_L_c49.npz` (seeds 300-311, lines
   489-545): base 0.659802 and PRIMARY 0.681568 within 1e-6; (c) E: v2-7 0.7035676497986656 bit for bit through the
   A66 function with the new parts off. Any failure: stop, do not judge.
9. **Judgement**: once, E 20 pools, A66 vs v2-7 (prereg49 eval.json per pool, rebuilt bit for bit in the judge):
   mean >= +0.01, pool bootstrap (10,000, seed 43, `Q42.bootstrap` as prereg63) lower bound > 0, >= 18 of 20 higher.
   Recorded only: by k; ablations `02_step1_CNN_lam078`, `03_step2_plus_P1_C13_sp128`, `06`-style rank term off
   (= 03); A66 with the rank weights only.
10. **E use number** = (largest `e_pools_use_number` in any committed `results/prereg*/explore/eval*.json`) + 1,
    cross-checked with prereg63's counter (`p63_eval.e_uses_so_far` + prereg63's own judgement). Expected 10.

## Information that would not exist for the official pools / that touches E (flags)

F1. **Rank models saw the E (and D) players with their rank labels.** All 600 E players are split v1 `train`
    players; every one of them (and every D player) has games in the competition-2 probe sets that trained the Q09
    P1TO members (`data/prereg08/p2`, `data/prereg09/p3` probe: E 600/600) and the S19/S25 set models (fit probe +
    validation players). The official pub2/priv2 players are not training players (prereg56). So z_rk_tri on E (and
    on D) can be stronger than on the official pools: the E gain of the rank part is optimistic. Gemini's own
    L_val check (players unseen by S0, s0_head, P1TO, S19, L+M1) is the only evidence it transfers. prereg63's C
    term had the same property on E. Recorded in eval.json: A66 with and without the rank term (`03` = no rank).
F2. The c49 head (`c49_head.npz`) has 5,771 classes = c49's training players; E is excluded from c49 (prereg49), so
    on E it behaves as on unseen players. Fine for the official pools (unseen).
F3. L-LDA (z_p1_orig), L+M1 Fisher LDAs (z_multi, sp128): fit on competition-1 training players only (no D/E,
    prereg63 / Gemini 00b audit).
F4. Gemini's development never opened E (its FORBIDDEN list, line 28-31). Its D numbers use D (S0 era); D is a
    seam here, not a selection set. The recipe was chosen by Gemini on L_c49 / L_val / D / official-candidate
    pseudo pools (candidate games only, no question files).
F5. The official apply uses only on-disk features: c49 pool vectors (`data/prereg49/emb/.../pool_*.npz`), P1 chunks
    `data/prereg39/p1/{pub2,priv2}` (both sides: means / counts), `c49_head.npz`, Q09 map, S25 models,
    p1multi.npz, sp128.npz. Nothing is computed on GPU; no label of an official question is used (k = its game count).
