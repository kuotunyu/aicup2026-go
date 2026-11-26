# prereg66 (competition 1, new questions): back end "04_R_final_PRIMARY" (Gemini Carbon 11/21)

Date 2026-11-21. Written before any code of this registration and before any E number of it. Source: Gemini
Carbon's review `data/advice/gemini_1121b/REPORT.md` section 3 and its frozen recipe
`data/advice/gemini_1121b/04_final_registerable_recipe.py` (FROZEN_RECIPE, lines 37-72) / `.json`, development on
44 pools (light unseen players L_c49 12, L_val 12, D 10, official-candidate pseudo pools 10). E not used by it.
E has been used 9 times (last: prereg63); this is the next use; the report says so.

## Method = Carbon's 04_R_final_PRIMARY exactly (every number into `frozen.json` with file/line pointers)

v2-7 vectors (c49) with: cnn_whiten_lam 0.78; P1 combination 5.5 z_p1_orig + 3.0 z_multi + 3.0 z_sp128 (its
"c13 approximate counterfactual" sp128 term as its script builds it from features already on disk); rank-compatibility
term z_rk_tri with weight 1.8 inside CSLS_20 and a post-CSLS weight by number of query games {1: 2.8, 2: 2.2,
3: 1.5, 5: 0.8}; colour_terms as frozen (no colour-fill change). Seam stated by Carbon: with lam 0.70, w_p1_orig 8 and
the new terms 0 it equals v2-7.

## Seams before the judgement

v2-7 E 0.7035676497986656 bit for bit with the new parts switched off; Carbon's D numbers (base 0.746777, PRIMARY
0.768315) and its L_c49 PRIMARY number reproduced (|diff| <= 1e-6) - checks the port. If the port does not
reproduce, stop.

## Judgement (E, 20 pools, once)

A66 vs v2-7: mean >= +0.01, bootstrap lower bound (10,000, seed 43) > 0, >= 18 of 20 pools higher. Recorded: by k.
Hard stop 11/23 12:00. If prereg61 is adopted later, combining is a separate registration.

## Apply

pub2 / priv2 with c49 vectors, original 800 rows unchanged, checks; upload as competition-1 v2-8 if adopted.
