# prereg67 (competition 1, new questions): score-level ensemble on top of v2-8 (Gemini Hydrogen 11/21)

Date 2026-11-21 20:20. Written before any vector of prereg61's final model or prereg62-X exists and before any E
number of an ensemble. Source: `data/advice/gemini_1121h/REPORT.md` section 4 ("PREREG_ENS_V29") and its scripts /
results in `data/advice/gemini_1121h/` (development on candidate pseudo pools only; E not used).
Base = v2-8 (prereg66 A66, E 0.7183609308316172). E has been used 10 times; this is the next use.

## Method (copied from Hydrogen's section 4; every number into `frozen.json` with file/line pointers)

- Members (raw, tta1 vectors; store_E, pool_public, pool_private): c49 (data/prereg49/emb), S (prereg55),
  B (prereg51), S1, S2 (prereg62), p61 (prereg61 final 200k model), X (prereg62-X, 400k).
- Each member's CNN term computed independently with cfg_ens: gnorm, mean pooling, centre_colour = True,
  whiten lam 0.85 (no LDA), proto similarity; Z_m = zscore_rows(E_m, ok_m); E_ens = sum_m (w_m / sum w) Z_m,
  ok = all members ok. Weights: c49 1, S 1, B 1, p61 2, S1 2, S2 2, X 2.5 if X's single-model pseudo-pool score
  (v2-8 back end, std10) is "clearly" above S1's - since "clearly" is not defined, X gets 2.0 (narrow reading).
- w_E = 14.0; everything else exactly prereg66's frozen back end (F_G with the frozen colour_terms, P1 combo
  5.5 / 3 / 3 with z_sp128 as in prereg66 - no breg128, z_rk_tri 1.8 inside CSLS_20, post weights by k
  2.8 / 2.2 / 1.5 / 0.8).
- Version by availability, decided mechanically before E: 7 members if p61 and X vectors exist and each passes the
  single-model check (v2-8 back end on Hydrogen's std10 pseudo pools >= 0.6406 and no game without a sample);
  otherwise drop the failing / missing member(s) (6-member A without X; 5-member B without p61 and X).
  Hard time rule: whatever exists at 11/22 02:00 is used.

## Seams before the judgement

With c49 alone (others weight 0), centre_colour False, lam 0.78, w_E 8: Hydrogen's std10 pseudo pools 0.639759
and E = 0.7183609308316172 bit for bit.

## Judgement (E, 20 pools, once)

Ens vs v2-8: mean >= +0.01, bootstrap lower bound (10,000, seed 43) > 0, >= 18/20 pools higher. Recorded: by k,
the version used, single-model E scores are NOT computed (only the ensemble is judged).

## Apply

pub2 / priv2, original 800 rows unchanged, checks, upload as v2-9 if adopted.

## prereg61's own judgement

prereg61's model enters here as a member; its standalone E judgement (prereg61 PLAN) is not run, to save E uses.
