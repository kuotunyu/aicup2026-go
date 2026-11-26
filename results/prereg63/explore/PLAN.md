# prereg63 (competition 1, new questions): back-end upgrades found by the 11/21 Gemini review

Date 2026-11-21. Written before any code of this registration and before any E number of it. Source:
`data/advice/gemini_1121/REPORT.md` sections 2.4 and 3.1 (C1-1), scripts `02_hypotheses_and_comp1_verify.py`,
`03_comp2_and_final_checks.py`, results `02_comp1_and_hypotheses.json` ("Comp1_upgrades_verification"),
`03_comp2_and_final_checks.json`. Those D-pool and candidate-pseudo-pool numbers are the development evidence; the
settings below are taken from them as they are, nothing is re-tuned here. E has not been used by them.

## Changes to v2-7 (c49 vectors + v2-6 back end + P1 term)

A. Bug fix in the hand-crafted terms: `tools/p06_style.py: colour_terms` fills a candidate WITHOUT games of the query's
   colour with the row minimum (written for v1 where every candidate had 100 black games). New rule: such candidates
   get the neutral value 0 after the row z-score computed over the candidates that do have that colour (the same
   convention as `p38_backend._z_valid` for the CNN term). Implemented in a new module (tools/p06_style.py is frozen).
B. P1 term from L+M1 (`results/prereg45` loaders) with two projections as in Gemini's U1a "P1_multi_lam05"
   (full 770-d within-class whitening lam 0.50, and LDA 32 with lam 0.20), each pool-whitened as now.
C. Rank-compatibility term as in Gemini's U1b: S25 (competition-2 set model, `tools/p25_forward.py`) rank posteriors
   of every candidate (10 games) and question (1-5 games) from the P1 means already on disk; term
   zrows(log(P_q K P_c^T + 1e-6)), K_ij = exp(-0.6 |i - j|).
Weights of B and C exactly as Gemini's "U1_combined_P1multi_plus_rank" (copy them into `frozen.json` with a pointer to
the line of Gemini's script that defines them, before computing anything on E).

## Seams before any E number of the method

v2-7's E 0.7035676497986656 bit for bit with A/B/C switched off; Gemini's D numbers for v2-6 base (0.746777) and for
U1_combined (0.756796) reproduced with A off (agreement to 1e-6) - this checks the port.

## Judgement (E, 20 pools, once)

A63 = v2-7 + A + B + C. Adopt if mean(A63 - v2-7) >= +0.01, bootstrap lower bound (10,000, seed 43) > 0, >= 18/20
pools higher. Also recorded (not gates): A alone, B+C alone, by k, and the number of E questions whose true candidate
lacks the query colour. E use number = committed E judgements + 1. Hard stop 11/23 12:00.
If prereg61 is adopted, combining is a separate registration (the back end applies to any vector set).

## Apply

pub2 / priv2 with c49 vectors (or the adopted vectors), original 800 rows unchanged, checks, upload.
