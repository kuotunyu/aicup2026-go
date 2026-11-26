# prereg63 notes (narrow readings of PLAN.md, written before any code or number of this registration)

2026-11-21 02:40. Every setting is in `frozen.json` with a pointer to the line of Gemini's script it comes from.
Gemini's files: `data/advice/gemini_1121/` (not in git; sha256 in `frozen.json`).

1. **B and C follow the code that produced Gemini's numbers, not the REPORT prose.** PLAN.md B/C were transcribed from
   REPORT.md 3.1, whose text ("full 770-d within-class whitening lam 0.50 and LDA 32 lam 0.20", "K = exp(-0.6|i-j|)",
   "+1e-6") describes a design Gemini never ran. The only computation behind the D number 0.756796 and the pseudo-pool
   +0.0148 is `02_hypotheses_and_comp1_verify.py` `U1_combined_P1multi_plus_rank` (line 813):
   - three Fisher LDAs fit on L+M1 side-0 P1 means (lines 579-640): full 770-d -> 128, moves [0:384] -> 64,
     board [384:768] -> 64 (standardise, within-class reg 1e-2, generalized eigh);
   - each projected through the pool's candidate whitening (`BK.whitening`, lam 0.5, no LDA; lines 652-687, 791-793),
     prototype cosine, row z; `z_multi = zrows(0.5 z128 + 0.3 zmove + 0.2 zboard)` (line 794);
   - rank term (lines 726-744, 797): S25 `question_set_term` of the pool's candidates and questions from both-side
     P1 means (q09 map standardisation), softmax, Gaussian kernel sigma 0.85, `log(max(p_q K p_c^T, 1e-6))`,
     z over rows with a usable question;
   - `S = CSLS_k(F_base + 9.0 z_multi + 2.5 z_rank)` with k = v2-6's k: B+C **replace** v2-7's P1 term (wP z_P1).
   PLAN.md asks for "weights exactly as U1_combined" and a seam reproducing 0.756796, which only this code can meet,
   so the code is ported line for line (the seam is the check). The PLAN prose is superseded where it disagrees.
2. **A (colour fill)** is implemented as PLAN.md A says (not Gemini's mean fill of script 03 line 909-929): per
   colour, the row z-score is computed over the candidates that have games of that colour only; the others get 0;
   no candidate with the colour -> 0 (as before). When every candidate has the colour the result equals the frozen
   `colour_terms` exactly. It enters `tools/p06_sim` (S2, PA, L terms) by replacing `p06_style.colour_terms` for the
   call only; `tools/p06_style.py` is not edited.
3. **A on the E pools**: the stored terms (`data/prereg36/terms_E/pool_*.npz`) were computed with the old fill, so the
   S2, PA and L terms are recomputed exactly as `p36_eval.cmd_pools` (same pool from `make_pool`, LDA refit
   excluding the pool's players). The recomputation with the frozen fill must equal the stored float32 terms bit for
   bit before the fixed terms (cast to float32 the same way) are used; O and JO do not use `colour_terms` and stay.
   The fixed terms are cached under `data/prereg63/terms_E_fix/` (no scores; sha256 in seam.json).
4. **P1 inputs**: Gemini's `load_p1_full` (side 0 and 1 means, counts, usable = not failed and side-0 count > 0, the
   PREREG_10 mock fallback) for L, M1, D, E, E2, pub2, priv2. E pools use E + E2 indexed by store row; the official
   pools use the pool's full list (candidates and questions). The three Fisher LDAs are fit once and cached
   (`data/prereg63/p1multi.npz`, sha256 in seam.json); every later step loads the cache.
5. **Base on E** = v2-7 exactly as prereg49's judgement computes it (`p49_eval.e_context` + `system` with the c49
   store_E file); seam = its per-pool scores equal prereg49 eval.json `per_pool.c49` bit for bit.
6. **Recorded only**: A alone (= v2-7 with fixed terms), B+C alone (= v2-7 terms, U1 combination), by k, and the E
   questions whose true candidate lacks a colour of the question (store colour of the 10 candidate games vs the
   question's games, as Gemini's script 03 lines 993-999), with their mean points under v2-7 / A63.
7. **E use number** = (p61_eval.e_uses_so_far: through prereg49 + later eval*.json with "e_pools_use_number",
   prereg60 included) + prereg61's own eval*.json with that key, prereg63 excluded; + 1. Computed in `judge`.
8. **Apply**: official pools as `p43_eval.official_rec` (terms computed in float64 on the fly, with the fixed fill),
   v2-7 rebuilt first and compared with `results/prereg49/apply/<pool>/top5.csv` (CRLF normalised to LF before the
   comparison: working copies may be CRLF). Assemble = `p35_apply.cmd_assemble` (v1 rows from the 06-ON file, as
   prereg49), then the original 800 rows compared with prereg49's submission.
9. Resources: CPU only, `BK.polite()` (4 cores, 1 BLAS thread as Gemini's run), one process, free RAM >= 7000 MB
   before start, RSS watched (stop above 3 GB).
10. **Amendment 11/21 03:2x (prep-e stopped on its own check; no score had been computed):** recomputing the E
    pool 0 L term with the frozen fill does not reproduce the stored float32 term bit for bit: max abs diff
    9.3e-10 (1 float32 ulp) in 0.0008% of the cells (S2 and PA were identical; the per-pool LDA refit is not
    bit-reproducible across BLAS thread settings). New rule: the frozen-fill recomputation must match the stored term
    to <= 1e-6 with identical ok flags; rows that the fill fix does not change keep the stored values, rows it
    changes take the recomputed fixed values. So "A off" stays v2-7 bit for bit and A differs only where the fill
    matters. Diagnostic script in the session scratchpad (diag_L.py), nothing written to the repo.
11. **Amendment 11/21 (memory; no score yet):** the rank term called on all 92,703 E+E2 games (Gemini's call shape)
    needs > 3 GB (float64 [N, 1540] copies inside `p19_forward.question_set_term`) and was stopped by the watchdog
    three times. E and the official pools now call the same function on only the rows the pool uses (indices
    remapped). The per-game network is row-wise; `check-subset` compares both calls on the 10 D pools and is
    committed (subset_check.json) before prep-e --part zbc. If they are not equal to <= 1e-9, stop.
