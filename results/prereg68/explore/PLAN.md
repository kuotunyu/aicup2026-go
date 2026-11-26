# prereg68 (competition 1, new questions): two more 400k members + 9-member ensemble on v2-9's recipe

Date 2026-11-22 09:50. Written before any training of these members and before any E number of a 9-member ensemble.

## Why

v2-9 (prereg67) = 7-member score-level ensemble on the v2-8 back end, E +0.0196 vs v2-8 (20/20), public 0.8417
(29/125). Gemini Hydrogen 11/22 (`data/advice/gemini_1122h/REPORT.md`): back-end tweaks are exhausted (Soft-CSLS
hurts on light-player pools); remaining room is more model information. Single-model pseudo-pool scores: c49 0.6398,
p61 0.6445, S1 0.6438, S2 0.6444, X (400k) 0.6466 - the longest-trained 18-plane member is the strongest.

## Members to train (Colab G4, prereg61 recipe = prereg62 arm X with another seed; nothing else changes)

X2 = seed 1, 400,000 steps; X3 = seed 2, 400,000 steps (wrapper `p68_colab_run.py` adds the two arms to p62_run;
checkpoints kept at 100k/200k/300k/400k; result zips `C10/p62/C10_p62_X2_results.zip`, `..._X3_...`).
Vectors on the 4090 (store_E, pool_public, pool_private; raw, tta1).

## Judgement (E, 20 pools, once) - after both members exist, latest 11/24 12:00

Ensemble = prereg67's frozen recipe with the member list extended by X2 and X3 at weight 2 each (all other weights,
lam 0.85, centre_colour, w_E 14, back end unchanged). Members pass prereg67's single-model check (std10 >= 0.6406,
no zero-count games) or are dropped. Seam: with X2 = X3 = 0 weight the E score equals prereg67's 0.7379199600122529
bit for bit. Adopt if mean vs v2-9 >= +0.005 (lower than the usual +0.01 because this only adds two members of an
adopted family), bootstrap lower bound (10,000, seed 43) > 0, >= 16 of 20 pools higher. E use = committed E
judgements + 1 (12 if nothing else is judged first). Apply / assemble / upload as v2-10 if adopted.
If only one of X2 / X3 finishes by 11/24 12:00, the 8-member version is judged with the same rule.
