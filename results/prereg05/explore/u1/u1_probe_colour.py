"""prereg05 exploration U1, step 3 (headroom probe): how much is the colour bias worth on single-colour questions?

    .venv/Scripts/python.exe results/prereg05/explore/u1/u1_probe_colour.py   -> u1_probe_colour.json

Seen in u1_analysis.json: on the PREREG_03 single-colour inner pseudo-questions Q02 scores 0.5949 when the target
plays black and 0.6264 when white, and black questions are pushed upward (mean signed error larger at every level).
Official competition-2 questions are single colour, so a colour-blind calibration may leave points there.

(a) per-game diagnostic: mean expected level minus true level of each source (C5, O4, SWA, SWB) by the target's colour
    on the 73,556 inner games.
(b) diagnostic upper-ish bound: a per-colour level-bias vector (9 free numbers per colour) added to Q02's OOF log P on
    the single-colour questions, fitted (level-weighted NLL) on 4 player folds, applied to the 5th; control: one
    colour-blind bias vector fitted the same way.
(c) a test-day-usable form: Q02's generalised calibration (tools/p04_rank_fuse.py gen_fit, unchanged) with two extra
    per-game terms, black-indicator x (+ordinal) and black-indicator x (-ordinal) (ordinal = (k - 4.5) / 4.5, k the
    level index), so the net black shift is w_up - w_down; fitted on the registered fit questions (mixed colour) of
    the other folds exactly like Q02, evaluated OOF on the eval questions and the single-colour questions.
Inner pseudo-questions only; design-stage exploration after looking at inner -- not validation results.
"""

import json
import os
import time

import numpy as np
from scipy import optimize

import u1_lib as U

T0 = time.time()
L = 10
LV = U.LEVELS


def pb_diff(sa, sb, player, y):
    b = U.PlayerBootstrap(player, y, 636)
    d = b.stats(sa) - b.stats(sb)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def per_level(sc, y):
    return {LV[i]: float(sc[y == i].mean()) for i in range(L)}


def fit_bias(lp, y, Bm):
    """Level-weighted NLL of softmax(lp + B beta) over beta (9), small ridge as the calibration (0.01)."""
    cnt = np.bincount(y, minlength=L).astype(float)
    wq = 1.0 / (np.count_nonzero(cnt) * cnt[y])
    Q = len(y)

    def obj(beta):
        z = lp + Bm @ beta
        ls = U.rf.log_softmax(z)
        loss = -np.sum(wq * ls[np.arange(Q), y]) + 0.01 * beta @ beta
        G = np.exp(ls)
        G[np.arange(Q), y] -= 1
        G *= wq[:, None]
        return loss, Bm.T @ G.sum(0) + 0.02 * beta

    r = optimize.minimize(obj, np.zeros(L - 1), jac=True, method="L-BFGS-B")
    return r.x


def main():
    d = U.load_inputs()
    inner, lp = d["inner"], d["lp"]
    keys = inner["keys"]
    black = (np.array(keys["color"]) == "B").astype(np.float64)
    y_game = np.asarray(keys["label"])
    out = {"what": __doc__.split("\n")[0], "note": "design-stage exploration on inner pseudo-questions; not validation"}

    # (a) per-game colour diagnostic
    lv = np.arange(L, dtype=float)
    diag = {}
    for s in ("C5", "O4", "SWA", "SWB"):
        p = np.exp(lp[s])
        el = (p * lv).sum(1) - y_game
        diag[s] = {c: {"mean_expected_level_minus_true": float(el[black == (c == "B")].mean()),
                       "top1_accuracy": float((lp[s].argmax(1) == y_game)[black == (c == "B")].mean())}
                   for c in ("B", "W")}
        diag[s]["black_minus_white_expected_level"] = (diag[s]["B"]["mean_expected_level_minus_true"]
                                                       - diag[s]["W"]["mean_expected_level_minus_true"])
    out["a_per_game_by_colour"] = diag
    print("(a)", {s: round(v["black_minus_white_expected_level"], 3) for s, v in diag.items()}, flush=True)

    z = U.load_saved()
    Bm = U.p4.B
    # (b) per-colour bias on Q02's OOF probabilities, single-colour questions, cross-fitted over player folds
    y, fold, player, col = z["mono_y"], z["mono_fold"], z["mono_player"], z["mono_colour"]
    P2 = z["mono_P_Q02"]
    lp2 = np.log(P2)
    sc2 = U.score(U.rule_b_np(P2), y)
    pred_c = np.empty(len(y), dtype=np.int64)
    pred_0 = np.empty(len(y), dtype=np.int64)
    betas = []
    for f in range(5):
        tr, te = fold != f, fold == f
        b0 = fit_bias(lp2[tr], y[tr], Bm)
        pred_0[te] = U.rule_b_np(U.rf.softmax(lp2[te] + Bm @ b0))
        rec = {"fold": f, "blind": (Bm @ b0).round(4).tolist()}
        for c in ("B", "W"):
            mtr, mte = tr & (col == c), te & (col == c)
            bc = fit_bias(lp2[mtr], y[mtr], Bm)
            pred_c[mte] = U.rule_b_np(U.rf.softmax(lp2[mte] + Bm @ bc))
            rec[c] = (Bm @ bc).round(4).tolist()
        betas.append(rec)
    res_b = {}
    for name, pr in (("colour_blind_bias", pred_0), ("per_colour_bias", pred_c)):
        sc = U.score(pr, y)
        res_b[name] = {"rank_balanced": U.rank_balanced(sc, y), "minus_Q02": U.rank_balanced(sc, y) -
                       U.rank_balanced(sc2, y), "minus_Q02_pboot_ci95": pb_diff(sc, sc2, player, y),
                       "by_colour_minus_Q02": {c: U.rank_balanced(sc, y, col == c) - U.rank_balanced(sc2, y, col == c)
                                               for c in ("B", "W")}}
    sc_c, sc_0 = U.score(pred_c, y), U.score(pred_0, y)
    res_b["per_colour_minus_blind"] = {"delta": U.rank_balanced(sc_c, y) - U.rank_balanced(sc_0, y),
                                       "pboot_ci95": pb_diff(sc_c, sc_0, player, y)}
    res_b["bias_vectors_by_fold"] = betas
    out["b_mono_crossfit_bias"] = res_b
    print("(b)", {k: (round(v["minus_Q02"], 4), [round(x, 4) for x in v["minus_Q02_pboot_ci95"]])
                  for k, v in res_b.items() if "minus_Q02" in v}, flush=True)

    # (c) Q02's calibration + black x ordinal terms, fitted on the fit questions (gen_fit unchanged)
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    Xm, Sm = inner["X_mono"], inner["S_mono"]
    ordv = (np.arange(L) - 4.5) / 4.5
    lp5 = lp["C5"]
    mf, me, mm = (U.pooled(S, lp5, "log") for S in (Sf, Se, Sm))
    terms = ["OP", "O4", "SWA", "SWB"]
    Ef, Ee, Em = (U.p4.question_terms(X, S, lp, terms) for X, S in ((Xf, Sf), (Xe, Se), (Xm, Sm)))
    fb = [np.asarray(S @ black) for S in (Sf, Se, Sm)]
    col_terms = [[f[:, None] * ordv[None, :], -f[:, None] * ordv[None, :]] for f in fb]
    Pq, folds_q = U.p4.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
    Pc, folds_c = U.p4.gen_oof_record(Xf, Xe, mf, me, Ef + col_terms[0], Ee + col_terms[1])
    Pqm = U.p4.gen_fold_predict(folds_q, mm, Em, Xm.n, Xm.fold)
    Pcm = U.p4.gen_fold_predict(folds_c, mm, Em + col_terms[2], Xm.n, Xm.fold)
    # sanity: the no-colour run is Q02
    assert np.array_equal(Pq, z["eval_P_Q02"]) and np.array_equal(Pqm, z["mono_P_Q02"])
    res_c = {"weights_by_fold": [{"w": fr["theta"][2:2 + 6], "net_black_shift": fr["theta"][6] - fr["theta"][7]}
                                 for fr in folds_c]}
    for setname, Pa, Pb_, X, extra in (("eval_mixed", Pc, Pq, Xe, None), ("single_colour", Pcm, Pqm, Xm, col)):
        sa, sb = U.score(U.rule_b_np(Pa), X.y), U.score(U.rule_b_np(Pb_), X.y)
        r = {"Q02": U.rank_balanced(sb, X.y), "Q02_plus_colour": U.rank_balanced(sa, X.y),
             "delta": U.rank_balanced(sa, X.y) - U.rank_balanced(sb, X.y),
             "pboot_ci95": pb_diff(sa, sb, X.player, X.y),
             "per_level_delta": {k: a - b for (k, a), b in zip(per_level(sa, X.y).items(), per_level(sb, X.y).values())}}
        if extra is not None:
            r["by_colour"] = {c: {"Q02": U.rank_balanced(sb, X.y, extra == c),
                                  "Q02_plus_colour": U.rank_balanced(sa, X.y, extra == c)} for c in ("B", "W")}
            r["by_k_delta"] = {str(k): U.rank_balanced(sa, X.y, X.n == k) - U.rank_balanced(sb, X.y, X.n == k)
                               for k in (5, 10, 15, 20)}
        res_c[setname] = r
    out["c_gen_fit_with_colour_terms"] = res_c
    print("(c)", {k: (round(v["delta"], 4), [round(x, 4) for x in v["pboot_ci95"]]) for k, v in res_c.items()
                  if isinstance(v, dict)}, "net shift", [round(x["net_black_shift"], 3) for x in res_c["weights_by_fold"]],
          flush=True)
    out["seconds"] = round(time.time() - T0, 1)
    with open(os.path.join(U.HERE, "u1_probe_colour.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote u1_probe_colour.json ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
