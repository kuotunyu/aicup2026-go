"""p04 proxy, step 3: fuse the per-game proxy 10-vector with R20 on the inner pseudo-questions.

Read-only on the repo. Reuses tools/p03_rank_fuse.py (prepare_inner, run_oof, combo_logprobs, load_c5_inner) and
tools/p02_rank_fuse.py (basis, metrics, rule B, PlayerBootstrap) unchanged. Reads only inner-side files:
results/prereg02/rank/{dumps/inner_*.npz, pseudo_inner.json, opening_inner.npz},
results/prereg03/rank_{a,b}/dumps/inner_*.npz. No validation dump, no answer file, no score.py.

Calibration (p02's, extended with extra terms E_j, each with its own weight w_j in [0, 4]):
    p = softmax( e^a (n/12)^gamma (m + sum_j w_j E_j + B beta) )
same objective (level-weighted NLL + 0.01 ||beta||^2), bounds, starts, maxiter, 5-fold OOF on the eval rounds.
"""
import json
import math
import os
import sys
import time
import types

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
from c10data import LEVELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = rf.basis()
L = 10


# ---------------------------------------------------------------- generalised calibration
def _unpack(th, J):
    return th[0], th[1], th[2:2 + J], th[2 + J:]


def gen_logits(th, m, Es, n):
    J = len(Es)
    a, g, w, beta = _unpack(th, J)
    t = math.exp(a) * np.power(n / 12.0, g)
    u = m + B @ beta
    for j in range(J):
        u = u + w[j] * Es[j]
    return t[:, None] * u, t


def gen_obj(th, m, Es, n, y, wq):
    J = len(Es)
    z, t = gen_logits(th, m, Es, n)
    ls = rf.log_softmax(z)
    Q = len(y)
    a, g, w, beta = _unpack(th, J)
    loss = -np.sum(wq * ls[np.arange(Q), y]) + rf.PENALTY * float(beta @ beta)
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    gz = np.sum(G * z, axis=1)
    grad = [np.sum(gz), np.sum(gz * np.log(n / 12.0))]
    for j in range(J):
        grad.append(np.sum(t * np.sum(G * Es[j], axis=1)))
    gb = (t[:, None] * G).sum(0)
    return loss, np.concatenate([grad, B.T @ gb + 2 * rf.PENALTY * beta])


def gen_fit(m, Es, n, y):
    J = len(Es)
    wq = rf.level_weights(y)
    bnds = [rf.BOUNDS["a"], rf.BOUNDS["gamma"]] + [rf.BOUNDS["w"]] * J + [(None, None)] * (L - 1)
    best = None
    for a0 in rf.STARTS:
        x0 = np.zeros(2 + J + L - 1)
        x0[0] = a0
        r = optimize.minimize(gen_obj, x0, args=(m, Es, n, y, wq), jac=True, method="L-BFGS-B",
                              bounds=bnds, options={"maxiter": rf.MAXITER})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def gen_oof(Xf, Xe, mf, me, Ef, Ee):
    P = np.zeros((len(Xe), L))
    ws = []
    for f in range(5):
        ev = Xe.fold == f
        tr = Xf.fold != f
        th = gen_fit(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
        z, _ = gen_logits(th, me[ev], [E[ev] for E in Ee], Xe.n[ev])
        P[ev] = rf.softmax(z)
        ws.append([float(x) for x in th[2:2 + len(Ef)]])
    return P, ws


# ---------------------------------------------------------------- per-game information (bank games)
def game_fit(Xs, y, wq):
    """p(y|g) = softmax(sum_j c_j X_j + B beta), c_j in [0, 20]; level-weighted NLL (+0.01||beta||^2)."""
    J = len(Xs)

    def obj(th):
        c, beta = th[:J], th[J:]
        z = B @ beta + sum(c[j] * Xs[j] for j in range(J))
        z = np.broadcast_to(z, (len(y), L)) if J == 0 else z
        ls = rf.log_softmax(z)
        loss = -np.sum(wq * ls[np.arange(len(y)), y]) + 0.01 * float(beta @ beta)
        G = np.exp(ls)
        G[np.arange(len(y)), y] -= 1.0
        G *= wq[:, None]
        gc = [np.sum(G * Xs[j]) for j in range(J)]
        return loss, np.concatenate([gc, B.T @ G.sum(0) + 0.02 * beta])
    x0 = np.concatenate([np.ones(J), np.zeros(L - 1)])
    r = optimize.minimize(obj, x0, jac=True, method="L-BFGS-B",
                          bounds=[(0.0, 20.0)] * J + [(None, None)] * (L - 1), options={"maxiter": 1000})
    return r.x


def game_ce(Xs, th, y, wq):
    J = len(Xs)
    z = B @ th[J:] + sum(th[j] * Xs[j] for j in range(J))
    z = np.broadcast_to(z, (len(y), L)) if J == 0 else z
    ls = rf.log_softmax(z)
    return float(-np.sum(wq * ls[np.arange(len(y)), y]) / np.sum(wq)), np.exp(ls)


def game_info(Xs, y, fold):
    """5-fold (player folds) OOF level-balanced cross-entropy -> information in bits = log2(10) - CE."""
    P = np.zeros((len(y), L))
    for f in range(5):
        tr, ev = fold != f, fold == f
        th = game_fit([X[tr] for X in Xs], y[tr], rf.level_weights(y[tr]))
        _, P[ev] = game_ce([X[ev] for X in Xs], th, y[ev], np.ones(int(ev.sum())))
    wq = rf.level_weights(y)
    ce = float(-np.sum(wq * np.log(np.maximum(P[np.arange(len(y)), y], 1e-300))))
    acc = float(np.mean([np.mean(P[y == i].argmax(1) == i) for i in range(L)]))
    return {"ce_bits": ce / math.log(2), "info_bits": math.log2(L) - ce / math.log(2), "acc_balanced": acc}


def centred(x):
    return x - x.mean(1, keepdims=True)


def main():
    t0 = time.time()
    c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK,
                              split=p3.SPLIT_CSV, smoke=False)
    inner = p3.prepare_inner(c)
    keys, Xf, Xe, Sf, Se = inner["keys"], inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    boot = inner["boot"]
    n = keys["n"]
    print(f"inner ready: fit {len(Xf)} eval {len(Xe)} questions ({time.time() - t0:.0f}s)", flush=True)

    logits = {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]}
    for arm, name in (("a", "p03a_d4"), ("b", "p03b_d4")):
        d = rf.load_dump(os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps", f"inner_{name}.npz"), n, 8)
        diff = p3.key_array_diff(os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps", "inner_keys.npz"),
                                 os.path.join(c.p02_dumps, "inner_keys.npz"), ("rank", "game_id", "player_id", "color"))
        if diff:
            raise SystemExit(f"arm {arm} inner keys differ: {diff}")
        logits[name] = d["logits"]
    lp = {cid: p3.combo_logprobs(dict(p3.COMBOS)[cid], logits) for cid in ("P00", "P01", "P02", "P03", "P04", "P05", "P06")}

    # proxy per game, aligned to the dump rows
    z = np.load(os.path.join(HERE, os.environ.get("P04_PROXY", "proxy_inner.npz")))
    row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
    idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
    if not all(z["player_id"][idx] == np.array(keys["player_id"])):
        raise SystemExit("proxy player ids differ from inner_keys")
    variants = [str(v) for v in z["variants"]]
    llv = z["ll"][idx]                                   # [n, 6, 3, 10]
    nmv = z["n_moves"][idx]
    lab = keys["label"]
    px = {v: rf.log_softmax(llv[:, j].sum(1)) for j, v in enumerate(variants)}
    px["to_choice_a20"] = rf.log_softmax(llv[:, variants.index("t_choice_a20")].sum(1)
                                         + llv[:, variants.index("o_choice_a20")].sum(1))
    for ph, nm in enumerate(("open", "mid", "late")):
        px[f"t_choice_a20_{nm}"] = rf.log_softmax(llv[:, variants.index("t_choice_a20"), ph])
    scored = z["scored"][idx] if "scored" in z.files else np.ones(len(idx), dtype=bool)
    per_game_acc = {v: float(np.mean(o[scored].argmax(1) == lab[scored])) for v, o in px.items()}
    print(f"per-game argmax accuracy ({int(scored.sum())} scored inner games):", {k: round(v * 100, 2) for k, v in per_game_acc.items()},
          flush=True)

    mf = {cid: rf.pooled(Sf, lp[cid], "log") for cid in lp}
    me = {cid: rf.pooled(Se, lp[cid], "log") for cid in lp}
    pf = {v: np.asarray(Sf @ o) for v, o in px.items()}
    pe = {v: np.asarray(Se @ o) for v, o in px.items()}
    opf, ope = Xf.mop, Xe.mop

    results = {}
    sc_ref = None

    def record(name, P, ws=None):
        nonlocal sc_ref
        pred = rf.rule_b_np(P)
        m, sc = rf.metrics(pred, Xe.y, Xe.n, Xe.player)
        wq = rf.level_weights(Xe.y)
        ce = float(-np.sum(wq * np.log2(np.maximum(P[np.arange(len(Xe.y)), Xe.y], 1e-300))))
        rec = {"rank_balanced": m["rank_balanced"], "question_mean": m["question_mean"],
               "per_level": m["per_level"], "weights_by_fold": ws, "ce_bits": ce, "info_bits": math.log2(L) - ce}
        if name == "R20":
            sc_ref = sc
        else:
            d = boot.stats(sc) - boot.stats(sc_ref)
            rec["diff_vs_R20"] = m["rank_balanced"] - results["R20"]["rank_balanced"]
            rec["diff_vs_R20_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            rec["question_mean_diff_vs_R20"] = float(np.mean(sc - sc_ref))
            rec["per_level_diff_vs_R20"] = {k: rec["per_level"][k] - results["R20"]["per_level"][k] for k in LEVELS}
        results[name] = rec
        print(f"{name:32s} rb {m['rank_balanced']:.4f} qm {m['question_mean']:.4f}"
              + (f"  d {rec['diff_vs_R20']:+.4f} [{rec['diff_vs_R20_ci95'][0]:+.4f}, {rec['diff_vs_R20_ci95'][1]:+.4f}]"
                 f" dqm {rec['question_mean_diff_vs_R20']:+.4f}" if name != "R20" else "")
              + (f"  w {np.round(ws, 4).tolist()}" if ws is not None else "") + f" I {rec['info_bits']:.4f}b ({time.time() - t0:.0f}s)",
              flush=True)
        return sc

    # R20 through p03's own path (must equal prereg02/03 tune: 0.5854702521055817)
    Xf.m[("P00", "log")] = mf["P00"]
    Xe.m[("P00", "log")] = me["P00"]
    B_ = rf.basis()
    m_r20, sc_r20, folds_r20, P_r20 = p3.run_oof(inner, "P00", True, False, B_)
    record("R20", P_r20)
    seam = {"R20_rank_balanced": m_r20["rank_balanced"], "registered": 0.5854702521055817,
            "abs_diff": abs(m_r20["rank_balanced"] - 0.5854702521055817)}
    # the generalised fit with E = [opening] must give the same predictions
    P_g, _ = gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf], [ope])
    seam["gen_vs_p03_max_abs_prob_diff"] = float(np.max(np.abs(P_g - P_r20)))
    seam["gen_vs_p03_same_rule_b"] = bool(np.array_equal(rf.rule_b_np(P_g), rf.rule_b_np(P_r20)))
    print("seam:", seam, flush=True)

    P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [], [])
    record("R10 (C5, no opening)", P, ws)
    main_v = "t_choice_a20"
    order = ["t_choice_a20", "t_choice_a5", "t_choice_a100", "t_nb", "o_choice_a20", "o_nb", "to_choice_a20",
             "t_choice_pat", "t_choice_ctx"]
    order = [v for v in order if v in px]
    for v in order:
        P, ws = gen_oof(Xf, Xe, pf[v], pe[v], [], [])
        record(f"PX[{v}] alone", P, ws)
    P, ws = gen_oof(Xf, Xe, pf[main_v], pe[main_v], [opf], [ope])
    record(f"PX[{main_v}] + opening (no CNN)", P, ws)
    P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [pf[main_v]], [pe[main_v]])
    record(f"R10 + PX[{main_v}]", P, ws)
    for v in order:
        P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf, pf[v]], [ope, pe[v]])
        record(f"R20 + PX[{v}]", P, ws)
    P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf, pf["t_choice_a20"], pf["o_choice_a20"]],
                    [ope, pe["t_choice_a20"], pe["o_choice_a20"]])
    record("R20 + PX[t] + PX[o] (separate w)", P, ws)
    P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"],
                    [opf] + [pf[f"t_choice_a20_{k}"] for k in ("open", "mid", "late")],
                    [ope] + [pe[f"t_choice_a20_{k}"] for k in ("open", "mid", "late")])
    record("R20 + PX[t] by phase (3 w)", P, ws)
    for cid in ("P01", "P02", "P03", "P06"):
        P, ws = gen_oof(Xf, Xe, mf[cid], me[cid], [opf], [ope])
        record(f"{cid} (C6 fusion, as prereg03)", P, ws)
        P, ws = gen_oof(Xf, Xe, mf[cid], me[cid], [opf, pf[main_v]], [ope, pe[main_v]])
        record(f"{cid} + PX[{main_v}]", P, ws)
    # question-level reference terms for the 'new fraction' (each alone, and added with its own weight)
    P, ws = gen_oof(Xf, Xe, opf, ope, [], [])
    record("OP alone (opening, no CNN)", P, ws)
    for cid, nm in (("P04", "C6A"), ("P05", "C6B")):
        P, ws = gen_oof(Xf, Xe, mf[cid], me[cid], [], [])
        record(f"{nm} alone (no opening)", P, ws)
        P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [mf[cid]], [me[cid]])
        record(f"R10 + {nm} (separate w)", P, ws)
        P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf, mf[cid]], [ope, me[cid]])
        record(f"R20 + {nm} (separate w)", P, ws)
        P, ws = gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf, mf[cid], pf[main_v]], [ope, me[cid], pe[main_v]])
        record(f"R20 + {nm} (separate w) + PX[{main_v}]", P, ws)
    I = {k: v["info_bits"] for k, v in results.items()}
    frac_q = {
        "PX_vs_R20": {"alone": I[f"PX[{main_v}] alone"], "added": I[f"R20 + PX[{main_v}]"] - I["R20"]},
        "PX_vs_R10": {"alone": I[f"PX[{main_v}] alone"], "added": I[f"R10 + PX[{main_v}]"] - I["R10 (C5, no opening)"]},
        "PXo_vs_R20": {"alone": I["PX[o_choice_a20] alone"], "added": I["R20 + PX[o_choice_a20]"] - I["R20"]},
        "OP_vs_R10": {"alone": I["OP alone (opening, no CNN)"], "added": I["R20"] - I["R10 (C5, no opening)"]},
        "C6A_vs_R10": {"alone": I["C6A alone (no opening)"], "added": I["R10 + C6A (separate w)"] - I["R10 (C5, no opening)"]},
        "C6A_vs_R20": {"alone": I["C6A alone (no opening)"], "added": I["R20 + C6A (separate w)"] - I["R20"]},
        "C6B_vs_R20": {"alone": I["C6B alone (no opening)"], "added": I["R20 + C6B (separate w)"] - I["R20"]},
    }
    for v in frac_q.values():
        v["new_fraction"] = v["added"] / v["alone"] if v["alone"] > 0 else None
    print("question-level information fractions:", json.dumps(frac_q), flush=True)
    # residual view: does the proxy's question-level level trend track what R20 gets wrong?
    lvl = np.arange(L) - 4.5
    E_r20 = P_r20 @ np.arange(L)
    resid = Xe.y - E_r20
    s_px = centred(pe[main_v]) @ lvl / (lvl @ lvl)
    s_c5 = centred(me["P00"]) @ lvl / (lvl @ lvl)
    s_op = centred(ope) @ lvl / (lvl @ lvl)
    resid_corr = {"corr(PX slope, y - E_R20)": float(np.corrcoef(s_px, resid)[0, 1]),
                  "corr(PX slope, C5 slope) questions": float(np.corrcoef(s_px, s_c5)[0, 1]),
                  "corr(PX slope, OP slope) questions": float(np.corrcoef(s_px, s_op)[0, 1]),
                  "corr(PX slope, y) questions": float(np.corrcoef(s_px, Xe.y)[0, 1]),
                  "corr(C5 slope, y) questions": float(np.corrcoef(s_c5, Xe.y)[0, 1])}
    print("question-level correlations:", json.dumps(resid_corr), flush=True)

    # ---------------- per-game information on the bank games (the games the pseudo-questions use)
    players = inner["players"]
    dump_row = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    bank_rows, bank_fold = [], []
    for p in players:
        for gid in p["bank"]:
            bank_rows.append(dump_row[(p["rank"], gid)])
            bank_fold.append(p["fold"])
    br = np.array(bank_rows)
    bf = np.array(bank_fold)
    yb = lab[br]
    op_full = np.where(inner["op_has"][:, None], inner["op_o"], 0.0)
    src = {"C5": lp["P00"], "C6A": p3.combo_logprobs(("C6A",), logits), "C6B": p3.combo_logprobs(("C6B",), logits),
           "OP": op_full, "PX": px[main_v], "PX_o": px["o_choice_a20"], "PX_nb": px["t_nb"]}
    info = {}
    for name, comb in (("C5", ["C5"]), ("PX", ["PX"]), ("PX_o", ["PX_o"]), ("PX_nb", ["PX_nb"]), ("OP", ["OP"]),
                       ("C6A", ["C6A"]), ("C6B", ["C6B"]),
                       ("C5+PX", ["C5", "PX"]), ("C5+PX_o", ["C5", "PX_o"]), ("C5+PX_nb", ["C5", "PX_nb"]),
                       ("C5+OP", ["C5", "OP"]), ("C5+C6A", ["C5", "C6A"]), ("C5+C6B", ["C5", "C6B"]),
                       ("C5+OP+PX", ["C5", "OP", "PX"]), ("C5+OP+PX+PX_o", ["C5", "OP", "PX", "PX_o"]),
                       ("C5+C6A+OP", ["C5", "C6A", "OP"]), ("C5+C6A+OP+PX", ["C5", "C6A", "OP", "PX"]),
                       ("PX+OP", ["PX", "OP"])):
        info[name] = game_info([src[s][br] for s in comb], yb, bf)
        print(f"per-game {name:16s} info {info[name]['info_bits']:.4f} bits, balanced acc "
              f"{info[name]['acc_balanced'] * 100:.2f}%", flush=True)
    frac = {}
    for s in ("PX", "PX_o", "PX_nb", "OP", "C6A", "C6B"):
        alone = info[s]["info_bits"]
        added = info[f"C5+{s}"]["info_bits"] - info["C5"]["info_bits"]
        frac[s] = {"alone_bits": alone, "added_to_C5_bits": added, "new_fraction": added / alone if alone > 0 else None}
    # correlations of the per-game level signal (bank games)
    uc, up = centred(src["C5"][br]), centred(src["PX"][br])
    uo = centred(src["OP"][br])
    lvl = np.arange(L) - 4.5
    slope = lambda u: (u @ lvl) / (lvl @ lvl)  # noqa: E731
    corr = {"C5_vs_PX_centred_logprob_vectors": float(np.corrcoef(uc.ravel(), up.ravel())[0, 1]),
            "C5_vs_PX_level_slope": float(np.corrcoef(slope(uc), slope(up))[0, 1]),
            "C5_vs_OP_level_slope": float(np.corrcoef(slope(uc), slope(uo[inner["op_has"][br]]
                                                                         if False else uo))[0, 1]),
            "C5_vs_C6A_level_slope": float(np.corrcoef(slope(uc), slope(centred(src["C6A"][br])))[0, 1]),
            "PX_level_slope_vs_label": float(np.corrcoef(slope(up), yb)[0, 1]),
            "C5_level_slope_vs_label": float(np.corrcoef(slope(uc), yb)[0, 1])}
    # partial: does PX's slope predict the label once C5's slope is known (residual correlation)
    A_ = np.c_[np.ones(len(yb)), slope(uc)]
    res_y = yb - A_ @ np.linalg.lstsq(A_, yb, rcond=None)[0]
    res_p = slope(up) - A_ @ np.linalg.lstsq(A_, slope(up), rcond=None)[0]
    corr["partial_PX_slope_vs_label_given_C5_slope"] = float(np.corrcoef(res_p, res_y)[0, 1])
    print("fractions:", json.dumps(frac), flush=True)
    print("correlations:", json.dumps(corr), flush=True)

    out = {"what": "p04 task B proxy fusion on inner (5-fold OOF, rule B), per-game information on bank games",
           "seam": seam, "per_game_argmax_accuracy_scored_games": per_game_acc, "scored_games": int(scored.sum()),
           "mean_moves_per_game": {"target": float(nmv[scored, 0].sum(1).mean()), "opponent": float(nmv[scored, 1].sum(1).mean())},
           "questions": {"fit": len(Xf), "eval": len(Xe)}, "bank_games": int(len(br)),
           "combinations": results, "question_info_fractions": frac_q, "question_correlations": resid_corr, "per_game_info": info, "new_fraction": frac, "correlations": corr,
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, os.environ.get("P04_FUSE_OUT", "fuse_report.json")), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
