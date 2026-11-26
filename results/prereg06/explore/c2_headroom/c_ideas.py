"""c2_headroom (c): cheap CPU probes for NEW information on top of Q02 (inner single-colour pseudo-questions only).

Probes (all OOF over the 5 player folds; scored with rule B on the 120,544 mono questions; PlayerBootstrap CIs):
  T4      add T4 (target's own moves scored with the O4 count tables) as a 5th E term in Q02's generalised fit
          (u1_lib.run_gen, same fit questions and folds as PREREG_04 tune)
  STACK   small multinomial correction on top of log P_Q02, fitted on the mono questions of the other 4 folds:
            bias      per-level bias only (control: pure recalibration, U1 said <= +0.004)
            noise     bias + 4 random features (control: what over-fitting alone gives)
            length    bias + game-length features (mean / sd of log moves, share of games where the target played
                      the last move, share of games >= 250 moves; the CNN only sees the last 250 board states)
            disp      bias + dispersion of the per-game evidence (sd over games of the per-game expected level of
                      Q02's per-game term, mean per-game entropy)
            all       bias + length + disp
  TRIM    robust pooling: per level column, drop the highest and lowest game before averaging (same fold
          parameters, no refit)
Game lengths come from the training CSVs through tools/c10data.load_rows(part="inner") (no data/local access).
"""

import json
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_lib as H  # noqa: E402
from b_decompose import Model, softmax  # noqa: E402

U = H.U
L = 10
E1 = H.E1


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def level_weights(y):
    cnt = np.bincount(y, minlength=L).astype(np.float64)
    present = (cnt > 0).sum()
    return 1.0 / (present * cnt[y])


def fit_stack(lp, F, y, l2=1e-3):
    Q, nf = F.shape
    wq = level_weights(y)

    def obj(x):
        b = x[:L]
        W = x[L:].reshape(L, nf) if nf else np.zeros((L, 0))
        z = lp + b + F @ W.T
        ls = log_softmax(z)
        loss = -np.sum(wq * ls[np.arange(Q), y]) + l2 * float(x @ x)
        G = np.exp(ls)
        G[np.arange(Q), y] -= 1.0
        G *= wq[:, None]
        gb = G.sum(0)
        gW = G.T @ F if nf else np.zeros((L, 0))
        return loss, np.concatenate([gb, gW.ravel()]) + 2 * l2 * x

    r = optimize.minimize(obj, np.zeros(L + L * nf), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


def stack_oof(lp, F, y, fold):
    P = np.zeros_like(lp)
    for f in range(5):
        tr, ev = fold != f, fold == f
        if F.shape[1]:
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
            Fs = (F - mu) / sd
        else:
            Fs = F
        x = fit_stack(lp[tr], Fs[tr], y[tr])
        b = x[:L]
        W = x[L:].reshape(L, F.shape[1]) if F.shape[1] else np.zeros((L, 0))
        P[ev] = softmax(lp[ev] + b + Fs[ev] @ W.T)
    return P


def main():
    t0 = time.time()
    z = U.load_saved()
    y, fold, player, mg = z["mono_y"], z["mono_fold"], z["mono_player"], z["mono_games"]
    P0 = z["mono_P_Q02"]
    boot = H.PlayerBoot(player, y, len(z["player_level"]))
    sc0 = H.score(H.rule_b(P0), y)
    base_rb = H.rank_balanced(sc0, y)
    out = {"what": "c2_headroom (c): cheap CPU probes for new information on top of Q02 (inner single-colour "
                   "pseudo-questions, OOF over player folds; design stage, NOT validation)",
           "Q02_mono_rb": base_rb, "Q02_mono_qm": float(sc0.mean()), "probes": {}}

    def record(name, P, note=""):
        sc = H.score(H.rule_b(P), y)
        ci, se = boot.ci(sc, sc0)
        rec = {"rb": H.rank_balanced(sc, y), "qm": float(sc.mean()), "delta_rb": H.rank_balanced(sc, y) - base_rb,
               "delta_rb_ci95": ci, "delta_rb_se": se, "delta_qm": float(sc.mean() - sc0.mean()),
               "by_k_delta_rb": {str(k): H.rank_balanced(sc[z["mono_n"] == k], y[z["mono_n"] == k]) -
                                 H.rank_balanced(sc0[z["mono_n"] == k], y[z["mono_n"] == k]) for k in H.KS}}
        if note:
            rec["note"] = note
        out["probes"][name] = rec
        print(f"{name}: {rec['delta_rb']:+.4f} {[round(v, 4) for v in ci]} ({time.time() - t0:.0f}s)", flush=True)
        return rec

    # ---------------- inputs
    d = U.load_inputs()
    inner, lp = d["inner"], d["lp"]
    keys = inner["keys"]

    # ---------------- T4 as an extra term (generalised fit, as PREREG_04 tune)
    _P, Pm_q02, _th = U.run_gen(d, ("OP", "O4", "SWA", "SWB"))
    out["rerun_Q02_max_abs_diff"] = float(np.abs(Pm_q02 - P0).max())
    _P, Pm_t4, th_t4 = U.run_gen(d, ("OP", "O4", "SWA", "SWB", "T4"))
    record("T4_extra_term", Pm_t4, "Q02 + T4 (own moves, count tables) refitted on the prereg02 fit questions")
    out["probes"]["T4_extra_term"]["w_T4_by_fold"] = [float(t[6]) for t in th_t4]

    # ---------------- per-game quantities
    with open(os.path.join(H.U1, "u1_repro.json"), encoding="utf-8") as fh:
        thetas = json.load(fh)["fold_thetas"]["Q02"]
    model = Model(lp, inner["op_o"], inner["op_has"], thetas)

    import c10data  # tools/ (on sys.path through u1_lib)
    rows = c10data.load_rows(os.path.join(H.REPO, "data", "training"),
                             os.path.join(H.REPO, "results", "split_v1", "players_split.csv"), part="inner")
    pos = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    nmov = np.full(keys["n"], -1, dtype=np.int64)
    for r in rows:
        i = pos.get((r["rank"], r["game_id"]))
        if i is not None:
            s = r["sgf_content"]
            nmov[i] = s.count(";B[") + s.count(";W[")
    del rows
    assert (nmov > 0).all(), "inner game without a move count"
    colour = np.asarray(keys["color"])
    last_is_black = (nmov % 2) == 1
    target_last = np.where(colour == "B", last_is_black, ~last_is_black).astype(np.float64)
    out["game_length"] = {"games": int(len(nmov)), "moves_quantiles": H.pct(nmov, (0, 10, 50, 90, 100)),
                          "share_ge_250": float((nmov >= 250).mean()),
                          "share_target_last_move": float(target_last.mean())}
    print(f"lengths loaded ({time.time() - t0:.0f}s)", flush=True)

    mask = mg >= 0
    gi = np.where(mask, mg, 0)
    n = mask.sum(1).astype(np.float64)

    def qmean(a):
        return (a[gi] * mask).sum(1) / n

    def qsd(a):
        m = qmean(a)
        return np.sqrt(np.maximum((((a[gi] - m[:, None]) ** 2) * mask).sum(1) / n, 0.0))

    logm = np.log(nmov.astype(np.float64))
    F_len = np.stack([qmean(logm), qsd(logm), qmean(target_last), qmean((nmov >= 250).astype(np.float64))], 1)
    # per-game expected level and entropy of Q02's per-game term (fold-specific weights -> use each question's fold)
    ev_g = np.zeros((5, keys["n"]))
    ent_g = np.zeros((5, keys["n"]))
    for f in range(5):
        p = softmax(model.v[f])
        ev_g[f] = (p * np.arange(L)).sum(1)
        ent_g[f] = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    F_disp = np.zeros((len(y), 2))
    for f in range(5):
        m = fold == f
        gi_f, mk, nn = gi[m], mask[m], n[m]
        e = ev_g[f][gi_f]
        mu = (e * mk).sum(1) / nn
        F_disp[m, 0] = np.sqrt(np.maximum((((e - mu[:, None]) ** 2) * mk).sum(1) / nn, 0.0))
        F_disp[m, 1] = (ent_g[f][gi_f] * mk).sum(1) / nn
    lp0 = np.log(np.maximum(P0, 1e-300))
    rng = np.random.default_rng(7)
    F_noise = rng.standard_normal((len(y), 4))
    variants = {"bias": np.zeros((len(y), 0)), "noise": F_noise, "length": F_len, "disp": F_disp,
                "all": np.concatenate([F_len, F_disp], 1)}
    for name, F in variants.items():
        record("stack_" + name, stack_oof(lp0, F, y, fold))

    # ---------------- trimmed pooling (drop max and min game per level column), same thetas
    Pt = np.zeros_like(P0)
    for f in range(5):
        sel = np.nonzero(fold == f)[0]
        for s in np.array_split(sel, 10):
            G = mg[s]
            mk = G >= 0
            gi_s = np.where(mk, G, 0)
            nn = mk.sum(1).astype(np.float64)
            v = model.v[f][gi_s]                                  # [q, 20, 10]
            big = np.where(mk[..., None], v, -np.inf).max(1)
            small = np.where(mk[..., None], v, np.inf).min(1)
            tot = (v * mk[..., None]).sum(1)
            mtrim = (tot - big - small) / (nn - 2)[:, None]
            u_full, _ = model.u(G, f)
            m_full = tot / nn[:, None]
            u = u_full - m_full + mtrim
            t = np.exp(model.a[f]) * np.power(nn / 12.0, model.g[f])
            Pt[s] = softmax(t[:, None] * u)
    record("trim_minmax", Pt, "per level column, mean of the per-game term without the highest and lowest game")
    out["seconds"] = round(time.time() - t0, 1)
    print(H.dump(out, "c_ideas.json"))


if __name__ == "__main__":
    main()
