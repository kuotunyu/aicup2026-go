"""c2_headroom (a): transductive decoding of a whole 400-question file under a level-count constraint.

Inner single-colour pseudo-questions (mono_*, official shape: one colour, k = 5/10/15/20) with Q02's OOF
probabilities (results/prereg05/explore/u1/u1_oof.npz). No validation data, nothing scored against validation.

A simulated "file" = 400 questions, 100 per k. How many per level depends on the scenario (see SCENARIOS).
Players: inside one file each level's slots go to the least-used players of that level first (random tie-break,
k = 20 first because fewer players are eligible), so a player appears twice only when the level has fewer
eligible players than slots. Every slot then takes one random pseudo-question of that (player, k) (any round,
either colour).

Decoders (all label-free):
  B     rule B per question (Q02 as submitted)
  EM    Saerens EM re-estimation of the level prior (model prior = uniform, the fit is level-weighted), then rule B
  SK    Sinkhorn prior matching to uniform (mean posterior = 1/10 per level), then rule B
  H40   Hungarian on the expected-score matrix, exactly 40 answers per level
  H10k  Hungarian inside each k, exactly 10 answers per level per k
  ORC   Hungarian with the TRUE level counts (oracle upper bound; not usable on test day)
Gate statistics (label-free, per file): chi2 of rule-B histogram vs 40 each; EM likelihood-ratio vs uniform;
self-assessed expected-score sacrifice of H40 (model's own expected score of B minus of H40).

Uncertainty: (i) spread over files (what one test file could show); (ii) the mean over files under a player
bootstrap (players resampled within level, as PlayerBootstrap; each replicate simulates several files).
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_lib as H  # noqa: E402

L = 10
KS = H.KS


# ---------------------------------------------------------------- scenarios: level counts [10, 4] (100 per k)
def c_bal_cell(rng):
    return np.full((L, 4), 10, dtype=np.int64)


def c_bal_level(rng):
    lab = np.repeat(np.arange(L), 40)
    rng.shuffle(lab)
    return np.stack([np.bincount(lab[i * 100:(i + 1) * 100], minlength=L) for i in range(4)], axis=1)


def c_near_bal(rng):
    # 800 questions = 20 per (level, k) over public + private; this file gets a random 100 of each k's 200
    out = np.zeros((L, 4), dtype=np.int64)
    for i in range(4):
        lab = np.repeat(np.arange(L), 20)
        out[:, i] = np.bincount(rng.choice(lab, 100, replace=False), minlength=L)
    return out


def c_prop(rng):
    pi = H.TRAIN_PLAYERS / H.TRAIN_PLAYERS.sum()
    return np.stack([rng.multinomial(100, pi) for _ in range(4)], axis=1)


def c_dir(alpha):
    def f(rng):
        pi = rng.dirichlet(np.full(L, alpha))
        return np.stack([rng.multinomial(100, pi) for _ in range(4)], axis=1)
    return f


SCENARIOS = {"bal_cell": c_bal_cell, "bal_level": c_bal_level, "near_bal": c_near_bal, "prop_players": c_prop,
             "dir10": c_dir(10.0), "dir3": c_dir(3.0)}


# ---------------------------------------------------------------- population and file sampler
class Pop:
    def __init__(self, d, ent_player):
        self.ent_player = np.asarray(ent_player)
        lvl = d["player_level"][self.ent_player]
        self.elig = [[None] * 4 for _ in range(L)]
        for ki, k in enumerate(KS):
            ok = np.array([(p, k) in d["qidx"] for p in self.ent_player])
            for l in range(L):
                self.elig[l][ki] = np.nonzero(ok & (lvl == l))[0]
        self.qidx = d["qidx"]


def draw_file(rng, counts, pop):
    q, kk = [], []
    reuse_slots = 0
    max_use = 0
    for l in range(L):
        usage = defaultdict(int)
        for ki in (3, 2, 1, 0):
            need = int(counts[l, ki])
            el = pop.elig[l][ki]
            if need == 0:
                continue
            if len(el) == 0:
                raise RuntimeError(f"no eligible player for level {l} k {KS[ki]}")
            chosen = []
            while len(chosen) < need:
                u = np.fromiter((usage[e] for e in el), dtype=np.int64, count=len(el))
                order = np.lexsort((rng.random(len(el)), u))
                for i in order[:need - len(chosen)]:
                    e = int(el[i])
                    if usage[e] > 0:
                        reuse_slots += 1
                    usage[e] += 1
                    chosen.append(e)
            for e in chosen:
                idx = pop.qidx[(int(pop.ent_player[e]), KS[ki])]
                q.append(idx[rng.integers(len(idx))])
                kk.append(ki)
        if usage:
            max_use = max(max_use, max(usage.values()))
    return np.array(q), np.array(kk), reuse_slots, max_use


# ---------------------------------------------------------------- decoders
def em_prior(P, iters=500, tol=1e-10):
    pi = np.full(L, 1.0 / L)
    for _ in range(iters):
        Q = P * pi
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    Q = P * pi
    return Q / Q.sum(1, keepdims=True), pi


def sinkhorn(P, iters=2000, tol=1e-9):
    lp = np.log(np.maximum(P, 1e-300))
    b = np.zeros(L)
    for _ in range(iters):
        z = lp + b
        z -= z.max(1, keepdims=True)
        Q = np.exp(z)
        Q /= Q.sum(1, keepdims=True)
        m = Q.mean(0)
        if np.max(np.abs(m - 0.1)) < tol:
            break
        b += np.log(0.1 / m)
    return Q, b


def hungarian(ES, counts):
    cols = np.repeat(np.arange(L), counts)
    assert len(cols) == ES.shape[0]
    r, c = linear_sum_assignment(ES[:, cols], maximize=True)
    pred = np.empty(ES.shape[0], dtype=np.int64)
    pred[r] = cols[c]
    return pred


def decode_all(P, kk, true_counts):
    ES = H.expected_scores(P)
    out = {}
    out["B"] = np.argmax(ES, axis=1)
    Pem, pi = em_prior(P)
    out["EM"] = H.rule_b(Pem)
    Psk, b = sinkhorn(P)
    out["SK"] = H.rule_b(Psk)
    out["H40"] = hungarian(ES, np.full(L, 40))
    h = np.empty(len(P), dtype=np.int64)
    for ki in range(4):
        m = kk == ki
        h[m] = hungarian(ES[m], np.full(L, 10))
    out["H10k"] = h
    out["ORC"] = hungarian(ES, true_counts)
    # label-free gate statistics
    hist = np.bincount(out["B"], minlength=L)
    gates = {"chi2_B": float(((hist - 40.0) ** 2 / 40.0).sum()),
             "lr_em": float(2.0 * np.sum(np.log(10.0 * (P @ pi)))),
             "sacrifice_H40": float(ES[np.arange(len(P)), out["B"]].mean() - ES[np.arange(len(P)), out["H40"]].mean()),
             "sk_bias_range": float(b.max() - b.min()),
             "em_pi_maxdev": float(np.max(np.abs(pi - 0.1)))}
    return out, gates


METHODS = ("B", "EM", "SK", "H40", "H10k", "ORC")
GATES = ("chi2_B", "lr_em", "sacrifice_H40", "sk_bias_range", "em_pi_maxdev")


def run_files(rng, scen, pop, d, n_files, combos):
    rows = []
    for _ in range(n_files):
        counts = SCENARIOS[scen](rng)
        q, kk, reuse, max_use = draw_file(rng, counts, pop)
        y = d["y"][q]
        tc = np.bincount(y, minlength=L)
        rec = {"reuse_slots": reuse, "max_use": max_use, "true_counts": tc}
        for cb in combos:
            P = d["P_" + cb][q]
            preds, gates = decode_all(P, kk, tc)
            for m in METHODS:
                sc = H.score(preds[m], y)
                rec[(cb, m, "qm")] = float(sc.mean())
                rec[(cb, m, "rb")] = H.rank_balanced(sc, y)
            for g in GATES:
                rec[(cb, "gate", g)] = gates[g]
            rec[(cb, "hist_B")] = np.bincount(preds["B"], minlength=L)
        rows.append(rec)
    return rows


def boot_pop(rng, d):
    ents = []
    pl = d["player_level"]
    present = np.unique(d["player"])
    for l in range(L):
        mem = present[pl[present] == l]
        ents.extend(mem[rng.integers(0, len(mem), size=len(mem))].tolist())
    return Pop(d, ents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot-files", type=int, default=10)
    ap.add_argument("--calib-files", type=int, default=1000)
    ap.add_argument("--scen", default=",".join(SCENARIOS))
    ap.add_argument("--out", default="a_transductive.json")
    a = ap.parse_args()
    t0 = time.time()
    m = H.load_mono("Q02")
    d = {"P_Q02": m["P"], "P_Q00": m["P_Q00"], "y": m["y"], "player_level": m["player_level"],
         "player": m["player"]}
    qidx = defaultdict(list)
    for i, (p, k) in enumerate(zip(m["player"], m["n"])):
        qidx[(int(p), int(k))].append(i)
    d["qidx"] = {key: np.array(v) for key, v in qidx.items()}
    present = np.unique(m["player"])
    pop0 = Pop(d, present)
    combos = ("Q02", "Q00")
    elig = {H.LEVELS[l]: {str(k): int(len(pop0.elig[l][ki])) for ki, k in enumerate(KS)} for l in range(L)}

    # gate thresholds from separate calibration files (bal_cell, original population, own seed)
    rng_c = np.random.default_rng(12345)
    calib = run_files(rng_c, "bal_cell", pop0, d, a.calib_files, ("Q02",))
    thr = {}
    for g in GATES:
        v = np.array([r[("Q02", "gate", g)] for r in calib])
        thr[g] = {"p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
                  "p99": float(np.percentile(v, 99)), "max": float(v.max())}
    print(f"calibration done {time.time() - t0:.0f}s", flush=True)

    out = {"what": "c2_headroom (a): transductive balanced decoding on simulated 400-question files of inner "
                   "single-colour pseudo-questions (Q02 OOF probabilities); design-stage numbers, NOT validation",
           "source": "results/prereg05/explore/u1/u1_oof.npz (mono_*)",
           "eligible_players_per_level_k": elig, "gate_thresholds_bal_cell_Q02": thr,
           "settings": vars(a), "scenarios": {}}
    for si, scen in enumerate(a.scen.split(",")):
        rng = np.random.default_rng(1000 + si)
        rows = run_files(rng, scen, pop0, d, a.files, combos)
        res = {"files": len(rows),
               "reuse_slots_mean": float(np.mean([r["reuse_slots"] for r in rows])),
               "reuse_slots_max": int(np.max([r["reuse_slots"] for r in rows])),
               "max_use_max": int(np.max([r["max_use"] for r in rows])),
               "true_count_range_mean": float(np.mean([r["true_counts"].max() - r["true_counts"].min()
                                                       for r in rows])),
               "combos": {}}
        for cb in combos:
            rc = {}
            base_qm = np.array([r[(cb, "B", "qm")] for r in rows])
            base_rb = np.array([r[(cb, "B", "rb")] for r in rows])
            rc["B_qm"] = H.summary(base_qm)
            rc["B_rb"] = H.summary(base_rb)
            for meth in METHODS[1:]:
                gq = np.array([r[(cb, meth, "qm")] for r in rows]) - base_qm
                gr = np.array([r[(cb, meth, "rb")] for r in rows]) - base_rb
                rc[meth] = {"gain_qm": H.summary(gq), "gain_rb": H.summary(gr),
                            "share_files_worse_qm": float((gq < 0).mean())}
            # gated H40 and gated H10k (fallback B) with Q02's calibration thresholds
            gated = {}
            for g in ("chi2_B", "lr_em", "sacrifice_H40"):
                for q in ("p95", "p99"):
                    t = thr[g][q]
                    acc = np.array([r[(cb, "gate", g)] <= t for r in rows])
                    for meth in ("H40", "H10k"):
                        gq = np.where(acc, np.array([r[(cb, meth, "qm")] for r in rows]), base_qm) - base_qm
                        gated[f"{meth}|{g}<={q}"] = {"accept_rate": float(acc.mean()), "gain_qm": H.summary(gq)}
            rc["gated"] = gated
            rc["gate_stats"] = {g: H.summary([r[(cb, "gate", g)] for r in rows]) for g in GATES}
            rc["hist_B_mean"] = np.mean([r[(cb, "hist_B")] for r in rows], axis=0).tolist()
            res["combos"][cb] = rc
        # player bootstrap of the mean gain (Q02 only)
        if a.boot > 0:
            rng_b = np.random.default_rng(5000 + si)
            reps = {meth: [] for meth in METHODS[1:]}
            reps_rb = {meth: [] for meth in METHODS[1:]}
            reps_g = {}
            for _ in range(a.boot):
                pop = boot_pop(rng_b, d)
                rr = run_files(rng_b, scen, pop, d, a.boot_files, ("Q02",))
                bq = np.array([r[("Q02", "B", "qm")] for r in rr])
                br = np.array([r[("Q02", "B", "rb")] for r in rr])
                for meth in METHODS[1:]:
                    reps[meth].append(float(np.mean(np.array([r[("Q02", meth, "qm")] for r in rr]) - bq)))
                    reps_rb[meth].append(float(np.mean(np.array([r[("Q02", meth, "rb")] for r in rr]) - br)))
                for g, q in (("lr_em", "p95"), ("chi2_B", "p95"), ("sacrifice_H40", "p95")):
                    acc = np.array([r[("Q02", "gate", g)] <= thr[g][q] for r in rr])
                    gq = np.where(acc, np.array([r[("Q02", "H40", "qm")] for r in rr]), bq) - bq
                    reps_g.setdefault(f"H40|{g}<={q}", []).append(float(gq.mean()))
            res["player_boot_mean_gain_Q02"] = {
                "replicates": a.boot, "files_per_replicate": a.boot_files,
                "qm": {meth: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                                   float(np.percentile(v, 97.5))]}
                       for meth, v in reps.items()},
                "rb": {meth: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                                   float(np.percentile(v, 97.5))]}
                       for meth, v in reps_rb.items()},
                "gated_qm": {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                                      float(np.percentile(v, 97.5))]}
                             for k, v in reps_g.items()}}
        out["scenarios"][scen] = res
        print(f"{scen} done {time.time() - t0:.0f}s: H40 gain qm "
              f"{res['combos']['Q02']['H40']['gain_qm']['mean']:+.4f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    print(H.dump(out, a.out))


if __name__ == "__main__":
    main()
