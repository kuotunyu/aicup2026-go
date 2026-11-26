"""Independent review of the D2 PILOT, part A: recompute from the saved arrays (pilot_arrays.npz + u1_oof.npz).
Own implementations of rule B, the official-type metric, the PlayerBootstrap draws, theta, triggers and the pools.
Nothing scored against validation; data/local guarded by u1_lib; writes only into this _review folder.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
EXPLORE = os.path.dirname(PILOT)
sys.path.insert(0, os.path.join(EXPLORE, "u1"))
import u1_lib as U  # noqa: E402  guard first

import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

E1 = np.exp(-1.0)
NS = np.array([5, 10, 15, 20])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ruleb(P):
    ex = P.copy()
    ex[:, 1:] += E1 * P[:, :-1]
    ex[:, :-1] += E1 * P[:, 1:]
    return np.argmax(ex, 1)


def score(pred, y):
    d = np.abs(pred - y)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def off_type(sc, y, kidx):
    lv = []
    for i in range(10):
        cells = []
        for j in range(4):
            m = (y == i) & (kidx == j)
            if m.any():
                cells.append(sc[m].mean())
        lv.append(np.mean(cells))
    return float(np.mean(lv))


def rank_bal(sc, y):
    return float(np.mean([sc[y == i].mean() for i in range(10)]))


def my_draws(player_level, n_boot=1000):
    rng = np.random.default_rng(0)
    out = []
    for i in range(10):
        mem = np.nonzero(player_level == i)[0]
        out.append(mem[rng.integers(0, len(mem), size=(n_boot, len(mem)))])
    return out


def boot_off(sc, player, kidx, draws, NP):
    S = np.zeros((NP, 4))
    C = np.zeros((NP, 4))
    np.add.at(S, (player, kidx), sc)
    np.add.at(C, (player, kidx), 1.0)
    per_level = []
    for d in draws:
        cell_s = S[d].sum(1)
        cell_c = C[d].sum(1)
        per_level.append((cell_s / cell_c).mean(1))
    return np.mean(np.stack(per_level, 0), 0)


def boot_rb(sc, player, draws, NP):
    s = np.bincount(player, weights=sc, minlength=NP)
    c = np.bincount(player, minlength=NP).astype(float)
    return np.mean(np.stack([s[d].sum(1) / c[d].sum(1) for d in draws], 0), 0)


def my_pools(player, kidx, NP, n_pools=1000, per_k=100):
    rng = np.random.default_rng(0)
    qs = {}
    for q in range(len(player)):
        qs.setdefault((int(player[q]), int(kidx[q])), []).append(q)
    pools = np.zeros((n_pools, 4 * per_k), np.int64)
    reused = 0
    for t in range(n_pools):
        order = rng.permutation(NP)
        used = set()
        slots = []
        for j in (3, 2, 1, 0):
            chosen = []
            for p in order:
                p = int(p)
                if p not in used and (p, j) in qs:
                    chosen.append(p)
                    if len(chosen) == per_k:
                        break
            if len(chosen) < per_k:
                for p in order:
                    p = int(p)
                    if p in used and (p, j) in qs and p not in chosen:
                        chosen.append(p)
                        reused += 1
                        if len(chosen) == per_k:
                            break
            used.update(chosen)
            for p in chosen:
                lst = qs[(p, j)]
                slots.append(lst[int(rng.integers(0, len(lst)))])
        pools[t] = slots
    return pools, reused


def main():
    out = {}
    rep = json.load(open(os.path.join(PILOT, "pilot_report.json"), encoding="utf-8"))
    out["sha"] = {"pilot_variants.json": sha(os.path.join(PILOT, "pilot_variants.json")),
                  "report_spec_sha_start": rep["spec"]["sha256_start"],
                  "pilot_arrays.npz": sha(os.path.join(PILOT, "pilot_arrays.npz")),
                  "u1_oof.npz": sha(os.path.join(EXPLORE, "u1", "u1_oof.npz"))}
    z = U.load_saved()
    a = np.load(os.path.join(PILOT, "pilot_arrays.npz"))
    y, n, fold, player = z["mono_y"], z["mono_n"], z["mono_fold"], z["mono_player"]
    kidx = np.searchsorted(NS, n)
    assert (NS[kidx] == n).all()
    NP = len(z["player_id"])
    P02 = z["mono_P_Q02"]
    out["questions"] = int(len(y))
    out["players_with_questions"] = int(len(np.unique(player)))
    # sanity of player/fold/level consistency
    out["player_level_consistent"] = bool((z["player_level"][player] == y).all())
    out["player_fold_consistent"] = bool((z["player_fold"][player] == fold).all())
    mg = z["mono_games"]
    dup = 0
    for q in range(len(mg)):
        g = mg[q][mg[q] >= 0]
        dup += len(g) != len(np.unique(g))
    out["questions_with_duplicate_games"] = int(dup)
    out["games_per_question_equals_k"] = bool(((mg >= 0).sum(1) == n).all())
    out["colour_single"] = bool(all(len(set(z["game_colour"][mg[q][mg[q] >= 0]])) == 1 for q in range(0, len(mg), 7)))
    out["colour_matches_mono_colour_sample"] = bool(all(z["game_colour"][mg[q][0]] == z["mono_colour"][q]
                                                        for q in range(0, len(mg), 7)))

    # Q02 score
    sq = score(ruleb(P02), y)
    out["Q02_score_equal_saved"] = bool(np.array_equal(sq, a["score_Q02"]))
    out["Q02_official_type"] = off_type(sq, y, kidx)
    out["Q02_rank_balanced"] = rank_bal(sq, y)

    # theta
    th = np.zeros((5, 4))
    for f in range(5):
        for j in range(4):
            th[f, j] = np.percentile(a["S2_pi0"][(fold != f) & (kidx == j)], 99)
    out["theta_max_abs_diff_vs_saved"] = float(np.max(np.abs(th - a["theta_fold_by_k"])))
    out["theta_fold_by_k"] = th.tolist()
    out["theta_vs_report_max_abs_diff"] = float(np.max(np.abs(th - np.array(rep["theta_fold_by_k"]))))

    w1 = a["w_pi1"]
    out["w_pi1_saved_equal_report"] = bool(np.array_equal(w1, np.array(rep["w"]["pi1_per_fold"])))
    lvl_q = np.full(NP, -1)
    lvl_q[player] = y
    out["players_without_mono_questions"] = int((lvl_q < 0).sum())
    draws = my_draws(lvl_q)
    res = {}
    for sc in ("pi1", "pi0", "pi05", "pi0_refit"):
        base = "pi0" if sc == "pi0_refit" else sc
        w = a["w_pi0_refit"] if sc == "pi0_refit" else w1
        trig = a[f"S2_{base}"] > th[fold, kidx]
        r = {"trigger_equal_saved": bool(np.array_equal(trig, a[f"trig_{base}"])),
             "trigger_rate": float(trig.mean()), "triggered": int(trig.sum()),
             "E_rows_equal_trigger": bool(np.array_equal(a[f"E_rows_{base}"], np.nonzero(trig)[0]))}
        rows = a[f"E_rows_{base}"]
        E = a[f"E_{base}"]
        lg = np.log(P02[rows]) + w[fold[rows]][:, None] * E
        lg = lg - lg.max(1, keepdims=True)
        Pn = np.exp(lg)
        Pn /= Pn.sum(1, keepdims=True)
        s = sq.copy()
        s[rows] = score(ruleb(Pn), y[rows])
        r["score_mismatches_vs_saved"] = int((s != a[f"score_{sc}"]).sum())
        r["non_triggered_identical_to_Q02"] = bool(np.array_equal(a[f"score_{sc}"][~trig], sq[~trig]))
        ot = off_type(s, y, kidx)
        r["official_type_delta"] = ot - out["Q02_official_type"]
        r["official_type_delta_from_saved_scores"] = off_type(a[f"score_{sc}"], y, kidx) - out["Q02_official_type"]
        d = boot_off(s, player, kidx, draws, NP) - boot_off(sq, player, kidx, draws, NP)
        r["official_type_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        r["rank_balanced_delta"] = rank_bal(s, y) - out["Q02_rank_balanced"]
        d2 = boot_rb(s, player, draws, NP) - boot_rb(sq, player, draws, NP)
        r["rank_balanced_ci95"] = [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]
        rr = rep["results"][sc]
        r["report_official_type"] = [rr["official_type"]["delta"], rr["official_type"]["pboot_ci95"]]
        r["report_rank_balanced"] = [rr["rank_balanced"]["delta"], rr["rank_balanced"]["pboot_ci95"]]
        r["abs_diff_vs_report"] = {
            "ot_delta": abs(r["official_type_delta"] - rr["official_type"]["delta"]),
            "ot_ci": max(abs(r["official_type_ci95"][i] - rr["official_type"]["pboot_ci95"][i]) for i in range(2)),
            "rb_delta": abs(r["rank_balanced_delta"] - rr["rank_balanced"]["delta"]),
            "rb_ci": max(abs(r["rank_balanced_ci95"][i] - rr["rank_balanced"]["pboot_ci95"][i]) for i in range(2))}
        # per-k official-type contribution (record)
        r["per_k_rank_balanced_delta"] = {int(NS[j]): float(np.mean([s[(y == i) & (kidx == j)].mean() for i in range(10)])
                                                           - np.mean([sq[(y == i) & (kidx == j)].mean() for i in range(10)]))
                                          for j in range(4)}
        # triggered-only stats
        if trig.any():
            r["triggered_question_mean_gain"] = float((s[trig] - sq[trig]).mean())
        res[sc] = r
    out["scenarios"] = res

    # pools
    pools, reused = my_pools(player, kidx, NP)
    out["pools_equal_saved"] = bool(np.array_equal(pools, a["pools"].astype(np.int64)))
    out["pools_reused"] = int(reused)
    pp = {}
    for sc in ("pi1", "pi0", "pi05"):
        trig = a[f"S2_{sc}"] > th[fold, kidx]
        frac = trig[pools].mean(1)
        pp[sc] = {"P_ON": float((frac >= 0.10).mean()), "share_mean": float(frac.mean()),
                  "share_p1_p5_p50_p95_p99": [float(x) for x in np.percentile(frac, [1, 5, 50, 95, 99])],
                  "share_max": float(frac.max()),
                  "report_P_ON": rep["pools_400"][sc]["P_ON"]}
    out["pools"] = pp
    out["pool_distinct_players_min"] = int(min(len(np.unique(player[p])) for p in pools))
    out["pool_k_counts_ok"] = bool(all((np.bincount(kidx[p], minlength=4) == 100).all() for p in pools[:50]))

    # criteria
    r1, r0 = res["pi1"], res["pi0"]
    out["criteria_recomputed"] = {
        "c1": bool(r1["official_type_delta"] >= 0.03 and r1["official_type_ci95"][0] > 0),
        "c2": bool(r0["official_type_delta"] >= -0.005 and r0["trigger_rate"] <= 0.02),
        "c3": bool(pp["pi1"]["P_ON"] >= 0.9 and pp["pi0"]["P_ON"] <= 0.01)}
    out["report_criteria"] = {k: rep["criteria"][k]["pass"] for k in ("c1", "c2", "c3")}
    with open(os.path.join(HERE, "review_a.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
