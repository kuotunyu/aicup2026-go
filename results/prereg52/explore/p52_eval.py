"""prereg52 explore (PLAN.md): D development stage only - score-level ensembles and three backend modules on c*.
CPU only; the E pools are never loaded here.

    PY results/prereg52/explore/p52_eval.py dev            # -> results/prereg52/explore/dev.json
    PY results/prereg52/explore/p52_eval.py report         # dev.json -> RESULTS.md tables (stdout)

System(E term) = CSLS_20( fuse_cfg(pool, E, cfg_v26) + 8 z(T_P1 joint/b4) ), cfg_v26 = prereg43 tune.json c_star.backend.cfg.
Seams (all bit for bit, else stop):
  1. v2-6 on D rebuilt by prereg45's base_fused / seam_scores (backend-only and with P1) == prereg43 tune.json;
  2. the ensemble code with the single member S0 == v2-6 per pool;
  3. the per-k code with (wE_k, wP_k) = (8, 8) for every k == v2-6 per pool.
Candidates, in the fixed order of PLAN.md 2: ENS2, ENS3, PK, B1a, B1b, B3b_top3, B3b_fuse. Gate (PLAN.md 3): mean
difference to v2-6 >= +0.01, pool bootstrap (10,000, seed 43) 95% low > 0, higher in >= 8 of 10 pools.
Record only: S1 alone, M37 alone, concatenation ensembles (p43_ens.py's operation) with the v2-6 backend.
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P45 = os.path.join(REPO, "results", "prereg45", "explore")
for p in (P45, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p45_eval as E45  # noqa: E402   (v26(), base_fused, seam_scores, peak_mb; unchanged)

E43 = E45.E43
P39E, Q42 = E43.P39E, E43.Q42
BK, EV, st, L = E43.BK, E43.EV, E43.st, E43.L
pe, ps = BK.pe, BK.ps

D_V26 = E45.D_V26
DEV_JSON = os.path.join(HERE, "dev.json")
RAM_NEED_MB = 9000
GATE_MEAN, GATE_POOLS = 0.01, 8
HALVES = ((100, 101, 102, 103, 104), (105, 106, 107, 108, 109))
KS = ("1", "2", "3", "5")
PK_GRID = [(8, 8)] + [(a, b) for a in (4, 8, 16) for b in (4, 8, 16) if (a, b) != (8, 8)]
MEMBERS = {
    "S0": os.path.join(REPO, "data", "prereg43", "emb_S0", "V2_step_320000_raw_tta1", "store_D.npz"),
    "S1": os.path.join(REPO, "data", "prereg43", "emb_S1", "V2_step_320000_raw_tta1", "store_D.npz"),
    "M37": os.path.join(REPO, "data", "prereg37", "emb", "step_160000", "store_D.npz"),
}
CANDIDATES = ("ENS2", "ENS3", "PK", "B1a", "B1b", "B3b_top3", "B3b_fuse")
MODULE_CFG = {
    "B1a": {"centre_colour": True},
    "B1b": {"sim": "colour", "alpha": 0.5},
    "B3b_top3": {"sim": "pair", "pair": {"agg": "top", "m": 3}},
}
LSE = {"sim": "pair", "pair": {"agg": "lse", "tau": 0.2}}


# ================================================================ helpers
def ens_E(Es):
    """Score-level ensemble of E terms [(M, ok)]: one member -> itself (bit for bit); else the mean of the row-z
    matrices, ok = intersection (PLAN.md 2.1)."""
    if len(Es) == 1:
        return Es[0]
    Z = np.mean([st.zscore_rows(M, ok) for M, ok in Es], axis=0)
    ok = np.logical_and.reduce([np.asarray(ok, bool) for _M, ok in Es])
    return Z, ok


def run_system(pools, Es, zs, cfg, wP, k):
    """Per-pool scores and by-k means of CSLS_k(fuse_cfg(pool, E, cfg) + wP z_P1)."""
    per, byk = [], []
    for p, E, z in zip(pools, Es, zs):
        S = st.csls(BK.fuse_cfg(p, E, cfg) + wP * z, k)
        per.append(float(st.score(S, p["owner"])))
        byk.append(EV.score_by_k(S, p))
    return per, byk


def g_part(pool):
    """sum_n w_n z(T_n) in the order of st.fuse (the G part of fuse_cfg's scalar path)."""
    w, _kG = pe.g_combo()
    return st.fuse([pool["T"][n] for n in ps.TERMS], list(w))


def pk_scores(pool, Gp, E, z, setting, k):
    """F = G part + wE_k z(E) + wP_k z(P1), row weights by the question's number of games; CSLS k."""
    ks = BK.q_sizes(pool)
    we = BK._per_k({x: setting[x][0] for x in KS}, ks, KS)
    wp = BK._per_k({x: setting[x][1] for x in KS}, ks, KS)
    F = Gp + st.zscore_rows(*E) * we[:, None]
    S = st.csls(F + z * wp[:, None], k)
    return S


def load_member(name):
    path = MEMBERS[name]
    E43.check_players(path, "D")
    e = E43.load_emb(path)
    return e, {"file": E43._rel(path), "file_sha256": L.sha256_file(path), "ckpt": e.name, "ckpt_sha256": e.sha}


def concat_emb(names):
    """p43_ens.py's operation in memory: per-model L2 (float64), concatenation / sqrt(m), float32; count = min."""
    zs = [np.load(MEMBERS[n], allow_pickle=False) for n in names]
    ref = zs[0]
    for z in zs[1:]:
        for key in ("store_row", "players_sha256_sorted_ids"):
            if not np.array_equal(ref[key], z[key]):
                raise SystemExit("concat: %s differs" % key)
    parts = []
    for z in zs:
        e = z["emb"].astype(np.float64)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        parts.append(np.where(n > 0, e / np.where(n > 0, n, 1), 0.0))
        del e
    emb = (np.concatenate(parts, axis=1) / np.sqrt(len(zs))).astype(np.float32)
    del parts
    count = np.min(np.stack([z["count"] for z in zs]), axis=0)
    rows = ref["store_row"]
    for z in zs:
        z.close()
    return EV.Emb(emb, count, rows, E43.store().color, "concat", "+".join(names))


def diff(per, base_per):
    r = E43.diff_rec(per, base_per)          # pool bootstrap 10,000, seed 43 (prereg42's bootstrap)
    r["gate"] = bool(r["mean"] >= GATE_MEAN and r["ci95"][0] > 0 and r["pools_higher"] >= GATE_POOLS)
    return r


def row(name, per, byk, base_per, gated, extra=None):
    r = {"name": name, "gated": gated, "mean": float(np.mean(per)), "per_pool": per, "by_k": EV.mean_by_k(byk),
         "minus_v26": diff(per, base_per)}
    if extra:
        r.update(extra)
    print(json.dumps({"name": name, "mean": r["mean"], "minus_v26": r["minus_v26"]}), flush=True)
    return r


# ================================================================ dev
def cmd_dev(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the D stage is run once" % a.out)
    for f in (os.path.join(HERE, "PLAN.md"), os.path.abspath(__file__)):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % E43._rel(f))
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    if [p["seed"] for p in pools] != [s for h in HALVES for s in h]:
        raise SystemExit("D pool seeds")
    # P1 term (shared by every system), as prereg43 / 45
    ctx = E43.p1_ctx("D")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    p1_sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    # seam 1: v2-6 on D (backend-only and with P1) by prereg45's code
    Fb, vec = E45.base_fused(base, "D", pools)
    base_per = E45.seam_scores(base, "D", Fb, zs, pools)
    del Fb
    rec = {"format": "c10-prereg52 dev v1", "git_head": L.git_head(), "d_seeds": EV.D_SEEDS, "base": base["record"],
           "base_vectors_D": vec, "p1_sets": p1_sets, "cfg_v26": cfg, "wP": wP, "csls_k": k,
           "v26_D": {"per_pool": base_per, "mean": float(np.mean(base_per)), "identical_to_tune_json": True},
           "seams": {}, "members": {}, "candidates": {}, "record_only": {}}
    # members' E terms (v2-6 backend, each member separately)
    E_of, extra_S0 = {}, {}
    for name in ("S0", "S1", "M37"):
        e, info = load_member(name)
        if name == "S0" and e.sha != base["tune"]["c_star"]["ckpt_sha256"]:
            raise SystemExit("S0 is not c*")
        E_of[name] = [BK.e_term(p, e, cfg) for p in pools]
        if name == "S0":
            for lab, over in MODULE_CFG.items():
                extra_S0[lab] = [BK.e_term(p, e, BK.with_(cfg, **over)) for p in pools]
            extra_S0["LSE"] = [BK.e_term(p, e, BK.with_(cfg, **LSE)) for p in pools]
        rec["members"][name] = info
        del e
        print(json.dumps({"member": name, "peak_mb": E45.peak_mb()}), flush=True)
    # seam 2: ensemble code, single member S0 == v2-6
    per, _b = run_system(pools, [ens_E([E]) for E in E_of["S0"]], zs, cfg, wP, k)
    if per != list(base_per):
        raise SystemExit("seam 2: the ensemble path with S0 alone differs from v2-6")
    rec["seams"]["ensemble_S0_alone_equals_v26"] = True
    # seam 3: per-k code at (8, 8) == v2-6
    Gps = [g_part(p) for p in pools]
    eight = {x: (8, 8) for x in KS}
    per = [float(st.score(pk_scores(p, G, E, z, eight, k), p["owner"])) for p, G, E, z in
           zip(pools, Gps, E_of["S0"], zs)]
    if per != list(base_per):
        raise SystemExit("seam 3: the per-k path at (8, 8) differs from v2-6")
    rec["seams"]["per_k_8_8_equals_v26"] = True
    print(json.dumps({"seams": rec["seams"], "v26_D": rec["v26_D"]["mean"]}), flush=True)

    C = rec["candidates"]
    # ENS2, ENS3
    for lab, names in (("ENS2", ("S0", "S1")), ("ENS3", ("S0", "S1", "M37"))):
        Es = [ens_E([E_of[n][i] for n in names]) for i in range(len(pools))]
        per, byk = run_system(pools, Es, zs, cfg, wP, k)
        C[lab] = row(lab, per, byk, base_per, True, {"members": list(names)})
    # PK (cross-fitted, PLAN.md 2.2)
    idx = {p["seed"]: i for i, p in enumerate(pools)}
    chosen = {}
    for h, half in enumerate(HALVES):
        setting, grids = {}, {}
        for kk in KS:
            rows_k = []
            for g in PK_GRID:
                s = dict(eight)
                s[kk] = g
                vals = []
                for seed in half:
                    i = idx[seed]
                    S = pk_scores(pools[i], Gps[i], E_of["S0"][i], zs[i], s, k)
                    vals.append(EV.score_by_k(S, pools[i])[kk])
                rows_k.append({"wE": g[0], "wP": g[1], "mean_k": float(np.mean(vals))})
            best = E43.first_max(rows_k, key=lambda r: r["mean_k"])
            setting[kk] = (best["wE"], best["wP"])
            grids[kk] = rows_k
        chosen["chosen_on_%d_%d" % (half[0], half[-1])] = {"setting": {x: list(v) for x, v in setting.items()},
                                                          "grid": grids}
    per, byk = [0.0] * len(pools), [None] * len(pools)
    for h, half in enumerate(HALVES):
        other = HALVES[1 - h]
        s = chosen["chosen_on_%d_%d" % (other[0], other[-1])]["setting"]
        for seed in half:
            i = idx[seed]
            S = pk_scores(pools[i], Gps[i], E_of["S0"][i], zs[i], {x: tuple(v) for x, v in s.items()}, k)
            per[i] = float(st.score(S, pools[i]["owner"]))
            byk[i] = EV.score_by_k(S, pools[i])
    C["PK"] = row("PK", per, byk, base_per, True, {"cross_fit": chosen})
    # modules
    for lab in ("B1a", "B1b", "B3b_top3"):
        per, byk = run_system(pools, extra_S0[lab], zs, cfg, wP, k)
        C[lab] = row(lab, per, byk, base_per, True, {"cfg_override": MODULE_CFG[lab]})
    Es = [ens_E([P, Q]) for P, Q in zip(E_of["S0"], extra_S0["LSE"])]
    per, byk = run_system(pools, Es, zs, cfg, wP, k)
    C["B3b_fuse"] = row("B3b_fuse", per, byk, base_per, True, {"fused_with": LSE})
    del extra_S0
    # record only: single members, concatenation ensembles
    R = rec["record_only"]
    for name in ("S1", "M37"):
        per, byk = run_system(pools, E_of[name], zs, cfg, wP, k)
        R[name + "_alone"] = row(name + "_alone", per, byk, base_per, False)
    del E_of
    for names in (("S0", "S1"), ("S0", "S1", "M37")):
        e = concat_emb(names)
        Es = [BK.e_term(p, e, cfg) for p in pools]
        del e
        per, byk = run_system(pools, Es, zs, cfg, wP, k)
        lab = "CONCAT_" + "".join(names)
        R[lab] = row(lab, per, byk, base_per, False, {"members": list(names)})
        print(json.dumps({"concat": lab, "peak_mb": E45.peak_mb()}), flush=True)
    # gate summary (PLAN.md 3)
    passed = [c for c in CANDIDATES if C[c]["minus_v26"]["gate"]]
    best = E43.first_max(passed, key=lambda c: C[c]["mean"]) if passed else None
    rec["gate"] = {"rule": "mean - v2-6 D >= %.2f, pool bootstrap (10,000, seed 43) 95%% low > 0, higher in >= %d of 10"
                   % (GATE_MEAN, GATE_POOLS), "order": list(CANDIDATES), "passed": passed, "to_E": best}
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps(rec["gate"], indent=1))
    return 0


# ================================================================ report
def cmd_report(a):
    r = E43.read_json(a.dev)
    print("v2-6 D %.10f\n" % r["v26_D"]["mean"])
    print("| 代號 | D 平均 | 對 v2-6 | 95% 區間 | 較高池數 | 閘門 | 1 盤 | 2 盤 | 3 盤 | 5 盤 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    b = {"name": "v2-6", "mean": r["v26_D"]["mean"]}
    print("| v2-6 | %.4f | — | — | — | — | | | | |" % b["mean"])
    for sect in ("candidates", "record_only"):
        for name, x in r[sect].items():
            d = x["minus_v26"]
            print("| %s | %.4f | %+.4f | [%+.4f, %+.4f] | %d/10 | %s | %.3f | %.3f | %.3f | %.3f |" % (
                name + ("" if x["gated"] else "（只記錄）"), x["mean"], d["mean"], d["ci95"][0], d["ci95"][1],
                d["pools_higher"], ("過" if d["gate"] else "沒過") if x["gated"] else "—",
                *[x["by_k"][kk] for kk in KS]))
    print("\n每池（池 100–109）：")
    print("| 代號 | " + " | ".join(str(s) for s in r["d_seeds"]) + " |")
    print("|---|" + "---|" * len(r["d_seeds"]))
    print("| v2-6 | " + " | ".join("%.4f" % v for v in r["v26_D"]["per_pool"]) + " |")
    for sect in ("candidates", "record_only"):
        for name, x in r[sect].items():
            print("| %s | " % name + " | ".join("%.4f" % v for v in x["per_pool"]) + " |")
    print("\nPK 交叉挑選：")
    for key, v in r["candidates"]["PK"]["cross_fit"].items():
        print(key, json.dumps(v["setting"]))
    print("\ngate:", json.dumps(r["gate"], ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--out", default=DEV_JSON)
    r = sub.add_parser("report")
    r.add_argument("--dev", default=DEV_JSON)
    a = ap.parse_args(argv)
    if a.cmd == "dev":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"dev": cmd_dev, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
