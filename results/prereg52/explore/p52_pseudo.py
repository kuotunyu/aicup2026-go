"""prereg52 PLAN revision 1: candidate self-retrieval pseudo-pools from the official pub2 / priv2 candidate stores.
CPU only. Reads the two candidate files only (never the official question files); the official pool vector files and
P1 lists are cut to their first 6,000 rows (= the candidate games) right after loading.

    PY results/prereg52/explore/p52_pseudo.py run          # once -> results/prereg52/explore/pseudo.json

System(E term) = CSLS_20( fuse_cfg(pool, E, cfg_v26) + 8 z(T_P1 joint/b4) ), G terms as p43_eval.official_rec.
A49 = c49 alone (= v2-7's system); ENS49_S0 = c49 + S0 score-level (p52_eval.ens_E); S0 alone (record only).
Gate (revision 1, R1.3): mean(ENS49_S0 - A49) >= +0.01, bootstrap over the 10 pseudo-pools (10,000, seed 43) 95% low
> 0, higher in >= 8 of 10. Nothing here opens the validation-answer folder, the E pools, or runs tools/score.py.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p52_eval as P52  # noqa: E402

E45, E43, P39E, BK, EV, st, L = P52.E45, P52.E43, P52.P39E, P52.BK, P52.EV, P52.st, P52.L
REPO = P52.REPO
SRC = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
OUT = os.path.join(HERE, "pseudo.json")
POOLS = {"public": ("pub2", 52000), "private": ("priv2", 52100)}
SPLITS = (0, 1, 2, 3, 4)
K_LIST = (1, 2, 3, 5)
N_OWN_PER_K = 100
VEC = {
    "c49": (os.path.join(REPO, "data", "prereg49", "emb", "V2_step_320000_raw_tta1"),
            "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"),
    "S0": (os.path.join(REPO, "data", "prereg43", "emb_S0", "V2_step_320000_raw_tta1"),
           "49c1c8f10c927922ca9f3ab3f7d2aaa37a24ba852d6acf77d3042ce1fd495939"),
}
GATE_MEAN, GATE_POOLS = 0.01, 8


def read_candidates(pool):
    """Candidate file only -> items [(sgf, colour)] in file order, sorted ids, rows per sorted id."""
    import p35_apply as pa
    path = os.path.join(SRC, "player_identification_candidates2_%s.csv" % pool)
    with open(path, encoding="utf-8", newline="") as f:
        cands = list(csv.DictReader(f))
    items, cand_ids, cand_rows, _q, _ids = L.pool_items(cands, [])
    if not all(c.startswith(pa.PREFIX[pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    return items, cand_ids, cand_rows, {"file": E43._rel(path), "sha256": L.sha256_file(path)}


def g_store(items):
    """The candidate games in p35_apply's PoolStore + the fit-store LDA models (as p43_eval.official_rec)."""
    import p35_apply as pa
    fit = E43.sim.Store(L.FEATS, load=())
    pst = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, lm = {}, {}
    for c in ("B", "W"):
        ld = E43.sim.LdaData(L.FEATS, c, fit)
        lm[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(pst, ld.cols)
    del fit
    return pst, ldas, lm


def split(cand_rows, seed):
    """R1.1: owners, held-out question games, reference rows."""
    rng = np.random.default_rng(seed)
    n = len(cand_rows)
    if any(len(r) != 10 for r in cand_rows):
        raise SystemExit("a candidate does not have 10 games")
    perm = rng.permutation(n)
    owners = perm[:N_OWN_PER_K * len(K_LIST)]
    ref = [list(r) for r in cand_rows]
    q_rows, owner, ks = [], [], []
    for i, c in enumerate(owners):
        k = K_LIST[i // N_OWN_PER_K]
        held = rng.choice(np.asarray(cand_rows[c]), size=k, replace=False)
        hs = set(int(x) for x in held)
        q_rows.append([int(x) for x in held])
        ref[c] = [r for r in cand_rows[c] if r not in hs]
        owner.append(int(c))
        ks.append(k)
    return ref, q_rows, np.asarray(owner), np.asarray(ks)


def load_vec(name, pool, n_cand, n_all, colour):
    d, sha = VEC[name]
    path = os.path.join(d, "pool_%s.npz" % pool)
    with np.load(path, allow_pickle=False) as z:
        if str(z["ckpt_sha256"]) != sha or len(z["emb"]) != n_all:
            raise SystemExit("%s: checkpoint / number of rows" % E43._rel(path))
        emb, count = np.array(z["emb"][:n_cand]), np.array(z["count"][:n_cand])
    return EV.Emb(emb, count, None, colour, sha, name), {"file": E43._rel(path), "file_sha256": L.sha256_file(path)}


def cmd_run(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: run once" % a.out)
    for f in (os.path.join(HERE, "PLAN.md"), os.path.abspath(__file__), os.path.join(HERE, "p52_eval.py")):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % E43._rel(f))
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    Lset = P39E.load_set("L")
    models = P39E.fit_models(Lset)
    meta_L = Lset["meta"]
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    rec = {"format": "c10-prereg52 pseudo v1", "git_head": L.git_head(), "plan_revision": 1, "cfg_v26": cfg,
           "wP": wP, "csls_k": k, "lda_sha256": base["lda_sha256"], "pools": [], "inputs": {}}
    per = {"A49": [], "ENS49_S0": [], "S0": []}
    byk = {x: [] for x in per}
    for pool, (pname, seed0) in POOLS.items():
        items, cand_ids, cand_rows, cinfo = read_candidates(pool)
        n = len(items)
        colour = np.array([c for _s, c in items])
        Pset = P39E.load_set(pname)
        if Pset["meta"] != meta_L:
            raise SystemExit("L and %s chunks have different metadata" % pname)
        n_all = len(Pset["games"])
        if any(g["item"] != i or g["color"] != colour[i] for i, g in enumerate(Pset["games"][:n])):
            raise SystemExit("%s P1 list does not match the candidate games" % pname)
        X, usable = np.array(Pset["X"][:n]), np.array(Pset["usable"][:n])
        del Pset
        Zs = P39E.project_all(X, models)
        embs, vinfo = {}, {}
        for name in ("c49", "S0"):
            embs[name], vinfo[name] = load_vec(name, pool, n, n_all, colour)
        rec["inputs"][pool] = {"candidates": cinfo, "vectors": vinfo, "games": n, "rows_in_files": n_all,
                               "games_without_P1": int((~usable).sum())}
        pst, ldas, lm = g_store(items)
        for s in SPLITS:
            ref, q_rows, owner, ks = split(cand_rows, seed0 + s)
            p = {"cand_rows": ref, "q_rows": q_rows, "owner": owner, "k": ks}
            p["T"] = E43.sim.pool_terms(pst, p, ldas, lm, None, want=("O", "JO", "S2", "L", "PA"))
            z, has = P39E.p1_terms(X, usable, colour, ref, q_rows, models, E43.P1_PW, Zs=Zs)[E43.P1_VARIANT]
            E = {name: BK.e_term(p, embs[name], cfg) for name in embs}
            systems = {"A49": E["c49"], "ENS49_S0": P52.ens_E([E["c49"], E["S0"]]), "S0": E["S0"]}
            row = {"pool": pool, "split": s, "seed": seed0 + s, "questions": len(q_rows),
                   "questions_without_P1": int((~np.asarray(has)).sum()), "scores": {}}
            for name, Et in systems.items():
                S = st.csls(BK.fuse_cfg(p, Et, cfg) + wP * z, k)
                sc = float(st.score(S, owner))
                bk = EV.score_by_k(S, p)
                per[name].append(sc)
                byk[name].append(bk)
                row["scores"][name] = {"score": sc, "by_k": bk}
            rec["pools"].append(row)
            print(json.dumps({"pool": pool, "split": s, **{x: row["scores"][x]["score"] for x in systems}}), flush=True)
        del pst, ldas, lm, embs, X, Zs
    rec["means"] = {x: float(np.mean(v)) for x, v in per.items()}
    rec["by_k"] = {x: EV.mean_by_k(v) for x, v in byk.items()}
    rec["per_pool"] = per
    r = E43.diff_rec(per["ENS49_S0"], per["A49"])
    r["gate"] = bool(r["mean"] >= GATE_MEAN and r["ci95"][0] > 0 and r["pools_higher"] >= GATE_POOLS)
    rec["ENS49_S0_minus_A49"] = r
    rec["S0_minus_A49_record_only"] = E43.diff_rec(per["S0"], per["A49"])
    rec["gate_rule"] = ("mean(ENS49_S0 - A49) >= %.2f, pseudo-pool bootstrap (10,000, seed 43) 95%% low > 0, higher in "
                        ">= %d of 10" % (GATE_MEAN, GATE_POOLS))
    rec["deferred"] = "ENS49_S0_S1: S1 official-pool vectors do not exist yet (PLAN revision 1, R1.2)"
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "by_k", "ENS49_S0_minus_A49", "S0_minus_A49_record_only",
                                          "peak_working_set_mb", "seconds")}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    BK.polite()
    BK.wait_for_ram(need_mb=P52.RAM_NEED_MB)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
