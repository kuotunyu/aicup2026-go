"""prereg49 explore (PLAN.md 3, 4): the one E judgement of c49 (the c* recipe retrained on T49) with v2-6's backend and
P1 term unchanged; official pools. CPU only.

    PY results/prereg49/explore/p49_eval.py seam                    # the judge code path on c*'s E vectors == v2-6
    PY results/prereg49/explore/p49_eval.py judge --e-use 6         # once -> eval.json
    PY results/prereg49/explore/p49_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg49/explore/p49_eval.py assemble --src <official tests dir>

System(vectors) = CSLS_k( fuse_cfg(pool, e_term(pool, emb, cfg), cfg) + wP z(T_P1 joint/b4) ) with cfg, wP, k = v2-6's
(results/prereg43/explore/tune.json c_star) - nothing is tuned (D is in T49).
c49 = data/prereg49/emb/V2_step_320000_raw_tta1/<store_E | pool_public | pool_private>.npz: checkpoint =
data/prereg49/S0/ckpt/step_320000.pt (sha256 checked), embed settings = c*'s except the checkpoint, the same games /
counts / store rows as c*'s file; written after the commit of PLAN.md and of this file.
judge: adopt if mean(c49 - v2-6) >= +0.01, pool bootstrap (10,000, seed 43) 95% low > 0, higher in >= 18 of 20.
apply: v2-6 rebuilt on the pool == results/prereg43/apply/<pool>/top5.csv byte for byte first.
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P46 = os.path.join(REPO, "results", "prereg46", "explore")
for p in (P46, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p46_eval as E46  # noqa: E402

E43, E45, P39E = E46.E43, E46.E45, E46.P39E
BK, EV, st, L = E43.BK, E43.EV, E43.st, E43.L
_abs, _rel, read_json, peak_mb = E43._abs, E43._rel, E43.read_json, E45.peak_mb

E_V26 = E45.E_V26
V26_APPLY = E45.V26_APPLY
V1_SUBMISSION = E45.V1_SUBMISSION
DATA = os.path.join(REPO, "data", "prereg49")
CKPT = os.path.join(DATA, "S0", "ckpt", "step_%d.pt" % E43.STEP)
C49_DIR = os.path.join(DATA, "emb", "V2_step_%d_raw_tta1" % E43.STEP)
PLAN = os.path.join(HERE, "PLAN.md")
EVAL_JSON = os.path.join(HERE, "eval.json")
SEAM_JSON = os.path.join(HERE, "seam.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg49", "apply")
ADOPT_MEAN = 0.01
COMP1_E_JUDGES = E43.COMP1_E_JUDGES + ("prereg43", "prereg45", "prereg46", "prereg47", "prereg48", "prereg49")
RAM_NEED_MB = 9000


def c49_file(which):
    name = "store_%s.npz" % which if which in ("D", "E") else "pool_%s.npz" % which
    return os.path.join(C49_DIR, name)


def cstar_ref(which):
    return E43.member_file("S0", "raw", which)


def ckpt_sha():
    if not os.path.exists(CKPT):
        raise SystemExit("missing %s" % _rel(CKPT))
    return L.sha256_file(CKPT)


def check_c49_file(path, which, cstar):
    """c49 vectors: from data/prereg49's last checkpoint with c*'s embed settings, on the games of c*'s file."""
    if not os.path.exists(path):
        raise SystemExit("missing %s" % _rel(path))
    sha = ckpt_sha()
    cfg = E43.embed_cfg_of(path) or {}
    ref = cstar_ref(which)
    rcfg = E43.embed_cfg_of(ref) or {}
    problems = []
    k49 = {k: v for k, v in (E43.cfg_key(cfg) or {}).items() if k != "ckpt_sha256"}
    kc = {k: v for k, v in (E43.cfg_key(cstar["embed_cfg"]) or {}).items() if k != "ckpt_sha256"}
    if k49 != kc:
        problems.append("embed settings %r != c*'s %r" % (k49, kc))
    if cfg.get("tta") != 1 or cfg.get("weights") != "raw":
        problems.append("tta %r weights %r" % (cfg.get("tta"), cfg.get("weights")))
    if (cfg.get("games"), cfg.get("moves")) != (rcfg.get("games"), rcfg.get("moves")):
        problems.append("games / moves differ from c*'s file")
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != sha or cfg.get("ckpt_sha256") != sha:
            problems.append("checkpoint sha %s != %s" % (str(z["ckpt_sha256"]), sha))
        if int(z["ckpt_step"]) != E43.STEP:
            problems.append("checkpoint step %s" % z["ckpt_step"])
        if not np.array_equal(z["count"], r["count"]):
            problems.append("count differs from c*'s file")
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in z.files) != (k in r.files) or (k in z.files and not np.array_equal(z[k], r[k])):
                problems.append("%s differs from c*'s file" % k)
    if problems:
        raise SystemExit("%s: %s" % (_rel(path), "; ".join(problems)))
    return {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt": _rel(CKPT), "ckpt_sha256": sha,
            "embed_cfg": cfg, "same_games_as": _rel(ref)}


def e_context(base):
    """E pools (players all in E), v2-6's fused part, the P1 z terms, v2-6's per-pool scores (checked bit for bit)."""
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    Fb, vec = E45.base_fused(base, "E", pools)
    ctx = E43.p1_ctx("E")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    bper = E45.seam_scores(base, "E", Fb, zs, pools)
    del Fb
    return pools, zs, bper, vec, sets


def system(path, pools, zs, base):
    e = E43.load_emb(path)
    Fs = E43.fused(pools, e, base["cfg"])
    del e
    per = E43.scores(Fs, zs, base["wP"], base["k"], pools)
    byk = [EV.score_by_k(st.csls(F + base["wP"] * z, base["k"]), p) for F, z, p in zip(Fs, zs, pools)]
    return per, byk


def cmd_seam(a):
    t0 = time.time()
    base = E45.v26()
    pools, zs, bper, vec, _sets = e_context(base)
    path, _c = E43.cstar_file(base["tune"], "E")
    per, _byk = system(path, pools, zs, base)
    ok = per == list(bper) and float(np.mean(per)) == E_V26
    rec = {"format": "c10-prereg49 seam v1", "rule": "the judge's system() on c*'s E vectors == v2-6 per pool, bit for bit",
           "per_pool": per, "mean": float(np.mean(per)), "want_mean": E_V26, "identical": ok, "vectors": vec,
           "git_head": L.git_head(), "peak_working_set_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("mean", "identical", "seconds", "peak_working_set_mb")}))
    if not ok:
        raise SystemExit("seam failed")
    return 0


def e_judgements_so_far(repo=REPO):
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    for f in (PLAN, os.path.abspath(__file__)):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))
    prior = e_judgements_so_far()
    if "prereg49" in prior:
        raise SystemExit("prereg49 already has an E judgement")
    if a.e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, len(prior), prior))
    base = E45.v26()
    cs = base["tune"]["c_star"]
    path = c49_file("E")
    vec49 = check_c49_file(path, "E", cs)
    E43.made_after([path], max(E43.commit_time(PLAN), E43.commit_time(os.path.abspath(__file__))))
    E43.check_players(path, "E")
    pools, zs, bper, vec, sets = e_context(base)
    per, byk = system(path, pools, zs, base)
    rec = {"format": "c10-prereg49 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "e_vectors": vec49, "base_vectors_E": vec,
           "base": base["record"], "cfg": base["cfg"], "wP": base["wP"], "k": base["k"],
           "seam_base_E_identical": True, "git_head": L.git_head(), "p1_sets": sets,
           "means": {"c49": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"c49": per, "base": list(bper)}, "by_k": {"c49": EV.mean_by_k(byk)}}
    j = E43.judge_rule(per, bper)
    rec["c49_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "c49")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c49_minus_base", "by_k", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    t0 = time.time()
    ev = read_json(a.eval)
    if not ev["adopt"]:
        raise SystemExit("eval.json does not adopt c49: keep v2-6")
    base = E45.v26()
    cs = base["tune"]["c_star"]
    rec, items, cand_ids, qids = E43.official_rec(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if Pset["meta"] != Lset["meta"]:
        raise SystemExit("L and pool chunks have different metadata")
    bad = int((~Pset["usable"]).sum())
    if bad > P39E.FAIL_MAX * len(items):
        raise SystemExit("%d of %d pool games have no usable P1 (> 0.5%%): stop" % (bad, len(items)))
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    colour = np.array([c for _s, c in items])
    z, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                           E43.P1_PW)[E43.P1_VARIANT]
    del models

    def top5_of(e):
        F = BK.fuse_cfg(rec, BK.e_term(rec, e, base["cfg"]), base["cfg"])
        return P39E.top5_bytes(qids, st.top5(st.csls(F + base["wP"] * z, base["k"]), cand_ids))

    path_b, _choice = E43.cstar_file(base["tune"], a.pool)
    data_b = top5_of(E43.pool_emb(path_b, items, cs["ckpt_sha256"]))
    with open(os.path.join(V26_APPLY, a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: v2-6 rebuilt on %s differs from results/prereg43/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    path = c49_file(a.pool)
    vec49 = check_c49_file(path, a.pool, cs)
    e = E43.pool_emb(path, items, vec49["ckpt_sha256"])
    data = top5_of(e)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg49 apply v1", "pool": a.pool, "vectors": vec49, "cfg": base["cfg"],
           "wP": base["wP"], "k": base["k"], "v26_top5_equals_prereg43_apply": True, "questions": len(qids),
           "candidates": len(cand_ids), "games": len(items), "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": st.sha256_lf(a.eval), "top5_sha256": L.sha256_file(out),
           "differs_from_v26_rows": int(sum(x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return E43.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--out", default=SEAM_JSON)
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--eval", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
