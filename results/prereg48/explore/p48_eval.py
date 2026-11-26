"""prereg48 explore: the CNN vectors of v2-6's checkpoint with TTA-8 (PLAN.md + revision 1). Seam, D selection,
the one E judgement, official pools. CPU only.

    PY results/prereg48/explore/p48_eval.py seam                # -> seam.json (v2-6 rebuilt by the select code path)
    PY results/prereg48/explore/p48_eval.py select              # -> tune.json (commit it before any E vector)
    PY results/prereg48/explore/p48_eval.py judge --e-use 6     # once -> eval.json
    PY results/prereg48/explore/p48_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg48/explore/p48_eval.py assemble --src <official tests dir>

Base = v2-6 (results/prereg43/explore/tune.json + eval.json), rebuilt by prereg45's v26 / base_fused / seam_scores.
A system = (CNN vectors, prereg38 backend cfg, wP, k): S = CSLS_k( fuse_cfg(pool, e_term(pool, emb, cfg), cfg)
+ wP z(T_P1 joint/b4) ), the P1 term = v2-6's (prereg39's L-LDA; it does not depend on the CNN).
system_D(vectors) is prereg43's select step 3 (p43_eval functions, nothing copied): prereg38 retune_structure (kept
['B4'], tune_D_B4 structure, start wE 4 / CSLS 20) on D, then (wP, k) over WP_GRID x K_GRID (first maximum) -> R.
seam: system_D(the prereg43 c* tta1 store_D) must give prereg43 tune.json's backend cfg, backend per-pool scores,
  (wP, k) and R = 0.7467766213119217 bit for bit, pool by pool.
select: c48 = data/prereg48/emb_S0/V2_step_320000_raw_tta8/store_D.npz (p42_embed tag <run>_<stem>_<weights>_tta<n>).
  Checks before any score: tta 8, raw, prereg43 c*'s checkpoint sha / spec / chunk / games / moves; the same games
  (store_row, count) as the tta1 file; the vector seam of revision 1 computed here (not read from embed_cfg):
  data/prereg48/seam_tta1/V2_step_320000_raw_tta1/store_D_limit200.npz vs the first 200 games of the prereg43 tta1
  store_D: count equal, per-game cosine min >= 0.99999, max |diff| / median norm (of the reference) <= 1e-3.
  Gate: R(c48) - 0.7467766213119217 >= 0.01. Stops if any prereg48 E vector / pool vector / E or official result exists.
judge (E pools, use 6): c48 system vs v2-6; adopt if mean >= +0.01, pool bootstrap (10,000, seed 43) 95% low > 0,
  higher in >= 18 of 20. tune.json committed, the gate passed, store_E written after the commit of tune.json.
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
import p46_eval as E46  # noqa: E402   (imports p45 / p43 / p39; check_base_unchanged)

E43, E45, P39E = E46.E43, E46.E45, E46.P39E
BK, EV, st, L, Q42 = E43.BK, E43.EV, E43.st, E43.L, E43.Q42
_abs, _rel, read_json, peak_mb = E43._abs, E43._rel, E43.read_json, E45.peak_mb

D_V26, E_V26 = E45.D_V26, E45.E_V26
V26_APPLY = E45.V26_APPLY
V1_SUBMISSION = E45.V1_SUBMISSION
DATA = os.path.join(REPO, "data", "prereg48")
TAG8 = "V2_step_%d_raw_tta8" % E43.STEP                     # p42_embed: <run>_<ckpt stem>_<weights>_tta<n>
C48_DIR = os.path.join(DATA, "emb_S0", TAG8)
SEAM_VEC = os.path.join(DATA, "seam_tta1", "V2_step_%d_raw_tta1" % E43.STEP, "store_D_limit200.npz")
SEAM_N = 200
COS_MIN = 0.99999                                          # revision 1
REL_MAX = 1e-3                                             # revision 1: max |diff| / median norm of the reference
SEAM_JSON = os.path.join(HERE, "seam.json")
TUNE_JSON = os.path.join(HERE, "tune.json")
EVAL_JSON = os.path.join(HERE, "eval.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg48", "apply")
GATE = 0.01
COMP1_E_JUDGES = E43.COMP1_E_JUDGES + ("prereg43", "prereg45", "prereg46", "prereg47", "prereg48")
RAM_NEED_MB = 9000


# ================================================================ files
def c48_file(which):
    """which: D, E (store) or public, private (official pools)."""
    name = "store_%s.npz" % which if which in ("D", "E") else "pool_%s.npz" % which
    return os.path.join(C48_DIR, name)


def tta1_file(which):
    """prereg43 c*'s (S0 raw tta1) file of the same games."""
    return E43.member_file("S0", "raw", which)


def later_outputs():
    """prereg48 results / vectors that must not exist before select (PLAN.md 1, 3)."""
    found = []
    for w in ("E", "public", "private"):
        p = c48_file(w)
        found += [x for x in (p, p + ".parts") if os.path.exists(x)]
    found += glob.glob(os.path.join(DATA, "*", "*", "store_E*")) + glob.glob(os.path.join(DATA, "*", "*", "pool_*"))
    found += [x for x in (EVAL_JSON, OUT_APPLY) if os.path.exists(x)]
    return sorted(set(os.path.abspath(x) for x in found))


def no_later_outputs():
    found = later_outputs()
    if found:
        raise SystemExit("prereg48 E / pool vectors or results exist before select: %s" % [_rel(p) for p in found])


def check_c48_file(path, which, cstar):
    """The TTA-8 file was made from c*'s checkpoint with c*'s settings except tta, on the games of the tta1 file."""
    if not os.path.exists(path):
        raise SystemExit("missing %s (the 4090 step of run_embed.sh)" % _rel(path))
    cfg = E43.embed_cfg_of(path) or {}
    ref = tta1_file(which)
    rcfg = E43.embed_cfg_of(ref) or {}
    want = dict(E43.cfg_key(cstar["embed_cfg"]), tta=8)
    problems = []
    if cfg.get("tta") != 8 or cfg.get("weights") != "raw":
        problems.append("tta %r weights %r" % (cfg.get("tta"), cfg.get("weights")))
    if E43.cfg_key(cfg) != want:
        problems.append("embed settings %r != %r" % (E43.cfg_key(cfg), want))
    if (cfg.get("games"), cfg.get("moves")) != (rcfg.get("games"), rcfg.get("moves")):
        problems.append("games / moves %r != tta1 %r" % ((cfg.get("games"), cfg.get("moves")),
                                                         (rcfg.get("games"), rcfg.get("moves"))))
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != cstar["ckpt_sha256"] or cfg.get("ckpt_sha256") != cstar["ckpt_sha256"]:
            problems.append("checkpoint sha %s" % str(z["ckpt_sha256"]))
        if not np.array_equal(z["count"], r["count"]):
            problems.append("count differs from the tta1 file")
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in z.files) != (k in r.files) or (k in z.files and not np.array_equal(z[k], r[k])):
                problems.append("%s differs from the tta1 file" % k)
    if problems:
        raise SystemExit("%s: %s" % (_rel(path), "; ".join(problems)))
    return {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt_sha256": cstar["ckpt_sha256"],
            "embed_cfg": cfg, "same_games_as": _rel(ref)}


# ================================================================ vector seam (revision 1)
def vector_seam_stats(a, ca, ra, b, cb, rb):
    """a = the 4090 tta1 vectors, b = the reference (first n games of the prereg43 tta1 file)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if (na == 0).any() or (nb == 0).any():
        cos_min = float("nan")
    else:
        cos_min = float(((a * b).sum(1) / (na * nb)).min())
    max_abs = float(np.abs(a - b).max())
    med = float(np.median(nb))
    rel = max_abs / med if med > 0 else float("inf")
    out = {"n": int(len(a)), "store_row_equal": bool(np.array_equal(ra, rb)),
           "count_equal": bool(np.array_equal(ca, cb)), "cos_min": cos_min, "max_abs_diff": max_abs,
           "median_norm_ref": med, "max_abs_over_median_norm": rel,
           "rule": "count equal, store_row equal, cos min >= %g, max |diff| / median norm (ref) <= %g"
                   % (COS_MIN, REL_MAX)}
    out["pass"] = bool(out["store_row_equal"] and out["count_equal"] and cos_min >= COS_MIN and rel <= REL_MAX)
    return out


def vector_seam(cstar, path=SEAM_VEC):
    if not os.path.exists(path):
        raise SystemExit("missing the vector seam file %s" % _rel(path))
    cfg = E43.embed_cfg_of(path) or {}
    ref = tta1_file("D")
    if E43.cfg_key(cfg) != E43.cfg_key(cstar["embed_cfg"]) or cfg.get("games") != SEAM_N:
        raise SystemExit("%s is not c*'s tta1 setting on %d games" % (_rel(path), SEAM_N))
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != cstar["ckpt_sha256"]:
            raise SystemExit("%s: another checkpoint" % _rel(path))
        n = len(z["emb"])
        if n != SEAM_N:
            raise SystemExit("%s: %d games" % (_rel(path), n))
        s = vector_seam_stats(z["emb"], z["count"], z["store_row"], r["emb"][:n], r["count"][:n], r["store_row"][:n])
    s.update({"file": _rel(path), "file_sha256": L.sha256_file(path), "ref": _rel(ref),
              "embed_cfg_compare": cfg.get("compare")})
    return s


# ================================================================ the system on D (prereg43 select step 3)
def system_D(path, pools, zs):
    e = E43.load_emb(path)
    bench = BK.Bench(pools, e, BK.base_cfg_of(E43.RETUNE_START))
    cfg, r = BK.retune_structure(bench, ["B4"], Q42.b4_cfg())
    del bench
    Fs = E43.fused(pools, e, cfg)
    del e
    rows, top = E43.p1_grid(Fs, zs, pools)
    return cfg, r, rows, top


def d_context(base):
    """D pools, the shared P1 term, and v2-6 itself rebuilt bit for bit (prereg45's seam path)."""
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    Fs, vec = E45.base_fused(base, "D", pools)
    ctx = E43.p1_ctx("D")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit on L differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    base_per = E45.seam_scores(base, "D", Fs, zs, pools)
    sets = ctx["sets"]
    del ctx, Fs
    P39E._MOCK.clear()
    return pools, zs, base_per, vec, sets


def check_seam_D(base, cfg, r, top):
    """system_D on the tta1 vectors == prereg43 tune.json (v2-6), pool by pool, bit for bit."""
    cs = base["tune"]["c_star"]
    bad = []
    if cfg != cs["backend"]["cfg"]:
        bad.append("backend cfg %r != %r" % (cfg, cs["backend"]["cfg"]))
    if r["per_pool"] != cs["backend"]["per_pool"] or r["mean"] != cs["backend"]["mean"]:
        bad.append("backend scores")
    if (top["wP"], top["k"]) != (cs["p1"]["wP"], cs["p1"]["csls_k"]):
        bad.append("(wP, k) %r != %r" % ((top["wP"], top["k"]), (cs["p1"]["wP"], cs["p1"]["csls_k"])))
    if top["per_pool"] != cs["p1"]["per_pool"] or top["mean"] != D_V26:
        bad.append("R %r != %r" % (top["mean"], D_V26))
    if bad:
        raise SystemExit("seam: " + "; ".join(bad))


def code_sha():
    return st.sha256_lf(os.path.abspath(__file__))


def cmd_seam(a):
    t0 = time.time()
    base = E45.v26()
    cs = base["tune"]["c_star"]
    pools, zs, base_per, vec, sets = d_context(base)
    tta1 = tta1_file("D")
    if _rel(tta1) != cs["file"] or L.sha256_file(tta1) != cs["file_sha256"]:
        raise SystemExit("the tta1 store_D is not prereg43 c*'s file")
    cfg, r, rows, top = system_D(tta1, pools, zs)
    check_seam_D(base, cfg, r, top)
    rec = {"format": "c10-prereg48 seam v1", "p48_eval_sha256_lf": code_sha(), "base": base["record"],
           "base_D_prereg45_path": {"per_pool": base_per, "mean": float(np.mean(base_per)), "identical": True,
                                    "vectors": vec},
           "select_path_on_tta1": {"file": cs["file"], "file_sha256": cs["file_sha256"], "backend_cfg": cfg,
                                   "backend_mean": r["mean"], "wP": top["wP"], "csls_k": top["k"],
                                   "per_pool": top["per_pool"], "mean": top["mean"], "want_mean": D_V26,
                                   "identical_to_prereg43_tune": True},
           "p1_sets": sets, "git_head": L.git_head()}
    if os.path.exists(SEAM_VEC):
        rec["vector_seam"] = vector_seam(cs)
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in rec if k != "base"}, indent=1))
    return 0


# ================================================================ select (D)
def check_seam_record(path=SEAM_JSON):
    if not L.git_committed(path):
        raise SystemExit("seam.json is not committed (run seam and commit it first)")
    s = read_json(path)
    if s.get("p48_eval_sha256_lf") != code_sha():
        raise SystemExit("p48_eval.py changed since seam.json: rerun seam and commit it")
    if not s["select_path_on_tta1"]["identical_to_prereg43_tune"] or s["select_path_on_tta1"]["mean"] != D_V26:
        raise SystemExit("seam.json does not hold the bit-for-bit seam")
    return {"file": _rel(path), "sha256_lf": st.sha256_lf(path)}


def cmd_select(a):
    t0 = time.time()
    no_later_outputs()
    seam_rec = check_seam_record()
    base = E45.v26()
    cs = base["tune"]["c_star"]
    path = c48_file("D")
    c48 = check_c48_file(path, "D", cs)
    E43.check_players(path, "D")
    vs = vector_seam(cs)
    if not vs["pass"]:
        raise SystemExit("vector seam (revision 1) failed: %s" % json.dumps(vs))
    pools, zs, base_per, vec, sets = d_context(base)
    cfg, r, rows, top = system_D(path, pools, zs)
    gate = top["mean"] - D_V26
    c48.update({"label": "c48", "backend": {"cfg": cfg, "mean": r["mean"], "per_pool": r["per_pool"],
                                            "by_k": r["by_k"]},
                "p1": {"variant": E43.P1_VARIANT, "pw": E43.P1_PW, "wP": top["wP"], "csls_k": top["k"],
                       "mean": top["mean"], "per_pool": top["per_pool"]}})
    rec = {"format": "c10-prereg48 tune v1", "plan_revision": 1, "d_seeds": EV.D_SEEDS, "seam": seam_rec,
           "vector_seam": vs, "c48": c48, "p1_grid": [{k: x[k] for k in ("mean", "wP", "k")} for x in rows],
           "base": base["record"], "base_D": {"per_pool": base_per, "mean": float(np.mean(base_per)),
                                              "identical_to_recorded": True, "vectors": vec},
           "lda": {"sha256": base["lda_sha256"]}, "p1_sets": sets,
           "R_minus_base_D": E43.diff_rec(top["per_pool"], base_per),
           "gate": {"rule": "R(c48) - D(v2-6) >= %.2f, D(v2-6) = %r" % (GATE, D_V26), "value": gate,
                    "pass": bool(gate >= GATE)},
           "p48_eval_sha256_lf": code_sha(), "peak_working_set_mb": peak_mb(),
           "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({"backend": cfg, "R_backend_only": r["mean"], "wP": top["wP"], "k": top["k"], "R": top["mean"],
                      "R_minus_base_D": rec["R_minus_base_D"], "gate": rec["gate"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far(repo=REPO):
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def judge_ready(tune, e_use, prior):
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 2)")
    if "prereg48" in prior:
        raise SystemExit("prereg48 already has an E judgement")
    if e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (e_use, len(prior), prior))


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    tune = read_json(a.tune)
    prior = e_judgements_so_far()
    judge_ready(tune, a.e_use, prior)
    base = E45.v26()
    E46.check_base_unchanged(base, tune)
    cs = base["tune"]["c_star"]
    path = c48_file("E")
    vec48 = check_c48_file(path, "E", cs)
    E43.made_after([path], E43.commit_time(a.tune))
    E43.check_players(path, "E")
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    Fb, vec = E45.base_fused(base, "E", pools)
    ctx = E43.p1_ctx("E")
    if ctx["lda_sha256"] != tune["lda"]["sha256"]:
        raise SystemExit("the P1 LDA refit differs from tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    bper = E45.seam_scores(base, "E", Fb, zs, pools)
    del Fb
    c = tune["c48"]
    cfg = c["backend"]["cfg"]
    e = E43.load_emb(path)
    Fs = E43.fused(pools, e, cfg)
    del e
    wP, k = c["p1"]["wP"], c["p1"]["csls_k"]
    per = E43.scores(Fs, zs, wP, k, pools)
    byk = [EV.score_by_k(st.csls(F + wP * z, k), p) for F, z, p in zip(Fs, zs, pools)]
    backend_only = [float(st.score(st.csls(F, int(cfg["norm"][1])), p["owner"])) for F, p in zip(Fs, pools)]
    rec = {"format": "c10-prereg48 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "c48": {x: c[x] for x in ("file", "ckpt_sha256", "backend", "p1")},
           "e_vectors": vec48, "base_vectors_E": vec, "cfg": cfg, "base": tune["base"],
           "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune), "git_head": L.git_head(),
           "p1_sets": sets,
           "means": {"c48": float(np.mean(per)), "base": float(np.mean(bper)),
                     "c48_backend_only": float(np.mean(backend_only))},
           "per_pool": {"c48": per, "base": bper, "c48_backend_only": backend_only},
           "by_k": {"c48": EV.mean_by_k(byk)}}
    j = E43.judge_rule(per, bper)
    rec["c48_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "c48")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c48_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c48 (or belongs to another tune.json): keep v2-6")
    base = E45.v26()
    E46.check_base_unchanged(base, tune)
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
    # seam: v2-6 rebuilt on this pool == prereg43's uploaded top5
    path_b, _choice = E43.cstar_file(base["tune"], a.pool)
    eb = E43.pool_emb(path_b, items, cs["ckpt_sha256"])
    Fb = BK.fuse_cfg(rec, BK.e_term(rec, eb, base["cfg"]), base["cfg"])
    data_b = P39E.top5_bytes(qids, st.top5(st.csls(Fb + base["wP"] * z, base["k"]), cand_ids))
    with open(os.path.join(V26_APPLY, a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: v2-6 rebuilt on %s differs from results/prereg43/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    del eb, Fb
    path = c48_file(a.pool)
    vec48 = check_c48_file(path, a.pool, cs)
    c = tune["c48"]
    cfg = c["backend"]["cfg"]
    e = E43.pool_emb(path, items, cs["ckpt_sha256"])
    F = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
    data = P39E.top5_bytes(qids, st.top5(st.csls(F + c["p1"]["wP"] * z, c["p1"]["csls_k"]), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg48 apply v1", "pool": a.pool, "vectors": vec48, "cfg": cfg, "p1": c["p1"],
           "v26_top5_equals_prereg43_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "tune_sha256": st.sha256_lf(a.tune), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "differs_from_v26_rows": int(sum(
               x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
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
    s = sub.add_parser("select")
    s.add_argument("--out", default=TUNE_JSON)
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--tune", default=TUNE_JSON)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--tune", default=TUNE_JSON)
    p.add_argument("--eval", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()                 # BELOW_NORMAL, 4 logical processors, 1 BLAS thread (prereg38's manners)
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
