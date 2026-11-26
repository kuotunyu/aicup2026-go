"""prereg39 explore: the KataGo P1 LDA term on the prereg36 pools, selection on D, the one judgement on E, apply
(PLAN.md 3-6). CPU only.

    PY results/prereg39/explore/p39_eval.py select --base p38 [--base-tune T --base-eval E --base-emb-dir D]
                                                                      # -> tune.json (commit it before judge)
    PY results/prereg39/explore/p39_eval.py select --base p36|p37     # pre-revision-2 bases (not used any more)
    PY results/prereg39/explore/p39_eval.py judge --e-use N [--e10]   # -> eval.json (once; main session only)
    PY results/prereg39/explore/p39_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg39/explore/p39_eval.py assemble --v1 <v1 submission> --src <official tests dir>

Base (PLAN.md revision 2) = the adopted M36 + prereg38 backend (v2-3): results/prereg38/explore/tune_D_B4.json +
eval.json, vectors data/prereg36/emb/step_40000; F_base = prereg38 fuse_cfg(pool, e_term(pool, emb, cfg), cfg), the
backend's score before its CSLS. If M37 is adopted first: --base-tune <prereg38 retune file> --base-eval <its
judgement> --base-emb-dir data/prereg37/emb/<ckpt>. The old p36 / p37 bases (prereg36 tune.json format) are kept.
M39 = CSLS_k(F_base + wP z(T_P1)); T_P1 with pw "none" (PLAN.md 3) or "b4" (revision 2: the L-LDA projections
through the B4 candidate-pool whitening of prereg38, cosine of prototypes).
The P1 features come from data/prereg39/p1/<list>/ (p39_extract.py) and, for the games with mock_ref >= 0, from the
PREREG_10 mock chunks (data/prereg10/mock; competition 1's own extraction). No competition-2 feature is read.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
P37 = os.path.join(REPO, "results", "prereg37", "explore")
P38 = os.path.join(REPO, "results", "prereg38", "explore")
for p in (P36, P38, os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p36_eval as EV  # noqa: E402   (TERMS_DIR = data/prereg36/terms_*)
import p36_lib as L  # noqa: E402
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
import p10_fuse as F10  # noqa: E402   (load_side0, fit_p1_lda, project_p1, t_p1: PREREG_10 2.1)
import p35_sim as ps  # noqa: E402
import p38_backend as BK  # noqa: E402   (fuse_cfg, e_term, transform, proto, polite: prereg38, unchanged)

DATA = os.path.join(REPO, "data", "prereg39")
LISTS = os.path.join(DATA, "lists")
P1 = os.path.join(DATA, "p1")
MOCK_DIR = os.path.join(REPO, "data", "prereg10", "mock")
MOCK_SETS = os.path.join(MOCK_DIR, "sets.json")
RECORD = os.path.join(HERE, "lists_record.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg39", "apply")
BASES = {"p36": {"dir": P36, "emb": os.path.join(REPO, "data", "prereg36", "emb"), "per_pool": "M"},
         "p37": {"dir": P37, "emb": os.path.join(REPO, "data", "prereg37", "emb"), "per_pool": "M37"}}
P38_DEFAULT = {"tune": os.path.join(P38, "tune_D_B4.json"), "eval": os.path.join(P38, "eval.json"),
               "emb_dir": os.path.join(REPO, "data", "prereg36", "emb", "step_40000"),
               "apply_dir": os.path.join(REPO, "results", "prereg38", "apply")}
VARIANTS = ("joint", "split")
PW = ("none", "b4")
B4_P1 = {"gnorm": True, "pool_w": "mean", "centre_colour": False, "whiten": {"lam": 0.02, "lda": 32}}
WP_GRID = (0.25, 0.5, 1, 2, 4, 8)
K_GRID = (10, 20, 40)
P1_ONLY_K = 10
BOOT_SEED = 39
ADOPT_MEAN = 0.01
FAIL_MAX = 0.005
META_KEYS = ("format", "meta_sha256", "ckpt_bytes", "precision", "code_sha256")
CHUNK = 200


# ================================================================ features
def read_list(name):
    p = os.path.join(LISTS, name + ".json")
    with open(RECORD, encoding="utf-8") as f:
        want = json.load(f)["lists_sha256"][name]
    if L.sha256_file(p) != want:
        raise SystemExit("%s differs from lists_record.json" % p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)["games"]


def chunk_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return tuple(str(z[k]) for k in META_KEYS)


_MOCK = {}


def mock_side0():
    if not _MOCK:
        with open(MOCK_SETS, encoding="utf-8") as f:
            n = len(json.load(f)["games"])
        paths = F10.chunk_files(MOCK_DIR)
        X, usable, _failed, have, _cnt = F10.load_side0(paths, n, "p10mock")
        _MOCK.update(X=X, usable=usable, have=have, meta={chunk_meta(p) for p in paths})
    return _MOCK


def load_set(name, p1_dir=P1, games=None):
    """-> dict(games, X float32 [n, 770], usable [n], failed (count of fresh failed), meta (set of chunk metadata)).
    Every game must be there: fresh ones in <p1_dir>/<name>/chunk_*.npz, the others in the PREREG_10 mock chunks
    whose metadata (format, neutral table, checkpoint bytes, precision, code SHAs) must equal the fresh chunks'."""
    games = read_list(name) if games is None else games
    n = len(games)
    d = os.path.join(p1_dir, name)
    paths = F10.chunk_files(d) if os.path.isdir(d) else []
    X, usable, failed, have, _cnt = F10.load_side0(paths, n, "p39" + name)
    meta = {chunk_meta(p) for p in paths}
    reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
    if have[reuse >= 0].any():
        raise SystemExit("%s: a reusable game was also extracted" % name)
    if (reuse >= 0).any():
        m = mock_side0()
        idx = np.flatnonzero(reuse >= 0)
        if not m["have"][reuse[idx]].all():
            raise SystemExit("%s: a PREREG_10 mock chunk is missing" % name)
        X[idx] = m["X"][reuse[idx]]
        usable[idx] = m["usable"][reuse[idx]]
        have[idx] = True
        if meta and meta != m["meta"]:
            raise SystemExit("%s: the PREREG_10 mock chunks were made with other code / weights than the new ones "
                             "(PLAN.md 1): re-extract everything under a revision" % name)
        meta = meta | m["meta"]
    if not have.all():
        raise SystemExit("%s: %d games have no P1 chunk yet (run p39_extract.py)" % (name, int((~have).sum())))
    if len(meta) != 1:
        raise SystemExit("%s: chunks with different metadata" % name)
    return {"games": games, "X": X, "usable": usable, "failed": int(failed.sum()), "meta": meta}


def fit_models(Lset):
    """joint: one LDA on every usable L game; split: one per colour. Labels = players."""
    lab = np.array([g["player_id"] for g in Lset["games"]])
    col = np.array([g["color"] for g in Lset["games"]])
    models = {"joint": F10.fit_p1_lda(Lset["X"], lab, Lset["usable"])}
    for c in ("B", "W"):
        models[c] = F10.fit_p1_lda(Lset["X"], lab, Lset["usable"] & (col == c))
    return models


def models_sha(models):
    h = hashlib.sha256()
    for k in ("joint", "B", "W"):
        for a in ("m", "s", "mu", "sd", "W"):
            h.update(np.ascontiguousarray(models[k][a], np.float64).tobytes())
    return h.hexdigest()


def project_all(X, models):
    """Per-game LDA projections for the joint and the two colour models (computed once per store)."""
    return {k: F10.project_p1(X, models[k]) for k in ("joint", "B", "W")}


def b4_term(Z, cand_groups, q_groups):
    """pw "b4" (revision 2): the projected games of the pool through prereg38's B4 (per-game L2, within-candidate
    whitening lam 0.02 of the pool's candidate games, top-32 between-candidate directions; candidate ids only),
    then the cosine of the L2-normalised prototypes. Groups = usable game indices into Z. -> (T, has_q); a
    candidate without a game takes the row minimum of the others (as p10_fuse.t_p1)."""
    nq, nc = len(q_groups), len(cand_groups)
    gc = np.array([i for i, g in enumerate(cand_groups) for _ in g], np.int64)
    gq = np.array([i for i, g in enumerate(q_groups) for _ in g], np.int64)
    ic = np.array([j for g in cand_groups for j in g], np.int64)
    iq = np.array([j for g in q_groups for j in g], np.int64)
    has_c = np.bincount(gc, minlength=nc) > 0 if len(gc) else np.zeros(nc, bool)
    if has_c.sum() < 2 or not len(iq):
        return np.zeros((nq, nc)), np.zeros(nq, bool)
    G = {"nc": nc, "nq": nq, "Xc": np.asarray(Z[ic], np.float64), "Xq": np.asarray(Z[iq], np.float64),
         "gc": gc, "gq": gq, "wc": np.zeros(len(ic), bool), "wq": np.zeros(len(iq), bool),
         "ncg": np.ones(len(ic)), "nqg": np.ones(len(iq))}
    Xc, Xq = BK.transform(G, B4_P1)
    C, hc = BK.proto(Xc, gc, nc)
    Q, hq = BK.proto(Xq, gq, nq)
    T = Q @ C.T
    if not hc.all():
        T[:, ~hc] = T[:, hc].min(1, keepdims=True) if hc.any() else 0.0
    return T, hq


def _term(Z, cg, qg, pw):
    if pw == "none":
        return F10.t_p1(Z, cg, qg)
    if pw == "b4":
        return b4_term(Z, cg, qg)
    raise ValueError(pw)


def p1_terms(X, usable, colour, cand_groups, q_groups, models, pw="none", Zs=None):
    """-> {variant: (z [nq x nc], has_q [nq])}. Groups hold indices into X. pw: "none" (PLAN.md 3) or "b4"
    (revision 2). Zs: project_all(X, models) when the caller has it already (same values)."""
    Zs = project_all(X, models) if Zs is None else Zs
    out = {}
    cg = [[g for g in grp if usable[g]] for grp in cand_groups]
    qg = [[g for g in grp if usable[g]] for grp in q_groups]
    T, has = _term(Zs["joint"], cg, qg, pw)
    out["joint"] = (st.zscore_rows(T, has), has)
    nq, nc = len(q_groups), len(cand_groups)
    acc, wsum = np.zeros((nq, nc)), np.zeros(nq)
    for c in ("B", "W"):
        use = usable & (colour == c)
        cgc = [[g for g in grp if use[g]] for grp in cand_groups]
        qgc = [[g for g in grp if use[g]] for grp in q_groups]
        Tc, hc = _term(Zs[c], cgc, qgc, pw)
        w = np.array([len(g) for g in qgc], np.float64) * hc
        acc += w[:, None] * st.zscore_rows(Tc, hc)
        wsum += w
    ok = wsum > 0
    acc[ok] /= wsum[ok, None]
    out["split"] = (acc, ok)
    return out


def store_index(sets):
    """Concatenate store-row sets (D, or E + E2) -> (X, usable, colour, {store row: index})."""
    X = np.concatenate([s["X"] for s in sets])
    usable = np.concatenate([s["usable"] for s in sets])
    games = [g for s in sets for g in s["games"]]
    colour = np.array([g["color"] for g in games])
    pos = {g["row"]: i for i, g in enumerate(games)}
    if len(pos) != len(games):
        raise SystemExit("a store row is in two lists")
    return X, usable, colour, pos


def pool_p1(pool, X, usable, colour, pos, models, pw="none", Zs=None):
    cg = [[pos[int(r)] for r in g] for g in pool["cand_rows"]]
    qg = [[pos[int(r)] for r in g] for g in pool["q_rows"]]
    return p1_terms(X, usable, colour, cg, qg, models, pw, Zs)


# ================================================================ base system
def _abs(p):
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(REPO, p))


def _rel(p):
    return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")


def base_info(name, tune=None, eval_=None, emb_dir=None):
    """The base system as one dict: name, k (its CSLS k), fused(pools, which) -> [F before CSLS],
    fused_official(rec, items, pool), the D and E seam values, record (what tune.json keeps).
    p38 (revision 2): a prereg38 variant / retune file + its adopting judgement + the vectors directory;
    p36 / p37: the prereg36-format tune.json + eval.json (pre-revision-2, not used any more)."""
    if name == "p38":
        return _base_p38(tune or P38_DEFAULT["tune"], eval_ or P38_DEFAULT["eval"],
                         emb_dir or P38_DEFAULT["emb_dir"])
    if tune or eval_ or emb_dir:
        raise SystemExit("--base-tune / --base-eval / --base-emb-dir go with --base p38 only")
    b = BASES[name]
    with open(os.path.join(b["dir"], "tune.json"), encoding="utf-8") as f:
        t = json.load(f)
    with open(os.path.join(b["dir"], "eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    if not ev["adopt"]:
        raise SystemExit("base %s was not adopted" % name)
    stem = os.path.splitext(t["checkpoint"]["name"])[0]
    emb = os.path.join(b["emb"], stem)

    def fused(pools, which):
        e = EV.Emb.load(os.path.join(emb, "store_%s.npz" % which))
        if e.sha != t["checkpoint"]["sha256"]:
            raise SystemExit("base embeddings do not match the base tune.json")
        w, _kG = EV.g_combo()
        return [EV.fused(p, EV.e36(e, p), w, t["M"]["wE"]) for p in pools]

    def fused_official(rec, items, pool):
        e = EV.Emb.load(os.path.join(emb, "pool_%s.npz" % pool))
        if e.sha != t["checkpoint"]["sha256"] or len(e.emb) != len(items):
            raise SystemExit("pool embeddings do not match the base checkpoint / the pool's games")
        return EV.fused(rec, EV.e36(e, rec), tuple(t["G"]["weights_O_L_JO_S2_PA"]), t["M"]["wE"])

    tsha = L.sha256_file(os.path.join(b["dir"], "tune.json"))
    esha = L.sha256_file(os.path.join(b["dir"], "eval.json"))
    return {"name": name, "k": t["M"]["csls_k"], "fused": fused, "fused_official": fused_official,
            "d_seam": {"mean": t["M"]["tune_mean"]}, "e_per_pool": ev["per_pool"][b["per_pool"]],
            "record": {"name": name, "tune_sha256": tsha, "eval_sha256": esha, "checkpoint": t["checkpoint"],
                       "M": t["M"], "seam_D_identical": True}}


def _base_p38(tune_path, eval_path, emb_dir):
    tune_path, eval_path, emb_dir = _abs(tune_path), _abs(eval_path), _abs(emb_dir)
    with open(tune_path, encoding="utf-8") as f:
        t = json.load(f)
    with open(eval_path, encoding="utf-8") as f:
        ev = json.load(f)
    tsha, esha = st.sha256_lf(tune_path), st.sha256_lf(eval_path)
    if not ev.get("adopt") or ev.get("tune_sha256") != tsha:
        raise SystemExit("--base-eval does not adopt --base-tune (adopt false, or it judged another tune file)")
    cfg = t["recommended"]["cfg"]
    if cfg.get("norm", [None])[0] != "csls" or cfg.get("aqe") or cfg.get("qsmooth"):
        raise SystemExit("base backend: only a configuration ending in CSLS without aqe / qsmooth is supported")
    ckpt = t["emb"]["ckpt_sha256"]
    k = int(cfg["norm"][1])
    colour = {}

    def store_colour():
        if "c" not in colour:
            colour["c"] = sim.Store(L.FEATS, load=()).color
        return colour["c"]

    def fused(pools, which):
        e = EV.Emb.load(os.path.join(emb_dir, "store_%s.npz" % which), colour=store_colour())
        if e.sha != ckpt:
            raise SystemExit("base vectors (%s) were not made from the checkpoint of --base-tune" % which)
        return [BK.fuse_cfg(p, BK.e_term(p, e, cfg), cfg) for p in pools]

    def fused_official(rec, items, pool):
        e = EV.Emb.load(os.path.join(emb_dir, "pool_%s.npz" % pool), colour=np.asarray([c for _s, c in items]))
        if e.sha != ckpt or len(e.emb) != len(items):
            raise SystemExit("pool vectors do not match the tuned checkpoint / the pool's games")
        return BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)

    default = (tune_path, eval_path, emb_dir) == (_abs(P38_DEFAULT["tune"]), _abs(P38_DEFAULT["eval"]),
                                                  _abs(P38_DEFAULT["emb_dir"]))
    v24 = tune_path == _abs(os.path.join("results", "prereg37", "explore", "tune_D_B4_M37.json"))
    apply_dir = P38_DEFAULT["apply_dir"] if default else (
        os.path.join(REPO, "results", "prereg37", "apply") if v24 else None)      # revision 3: v2-4 base
    return {"name": "p38", "k": k, "fused": fused, "fused_official": fused_official, "default": default,
            "apply_dir": apply_dir,
            "d_seam": {"per_pool": t["recommended"]["per_pool"], "mean": t["recommended"]["mean"]},
            "e_per_pool": ev["per_pool"]["backend"], "e_mean": ev["means"]["backend"],
            "record": {"name": "p38", "tune": _rel(tune_path), "eval": _rel(eval_path), "emb_dir": _rel(emb_dir),
                       "tune_sha256_lf": tsha, "eval_sha256_lf": esha, "cfg": cfg, "emb": t["emb"],
                       "csls_k": k, "tune_mean": t["recommended"]["mean"], "e_mean": ev["means"]["backend"],
                       "seam_D_identical": True}}


def base_from_record(rec):
    """judge / apply: the base exactly as tune.json recorded it (no other paths accepted)."""
    if rec["name"] == "p38":
        b = base_info("p38", rec["tune"], rec["eval"], rec["emb_dir"])
        if b["record"]["tune_sha256_lf"] != rec["tune_sha256_lf"] or \
                b["record"]["eval_sha256_lf"] != rec["eval_sha256_lf"]:
            raise SystemExit("the base changed since tune.json")
        return b
    b = base_info(rec["name"])
    if b["record"]["tune_sha256"] != rec["tune_sha256"] or b["record"]["eval_sha256"] != rec["eval_sha256"]:
        raise SystemExit("the base changed since tune.json")
    return b


def base_scores(base, Fs, pools):
    return [st.score(st.csls(F, base["k"]), p["owner"]) for F, p in zip(Fs, pools)]


def check_d_seam(base, Fs, pools):
    """The base on the D pools must give its recorded per-pool scores (p38) / mean bit for bit."""
    per = base_scores(base, Fs, pools)
    want = base["d_seam"]
    if "per_pool" in want and per != list(want["per_pool"]):
        raise SystemExit("seam: base per-pool D scores %r != recorded %r" % (per, want["per_pool"]))
    if float(np.mean(per)) != want["mean"]:
        raise SystemExit("seam: base on the D pools %r != recorded %r" % (float(np.mean(per)), want["mean"]))
    return per


def check_e_seam(base, b, n):
    """The base on the first n E pools must give its eval.json per-pool scores (and, on all 20, its mean)."""
    if np.asarray(b).tolist() != list(base["e_per_pool"])[:n]:
        raise SystemExit("seam: the base on the E pools differs from its eval.json")
    if n == len(base["e_per_pool"]) and "e_mean" in base and float(np.mean(b)) != base["e_mean"]:
        raise SystemExit("seam: base E mean %r != recorded %r" % (float(np.mean(b)), base["e_mean"]))


def mean_score(Fs, zs, wP, k, pools):
    return float(np.mean([st.score(st.csls(F + wP * z, k), p["owner"]) for F, z, p in zip(Fs, zs, pools)]))


def grid_for(base):
    """(variant, pw, wP, k) in grid order. p38 (revision 2): pw in {none, b4}; p36 / p37: pw none only."""
    pws = PW if base["name"] == "p38" else ("none",)
    return [(v, pw, wP, k) for v in VARIANTS for pw in pws for wP in WP_GRID for k in K_GRID]


def m39_of(tune):
    m = tune["M39"]
    return m["variant"], m.get("pw", "none"), m["wP"], m["csls_k"]


# ================================================================ select / judge
def select_grid(Fs, terms, pools, grid):
    """-> rows (mean, variant, pw, wP, k) in grid order and the first maximum. terms[pw][i] = pool_p1(...)."""
    rows = []
    for v, pw, wP, k in grid:
        zs = [t[v][0] for t in terms[pw]]
        rows.append((mean_score(Fs, zs, wP, k, pools), v, pw, wP, k))
    return rows, max(rows, key=lambda r: r[0])


def cmd_select(a):
    base = base_info(a.base, a.base_tune, a.base_eval, a.base_emb_dir)
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    Fs = base["fused"](pools, "D")
    check_d_seam(base, Fs, pools)
    Lset, Dset = load_set("L"), load_set("D")
    if Lset["meta"] != Dset["meta"]:
        raise SystemExit("L and D chunks have different metadata")
    models = fit_models(Lset)
    X, usable, colour, pos = store_index([Dset])
    Zs = project_all(X, models)
    grid = grid_for(base)
    pws = [pw for pw in PW if any(g[1] == pw for g in grid)]
    terms = {pw: [pool_p1(p, X, usable, colour, pos, models, pw, Zs) for p in pools] for pw in pws}
    rows, top = select_grid(Fs, terms, pools, grid)
    only = {"%s/%s" % (v, pw): float(np.mean([st.score(st.csls(t[v][0], P1_ONLY_K), p["owner"])
                                              for t, p in zip(terms[pw], pools)]))
            for pw in pws for v in VARIANTS}
    rec = {"format": "c10-prereg39 tune v2", "plan_revision": 2 if base["name"] == "p38" else 1,
           "d_seeds": EV.D_SEEDS, "base": base["record"],
           "lda": {"sha256": models_sha(models),
                   "fit": {k: [models[k]["n_players"], models[k]["n_games"]] for k in ("joint", "B", "W")}},
           "sets": {n: {"games": len(s["games"]), "usable": int(s["usable"].sum()), "failed_fresh": s["failed"]}
                    for n, s in (("L", Lset), ("D", Dset))},
           "p1_only_csls%d_on_D" % P1_ONLY_K: only,
           "grid": [{"mean": r[0], "variant": r[1], "pw": r[2], "wP": r[3], "k": r[4]} for r in rows],
           "M39": {"variant": top[1], "pw": top[2], "wP": top[3], "csls_k": top[4], "tune_mean": top[0]},
           "lists_record_sha256": L.sha256_file(RECORD), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("base", "lda", "sets", "p1_only_csls%d_on_D" % P1_ONLY_K, "M39")}, indent=1))
    return 0


def judge_numbers(Fs, terms, pools, base, m39):
    """Per-pool M39 and base scores on the judged pools and the rule (PLAN.md 5, revision 2).
    terms[i] = pool_p1(...) of pool i with the chosen pw."""
    v, _pw, wP, k = m39
    m, b, byk, byk_b, only = [], [], [], [], []
    for F, t, p in zip(Fs, terms, pools):
        z = t[v][0]
        S, Sb = st.csls(F + wP * z, k), st.csls(F, base["k"])
        m.append(st.score(S, p["owner"]))
        b.append(st.score(Sb, p["owner"]))
        byk.append(EV.score_by_k(S, p))
        byk_b.append(EV.score_by_k(Sb, p))
        only.append(st.score(st.csls(z, P1_ONLY_K), p["owner"]))
    m, b = np.array(m), np.array(b)
    d = m - b
    lo, hi = EV.bootstrap(d, seed=BOOT_SEED)
    posn, need = int((d > 0).sum()), int(np.ceil(0.9 * len(d)))
    return {"means": {"M39": float(m.mean()), "base": float(b.mean()),
                      "P1_only_csls%d" % P1_ONLY_K: float(np.mean(only))},
            "per_pool": {"M39": m.tolist(), "base": b.tolist()},
            "M39_minus_base": {"mean": float(d.mean()), "ci95": [lo, hi], "pools_positive": posn},
            "by_k": {"M39": EV.mean_by_k(byk), "base": EV.mean_by_k(byk_b)},
            "rule": "adopt M39 if mean(M39 - base) >= %.2f and ci95 low > 0 and M39 > base in >= %d of %d pools "
                    "(pool bootstrap, %d samples, seed %d)" % (ADOPT_MEAN, need, len(d), EV.BOOT, BOOT_SEED),
            "adopt": bool(d.mean() >= ADOPT_MEAN and lo > 0 and posn >= need)}


def cmd_judge(a):
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    base = base_from_record(tune["base"])
    e2 = os.path.isdir(os.path.join(P1, "E2")) and _complete("E2")
    if a.e10 and e2:
        raise SystemExit("--e10 is the fall-back for an unfinished E2 extraction (PLAN.md 2); E2 is complete")
    if not a.e10 and not e2:
        raise SystemExit("E2 is not complete: finish it, or after 11/14 23:59 use --e10 (PLAN.md 2)")
    seeds = EV.E_SEEDS[:10] if a.e10 else EV.E_SEEDS
    pools = EV.load_pools(EV.TERMS_DIR["E"], seeds)
    Fs = base["fused"](pools, "E")
    check_e_seam(base, base_scores(base, Fs, pools), len(seeds))
    Lset = load_set("L")
    models = fit_models(Lset)
    if models_sha(models) != tune["lda"]["sha256"]:
        raise SystemExit("the LDA refit differs from tune.json")
    esets = [load_set("E")] + ([] if a.e10 else [load_set("E2")])
    if any(s["meta"] != Lset["meta"] for s in esets):
        raise SystemExit("L and E chunks have different metadata")
    X, usable, colour, pos = store_index(esets)
    m39 = m39_of(tune)
    Zs = project_all(X, models)
    terms = [pool_p1(p, X, usable, colour, pos, models, m39[1], Zs) for p in pools]
    rec = {"format": "c10-prereg39 eval v2", "e_seeds": list(seeds), "fallback_e10": bool(a.e10),
           "e_pools_use_number": a.e_use, "base": tune["base"], "M39": tune["M39"],
           "tune_sha256": st.sha256_lf(a.tune), "git_head": L.git_head(), "seam_base_identical": True,
           "sets": {n: {"games": len(s["games"]), "usable": int(s["usable"].sum()), "failed_fresh": s["failed"]}
                    for n, s in zip(("E", "E2"), esets)}}
    rec.update(judge_numbers(Fs, terms, pools, base, m39))
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "M39_minus_base", "by_k", "adopt")}, indent=1))
    return 0


def _complete(name):
    games = read_list(name)
    n = sum(1 for g in games if g.get("mock_ref", -1) < 0)
    return len(F10.chunk_files(os.path.join(P1, name))) == (n + CHUNK - 1) // CHUNK


# ================================================================ official pools
def top5_bytes(qids, tops):
    """The bytes of a top5.csv (the layout of p36 / p38 apply: header, LF line ends)."""
    buf = io.StringIO(newline="")
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
    for q, t in zip(qids, tops):
        wr.writerow([q] + t)
    return buf.getvalue().encode("utf-8")


def check_base_top5(base, pool, data, apply_dir=None):
    """Default p38 base (revision 2): the base without P1 must give exactly the v2-3 new-pool top5 of prereg38's
    apply. -> True when checked, None when the base is not the default one."""
    if not base.get("apply_dir"):
        return None
    d = os.path.join(apply_dir or base["apply_dir"], pool)
    with open(os.path.join(d, "apply_report.json"), encoding="utf-8") as f:
        r = json.load(f)
    if r["tune_sha256"] != base["record"]["tune_sha256_lf"]:
        raise SystemExit("the base's apply of %s was made with another tune file" % pool)
    with open(os.path.join(d, "top5.csv"), "rb") as f:
        if f.read() != data:
            raise SystemExit("seam: the base's %s top5 differs from %s" % (pool, d))
    return True


def cmd_apply(a):
    import p35_apply as pa
    t0 = time.time()
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    with open(a.eval, encoding="utf-8") as f:
        ev = json.load(f)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt M39 (or belongs to another tune.json): keep the base")
    base = base_from_record(tune["base"])
    cands, qs = pa.read_pool(a.src, a.pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[a.pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    fit = sim.Store(L.FEATS, load=())
    store = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, lm = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        lm[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(store, ld.cols)
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    T = sim.pool_terms(store, rec, ldas, lm, None, want=("O", "JO", "S2", "L", "PA"))
    rec["T"] = T
    Fb = base["fused_official"](rec, items, a.pool)
    base_seam = check_base_top5(base, a.pool, top5_bytes(qids, st.top5(st.csls(Fb, base["k"]), cand_ids)))
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    Pset, Lset = load_set(name), load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if Pset["meta"] != Lset["meta"]:
        raise SystemExit("L and pool chunks have different metadata")
    bad = int((~Pset["usable"]).sum())
    if bad > FAIL_MAX * len(items):
        raise SystemExit("%d of %d pool games have no usable P1 (> 0.5%%): stop (PLAN.md revision 1)"
                         % (bad, len(items)))
    models = fit_models(Lset)
    if models_sha(models) != tune["lda"]["sha256"]:
        raise SystemExit("the LDA refit differs from tune.json")
    colour = np.array([c for _s, c in items])
    v, pw, wP, k = m39_of(tune)
    z, has = p1_terms(Pset["X"], Pset["usable"], colour, cand_rows, q_rows, models, pw)[v]
    data = top5_bytes(qids, st.top5(st.csls(Fb + wP * z, k), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, "top5.csv")
    with open(path, "wb") as f:
        f.write(data)
    rep = {"format": "c10-prereg39 apply v2", "pool": a.pool, "base": tune["base"], "M39": tune["M39"],
           "base_top5_equals_prereg38_apply": base_seam,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "terms_rows_without_signal": {n: int((~np.asarray(T[n][1])).sum()) for n in ps.TERMS},
           "tune_sha256": st.sha256_lf(a.tune), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(path), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return EV.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--base", required=True, choices=["p38"] + sorted(BASES))
    s.add_argument("--base-tune", default=None, help="p38 only: prereg38 variant / retune file (default tune_D_B4)")
    s.add_argument("--base-eval", default=None, help="p38 only: the judgement adopting it (default prereg38 eval)")
    s.add_argument("--base-emb-dir", default=None,
                   help="p38 only: its vectors directory (default data/prereg36/emb/step_40000)")
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    j.add_argument("--e-use", type=int, required=True, choices=[3, 4],
                   help="which use of the E pools this is (3 or 4 by the actual order; revision 2), recorded")
    j.add_argument("--e10", action="store_true", help="PLAN.md 2 fall-back: seeds 0-9 only (E2 unfinished)")
    q = sub.add_parser("apply")
    q.add_argument("--src", required=True)
    q.add_argument("--pool", required=True, choices=["public", "private"])
    q.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    q.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    q.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", required=True)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()                 # BELOW_NORMAL, 4 logical processors, 1 BLAS thread (prereg38's manners)
        BK.wait_for_ram(need_mb=4096)
    return {"select": cmd_select, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
