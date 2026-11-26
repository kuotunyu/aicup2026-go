"""prereg58 (competition 2): the opponent-identity term added to Q25 (PLAN.md; NOTES.md for the narrow readings).
CPU only, .venv (numpy / scipy), 4 cores, working set <= 3 GB (p58_lib.limits).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8; no OMP / OPENBLAS / MKL variable
    S=results/prereg58/explore;  PY=.venv/Scripts/python.exe
    $PY $S/p58_eval.py rows          -> data/prereg58/inner_rows.npy, sanity_rows.npy; $S/rows.json
                                        (the inner games used by the inner questions; no model number)
    (GPU: $S/embed_inner.sh -> data/prereg58/emb_opp_inner/...; CPU: the two 64-game sanity vectors)
    $PY $S/p58_eval.py vectors       -> $S/vectors.json  (the vector files are the registered ones)
    $PY $S/p58_eval.py kernel        -> $S/kernel.json   (K, tau; D and E players only; chosen once)
    $PY $S/p58_eval.py seam          -> $S/seam.json     (Q25 on inner through this code path, bit for bit)
    $PY $S/p58_eval.py eval          -> $S/eval.json     (judged once: kernel.json and seam.json committed, tree clean)
    $PY $S/p58_eval.py eval --smoke  -> data/prereg58/smoke/eval.json (the OPP term replaced by default_rng(12345)
                                        noise: pipeline and memory only, never a result)

No inner number of the new term exists before `eval`; `rows`, `vectors`, `kernel` and `seam` never compute one.
Never data/local, never tools/score.py, never a test-set file (p58_lib.guard), no public score.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
P31 = os.path.join(REPO, "results", "prereg31", "explore")
P36 = os.path.join(REPO, "results", "prereg36", "explore")
P41 = os.path.join(REPO, "results", "prereg41", "explore")
for p in (os.path.join(REPO, "tools"), P23, P41, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p58_lib as K  # noqa: E402

ROWS_JSON = os.path.join(HERE, "rows.json")
VECTORS_JSON = os.path.join(HERE, "vectors.json")
SANITY_ROWS = os.path.join(K.DATA, "sanity_rows.npy")
SANITY_N = 64
SANITY_OWN = os.path.join(K.DATA, "sanity", "own", K.TAG, "rows_sanity_own.npz")
SANITY_OPP = os.path.join(K.DATA, "sanity", "opp", K.TAG, "rows_sanity_opp.npz")
S0_D_OWN = os.path.join(REPO, "data", "prereg43", "emb_S0", K.TAG, "store_D.npz")     # S0's own D vectors (GPU)
TERM = "OPP"
NS = (5, 10, 20)
DRAWS = 20
SEED = 58
NOISE_SEED = 12345


def log(*x):
    print(*x, flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def rel(p):
    return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")


def committed_clean(path):
    return bool(git("ls-files", "--", rel(path))) and not git("status", "--porcelain", "--", rel(path))


def loaded_repo_modules():
    root = os.path.join(os.path.abspath(REPO), "").lower()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(root):
            out.add(rel(f))
    return sorted(out)


def code_state():
    """HEAD, the state of results/prereg58 (must be empty for a judged run) and of every imported repository file."""
    mods = loaded_repo_modules()
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prereg58_dirty": git("status", "--porcelain", "--", "results/prereg58"),
            "imported_dirty": git("status", "--porcelain", "--", *mods) if mods else "",
            "sha256_lf": {x: K.sha_lf(os.path.join(REPO, *x.split("/"))) for x in mods}}


def peak_gb(st):
    return round(st["peak"] / 2 ** 30, 2)


# ================================================================ the inner harness (imported, never edited)
def harness():
    import p41_eval as E41     # -> p23_eval (E23), tools/p09_rank_fuse (r), ft_eval, p09_forward (FW)
    import p23_lib as K23
    return E41, E41.E23, E41.r, K23


def lean_inner(r, prep):
    """tools/p09_rank_fuse.Inner without the Q02 / Q02H / Q08 fits of its constructor: Inner.inner_lr_lp (Q08) holds
    the whole inner feature matrix in float64 three times over (the 3.8 GB peak of every earlier Q25 rebuild), which
    the 3 GB rule of this run does not allow and Q25 does not need. Every attribute Q25 uses is built by the same
    calls in the same order; run / E_p1 / compare / short are inherited. The seam (Q25 bit for bit) is the check."""
    R = r.R

    class LeanInner(r.Inner):
        def __init__(self, prep):
            self.PF = R.import_pilot_fuse()
            _L1, self.F = R.import_p1()
            self.U = R._import_u1()
            d = self.U.load_inputs(verbose=False)
            self.d = d
            inner = d["inner"]
            self.n_inner = inner["keys"]["n"]
            lp_hs, _cov, self.hs_rec = self.F.hs_baseline(self.PF, self.U, d)
            d["lp"]["HS"] = lp_hs
            hs_usable, _f, _c = R.pilot_hs_usable(self.PF, self.n_inner)
            self.E_hs, self.hs_info = R.masked_inner(d, lp_hs, hs_usable)
            self.ref = prep.inner_ref
            if len(np.unique(self.ref)) != len(self.ref) or (len(self.ref) and self.ref.max() >= self.n_inner):
                raise SystemExit("prepared inner refs are not distinct inner rows")
            self.covered = np.zeros(self.n_inner, dtype=bool)
            self.covered[self.ref] = True
            self.Xf, self.Xe, self.Xm, self.ms = inner["X_fit"], inner["X_eval"], inner["X_mono"], inner["ms"]
            self.boot, self.mboot = inner["boot"], inner["mono_boot"]

    return LeanInner(prep)


def build_q25():
    """-> (harness modules, inner, terms7, Emap, q25, seam record). Q25 exactly as prereg44's judgement built it
    (p41_eval.s25_models, Q09's p1e_map P1TO, Inner.E_hs, p23_eval.s19_term with the frozen S25 models)."""
    E41, E23, r, K23 = harness()
    r.blas_env_check(False)
    t0 = time.time()
    s25, sm = E41.s25_models()
    prep = E41.ft_eval.InnerPrep(E23.guard_path(E23.INNER_PREP))
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    sm["inner_X_sha_equal_check"] = bool(E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))
                                         == want.get("inner_X.npy"))
    inner = lean_inner(r, prep)
    mp = E41.FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    E_s25, _c, _d = E23.s19_term(prep, inner, mp, s25)
    del _c, _d
    Emap = {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25}
    terms7 = list(r.Q08_TERMS) + ["SET"]
    q25 = inner.run(terms7, Emap)
    rep25 = r.read_json(E41.S25_EVAL_REPORT)
    bound = bool(rep25["candidate"]["inner_rank_balanced"] == K.REF_Q25
                 and rep25["candidate"]["single_colour_rank_balanced"] == K.REF_Q25_MONO)
    seam = {"Q25": [q25["rb"], K.REF_Q25], "Q25_mono": [q25["mono_rb"], K.REF_Q25_MONO],
            "Q25_bitwise": bool(q25["rb"] == K.REF_Q25), "Q25_mono_bitwise": bool(q25["mono_rb"] == K.REF_Q25_MONO),
            "ref_is_prereg25_eval_report": bound, "s25_models": sm, "terms": terms7,
            "eval_questions": int(len(inner.Xe)), "mono_questions": int(len(inner.Xm)),
            "fit_questions": int(len(inner.Xf)), "n_inner": int(inner.n_inner),
            "path": "lean_inner (no Q02 / Q02H / Q08 fit) -> p09 inner_map_lp -> p23_eval.s19_term(S25) -> Inner.run",
            "seconds": round(time.time() - t0, 1)}
    seam["ok"] = bool(seam["Q25_bitwise"] and seam["Q25_mono_bitwise"] and bound and sm["ok"]
                      and sm["inner_X_sha_equal_check"])
    return (E41, E23, r, K23), inner, terms7, Emap, q25, seam


# ================================================================ rows
def inner_keys_arrays():
    with np.load(K.guard(K.INNER_KEYS), allow_pickle=False) as z:
        return {k: np.asarray(z[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}


def cmd_rows(a):
    st = K.limits(log)
    if os.path.exists(ROWS_JSON):
        raise SystemExit("rows.json exists")
    _E41, _E23, r, _K23 = harness()
    d = r.R._import_u1().load_inputs(verbose=False)
    inner = d["inner"]
    n = inner["keys"]["n"]
    used = np.zeros(n, bool)
    per = {}
    for name in ("S_fit", "S_eval", "S_mono"):
        cols = np.unique(inner[name].tocsr().indices)
        used[cols] = True
        per[name] = {"questions": int(inner[name].shape[0]), "games": int(len(cols))}
    rows = np.flatnonzero(used).astype(np.int64)
    ik = inner_keys_arrays()
    same_keys = all(np.array_equal(ik[k], np.asarray(inner["keys"][k]).astype(str))
                    for k in ("rank", "game_id", "player_id", "color"))
    sys.path.insert(0, P36)
    import p36_lib as L36
    cache = L36.Cache(K.INNER_CACHE)
    cache_ok = bool(cache.n == n and np.array_equal(np.asarray(cache.rank).astype(str), ik["rank"])
                    and np.array_equal(np.asarray(cache.gid).astype(str), ik["game_id"])
                    and np.array_equal(np.asarray(cache.pid).astype(str), ik["player_id"])
                    and np.array_equal(np.asarray(cache.color).astype(str), ik["color"])
                    and not np.asarray(cache.bad).any()
                    and cache.meta.get("inner_keys_sha256") == K.sha256_file(K.INNER_KEYS))
    # the 64 D games of the vector sanity check: S0's own store_D rows -> rows of the prereg36 move cache
    with np.load(K.guard(S0_D_OWN), allow_pickle=False) as z:
        srow = np.asarray(z["store_row"][:SANITY_N])
    with np.load(K.guard(K.FEATS_META), allow_pickle=False) as z:
        fr, fg, fp, fc = (np.asarray(z[k])[srow].astype(str) for k in ("rank", "game_id", "player_id", "color"))
    c36 = L36.Cache()
    key = c36.key_index()
    srows = np.array([key[(x, y, w)] for x, y, w in zip(fr.tolist(), fg.tolist(), fp.tolist())], np.int64)
    sanity_ok = bool((np.asarray(c36.color)[srows].astype(str) == fc).all())
    if not (same_keys and cache_ok and sanity_ok):
        raise SystemExit(f"rows: keys {same_keys}, inner cache {cache_ok}, sanity rows {sanity_ok}")
    os.makedirs(K.DATA, exist_ok=True)
    np.save(K.INNER_ROWS, rows)
    np.save(SANITY_ROWS, srows)
    rep = {"format": "c10-prereg58 rows v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "what": "the inner games that appear in at least one inner question (fit / eval / single-colour); their "
                   "rows in data/prereg44/cache/inner (prereg36 cache format, inner_keys order)",
           "n_inner": int(n), "used_rows": int(len(rows)), "per_set": per,
           "inner_players": int(len(set(ik["player_id"].tolist()))),
           "inner_keys_equal_harness_keys": same_keys, "inner_cache_equals_inner_keys": cache_ok,
           "inner_cache_codes_sha256": cache.meta.get("codes_sha256"),
           "inner_rows_sha256": K.sha256_file(K.INNER_ROWS),
           "sanity": {"games": int(len(srows)), "source": rel(S0_D_OWN), "rows_sha256": K.sha256_file(SANITY_ROWS)},
           "peak_working_set_gb": peak_gb(st)}
    K.write_json(ROWS_JSON, rep)
    log(json.dumps({k: rep[k] for k in ("n_inner", "used_rows", "per_set", "peak_working_set_gb")}))
    return 0


# ================================================================ vectors
def load_vec(path):
    with np.load(K.guard(path), allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in ("emb", "count", "store_row")}
        d["ckpt_sha256"] = str(z["ckpt_sha256"])
        d["embed_cfg"] = json.loads(str(z["embed_cfg"]))
        d["rows_cfg"] = json.loads(str(z["rows_cfg"])) if "rows_cfg" in z.files else None
    return d


def cos_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def cmd_vectors(a):
    st = K.limits(log)
    _Wn, _cr, hrec = K.load_head()
    rows = np.load(K.INNER_ROWS)
    v = load_vec(K.OPP_INNER)
    cfg = v["embed_cfg"]
    inner_rec = {"file": rel(K.OPP_INNER), "sha256": K.sha256_file(K.OPP_INNER), "games": int(len(v["emb"])),
                 "rows_equal_inner_rows": bool(np.array_equal(v["store_row"], rows)),
                 "games_without_sample": int((v["count"] == 0).sum()),
                 "ckpt_is_s0": bool(v["ckpt_sha256"] == hrec["ckpt_sha256"]),
                 "weights": cfg["weights"], "tta": cfg["tta"], "mode": cfg["mode"], "side": v["rows_cfg"]["side"],
                 "cache_dir": v["rows_cfg"]["cache_dir"], "finite": bool(np.isfinite(v["emb"]).all())}
    inner_rec["ok"] = bool(inner_rec["rows_equal_inner_rows"] and inner_rec["ckpt_is_s0"] and cfg["weights"] == "raw"
                           and cfg["tta"] == 1 and cfg["mode"] is None and inner_rec["side"] == "opp"
                           and inner_rec["finite"])
    chain = {}
    for name, path in (("D", K.OPP_D), ("E", K.OPP_E)):
        w = load_vec(path)
        chain[name] = {"file": rel(path), "sha256": K.sha256_file(path), "games": int(len(w["emb"])),
                       "games_without_sample": int((w["count"] == 0).sum()),
                       "ckpt_is_s0": bool(w["ckpt_sha256"] == hrec["ckpt_sha256"]),
                       "weights": w["embed_cfg"]["weights"], "tta": w["embed_cfg"]["tta"]}
    # sanity: this wrapper (CPU, fp32) against the existing GPU (fp16) vectors of the same 64 D games
    sr = np.load(SANITY_ROWS)
    san = {"references_same_games": bool(np.array_equal(load_vec(S0_D_OWN)["store_row"][:SANITY_N],
                                                        load_vec(K.OPP_D)["store_row"][:SANITY_N]))}
    if not san["references_same_games"]:
        raise SystemExit("the two reference files do not start with the same 64 store rows")
    for name, mine, ref_path in (("own_vs_S0_store_D", SANITY_OWN, S0_D_OWN), ("opp_vs_chain_store_D", SANITY_OPP, K.OPP_D)):
        m, ref = load_vec(mine), load_vec(ref_path)
        e_ref, c_ref = ref["emb"][:SANITY_N], ref["count"][:SANITY_N]
        cs = cos_rows(m["emb"], e_ref)
        san[name] = {"games": int(len(m["emb"])), "rows_equal": bool(np.array_equal(m["store_row"], sr)),
                     "count_equal": bool(np.array_equal(m["count"], c_ref)),
                     "max_abs_diff": float(np.abs(m["emb"] - e_ref).max()), "min_cosine": float(cs.min()),
                     "ref_abs_mean": float(np.abs(e_ref).mean()), "side": m["rows_cfg"]["side"],
                     "device_note": "this wrapper on CPU fp32, the reference on cuda fp16 autocast"}
        san[name]["ok"] = bool(san[name]["rows_equal"] and san[name]["count_equal"] and san[name]["min_cosine"] > 0.9999)
    own, opp = load_vec(SANITY_OWN), load_vec(SANITY_OPP)
    san["own_vs_opp_mean_cosine_record"] = float(cos_rows(own["emb"], opp["emb"]).mean())
    rep = {"format": "c10-prereg58 vectors v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "head": hrec, "inner_opponent_side": inner_rec, "kernel_opponent_side": chain, "sanity": san,
           "ok": bool(inner_rec["ok"] and all(c["ckpt_is_s0"] and c["weights"] == "raw" and c["tta"] == 1
                                              for c in chain.values())
                      and san["own_vs_S0_store_D"]["ok"] and san["opp_vs_chain_store_D"]["ok"]),
           "peak_working_set_gb": peak_gb(st)}
    K.write_json(VECTORS_JSON, rep)
    log(json.dumps({"ok": rep["ok"], "inner": inner_rec, "sanity": san}))
    return 0 if rep["ok"] else 1


# ================================================================ kernel
def hd_join(rank, game_id, player_id, colour):
    sys.path.insert(0, P31)
    import p31_lib as L31
    tab = L31.HdTable()
    hd, rec = tab.join(rank, game_id, player_id, colour)
    rec["table_sha256"] = tab.sha256
    if not rec["ok"]:
        raise SystemExit(f"HD join failed: {rec}")
    return hd, rec


def kernel_games():
    """The opponent-side games of the D and E players (S0 never saw them, none is an inner player)."""
    Wn, cr, hrec = K.load_head()
    with np.load(K.guard(K.FEATS_META), allow_pickle=False) as z:
        meta = {k: np.asarray(z[k]) for k in ("rank", "game_id", "player_id", "color")}
    emb, cnt, col, src, files = [], [], {k: [] for k in meta}, [], {}
    for name, path in (("D", K.OPP_D), ("E", K.OPP_E)):
        v = load_vec(path)
        if v["ckpt_sha256"] != hrec["ckpt_sha256"] or v["embed_cfg"]["weights"] != "raw" or v["embed_cfg"]["tta"] != 1:
            raise SystemExit(f"{path}: not S0 raw tta1 vectors")
        emb.append(v["emb"])
        cnt.append(v["count"])
        for k in meta:
            col[k].append(meta[k][v["store_row"]].astype(str))
        src += [name] * len(v["emb"])
        files[name] = {"file": rel(path), "sha256": K.sha256_file(path), "games": int(len(v["emb"]))}
    g = {k: np.concatenate(v) for k, v in col.items()}
    g.update(emb=np.concatenate(emb), count=np.concatenate(cnt), src=np.array(src))
    return g, Wn, cr, hrec, files


def cmd_kernel(a):
    st = K.limits(log)
    t0 = time.time()
    if os.path.exists(K.KERNEL):
        raise SystemExit("kernel.json exists (chosen once)")
    if not committed_clean(VECTORS_JSON) or not K.read_json(VECTORS_JSON)["ok"]:
        raise SystemExit("vectors.json is not committed, clean and ok")
    cs = code_state()
    if cs["prereg58_dirty"]:
        raise SystemExit(f"results/prereg58 not committed and clean: {cs['prereg58_dirty']}")
    g, Wn, cr, hrec, files = kernel_games()
    with np.load(K.guard(K.HEAD_NPZ), allow_pickle=False) as z:
        class_ids = set(np.asarray(z["class_id"]).astype(str).tolist())
    ik = inner_keys_arrays()
    inner_pid = set(ik["player_id"].tolist())
    players = sorted(set(g["player_id"].tolist()))
    overlap = {"kernel_and_inner_players": len(set(players) & inner_pid),
               "kernel_and_s0_classes": len(set(players) & class_ids),
               "inner_and_s0_classes": len(inner_pid & class_ids)}
    if any(overlap.values()):
        raise SystemExit(f"player overlap: {overlap}")
    hd, jrec = hd_join(g["rank"], g["game_id"], g["player_id"], g["color"])
    r_g = K.rank_index(g["rank"])
    h_g, c_g = K.hd_group(hd), K.colour_index(g["color"])
    ok = g["count"] > 0
    pidx = {p: i for i, p in enumerate(players)}
    gp = np.array([pidx[p] for p in g["player_id"].tolist()], np.int64)
    p_rank = np.zeros(len(players), np.int64)
    p_rank[gp] = r_g
    if not (p_rank[gp] == r_g).all():
        raise SystemExit("a kernel player with two ranks")
    p_src = np.empty(len(players), dtype="U1")
    p_src[gp] = g["src"]
    p_fold = np.array([K.player_fold(p) for p in players], np.int64)
    f_g = p_fold[gp]
    log(f"kernel games {len(gp)} ({int(ok.sum())} with a sample), players {len(players)}; q for tau {K.TAUS} ...")
    Q = K.class_q(g["emb"], Wn, cr, K.TAUS)
    log(f"q done ({time.time() - t0:.0f}s)")
    idx = np.arange(len(gp))
    per_tau = {}
    for t in K.TAUS:
        ll = np.zeros(len(gp))
        lln = np.zeros(len(gp))
        for f in range(K.FOLDS):
            Kf = K.kernel(K.kernel_counts(Q[t], r_g, h_g, c_g, ok & (f_g != f)))
            te = ok & (f_g == f)
            lg = K.game_ll(Q[t][te], Kf, h_g[te], c_g[te])
            ll[te] = lg[np.arange(int(te.sum())), r_g[te]]
            m = lg.max(axis=1, keepdims=True)
            lgn = lg - m - np.log(np.exp(lg - m).sum(axis=1, keepdims=True))
            lln[te] = lgn[np.arange(int(te.sum())), r_g[te]]
        per_tau[str(t)] = {"heldout_mean_loglik": float(ll[ok].mean()),
                           "heldout_mean_loglik_by_fold": [float(ll[ok & (f_g == f)].mean()) for f in range(K.FOLDS)],
                           "record_only_heldout_mean_log_posterior_true_rank": float(lln[ok].mean())}
        log(f"tau {t}: {per_tau[str(t)]}")
    tau = max(K.TAUS, key=lambda t: (per_tau[str(t)]["heldout_mean_loglik"], -t))
    counts = K.kernel_counts(Q[tau], r_g, h_g, c_g, ok)
    Kfull = K.kernel(counts)
    # record: the standalone competition score of the term on the kernel players, kernel cross-fitted by player fold
    rng = np.random.default_rng(SEED)
    SM = K.score_matrix()
    by_p = [idx[ok & (gp == i)] for i in range(len(players))]
    res = {n: {"score": [], "rank": [], "src": []} for n in NS}
    prior_only = {"score": [], "rank": []}
    for f in range(K.FOLDS):
        Kf = K.kernel(K.kernel_counts(Q[tau], r_g, h_g, c_g, ok & (f_g != f)))
        te = np.flatnonzero(ok & (f_g == f))
        lg = np.zeros((len(gp), K.L))
        lg[te] = K.game_ll(Q[tau][te], Kf, h_g[te], c_g[te])
        prior = np.bincount(p_rank[p_fold != f], minlength=K.L) + 1.0
        lprior = np.log(prior / prior.sum())
        for i in np.flatnonzero(p_fold == f).tolist():
            prior_only["score"].append(float(SM[int(K.decode(lprior[None])[0]), p_rank[i]]))
            prior_only["rank"].append(int(p_rank[i]))
            for n in NS:
                if len(by_p[i]) < n:
                    continue
                draws = np.stack([lg[rng.choice(by_p[i], n, replace=False)].sum(0) for _ in range(DRAWS)])
                pred = K.decode(lprior[None] + draws)
                res[n]["score"] += SM[pred, p_rank[i]].tolist()
                res[n]["rank"] += [int(p_rank[i])] * DRAWS
                res[n]["src"] += [str(p_src[i])] * DRAWS
    standalone = {"what": "opponent-identity term alone (prior = rank frequencies of the other folds' players), n "
                          "random games of one kernel player, 20 draws per player, kernel from the other 4 player "
                          "folds at the chosen tau; decoded by expected competition score; record only",
                  "prior_only_mean": float(np.mean(prior_only["score"]))}
    for n in NS:
        sc, rk, sr = np.array(res[n]["score"]), np.array(res[n]["rank"]), np.array(res[n]["src"])
        standalone[f"n{n}"] = {"mean": float(sc.mean()), "rank_balanced": K.rank_balanced(sc, rk),
                               "questions": int(len(sc)), "D_players_mean": float(sc[sr == "D"].mean()),
                               "E_players_mean": float(sc[sr == "E"].mean()),
                               "by_rank": [float(sc[rk == k].mean()) for k in range(K.L)]}
    hc_games = [[int((ok & (h_g == h) & (c_g == c)).sum()) for c in range(K.N_COL)] for h in range(K.N_HD)]
    rep = {"format": "c10-prereg58 kernel v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seconds": round(time.time() - t0, 1), "code": cs,
           "what": "PLAN.md: K[h, c][d] = P(opponent's class rank - player's rank = d | HD group h, player's colour "
                   "c) from soft counts q_g[r_g + d] with add-0.5 smoothing, on the D and E players of prereg36's "
                   "split (labelled, never seen by S0, no inner player); tau by 5-fold (by player) held-out "
                   "likelihood. No inner game is read.",
           "taus": list(K.TAUS), "tau": int(tau), "per_tau": per_tau,
           "selection_rule": "largest held-out mean log(sum_s q_g[s] K[h_g, c_g][s - r_g] + 1e-9) over the games of "
                             "the 5 held-out player folds (NOTES.md 3); ties -> the smaller tau",
           "index": {"h": ["HD 0", "HD 1", "HD 2+"], "c": ["player B", "player W"],
                     "d": list(range(-(K.L - 1), K.L)), "ranks": list(K.RANKS)},
           "K": Kfull.tolist(), "soft_counts": counts.tolist(), "games_per_h_c": hc_games, "smoothing": K.SMOOTH,
           "players": {"total": len(players), "D": int((p_src == "D").sum()), "E": int((p_src == "E").sum()),
                       "per_rank": np.bincount(p_rank, minlength=K.L).tolist(),
                       "per_fold": np.bincount(p_fold, minlength=K.FOLDS).tolist()},
           "games": {"total": int(len(gp)), "with_sample": int(ok.sum()), "without_sample": int((~ok).sum()),
                     "D": int((g["src"] == "D").sum()), "E": int((g["src"] == "E").sum())},
           "overlap": overlap, "hd_join": jrec, "head": hrec, "vector_files": files,
           "standalone_score_record": standalone, "peak_working_set_gb": peak_gb(st)}
    K.write_json(K.KERNEL, rep)
    log(json.dumps({"tau": tau, "players": rep["players"], "games": rep["games"],
                    "standalone": {k: v for k, v in standalone.items() if k != "what"}}))
    return 0


def load_kernel():
    k = K.read_json(K.KERNEL)
    Km = np.asarray(k["K"], np.float64)
    if Km.shape != (K.N_HD, K.N_COL, K.ND) or not np.allclose(Km.sum(axis=2), 1.0):
        raise SystemExit("kernel.json: not a [3, 2, 19] kernel")
    return Km, int(k["tau"]), k


# ================================================================ seam
def cmd_seam(a):
    st = K.limits(log)
    if os.path.exists(K.SEAM):
        raise SystemExit("seam.json exists")
    if not committed_clean(K.KERNEL):
        raise SystemExit("kernel.json is not committed and clean")
    _mods, _inner, _terms, _Emap, _q25, seam = build_q25()
    cs = code_state()
    rep = {"format": "c10-prereg58 seam v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "code": cs,
           "what": "Q25 on inner through this code path (p58_eval.build_q25); no number of the new term is computed",
           **seam, "peak_working_set_gb": peak_gb(st)}
    K.write_json(K.SEAM, rep)
    log(json.dumps({k: rep[k] for k in ("Q25", "Q25_mono", "Q25_bitwise", "Q25_mono_bitwise", "ok", "seconds",
                                        "peak_working_set_gb")}))
    return 0 if seam["ok"] else 1


# ================================================================ the judgement (once)
def inner_term(inner, Km, tau):
    """Per-game l_g(r) on the full inner row space (0 for a game without an opponent-side sample or not in any
    question) and its record."""
    Wn, cr, hrec = K.load_head()
    v = load_vec(K.OPP_INNER)
    rows = np.load(K.INNER_ROWS)
    if not (np.array_equal(v["store_row"], rows) and v["ckpt_sha256"] == hrec["ckpt_sha256"]
            and v["rows_cfg"]["side"] == "opp"):
        raise SystemExit("inner opponent-side vectors are not the registered file")
    keys = inner.d["inner"]["keys"]
    hd, jrec = hd_join(keys["rank"], keys["game_id"], keys["player_id"], keys["color"])
    h, c = K.hd_group(hd), K.colour_index(keys["color"])
    q = K.class_q(v["emb"], Wn, cr, (tau,))[tau]
    lg = np.zeros((inner.n_inner, K.L))
    has = np.zeros(inner.n_inner, bool)
    okv = v["count"] > 0
    lg[rows[okv]] = K.game_ll(q[okv], Km, h[rows[okv]], c[rows[okv]])
    has[rows[okv]] = True
    rec = {"games_with_term": int(has.sum()), "games_in_questions": int(len(rows)),
           "games_without_sample": int((~okv).sum()), "hd_join": jrec,
           "hd_groups_of_question_games": np.bincount(h[rows], minlength=K.N_HD).tolist(),
           "player_colour_of_question_games": np.bincount(c[rows], minlength=K.N_COL).tolist(),
           "mean_max_q_record": float(q[okv].max(axis=1).mean())}
    return lg, has, rec


def cmd_eval(a):
    st = K.limits(log)
    t0 = time.time()
    smoke = bool(a.smoke)
    out_path = os.path.join(K.DATA, "smoke", "eval.json") if smoke else K.EVAL
    if not smoke:
        for p in (K.KERNEL, K.SEAM, VECTORS_JSON, ROWS_JSON):
            if not committed_clean(p):
                raise SystemExit(f"{os.path.basename(p)} is not committed and clean")
        if not K.read_json(K.SEAM)["ok"]:
            raise SystemExit("seam.json is not ok")
        if os.path.exists(K.EVAL) or os.path.exists(K.STARTED):
            raise SystemExit("eval.json / eval_started.json exist: judged once")
    Km, tau, krec = load_kernel()
    mods, inner, terms7, Emap, q25, seam = build_q25()
    E41, E23, r, K23 = mods
    rf = r.rf
    cs = code_state()
    if not smoke and (cs["prereg58_dirty"] or cs["imported_dirty"]):
        raise SystemExit(f"not committed and clean: {cs['prereg58_dirty']} {cs['imported_dirty']}")
    log(f"seam {seam['ok']}: Q25 {q25['rb']!r}, single-colour {q25['mono_rb']!r} ({time.time() - t0:.0f}s)")
    if not seam["ok"]:
        raise SystemExit("seam failed inside the judgement run: no number of the new term is computed")
    if not smoke:
        K.write_json(K.STARTED, {"format": "c10-prereg58 eval started", "created": r.now_iso(), "code": cs,
                                 "tau": tau, "kernel_sha256_lf": K.sha_lf(K.KERNEL)})
    # ---- the new term (the first inner number of it is computed below)
    lg, has, trec = inner_term(inner, Km, tau)
    if lg.shape != (inner.n_inner, K.L) or not np.isfinite(lg).all():
        raise SystemExit("the per-game term is not a finite [n_inner, 10] array")
    if smoke:      # pipeline check only: the real per-game values are discarded, never pooled, never scored
        lg = np.random.default_rng(NOISE_SEED).standard_normal((inner.n_inner, K.L))
        trec = {"smoke": "per-game term replaced by default_rng(12345) normal noise"}
    di = inner.d["inner"]
    E58 = {w: rf.pooled(S, lg, "log") for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]),
                                                    ("mono", di["S_mono"]))}
    terms8 = terms7 + [TERM]
    cand = inner.run(terms8, {**Emap, TERM: E58})
    cmp_ = inner.compare(cand, q25)
    gates = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                      cmp_["mono_by_colour"])
    gates["G4"]["lower_bound_only_passed"] = bool(cmp_["mono_ci95"][0] is not None and cmp_["mono_ci95"][0] > 0)
    adopt = bool(gates["passed"]) and not smoke
    log(f"Q58 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(gates)}")
    # ---- record only
    ones = {w: np.ones(len(E58[w]), bool) for w in ("eval", "mono")}
    alone_mean = E23.alone(inner, {w: E58[w] for w in ("eval", "mono")}, ones)
    alone_sum = E23.alone(inner, {"eval": E58["eval"] * inner.Xe.n[:, None], "mono": E58["mono"] * inner.Xm.n[:, None]},
                          ones)
    rep = {"format": "c10-prereg58 eval v1", "smoke": smoke, "created": r.now_iso(),
           "seconds": round(time.time() - t0, 1), "code": cs,
           "smoke_note": "the OPP term replaced by noise: numbers meaningless" if smoke else None,
           "what": "PLAN.md: inner-only judgement, once, of Q58 = Q25's seven terms + OPP (the pooled mean over the "
                   "question's games of l_g(r) = log(sum_s q_g[s] K[h_g, c_g][s - r] + 1e-9), added exactly as "
                   "prereg44 added CNN44: one more E term of Inner.run, 5-fold player OOF fusion, rule B); no "
                   "validation scoring, no test data",
           "adopt": adopt,
           "status": ("all three gates passed: freeze, write PREREG_58 (PLAN.md)" if adopt
                      else "gates not passed: not adopted"),
           "tau": tau, "kernel_sha256_lf": K.sha_lf(K.KERNEL), "seam": seam, "term": trec,
           "candidate": {"name": "Q58", "terms": terms8, **inner.short(cand, q25)},
           "baseline": {"name": "Q25", "terms": terms7, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"],
                        "single_colour_by_colour": {c: q25["mrec"]["by_colour"][c]["rank_balanced"]
                                                    for c in q25["mrec"]["by_colour"]}},
           "gates": gates,
           "record_only": {"OPP_alone_rule_b_mean_pooled": alone_mean,
                           "OPP_alone_rule_b_sum_over_games": alone_sum,
                           "by_game_count": E23.by_count(inner, cand, q25),
                           "by_level": E23.by_level(inner, cand, q25),
                           "weights": E23.w_record(cand, terms8),
                           "q25_weights": E23.w_record(q25, terms7),
                           "fusion_a_gamma_by_fold": [{"fold": fr["fold"], "a": fr["params"]["a"],
                                                       "gamma": fr["params"]["gamma"]} for fr in cand["folds"]]},
           "peak_working_set_gb": peak_gb(st)}
    K.write_json(out_path, rep)
    log(f"-> {out_path}: {rep['status']} ({time.time() - t0:.0f}s, peak {peak_gb(st)} GB)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "vectors", "kernel", "seam", "eval"):
        p = sub.add_parser(name)
        if name == "eval":
            p.add_argument("--smoke", action="store_true",
                           help="pipeline check: the OPP term replaced by noise -> data/prereg58/smoke/eval.json")
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "vectors": cmd_vectors, "kernel": cmd_kernel, "seam": cmd_seam, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
