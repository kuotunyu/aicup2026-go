"""prereg18 exploration (competition 2): same-player linking + pooling layer L18, inner-only simulation of PLAN.md,
judged once. CPU only (.venv).

    Git Bash, repository root (a subst drive when the worktree path contains 'AppData\\Local'); PYTHONDONTWRITEBYTECODE=1;
    .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg18/explore
    $S/p18_eval.py eval            -> $S/eval_report.json (judged once; the code must be committed and clean)
    $S/p18_eval.py eval --smoke    -> data/prereg18/smoke/eval_report.json (noise features, 3 pools per scenario:
                                      pipeline and timing only, never a result)

Inputs (read only): data/prereg17/prep (probe P1 features, 09's prep rebuilt; seam P), data/prereg12/q09_inner (inner
P1 features), the probe sets.json files (colours), Q09's frozen p1e_map.npz, u1_lib's inner pseudo-questions and
per-game terms through tools/p09_rank_fuse.Inner. Never data/local, never validation features, never tools/score.py.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
FT_DIR = os.path.join(REPO, "results", "prereg12", "explore", "finetune")
for p in (TOOLS, FT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402  (pins the p08 tools before importing them)
import p09_forward as FW  # noqa: E402
import ft_eval  # noqa: E402  (prereg12: InnerPrep, guard_path)
import p18_lib as K  # noqa: E402

rf, p4, P7, R = r.rf, r.p4, r.P7, r.R
TOL = K.TOL
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
SETS_JSON = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
             "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "sets.json"),
             "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "sets.json")}
DATA = os.path.join(REPO, "data", "prereg18")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
OWN_FILES = ("PLAN.md", "p18_lib.py", "p18_eval.py", "test_p18.py")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p07_rank_fuse.py",
            "tools/p08_rank_fuse.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py", "tools/p09_map.py",
            "results/prereg12/explore/finetune/ft_eval.py", "results/prereg05/explore/u1/u1_lib.py",
            "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg07/explore/pilot_humansl/hs_lib.py",
            "results/prereg08/p1/p1_fuse.py", "results/prereg08/p1/p1_lib.py",
            "tools/p02_rank_pseudo.py", "tools/p02_rank_opening.py", "tools/p02_moves.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/check_report.json",
            "results/prereg09/rank/tune_report.json", "results/prereg09/rank/frozen_config.json",
            "results/prereg08/rank/p1_map.npz", "results/prereg08/feat/inner_manifest.json")
FORMAT = "c10-prereg18 explore eval v1"
BLOCK = 8192
NOISE_SEED = 12345
guard_path = ft_eval.guard_path


def log(*x):
    print(*x, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(guard_path(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = set()
    for p in prior:
        with open(p, encoding="utf-8") as f:
            heads.add(((json.load(f).get("code") or {}).get("head")))
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def tune_ref():
    with open(Q09_TUNE, encoding="utf-8") as f:
        c = json.load(f)["combinations"]["Q09"]
    return {"Q09": c["oof"]["rank_balanced"], "Q09_mono": c["single_colour"]["rank_balanced"]}


# ---------------------------------------------------------------- per-game arrays and the Q09 path (PLAN.md 4)
class Games:
    """Inner per-game inputs of Q09's terms (full inner row space) and the term builder for arbitrary game groups."""

    def __init__(self, inner, lp_q09):
        d = inner.d
        self.n = inner.n_inner
        self.lp = d["lp"]
        self.lp_hs = d["lp"]["HS"]
        hs_usable, _f, _c = R.pilot_hs_usable(inner.PF, inner.n_inner)
        self.hs_usable = np.asarray(hs_usable, dtype=bool)
        self.covered = inner.covered
        self.lp_p1 = inner.lp_full(lp_q09)
        self.op_o = d["inner"]["op_o"]
        self.op_has = d["inner"]["op_has"]
        self.terms = list(r.Q08_TERMS)

    def question_terms(self, groups):
        S = rf.pool_matrix(groups, self.n)
        Sop = rf.pool_matrix([[g for g in grp if self.op_has[g]] for grp in groups], self.n)
        m = rf.pooled(S, self.lp["C5"], "log")
        Es = []
        for t in self.terms:
            if t == "OP":
                Es.append(np.asarray(Sop @ self.op_o))
            elif t == "HS":
                Es.append(P7.masked_E(S, self.lp_hs, self.hs_usable)[0])
            elif t == "P1TO":
                Es.append(R.p1_term(S, self.lp_p1, self.covered, scale=r.SCALE)[0])
            else:
                Es.append(rf.pooled(S, self.lp[t], "log"))
        n = np.array([len(g) for g in groups], dtype=np.float64)
        return m, Es, n


def predict(thetas, keys, m, Es, n):
    """Rows grouped by key, each group with thetas[key] (p4.gen_logits, softmax) -> P."""
    P = np.full((len(n), r.L), np.nan)
    keys = list(keys)
    for k in sorted(set(keys), key=str):
        ev = np.array([x == k for x in keys])
        z, _ = p4.gen_logits(np.asarray(thetas[k], dtype=np.float64), m[ev], [E[ev] for E in Es], n[ev])
        P[ev] = rf.softmax(z)
    return P


class Thetas:
    """Q09's fold parameters and refits that leave out a set of folds (PLAN.md 4 item 5)."""

    def __init__(self, inner, q09):
        self.folds = {int(fr["fold"]): list(fr["theta"]) for fr in q09["folds"]}
        self.Xf = inner.d["inner"]["X_fit"]
        self.mf, self.Ef = q09["aux"]["mf"], q09["aux"]["Ef"]
        self.cache = {}
        self.refits = 0
        self.uses = 0

    def refit(self, F):
        F = tuple(sorted(set(int(f) for f in F)))
        if F not in self.cache:
            tr = ~np.isin(self.Xf.fold, np.array(F))
            th, _rec = p4.gen_fit_record(self.mf[tr], [E[tr] for E in self.Ef], self.Xf.n[tr], self.Xf.y[tr])
            self.cache[F] = [float(x) for x in th]
            self.refits += 1
        return self.cache[F]

    def get(self, F):
        F = tuple(sorted(set(int(f) for f in F)))
        if len(F) == 1:
            return self.folds[F[0]]
        self.uses += 1
        return self.refit(F)


# ---------------------------------------------------------------- features
def std_block(X, s, e, mu, sd, smoke, tag):
    if smoke:
        return np.random.default_rng([NOISE_SEED, tag, s]).standard_normal((e - s, X.shape[1]))
    return FW.standardise_x(np.asarray(X[s:e], dtype=np.float64), mu, sd)


def fit_plda(probe_X, t_pidx, n_t, mu, sd, smoke):
    """LDA (S_w, S_b over T players' games) + PLDA prior; t_pidx[row] = T player index or -1."""
    acc = K.Accum(n_t, probe_X.shape[1])
    for s in range(0, len(t_pidx), BLOCK):
        e = min(s + BLOCK, len(t_pidx))
        sel = t_pidx[s:e] >= 0
        if sel.any():
            Z = std_block(probe_X, s, e, mu, sd, smoke, 0)[sel]
            acc.add(Z, t_pidx[s:e][sel])
    Sw, Sb, c, means, cnt = acc.scatter()
    V, evals, eps = K.lda_fit(Sw, Sb)
    Um = (means - c) @ V
    b = K.plda_prior(Um, cnt)
    return {"V": V, "c": c, "b": b, "eigvals": evals, "eps": eps, "players": int(len(cnt)),
            "games": int(cnt.sum())}


def project(X, mu, sd, pl, smoke, tag):
    out = np.zeros((X.shape[0], pl["V"].shape[1]))
    for s in range(0, X.shape[0], BLOCK):
        e = min(s + BLOCK, X.shape[0])
        out[s:e] = (std_block(X, s, e, mu, sd, smoke, tag) - pl["c"]) @ pl["V"]
    return out


# ---------------------------------------------------------------- players
def probe_players(prep):
    """Probe colours from the sets.json files (player cross-checked) -> (colour [N], keys [N])."""
    games = {}
    for s, p in SETS_JSON.items():
        with open(guard_path(p), encoding="utf-8") as f:
            games[s] = json.load(f)["probe"]["games"]
    col = np.empty(len(prep.ref), dtype="<U1")
    for i, (s, j, pk) in enumerate(zip(prep.src, prep.ref, prep.players)):
        g = games[str(s)][int(j)]
        if f"{g['rank']}:{g['player_id']}" != str(pk):
            raise SystemExit(f"probe row {i}: sets.json player differs from probe_meta")
        col[i] = g["color"]
    return col, np.asarray(prep.players).astype(str)


def player_games(keys_by_row, colour_by_row, rows, wanted_keys):
    """-> PlayerGames over wanted_keys (order kept) with the given rows (game ids)."""
    idx = {k: i for i, k in enumerate(wanted_keys)}
    lists = [{c: [] for c in K.COLOURS} for _ in wanted_keys]
    for g in rows:
        i = idx.get(keys_by_row[g])
        if i is not None:
            lists[i][colour_by_row[g]].append(int(g))
    gl = [{c: np.array(sorted(d[c]), dtype=np.int64) for c in K.COLOURS} for d in lists]
    return K.PlayerGames(wanted_keys, gl)


# ---------------------------------------------------------------- pools
def pool_pairs(qs, U, row_of, b):
    M, n = K.question_means(qs, U, row_of)
    llr = K.plda_llr(M, n, b)
    same = K.same_matrix(qs)
    return llr, same


def build_pools(pg, sid, n_pools, seed):
    """All pools of one scenario (generation needs no feature, label or score) + seam S problems."""
    sc = K.scenario(sid)
    pools, atts, probs = [], [], []
    for i in range(n_pools):
        qs, att = K.make_pool(seed, sid, i, pg, sc["p_multi"], sc["sizes"])
        probs += K.check_pool(qs, sc["p_multi"], sc["sizes"], pg)
        pools.append(qs)
        atts.append(att)
    return pools, atts, probs


def cal_pairs(pools, U, row_of, b):
    llrs, sames = [], []
    for qs in pools:
        llr, same = pool_pairs(qs, U, row_of, b)
        llrs.append(K.upper_pairs(llr))
        sames.append(K.upper_pairs(same))
    return llrs, sames


def eval_scenario(pools, U, row_of, b, tau, c, games, thetas, player_fold, player_y):
    n_pools = len(pools)
    rows = {k: [] for k in ("player", "y", "k", "pool", "base", "layer", "oracle", "pooled")}
    st = {"pools": n_pools, "on": 0, "links_raw": 0, "tp_raw": 0, "true_pairs": 0, "links_used": 0, "tp_used": 0,
          "component_sizes": {}, "too_big_components": [], "levels": np.zeros(r.L, int), "levels_by_pool": [],
          "multi_questions": 0, "multi_questions_pooled_clean": 0}
    for i, qs in enumerate(pools):
        llr, same = pool_pairs(qs, U, row_of, b)
        iu = np.triu_indices(len(qs), 1)
        link_u = llr[iu] >= tau
        same_u = same[iu]
        st["links_raw"] += int(link_u.sum())
        st["tp_raw"] += int((link_u & same_u).sum())
        st["true_pairs"] += int(same_u.sum())
        on = bool(link_u.sum() >= c)
        players = np.array([q["player"] for q in qs])
        y = player_y[players]
        st["levels"] += np.bincount(y, minlength=r.L)
        st["levels_by_pool"].append(np.bincount(y, minlength=r.L))
        groups = [list(q["games"]) for q in qs]
        m, Es, n = games.question_terms(groups)
        fold = player_fold[players]
        P_base = predict(thetas.folds, fold.tolist(), m, Es, n)
        pred_base = rf.rule_b_np(P_base)
        pred_layer = pred_base.copy()
        pooled_flag = np.zeros(len(qs), dtype=bool)
        comps = []
        if on:
            st["on"] += 1
            st["links_used"] += int(link_u.sum())
            st["tp_used"] += int((link_u & same_u).sum())
            link = np.zeros(same.shape, dtype=bool)
            link[iu] = link_u
            comps, big = K.pooled_groups(link | link.T)
            st["too_big_components"] += big
            for comp in comps:
                st["component_sizes"][str(len(comp))] = st["component_sizes"].get(str(len(comp)), 0) + 1
                ug = sorted(set(int(g) for j in comp for g in qs[j]["games"]))
                th = thetas.get([int(fold[j]) for j in comp])
                mm, EE, nn = games.question_terms([ug])
                pc = rf.rule_b_np(predict({0: th}, [0], mm, EE, nn))[0]
                pred_layer[comp] = pc
                pooled_flag[comp] = True
        # oracle: the true groups pooled (record only)
        pred_or = pred_base.copy()
        by = {}
        for j, q in enumerate(qs):
            by.setdefault(q["player"], []).append(j)
        for p, js in by.items():
            if len(js) > 1:
                st["multi_questions"] += len(js)
                ug = sorted(set(int(g) for j in js for g in qs[j]["games"]))
                mm, EE, nn = games.question_terms([ug])
                pred_or[js] = rf.rule_b_np(predict({0: thetas.folds[int(player_fold[p])]}, [0], mm, EE, nn))[0]
                # clean: the layer pooled exactly this true group
                if on and any(set(cmp_) == set(js) for cmp_ in comps):
                    st["multi_questions_pooled_clean"] += len(js)
        rows["player"] += players.tolist()
        rows["y"] += y.tolist()
        rows["k"] += [q["k"] for q in qs]
        rows["pool"] += [i] * len(qs)
        rows["base"] += K.official_score(pred_base, y).tolist()
        rows["layer"] += K.official_score(pred_layer, y).tolist()
        rows["oracle"] += K.official_score(pred_or, y).tolist()
        rows["pooled"] += pooled_flag.tolist()
    st["levels"] = st["levels"].tolist()
    lb = np.array(st.pop("levels_by_pool"))
    st["levels_per_pool_min_max"] = [lb.min(0).tolist(), lb.max(0).tolist()]
    return {k: np.asarray(v) for k, v in rows.items()}, st


def summarise(rows, st, n_players):
    y, pl = rows["y"], rows["player"]
    boot = rf.PlayerBootstrap(pl, y, n_players)
    out = {"questions": int(len(y)), "q09": K.rank_balanced(rows["base"], y),
           "l18": K.rank_balanced(rows["layer"], y), "oracle": K.rank_balanced(rows["oracle"], y)}
    out["diff"] = out["l18"] - out["q09"]
    d = boot.stats(rows["layer"]) - boot.stats(rows["base"])
    out["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    do = boot.stats(rows["oracle"]) - boot.stats(rows["base"])
    out["oracle_diff"] = out["oracle"] - out["q09"]
    out["oracle_ci95"] = [float(np.percentile(do, 2.5)), float(np.percentile(do, 97.5))]
    out["by_k"] = {str(k): {"diff": K.rank_balanced(rows["layer"][rows["k"] == k], y[rows["k"] == k])
                            - K.rank_balanced(rows["base"][rows["k"] == k], y[rows["k"] == k])}
                   for k in K.KS}
    out["questions_pooled"] = int(rows["pooled"].sum())
    lr = st["links_raw"]
    out["linking"] = {"pairs_linked_raw": lr, "precision_raw": (st["tp_raw"] / lr) if lr else None,
                      "recall_raw": (st["tp_raw"] / st["true_pairs"]) if st["true_pairs"] else None,
                      "true_pairs": st["true_pairs"], "pairs_linked_used": st["links_used"],
                      "precision_used": (st["tp_used"] / st["links_used"]) if st["links_used"] else None,
                      "recall_used": (st["tp_used"] / st["true_pairs"]) if st["true_pairs"] else None}
    out["detector_on_rate"] = st["on"] / st["pools"]
    out["component_sizes"] = st["component_sizes"]
    out["too_big_components"] = len(st["too_big_components"])
    out["multi_questions"] = st["multi_questions"]
    out["multi_questions_pooled_clean"] = st["multi_questions_pooled_clean"]
    out["levels_questions"] = st["levels"]
    out["levels_per_pool_min_max"] = st["levels_per_pool_min_max"]
    return out


# ---------------------------------------------------------------- eval
def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state()
    earlier = None
    if not smoke and not a.pools_only:
        if cs["dirty"]:
            raise SystemExit(f"the prereg18 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    prep = ft_eval.InnerPrep(guard_path(INNER_PREP))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = tune_ref()
    mp = FW.load_map(Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    q09 = inner.run(list(r.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_q09})
    # ---- seam 1
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    # ---- seam Q: own term builder reproduces Q09's P / Pm bit for bit
    games = Games(inner, lp_q09)
    thetas = Thetas(inner, q09)
    din = inner.d["inner"]
    ms = din["ms"]
    m_, E_, n_ = games.question_terms(ms["groups"])
    Pm_own = predict(thetas.folds, din["X_mono"].fold.tolist(), m_, E_, n_)
    m_, E_, n_ = games.question_terms(din["eval_groups"])
    P_own = predict(thetas.folds, din["X_eval"].fold.tolist(), m_, E_, n_)
    seam_q = {"mono_equal": bool(np.array_equal(Pm_own, q09["Pm"])), "eval_equal": bool(np.array_equal(P_own, q09["P"]))}
    seam_q["ok"] = bool(seam_q["mono_equal"] and seam_q["eval_equal"])
    # ---- seam theta
    seam_t = {}
    for f in range(5):
        tr = ~np.isin(thetas.Xf.fold, np.array([f]))
        th, _rec = p4.gen_fit_record(thetas.mf[tr], [E[tr] for E in thetas.Ef], thetas.Xf.n[tr], thetas.Xf.y[tr])
        seam_t[str(f)] = bool([float(x) for x in th] == thetas.folds[f])
    seam_t["ok"] = all(seam_t.values())
    # ---- seam P
    want = (r.read_json(Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": sha_file(os.path.join(PROBE_PREP, "probe_X.npy")),
           "inner_X.npy": sha_file(os.path.join(INNER_PREP, "inner_X.npy"))}
    seam_p = {"want": {k: want.get(k) for k in got}, "got": got,
              "ok": bool(all(want.get(k) == v for k, v in got.items()))}
    log(f"seam 1 {seam1} {seam}; seam Q {seam_q}; seam theta {seam_t}; seam P {seam_p['ok']}")
    if not (seam1 and seam_q["ok"] and seam_t["ok"] and seam_p["ok"]):
        write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_q": seam_q,
                              "seam_theta": seam_t, "seam_p": seam_p})
        return 1
    # ---- players (seam D)
    pprep = r.Prep(guard_path(PROBE_PREP))
    pcol, pkeys = probe_players(pprep)
    pcount = {}
    for k in pkeys:
        pcount[k] = pcount.get(k, 0) + 1
    bits = {k: K.split_bit(k) for k in pcount}
    t_players = sorted(k for k in pcount if bits[k] == 0 and pcount[k] >= K.T_MIN_GAMES)
    c_players = sorted(k for k in pcount if bits[k] == 1)
    players = din["players"]
    i_keys = [f"{p['rank']}:{p['player_id']}" for p in players]
    seam_d = {"T_players": len(t_players), "C_players": len(c_players), "I_players": len(i_keys),
              "T_and_C": len(set(t_players) & set(c_players)),
              "probe_and_I": len(set(pcount) & set(i_keys)),
              "probe_and_I_player_id_only_record": len({k.split(":", 1)[1] for k in pcount}
                                                       & {p["player_id"] for p in players})}
    seam_d["ok"] = bool(seam_d["T_and_C"] == 0 and seam_d["probe_and_I"] == 0)
    log(f"seam D {seam_d}")
    if not seam_d["ok"]:
        write_json(out_path, {**base, "status": "stop: seam D failed", "seam_d": seam_d})
        return 1
    # ---- all pools first (no feature, label or score; seam S), then the judged part starts (PLAN.md 9)
    tp = time.time()
    keys = din["keys"]
    usable = games.covered & games.hs_usable
    row_inner = np.full(games.n, -1, dtype=np.int64)
    row_inner[prep.inner_ref] = np.arange(len(prep.inner_ref))
    i_row_keys = np.array([f"{rk}:{pid}" for rk, pid in zip(keys["rank"], keys["player_id"])])
    pg_i = player_games(i_row_keys, np.asarray(keys["color"]).astype(str), np.nonzero(usable)[0], i_keys)
    pg_c = player_games(pkeys, pcol, np.arange(len(pkeys)), c_players)
    row_probe = np.arange(len(pkeys))
    n_cal = {sid: (K.SMOKE_POOLS["cal"] if smoke else K.scenario(sid)["cal"]) for sid in (0, 1)}
    seed_cal = K.SEED_SMOKE if smoke else K.SEED_CAL
    seed_eval = K.SEED_SMOKE if smoke else K.SEED_EVAL
    cal_pools, eval_pools, attempts, sprobs = {}, {}, {}, []
    for sid in (0, 1):
        cal_pools[sid], att, pr = build_pools(pg_c, sid, n_cal[sid], seed_cal)
        attempts[f"cal_{K.scenario(sid)['name']}"] = int(sum(att))
        sprobs += pr
    for sc_ in K.SCENARIOS:
        n_pools = K.SMOKE_POOLS["eval"] if smoke else sc_[5]
        eval_pools[sc_[0]], att, pr = build_pools(pg_i, sc_[0], n_pools, seed_eval)
        attempts[f"eval_{sc_[1]}"] = int(sum(att))
        sprobs += pr
    pool_rec = {"seam_s_problems": sprobs[:20], "voided_attempts": attempts, "seconds": round(time.time() - tp, 1),
                "I_players_with_usable_games": int(sum(1 for c_ in pg_i.count if c_.sum() > 0))}
    log(f"pools {pool_rec}")
    if sprobs:
        write_json(out_path, {**base, "status": "stop: seam S failed", "pools": pool_rec})
        return 1
    if a.pools_only:
        write_json(os.path.join(DATA, "pools_check.json"), {**base, "pools": pool_rec, "created": r.now_iso(),
                                                             "note": "pool feasibility only; nothing judged"})
        return 0
    if not smoke:
        write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                             "rerun_reason": a.reason if a.rerun else None})
    # ---- LDA / PLDA on T
    tc = time.time()
    mu, sd = mp["mu_x"], mp["sd_x"]
    t_index = {k: i for i, k in enumerate(t_players)}
    t_pidx = np.array([t_index.get(k, -1) for k in pkeys], dtype=np.int64)
    pl = fit_plda(pprep.probe_X, t_pidx, len(t_players), mu, sd, smoke)
    out_dir = SMOKE_DIR if smoke else DATA
    os.makedirs(out_dir, exist_ok=True)
    plda_path = os.path.join(out_dir, "plda.npz")
    np.savez(plda_path, V=pl["V"], c=pl["c"], b=pl["b"], eigvals=pl["eigvals"])
    plda_rec = {"players": pl["players"], "games": pl["games"], "eps": pl["eps"], "sha256": sha_file(plda_path),
                "b_summary": [float(np.min(pl["b"])), float(np.median(pl["b"])), float(np.max(pl["b"]))],
                "eigvals_top5": [float(x) for x in pl["eigvals"][:5]], "seconds": round(time.time() - tc, 1)}
    log(f"PLDA {plda_rec}")
    U_probe = project(pprep.probe_X, mu, sd, pl, smoke, 0)  # same noise tag as the fit
    U_inner = project(prep.inner_X, mu, sd, pl, smoke, 2)
    # ---- calibration on C
    llr1, same1 = cal_pairs(cal_pools[1], U_probe, row_probe, pl["b"])
    llr1, same1 = np.concatenate(llr1), np.concatenate(same1)
    tau, tau_rec = K.choose_tau(llr1, same1)
    llr0, _s0 = cal_pairs(cal_pools[0], U_probe, row_probe, pl["b"])
    counts0 = [int((u >= tau).sum()) for u in llr0]
    c_thr = K.choose_c(counts0)
    smoke_forced = None
    if smoke and not np.isfinite(tau):
        # smoke only (PLAN.md 9): noise never links, so force a threshold to exercise the pooling path
        tau = float(np.percentile(llr1, 99.9))
        c_thr = 1
        smoke_forced = {"tau": tau, "c": c_thr}
    cal_rec = {"tau": tau if np.isfinite(tau) else "inf", "tau_record": tau_rec, "c": c_thr,
               "smoke_forced": smoke_forced, "S0_counts_max": max(counts0),
               "S0_pools_with_any_link": int(sum(1 for x in counts0 if x > 0)),
               "S0_cal_on_rate": float(np.mean(np.array(counts0) >= c_thr)), "pools": n_cal}
    log(f"calibration {cal_rec}")
    del llr1, same1, llr0
    # ---- evaluation on I
    player_fold = np.array([p["fold"] for p in players], dtype=np.int64)
    player_y = np.array([r.LEVELS.index(p["rank"]) for p in players], dtype=np.int64)
    results = {}
    for s in K.SCENARIOS:
        ts = time.time()
        rows, st = eval_scenario(eval_pools[s[0]], U_inner, row_inner, pl["b"], tau, c_thr, games, thetas,
                                 player_fold, player_y)
        results[s[1]] = summarise(rows, st, len(players))
        results[s[1]]["seconds"] = round(time.time() - ts, 1)
        log(f"{s[1]}: {json.dumps({k: results[s[1]][k] for k in ('q09', 'l18', 'diff', 'ci95', 'detector_on_rate')})}"
            f" linking {results[s[1]]['linking']}")
    g = K.gates(results["S0"]["diff"], results["S0"]["detector_on_rate"], results["S30"]["diff"],
                results["S30"]["ci95"][0])
    status = "gates passed (worth a PREREG_18; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("P1 features replaced by normal noise, 3 pools per scenario: numbers are meaningless"
                          if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: L18 (PLDA same-player linking on LDA-128 of the 1,540 P1 features, pooled Q09 decision) "
                   "on simulated 400-question single-colour pools of inner training players; linker trained (T) and "
                   "calibrated (C) on disjoint probe players; no validation data, no scoring",
           "status": status, "seam": seam, "seam_q": seam_q, "seam_theta": seam_t, "seam_p": seam_p,
           "seam_d": seam_d, "pools": pool_rec, "plda": plda_rec, "calibration": cal_rec, "scenarios": results, "gates": g,
           "theta_refits_distinct": thetas.refits, "theta_refit_uses": thetas.uses, "q09_map_sha256": map_sha}
    write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pools-only", action="store_true",
                   help="seams + generation of every pool with the real seeds (seam S), then stop; nothing judged")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
