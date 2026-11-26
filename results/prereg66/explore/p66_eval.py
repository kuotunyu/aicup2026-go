"""prereg66 explore (PLAN.md, NOTES.md, frozen.json): competition 1 new questions, Gemini Carbon's
04_R_final_PRIMARY back end on v2-7's c49 vectors. CPU only, one process at a time.

    PY results/prereg66/explore/p66_eval.py prep-sp                 # copy Carbon's sp128 arrays -> data/prereg66/sp128.npz
    PY results/prereg66/explore/p66_eval.py seam --part d|l|e       # D / L_c49 vs Carbon's numbers; E v2-7 bit for bit
    PY results/prereg66/explore/p66_eval.py prep-e                  # E: z_multi, z_sp128, z_rk_tri per pool (no scores)
    PY results/prereg66/explore/p66_eval.py e-use
    PY results/prereg66/explore/p66_eval.py judge --e-use N         # once -> eval.json
    PY results/prereg66/explore/p66_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg66/explore/p66_eval.py assemble --src <official tests dir>

A66 = csls_20(F_078 + (5.5 z_p1_orig + 3 z_multi + 3 z_sp128) + 1.8 z_rk) + w_post(k) z_rk
(Carbon 04_final_registerable_recipe.py lines 341-355). Nothing here opens the validation-answer folder or runs
tools/score.py.
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
P63_DIR = os.path.join(REPO, "results", "prereg63", "explore")
if P63_DIR not in sys.path:
    sys.path.insert(0, P63_DIR)
import p63_eval as P63  # noqa: E402   (imports prereg49 / 45 / 43 / 39 / 38 / 36 paths; unchanged)

P49, E43, E45, P39E = P63.P49, P63.E43, P63.E45, P63.P39E
BK, EV, st, L, sim, Q42 = P63.BK, P63.EV, P63.st, P63.L, P63.sim, P63.Q42
FW09, FW25 = P63.FW09, P63.FW25
_rel, read_json = P63._rel, P63.read_json

GEM = os.path.join(REPO, "data", "advice", "gemini_1121b")
DATA = os.path.join(REPO, "data", "prereg66")
SP128 = os.path.join(DATA, "sp128.npz")
Z_E = os.path.join(DATA, "z_E.npz")
PLAN = os.path.join(HERE, "PLAN.md")
NOTES = os.path.join(HERE, "NOTES.md")
FROZEN = os.path.join(HERE, "frozen.json")
SEAM_JSON = os.path.join(HERE, "seam.json")
PREP_JSON = os.path.join(HERE, "prep_e.json")
EVAL_JSON = os.path.join(HERE, "eval.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg66", "apply")
C49_HEAD = os.path.join(REPO, "results", "prereg64", "rank", "c49_head.npz")
S0_HEAD = os.path.join(GEM, "s0_head.npz")

# ---- frozen.json (Carbon FROZEN_RECIPE, 04_final_registerable_recipe.py lines 37-72)
R = {"cnn_whiten_lam": 0.78, "w_p1_orig": 5.5, "w_p1_multi": 3.0, "w_c13_sp128": 3.0, "c13_pool_lam": 0.50,
     "rk_w_s25": 1.0, "rk_w_p1to": 0.6, "rk_w_cnn": 0.5, "rk_cnn_tau": 35.0, "rk_sigma": 0.85, "rk_eps": 1e-6,
     "w_rk_in_csls": 1.8, "w_rk_post_by_k": {"1": 2.8, "2": 2.2, "3": 1.5, "5": 0.8}, "csls_k": 20}
OFF = {"lam": 0.70, "w_p1_orig": 8.0, "w_p1_multi": 0.0, "w_c13_sp128": 0.0, "w_rk_in_csls": 0.0, "post": False}
PRIMARY = {"lam": 0.78, "w_p1_orig": 5.5, "w_p1_multi": 3.0, "w_c13_sp128": 3.0, "w_rk_in_csls": 1.8, "post": True}
STEP1 = {"lam": 0.78, "w_p1_orig": 8.0, "w_p1_multi": 0.0, "w_c13_sp128": 0.0, "w_rk_in_csls": 0.0, "post": False}
STEP2 = {"lam": 0.78, "w_p1_orig": 5.5, "w_p1_multi": 3.0, "w_c13_sp128": 3.0, "w_rk_in_csls": 0.0, "post": False}
SYSTEMS = {"v27_off": OFF, "02_step1_lam078": STEP1, "03_step2_p1combo": STEP2, "A66": PRIMARY}

D_BASE, D_PRIMARY = 0.7467766213119217, 0.768315
D_PRIMARY_PER = [0.797383, 0.771075, 0.741022, 0.778487, 0.75622, 0.797174, 0.762371, 0.767635, 0.763328, 0.748452]
LC_BASE, LC_PRIMARY = 0.659802, 0.681568
LC_PRIMARY_PER = [0.696893, 0.681068, 0.673209, 0.663045, 0.683506, 0.700925, 0.686619, 0.684579, 0.685494, 0.679008,
                  0.696785, 0.647688]
TOL = 1e-6
E_V27 = 0.7035676497986656
ADOPT_MEAN = 0.01
RAM_MIN_MB = 7000
GEM_SHA = {"c13_breg_models.npz": "db2a9861c98627201f59819d5d42fcbfd3c2b941d20610fe6a16878ea5692a71",
           "s0_head.npz": "4b80bba8bdbd21a65557783322ac488e26ca22d2b45b348086279d9028273587",
           "bundle_L_c49.npz": "79b681ef5c49b929be49a526c8bfe8c044e4fb0b3ce7928bc92dfacb1ac3d920",
           "pstore_L_c49.npz": "462b3896c656470cc84119db6baa7546e6bb41b73c86e6f9818748f4f921671a",
           "01b_build_bundles_and_eval_task1.py":
               "823012b6396c8ffccd87ef7713d06bb7076130279ca222d766cb88e95de0b21b"}


def gem(name):
    p = os.path.join(GEM, name)
    if L.sha256_file(p) != GEM_SHA[name]:
        raise SystemExit("%s differs from frozen.json" % _rel(p))
    return p


# ================================================================ ported functions (Carbon 04 script)
l2_rows = P63.l2_rows


def c13_feature_385(p1_means):
    """lines 131-140."""
    b0 = np.asarray(p1_means[:, 0, :], np.float64)
    sp = b0[:, 0:384] - b0[:, 384:768]
    surp = (b0[:, 768] + b0[:, 769])[:, None]
    return np.concatenate([sp, surp], axis=1)


def project_sp128(p1_means, sp):
    """lines 290-291."""
    return ((c13_feature_385(p1_means) - sp["sp128_mu"]) / sp["sp128_sd"]) @ sp["sp128_W"]


def pool_whiten_proto_z(X_all, usable, cand_groups, q_groups, lam):
    """lines 143-175."""
    Xc_l, gc_l, Xq_l, gq_l = [], [], [], []
    has_c = np.zeros(len(cand_groups), bool)
    has_q = np.zeros(len(q_groups), bool)
    for ci, grp in enumerate(cand_groups):
        ug = [g for g in grp if usable[g]]
        if ug:
            has_c[ci] = True
            for g in ug:
                Xc_l.append(X_all[g])
                gc_l.append(ci)
    for qi, grp in enumerate(q_groups):
        ug = [g for g in grp if usable[g]]
        if ug:
            has_q[qi] = True
            for g in ug:
                Xq_l.append(X_all[g])
                gq_l.append(qi)
    Xc = l2_rows(np.array(Xc_l))
    Xq = l2_rows(np.array(Xq_l))
    gc_a = np.array(gc_l, np.int64)
    gq_a = np.array(gq_l, np.int64)
    mu, P = BK.whitening(Xc, gc_a, float(lam), None)
    Yc = (Xc - mu) @ P
    Yq = (Xq - mu) @ P
    Pc, _ = BK.proto(Yc, gc_a, len(cand_groups))
    Pq, _ = BK.proto(Yq, gq_a, len(q_groups))
    T = Pq @ Pc.T
    if not has_c.all() and has_c.any():
        T[:, ~has_c] = T[:, has_c].min(axis=1, keepdims=True)
    return st.zscore_rows(T, has_q)


def load_head(path):
    with np.load(path, allow_pickle=False) as zh:
        W_h = l2_rows(zh["weight"])
        rk_h = np.asarray(zh["class_rank"], np.int64)
    onehot = np.zeros((len(rk_h), 10), np.float64)
    onehot[np.arange(len(rk_h)), rk_h] = 1.0
    inv_cls = 1.0 / np.maximum(onehot.sum(axis=0), 1.0)
    return {"W": W_h, "onehot": onehot, "inv_cls": inv_cls, "sha256": L.sha256_file(path), "file": _rel(path)}


def single_game_rank_logprobs(means, counts, usable, emb_arr, cnt_arr, q09, head, tau):
    """lines 178-209 (head loaded once by load_head, same operations)."""
    n = len(usable)
    ms_5 = [(m["kind"], m["params"]) for m in q09["members"] if not FW09.uses_ex(m["kind"])]
    lp_p1to = np.zeros((n, 10), np.float64)
    idx_u = np.flatnonzero(usable)
    if len(idx_u):
        Xraw = FW09.x_raw(means[idx_u], counts[idx_u])
        for s in range(0, len(idx_u), 8192):
            Xs = FW09.standardise_x(Xraw[s:s + 8192], q09["mu_x"], q09["sd_x"])
            lps = [FW09.member_logp(k, p, Xs) for k, p in ms_5]
            lp_p1to[idx_u[s:s + 8192]] = FW09.lse_mean(lps)
    W_h, onehot, inv_cls = head["W"], head["onehot"], head["inv_cls"]
    lp_cnn = np.zeros((n, 10), np.float64)
    idx_e = np.flatnonzero(np.asarray(cnt_arr) > 0)
    for s in range(0, len(idx_e), 4096):
        sl = idx_e[s:s + 4096]
        logits = float(tau) * (l2_rows(emb_arr[sl]) @ W_h.T)
        mx = logits.max(axis=1, keepdims=True)
        probs_rk = (np.exp(logits - mx) @ onehot) * inv_cls[None, :]
        probs_rk /= np.maximum(probs_rk.sum(axis=1, keepdims=True), 1e-30)
        lp_cnn[sl] = np.log(np.maximum(probs_rk, 1e-12))
    return lp_p1to, lp_cnn


def tri_rank_compat_z(set_models, q09, means, counts, usable, lp_p1to, lp_cnn, cnt_arr, cand_groups, q_groups,
                      w_set, w_p1to, w_cnn, sigma_rk, eps_rk):
    """lines 212-268, unchanged."""
    from scipy import sparse
    rows = np.array(sorted({int(x) for g in list(cand_groups) + list(q_groups) for x in g}), np.int64)
    loc = {int(r): i for i, r in enumerate(rows.tolist())}
    cg = [[loc[int(x)] for x in g] for g in cand_groups]
    qg = [[loc[int(x)] for x in g] for g in q_groups]
    all_g = cg + qg
    r_i, c_i = [], []
    for gi, grp in enumerate(all_g):
        r_i.extend([gi] * len(grp))
        c_i.extend(grp)
    S_mat = sparse.csr_matrix((np.ones(len(r_i)), (r_i, c_i)), shape=(len(all_g), len(rows)))
    _E, lp_set, has_set = FW25.question_set_term(
        set_models, np.asarray(means[rows]), np.asarray(counts[rows]), np.asarray(usable[rows]),
        S_mat, q09["mu_x"], q09["sd_x"])
    lp_comb = float(w_set) * lp_set
    if w_p1to > 0:
        lp_sub_p1 = lp_p1to[rows]
        u_sub = usable[rows]
        mean_p1 = np.zeros((len(all_g), 10), np.float64)
        for gi, grp in enumerate(all_g):
            ug = [g for g in grp if u_sub[g]]
            if ug:
                v = lp_sub_p1[ug].mean(axis=0)
                v -= np.max(v)
                v -= np.log(np.sum(np.exp(v)))
                mean_p1[gi] = v
        lp_comb = lp_comb + float(w_p1to) * mean_p1
    if w_cnn > 0:
        lp_sub_cnn = lp_cnn[rows]
        ce_sub = np.asarray(cnt_arr)[rows] > 0
        mean_cnn = np.zeros((len(all_g), 10), np.float64)
        for gi, grp in enumerate(all_g):
            eg = [g for g in grp if ce_sub[g]]
            if eg:
                v = lp_sub_cnn[eg].mean(axis=0)
                v -= np.max(v)
                v -= np.log(np.sum(np.exp(v)))
                mean_cnn[gi] = v
        lp_comb = lp_comb + float(w_cnn) * mean_cnn
    prob = np.exp(lp_comb - lp_comb.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    nc = len(cand_groups)
    pc, pq = prob[:nc], prob[nc:]
    ii = np.arange(10)
    K_rk = np.exp(-((ii[:, None] - ii[None, :]) ** 2) / (2.0 * (float(sigma_rk) ** 2)))
    compat = np.log(np.maximum(pq @ K_rk @ pc.T, float(eps_rk)))
    return st.zscore_rows(compat, has_set[nc:])


def rank_z_pool(cg, qg, means, counts, usable, emb_fn, q09, s25, head):
    """NOTES.md 5: the single-game log-posteriors on the pool's rows only, then lines 212-268 on that subset."""
    rows = np.array(sorted({int(x) for g in list(cg) + list(qg) for x in g}), np.int64)
    loc = {int(r): i for i, r in enumerate(rows.tolist())}
    cgl = [[loc[int(x)] for x in g] for g in cg]
    qgl = [[loc[int(x)] for x in g] for g in qg]
    m, c, u = np.asarray(means[rows]), np.asarray(counts[rows]), np.asarray(usable[rows])
    emb, cnt = emb_fn(rows)
    lp_p1to, lp_cnn = single_game_rank_logprobs(m, c, u, emb, cnt, q09, head, R["rk_cnn_tau"])
    return tri_rank_compat_z(s25, q09, m, c, u, lp_p1to, lp_cnn, cnt, cgl, qgl, R["rk_w_s25"], R["rk_w_p1to"],
                             R["rk_w_cnn"], R["rk_sigma"], R["rk_eps"])


def cfg_lam(cfg, lam):
    """lines 303-304 (lam 0.70 = v2-6's cfg itself, unchanged)."""
    if lam == cfg["whiten"]["lam"] and cfg["whiten"].get("lda") is None:
        return cfg
    c = dict(cfg)
    c["whiten"] = {"lam": float(lam), "lda": None}
    return c


def w_post_vec(ks):
    """line 339."""
    return np.array([float(R["w_rk_post_by_k"][str(int(kv))]) for kv in ks], np.float64)[:, None]


def combine(F, z1, zm, zs, zr, ks, w):
    """lines 341-355 (terms with weight 0 are not added: NOTES.md 7)."""
    P1 = float(w["w_p1_orig"]) * z1
    if w["w_p1_multi"]:
        P1 = P1 + float(w["w_p1_multi"]) * zm
    if w["w_c13_sp128"]:
        P1 = P1 + float(w["w_c13_sp128"]) * zs
    inner = F + P1
    if w["w_rk_in_csls"]:
        inner = inner + float(w["w_rk_in_csls"]) * zr
    S = st.csls(inner, int(R["csls_k"]))
    if w["post"]:
        S = S + w_post_vec(ks) * zr
    return S


def all_systems(F070, F078, z1, zm, zs, zr, ks):
    return {n: combine(F070 if w["lam"] == 0.70 else F078, z1, zm, zs, zr, ks, w) for n, w in SYSTEMS.items()}


def load_sp(rec_sha=None):
    if not os.path.exists(SP128):
        raise SystemExit("missing %s (run prep-sp)" % _rel(SP128))
    if rec_sha is not None and L.sha256_file(SP128) != rec_sha:
        raise SystemExit("%s differs from the recorded sha256" % _rel(SP128))
    with np.load(SP128, allow_pickle=False) as z:
        return {k: np.asarray(z[k], np.float64) for k in ("sp128_mu", "sp128_sd", "sp128_W")}


def chunked(fn, means, block=8192):
    """Row-wise projections in row blocks (memory; prep-e was stopped at 3.05 GB with whole-array float64 copies)."""
    parts = [fn(means[s:s + block]) for s in range(0, len(means), block)]
    if isinstance(parts[0], dict):
        return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    return np.concatenate(parts)


def e_store_index_lean():
    """= p63_eval.e_store_index (E then E2, indexed by store row), filled set by set into one preallocated array
    (memory: holding both loads and their concatenation passed 3 GB)."""
    sizes = {nm: len(P39E.read_list(nm)) for nm in ("E", "E2")}
    n = sizes["E"] + sizes["E2"]
    means = np.zeros((n, 2, 770), np.float32)
    counts = np.zeros((n, 2), np.int64)
    usable = np.zeros(n, bool)
    rows, off = [], 0
    for nm in ("E", "E2"):
        g, m, c, u = P63.load_p1_full(nm)
        if len(g) != sizes[nm]:
            raise SystemExit("%s list size" % nm)
        means[off:off + len(g)] = m
        counts[off:off + len(g)] = c
        usable[off:off + len(g)] = u
        rows.extend(int(x["row"]) for x in g)
        off += len(g)
        del g, m, c, u
        P63._MOCK.clear()
    pos = {r: i for i, r in enumerate(rows)}
    if len(pos) != n:
        raise SystemExit("a store row is in E and E2")
    return means, counts, usable, pos


def p1multi_sha():
    return read_json(os.path.join(P63_DIR, "fit.json"))["sha256"]


# ================================================================ suite evaluation (D / L_c49 seams)
def eval_suite(pools, emb_obj, cols, pm, pc, pu, head_path, models_L, cfg):
    """Carbon's evaluate_suite for v27_off / 02 / 03 / A66; groups index pm / pc / pu and emb_obj rows."""
    mods = P63.load_multi(p1multi_sha())
    sp = load_sp()
    head = load_head(head_path)
    q09, s25 = P63.load_models()
    Zs_orig = P39E.project_all(pm[:, 0, :], models_L)
    Zp = P63.project_multi(pm, mods)
    Zsp = project_sp128(pm, sp)
    cfg78 = cfg_lam(cfg, R["cnn_whiten_lam"])

    def emb_fn(rows):
        return emb_obj.emb[rows], np.asarray(emb_obj.count)[rows]

    per = {n: [] for n in SYSTEMS}
    byk = {n: [] for n in SYSTEMS}
    for p in pools:
        cg, qg = p["cand_rows"], p["q_rows"]
        ks = np.asarray(p["k"], np.int64)
        F070 = BK.fuse_cfg(p, BK.e_term(p, emb_obj, cfg), cfg)
        F078 = BK.fuse_cfg(p, BK.e_term(p, emb_obj, cfg78), cfg78)
        z1, _h = P39E.p1_terms(pm[:, 0, :], pu, cols, cg, qg, models_L, "b4", Zs=Zs_orig)["joint"]
        zm = P63.z_multi_of(Zp, pu, cg, qg)
        zs = pool_whiten_proto_z(Zsp, pu, cg, qg, R["c13_pool_lam"])
        zr = rank_z_pool(cg, qg, pm, pc, pu, emb_fn, q09, s25, head)
        for n, S in all_systems(F070, F078, z1, zm, zs, zr, ks).items():
            per[n].append(float(st.score(S, p["owner"])))
            byk[n].append(EV.score_by_k(S, p))
        print(json.dumps({"seed": p.get("seed"), "v27_off": per["v27_off"][-1], "A66": per["A66"][-1]}), flush=True)
    return {n: {"mean": float(np.mean(per[n])), "per_pool": per[n], "by_k": EV.mean_by_k(byk[n])} for n in SYSTEMS}


def seam_d():
    """Carbon lines 567-614 (D, S0 vectors, s0_head)."""
    base = E45.v26()
    pools_raw = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    used = sorted({int(r) for p in pools_raw for g in list(p["cand_rows"]) + list(p["q_rows"]) for r in g})
    d_loc = {r: i for i, r in enumerate(used)}
    used_arr = np.asarray(used, np.int64)
    with np.load(os.path.join(L.FEATS, "meta.npz"), allow_pickle=False) as zm:
        col = np.asarray(zm["color"][used_arr]).astype(str)
    with np.load(P63.S0_D, allow_pickle=False) as zd:
        pos = {int(r): i for i, r in enumerate(np.asarray(zd["store_row"], np.int64).tolist())}
        take = np.asarray([pos[r] for r in used], np.int64)
        emb = np.asarray(zd["emb"][take], np.float32)
        cnt = np.asarray(zd["count"][take], np.int32)
    gD, mD, cD, uD = P63.load_p1_full("D")
    P63._MOCK.clear()
    ppos = {int(g["row"]): i for i, g in enumerate(gD)}
    n = len(used)
    pm = np.zeros((n, 2, 770), np.float32)
    pc = np.zeros((n, 2), np.int32)
    pu = np.zeros(n, bool)
    for i, r in enumerate(used):
        j = ppos.get(int(r))
        if j is not None:
            pm[i], pc[i], pu[i] = mD[j], cD[j], uD[j]
    del gD, mD, cD, uD
    pools = [{"cand_rows": [[d_loc[int(r)] for r in g] for g in p["cand_rows"]],
              "q_rows": [[d_loc[int(r)] for r in g] for g in p["q_rows"]], "owner": p["owner"], "k": p["k"],
              "seed": p["seed"], "T": p["T"]} for p in pools_raw]
    emb_obj = EV.Emb(emb, cnt, None, col, "sha", "D")
    models_L = P39E.fit_models(P39E.load_set("L"))
    P39E._MOCK.clear()
    out = eval_suite(pools, emb_obj, col, pm, pc, pu, gem("s0_head.npz"), models_L, base["cfg"])
    b, a = out["v27_off"], out["A66"]
    res = {"suite": out, "base_want": D_BASE, "base_identical": b["mean"] == D_BASE,
           "base_per_pool_equals_prereg43": b["per_pool"] == base["d_per_pool"],
           "primary_want": D_PRIMARY, "primary_abs_diff": abs(a["mean"] - D_PRIMARY),
           "primary_per_pool_max_abs_diff": max(abs(x - y) for x, y in zip(a["per_pool"], D_PRIMARY_PER))}
    res["ok"] = bool(res["base_identical"] and res["primary_abs_diff"] <= TOL
                     and res["primary_per_pool_max_abs_diff"] <= TOL)
    return res


def seam_l():
    """Carbon lines 474-545 (L_c49, seeds 300-311, c49 vectors, c49_head), with Carbon's 01b make_light_pool."""
    import importlib
    from collections import defaultdict
    from scipy import sparse
    import p35_apply as PA
    gem("01b_build_bundles_and_eval_task1.py")
    if GEM not in sys.path:
        sys.path.append(GEM)
    b1 = importlib.import_module("01b_build_bundles_and_eval_task1")
    base = E45.v26()
    fit_st = sim.Store(L.FEATS, load=())
    resp_vocab = fit_st.csr("resp").vocab  # noqa: F841  (as Carbon; the light store has its own resp)
    lda_cols, lm_fit = {}, {}
    for cch in ("B", "W"):
        ld = sim.LdaData(L.FEATS, cch, fit_st)
        lm_fit[cch] = ld.fit_excluding(fit_st, set())
        lda_cols[cch] = ld.cols
        del ld
    del fit_st
    with np.load(gem("bundle_L_c49.npz"), allow_pickle=False) as zb:
        pids = np.asarray(zb["player_id"]).astype(str)
        rks = np.asarray(zb["rank"]).astype(str)
        cols = np.asarray(zb["color"]).astype(str)
        hds = np.asarray(zb["hd"], np.int16)
        emb_arr = np.asarray(zb["emb"], np.float32)
        cnt_arr = np.asarray(zb["emb_count"], np.int32)
        pm = np.asarray(zb["p1_means"], np.float32)
        pc = np.asarray(zb["p1_counts"], np.int32)
        pu = np.asarray(zb["p1_usable"], bool)
    with np.load(gem("pstore_L_c49.npz"), allow_pickle=False) as zp:
        class Pst:
            color = np.asarray(zp["color"]).astype(str)
            alt = np.asarray(zp["alt"], bool)
            nmov = np.asarray(zp["nmov"])
            n = len(color)
            dense_m = {"s2": np.asarray(zp["s2"]), "s34": np.asarray(zp["s34"])}
            csr_m = {nm: sparse.csr_matrix((zp["%s_data" % nm], zp["%s_indices" % nm], zp["%s_indptr" % nm]),
                                           shape=tuple(zp["%s_shape" % nm].tolist())) for nm in ("oraw", "jo", "resp")}

            def csr(self, nm):
                return self.csr_m[nm]

            def dense(self, nm):
                return self.dense_m[nm]
        pst = Pst()
    ldas = {cch: PA.PoolLda(pst, lda_cols[cch]) for cch in ("B", "W")}
    clean = (hds <= 1) & pu & (cnt_arr > 0) & pst.alt
    by_p, p_rk = defaultdict(list), {}
    for i in range(len(pids)):
        p_rk[pids[i]] = rks[i]
        if clean[i]:
            by_p[pids[i]].append(i)
    pools = []
    for sd in range(300, 312):
        pl = b1.make_light_pool(by_p, p_rk, sd, stratified=False, n_cands=600, per_level=60)
        pl["T"] = sim.pool_terms(pst, pl, ldas, lm_fit, None, want=("O", "JO", "S2", "L", "PA"))
        pools.append(pl)
    del pst, ldas
    emb_obj = EV.Emb(emb_arr, cnt_arr, None, cols, "sha", "L_c49")
    models_L = P39E.fit_models(P39E.load_set("L"))
    P39E._MOCK.clear()
    out = eval_suite(pools, emb_obj, cols, pm, pc, pu, C49_HEAD, models_L, base["cfg"])
    b, a = out["v27_off"], out["A66"]
    res = {"suite": out, "base_want": LC_BASE, "base_abs_diff": abs(b["mean"] - LC_BASE),
           "primary_want": LC_PRIMARY, "primary_abs_diff": abs(a["mean"] - LC_PRIMARY),
           "primary_per_pool_max_abs_diff": max(abs(x - y) for x, y in zip(a["per_pool"], LC_PRIMARY_PER))}
    res["ok"] = bool(res["base_abs_diff"] <= TOL and res["primary_abs_diff"] <= TOL
                     and res["primary_per_pool_max_abs_diff"] <= TOL)
    return res


def seam_e():
    """E: v2-7 through combine() with the new parts off (zm, zs, zr unused), bit for bit."""
    base = E45.v26()
    cfg = base["cfg"]
    if (base["wP"], base["k"]) != (OFF["w_p1_orig"], R["csls_k"]) or cfg["whiten"] != {"lam": 0.7, "lda": None}:
        raise SystemExit("v2-6 cfg / wP / k differ from NOTES.md")
    pools, zs, _bper, vec, _sets = P49.e_context(base)
    path = P49.c49_file("E")
    e = E43.load_emb(path)
    per = []
    for p, z in zip(pools, zs):
        F = BK.fuse_cfg(p, BK.e_term(p, e, cfg), cfg)
        per.append(float(st.score(combine(F, z, None, None, None, p["k"], OFF), p["owner"])))
    want = read_json(P49.EVAL_JSON)["per_pool"]["c49"]
    return {"per_pool": per, "mean": float(np.mean(per)), "want_mean": E_V27,
            "identical": bool(per == want and float(np.mean(per)) == E_V27),
            "vectors": {"file": _rel(path), "sha256": L.sha256_file(path)}}


def cmd_prep_sp(a):
    src = gem("c13_breg_models.npz")
    with np.load(src, allow_pickle=False) as z:
        arrs = {k: np.asarray(z[k]) for k in ("sp128_mu", "sp128_sd", "sp128_W")}
    os.makedirs(DATA, exist_ok=True)
    np.savez(SP128 + ".tmp.npz", **arrs)
    os.replace(SP128 + ".tmp.npz", SP128)
    with np.load(SP128, allow_pickle=False) as z:
        same = all(np.array_equal(z[k], arrs[k]) and z[k].dtype == arrs[k].dtype for k in arrs)
    rec = {"file": _rel(SP128), "sha256": L.sha256_file(SP128), "source": _rel(src),
           "source_sha256": GEM_SHA["c13_breg_models.npz"], "arrays_equal_source": same,
           "shapes": {k: list(v.shape) for k, v in arrs.items()}}
    print(json.dumps(rec))
    if not same:
        raise SystemExit("copy differs")
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else {"format": "c10-prereg66 seam v1"}
    seam["sp128"] = rec
    L.write_json(SEAM_JSON, seam)
    return 0


def cmd_seam(a):
    t0 = time.time()
    seam = read_json(SEAM_JSON)
    if "sp128" not in seam:
        raise SystemExit("run prep-sp first")
    load_sp(seam["sp128"]["sha256"])
    res = {"d": seam_d, "l": seam_l, "e": seam_e}[a.part]()
    res.update(git_head=L.git_head(), peak_rss_mb=P63.peak_mb(), seconds=round(time.time() - t0, 1))
    key = {"d": "D", "l": "L_c49", "e": "E_v27"}[a.part]
    seam = read_json(SEAM_JSON)
    seam[key] = res
    seam["p1multi_sha256"] = p1multi_sha()
    seam["ok"] = bool(seam.get("D", {}).get("ok") and seam.get("L_c49", {}).get("ok")
                      and seam.get("E_v27", {}).get("identical"))
    L.write_json(SEAM_JSON, seam)
    print(json.dumps({k: v for k, v in res.items() if k != "suite" and k != "per_pool"}))
    if not (res.get("ok") or res.get("identical")):
        raise SystemExit("seam %s failed: stop (do not judge)" % a.part)
    return 0


# ================================================================ E preparation (no scores)
def cmd_prep_e(a):
    t0 = time.time()
    seam = read_json(SEAM_JSON)
    if not seam.get("ok") or not L.git_committed(SEAM_JSON):
        raise SystemExit("seam.json is not ok / not committed")
    # NOTES.md 2: Carbon's loader vs the v2-7 P1 sets on E and E2
    check = {}
    for nm in ("E", "E2"):
        s = P39E.load_set(nm)
        g, m, _c, u = P63.load_p1_full(nm)
        check[nm] = {"games": len(g), "usable_equal": bool(np.array_equal(s["usable"], u)),
                     "side0_equal": bool(np.array_equal(s["X"], m[:, 0, :])),
                     "rows_equal": [x["row"] for x in g] == [x["row"] for x in s["games"]]}
        del s, g, m, u
        P39E._MOCK.clear()
        P63._MOCK.clear()
    if not all(v["usable_equal"] and v["side0_equal"] and v["rows_equal"] for v in check.values()):
        print(json.dumps(check))
        raise SystemExit("Carbon's P1 loader differs from the v2-7 P1 sets on E: stop (NOTES.md 2)")
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    means, counts, usable, pos = e_store_index_lean()
    rows_of = np.empty(len(pos), np.int64)
    for r, i in pos.items():
        rows_of[i] = r
    mods = P63.load_multi(seam["p1multi_sha256"])
    sp = load_sp(seam["sp128"]["sha256"])
    Zp = chunked(lambda m: P63.project_multi(m, mods), means)
    Zsp = chunked(lambda m: project_sp128(m, sp), means)
    del mods
    path = P49.c49_file("E")
    e = E43.load_emb(path)
    head = load_head(C49_HEAD)
    q09, s25 = P63.load_models()

    def emb_fn(idx):
        j = np.array([e.pos[int(r)] for r in rows_of[idx]], np.int64)
        return e.emb[j], np.asarray(e.count)[j]

    arrs = {}
    for p in pools:
        cg = [[pos[int(r)] for r in g] for g in p["cand_rows"]]
        qg = [[pos[int(r)] for r in g] for g in p["q_rows"]]
        s = p["seed"]
        arrs["multi_%03d" % s] = P63.z_multi_of(Zp, usable, cg, qg)
        arrs["sp_%03d" % s] = pool_whiten_proto_z(Zsp, usable, cg, qg, R["c13_pool_lam"])
        arrs["rank_%03d" % s] = rank_z_pool(cg, qg, means, counts, usable, emb_fn, q09, s25, head)
        print("pool %d done" % s, flush=True)
    # z_multi equals prereg63's cached term (same code / inputs) - recorded
    with np.load(P63.ZBC_E) as z:
        multi_eq = all(np.array_equal(arrs["multi_%03d" % p["seed"]], z["multi_%03d" % p["seed"]]) for p in pools)
    os.makedirs(DATA, exist_ok=True)
    np.savez(Z_E + ".tmp.npz", **arrs)
    os.replace(Z_E + ".tmp.npz", Z_E)
    rec = {"format": "c10-prereg66 prep-e v1", "file": _rel(Z_E), "sha256": L.sha256_file(Z_E),
           "carbon_loader_equals_v27_p1_sets": check, "z_multi_equals_prereg63_zBC_E": multi_eq,
           "head": {"file": head["file"], "sha256": head["sha256"]},
           "vectors": {"file": _rel(path), "sha256": L.sha256_file(path)},
           "note": "features only: no owner is read, no score is computed",
           "git_head": L.git_head(), "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(PREP_JSON, rec)
    print(json.dumps({k: rec[k] for k in ("sha256", "z_multi_equals_prereg63_zBC_E", "peak_rss_mb", "seconds")}))
    return 0


# ================================================================ E use number (NOTES.md 10)
def e_uses_so_far(repo=REPO):
    nums = {}
    for f in sorted(glob.glob(os.path.join(repo, "results", "prereg*", "explore", "eval*.json"))):
        if os.path.abspath(f) == os.path.abspath(os.path.join(repo, "results", "prereg66", "explore", "eval.json")):
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "e_pools_use_number" in d:
            nums[os.path.relpath(f, repo).replace("\\", "/")] = int(d["e_pools_use_number"])
    p63 = P63.e_uses_so_far(repo)["count"] + (1 if "results/prereg63/explore/eval.json" in nums else 0)
    mx = max(nums.values()) if nums else 0
    return {"files": nums, "max": mx, "prereg63_counter_plus_prereg63": p63, "count": mx,
            "consistent": mx == p63 and len(set(nums.values())) == len(nums)}


# ================================================================ judge
def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    for f in (PLAN, NOTES, FROZEN, os.path.abspath(__file__), SEAM_JSON, PREP_JSON):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))
    seam, prep = read_json(SEAM_JSON), read_json(PREP_JSON)
    if not seam.get("ok"):
        raise SystemExit("seam not ok")
    uses = e_uses_so_far()
    if not uses["consistent"]:
        raise SystemExit("E use counters disagree: %s" % uses)
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before" % (a.e_use, uses["count"]))
    if L.sha256_file(Z_E) != prep["sha256"]:
        raise SystemExit("z_E.npz differs from prep_e.json")
    base = E45.v26()
    cfg = base["cfg"]
    cfg78 = cfg_lam(cfg, R["cnn_whiten_lam"])
    pools, zs, _bper, vec, sets = P49.e_context(base)
    path = P49.c49_file("E")
    vec49 = P49.check_c49_file(path, "E", base["tune"]["c_star"])
    e = E43.load_emb(path)
    per = {n: [] for n in SYSTEMS}
    byk = {n: [] for n in SYSTEMS}
    with np.load(Z_E) as z:
        for p, z1 in zip(pools, zs):
            s = p["seed"]
            F070 = BK.fuse_cfg(p, BK.e_term(p, e, cfg), cfg)
            F078 = BK.fuse_cfg(p, BK.e_term(p, e, cfg78), cfg78)
            zm, zsp, zr = (np.asarray(z["%s_%03d" % (n, s)]) for n in ("multi", "sp", "rank"))
            for n, S in all_systems(F070, F078, z1, zm, zsp, zr, np.asarray(p["k"], np.int64)).items():
                per[n].append(float(st.score(S, p["owner"])))
                byk[n].append(EV.score_by_k(S, p))
    del e
    if per["v27_off"] != read_json(P49.EVAL_JSON)["per_pool"]["c49"] or float(np.mean(per["v27_off"])) != E_V27:
        raise SystemExit("seam: v2-7 rebuilt in the judgement differs from prereg49 eval.json")
    j = P63.judge_rec(per["A66"], per["v27_off"])
    bk = {n: EV.mean_by_k(v) for n, v in byk.items()}
    rec = {"format": "c10-prereg66 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "e_vectors": vec49, "p1_sets": sets, "cfg": cfg, "cfg78": cfg78,
           "recipe": R, "systems": SYSTEMS, "seam_v27_E_identical": True,
           "seam_json_sha256": st.sha256_lf(SEAM_JSON), "prep_json_sha256": st.sha256_lf(PREP_JSON),
           "git_head": L.git_head(), "means": {n: float(np.mean(v)) for n, v in per.items()}, "per_pool": per,
           "by_k": bk, "by_k_diff_A66": {k: bk["A66"][k] - bk["v27_off"][k] for k in bk["A66"]},
           "A66_minus_v27": {x: j[x] for x in ("mean", "ci95", "pools_higher")},
           "recorded": {n + "_minus_v27": {x: v for x, v in P63.judge_rec(per[n], per["v27_off"]).items()
                                           if x in ("mean", "ci95", "pools_higher")}
                        for n in ("02_step1_lam078", "03_step2_p1combo")},
           "rule": j["rule"].replace("A63", "A66"), "adopt": j["adopt"],
           "note": "NOTES.md F1: the rank models (S25, Q09 P1TO) were trained on games of the E players; "
                   "03_step2_p1combo = A66 without the rank term",
           "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "A66_minus_v27", "by_k_diff_A66", "recorded", "adopt")}, indent=1))
    return 0


# ================================================================ apply / assemble
def cmd_apply(a):
    t0 = time.time()
    ev = read_json(a.eval)
    if not ev["adopt"]:
        raise SystemExit("eval.json does not adopt A66: keep v2-7")
    seam = read_json(SEAM_JSON)
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    cfg78 = cfg_lam(cfg, R["cnn_whiten_lam"])
    cs = base["tune"]["c_star"]
    rec, items, cand_ids, qids = E43.official_rec(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    colour = np.array([c for _s, c in items])
    ks = np.array([len(g) for g in rec["q_rows"]], np.int64)
    if not set(ks.tolist()) <= {1, 2, 3, 5}:
        raise SystemExit("question game counts outside {1, 2, 3, 5}")
    # v2-7 exactly as prereg49's apply
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    z1, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                            E43.P1_PW)[E43.P1_VARIANT]
    del models
    path = P49.c49_file(a.pool)
    vec49 = P49.check_c49_file(path, a.pool, cs)
    e = E43.pool_emb(path, items, vec49["ckpt_sha256"])
    F070 = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
    S27 = combine(F070, z1, None, None, None, ks, OFF)
    data27 = P39E.top5_bytes(qids, st.top5(S27, cand_ids))
    with open(os.path.join(P63.V27_APPLY, a.pool, "top5.csv"), "rb") as f:
        if P63.lf(f.read()) != P63.lf(data27):
            raise SystemExit("seam: v2-7 rebuilt on %s differs from results/prereg49/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    # A66 (NOTES.md 2: Carbon's loader must give the same side-0 / usable as the v2-7 P1 set)
    games, means, counts, usable = P63.load_p1_full(name)
    if len(games) != len(items) or any(g["color"] != c for g, (_s, c) in zip(games, items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if not (np.array_equal(usable, Pset["usable"]) and np.array_equal(means[:, 0, :], Pset["X"])):
        raise SystemExit("Carbon's P1 loader differs from the v2-7 P1 set on %s: stop" % name)
    del Pset
    mods = P63.load_multi(seam["p1multi_sha256"])
    sp = load_sp(seam["sp128"]["sha256"])
    head = load_head(C49_HEAD)
    q09, s25 = P63.load_models()
    cg, qg = rec["cand_rows"], rec["q_rows"]
    zm = P63.z_multi_of(P63.project_multi(means, mods), usable, cg, qg)
    zsp = pool_whiten_proto_z(project_sp128(means, sp), usable, cg, qg, R["c13_pool_lam"])

    def emb_fn(rows):
        return e.emb[rows], np.asarray(e.count)[rows]

    zr = rank_z_pool(cg, qg, means, counts, usable, emb_fn, q09, s25, head)
    F078 = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg78), cfg78)
    S = combine(F078, z1, zm, zsp, zr, ks, PRIMARY)
    data = P39E.top5_bytes(qids, st.top5(S, cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    r27 = [r.split(b",") for r in data27.split(b"\n")[1:] if r]
    r66 = [r.split(b",") for r in data.split(b"\n")[1:] if r]
    rep = {"format": "c10-prereg66 apply v1", "pool": a.pool, "vectors": vec49, "cfg78": cfg78, "recipe": R,
           "v27_top5_equals_prereg49_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "k_counts": {str(x): int((ks == x).sum()) for x in (1, 2, 3, 5)},
           "games_without_P1": int((~usable).sum()), "questions_without_P1": int((~np.asarray(has)).sum()),
           "head": {"file": head["file"], "sha256": head["sha256"]}, "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "rows_differ_from_v27": int(sum(x != y for x, y in zip(r66, r27))),
           "top1_differs_from_v27": int(sum(x[1] != y[1] for x, y in zip(r66, r27))),
           "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return P63.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep-sp")
    s = sub.add_parser("seam")
    s.add_argument("--part", required=True, choices=["d", "l", "e"])
    sub.add_parser("prep-e")
    sub.add_parser("e-use")
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--eval", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd == "e-use":
        u = e_uses_so_far()
        print(json.dumps(u))
        print(u["count"] + 1)
        return 0
    if a.cmd not in ("assemble", "prep-sp"):
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_MIN_MB)
        P63._WD["s"] = P63.rss_watchdog()
    return {"prep-sp": cmd_prep_sp, "seam": cmd_seam, "prep-e": cmd_prep_e, "judge": cmd_judge,
            "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
