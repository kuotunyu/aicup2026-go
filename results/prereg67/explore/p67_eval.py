"""prereg67 explore (PLAN.md, NOTES.md, frozen.json): competition 1 new questions, Gemini Hydrogen's score-level
CNN ensemble on top of v2-8 (prereg66 A66). CPU only, one process at a time.

    PY results/prereg67/explore/p67_eval.py seam --part pseudo|e         # NOTES.md 9 -> seam.json
    PY results/prereg67/explore/p67_eval.py check-member --name p61|X     # NOTES.md 7 -> members.json (once per name)
    PY results/prereg67/explore/p67_eval.py version                      # prints the version judge would use
    PY results/prereg67/explore/p67_eval.py e-use
    PY results/prereg67/explore/p67_eval.py judge --e-use N              # once -> eval.json
    PY results/prereg67/explore/p67_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg67/explore/p67_eval.py assemble --src <official tests dir>

ens = csls_20(G + 14 E_ens + (5.5 z_p1_orig + 3 z_multi + 3 z_sp128) + 1.8 z_rk) + w_post(k) z_rk,
E_ens = sum_m (w_m / W) zscore_rows(e_term(m, cfg_ens)). Nothing here opens the validation-answer folder or runs
tools/score.py; no single-model E score is computed.
"""
import argparse
import datetime
import gc
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P66_DIR = os.path.join(REPO, "results", "prereg66", "explore")
for _p in (P66_DIR, os.path.join(REPO, "results", "prereg52", "explore"),
           os.path.join(REPO, "results", "prereg35", "explore")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p66_eval as P66  # noqa: E402   (v2-8's code; imported, never edited)

P63, P49, E43, E45, P39E = P66.P63, P66.P49, P66.E43, P66.E45, P66.P39E
BK, EV, st, L, sim, Q42 = P66.BK, P66.EV, P66.st, P66.L, P66.sim, P66.Q42
_rel, read_json = P66._rel, P66.read_json

PLAN = os.path.join(HERE, "PLAN.md")
NOTES = os.path.join(HERE, "NOTES.md")
FROZEN = os.path.join(HERE, "frozen.json")
SEAM_JSON = os.path.join(HERE, "seam.json")
MEMBERS_JSON = os.path.join(HERE, "members.json")
EVAL_JSON = os.path.join(HERE, "eval.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg67", "apply")
V28_EVAL = os.path.join(P66_DIR, "eval.json")
V28_PREP = os.path.join(P66_DIR, "prep_e.json")
V28_SEAM = os.path.join(P66_DIR, "seam.json")
V28_APPLY = os.path.join(REPO, "results", "prereg66", "apply")
HYD = os.path.join(REPO, "data", "advice", "gemini_1121h")

ORDER = ("c49", "S", "B", "p61", "S1", "S2", "X")
WEIGHTS = {"c49": 1.0, "S": 1.0, "B": 1.0, "p61": 2.0, "S1": 2.0, "S2": 2.0, "X": 2.0}
FIXED = ("c49", "S", "B", "S1", "S2")
OPTIONAL = {"p61": 200000, "X": 400000}
WE_ENS = 14.0
WE_BASE = 8.0
ENS_LAM = 0.85
CHECK_MIN = 0.6406
CUTOFF = datetime.datetime(2026, 11, 22, 2, 0, 0)       # Taiwan local time (the machine's clock)
FILES = ("store_E.npz", "pool_public.npz", "pool_private.npz")
SEAM_PSEUDO_MEAN = 0.6397593752945936
SEAM_E_MEAN = 0.7183609308316172
RECIPE_STD10 = 0.658555
RECIPE_TOL = 1e-6
ADOPT_MEAN = 0.01
RAM_MIN_MB = 7000
HYD_SHA = {"01_prep_summary.json": "075f709357276498a2ab8ffb91e6ce0d502262832087804cd016e94b1b4765a3",
           "03_beyond_ensemble_results.json": "56414c6ee41264aff143b7d91dc17134e7b3ec4bcab4bda292f16e79235cf343"}


def hyd(name):
    p = os.path.join(HYD, name)
    if L.sha256_file(p) != HYD_SHA[name]:
        raise SystemExit("%s differs from frozen.json" % _rel(p))
    return p


def frozen():
    return read_json(FROZEN)


# ================================================================ pure parts (tested in test_p67.py)
def cfg_ens(cfg):
    """NOTES.md 1: v2-6 cfg with centre_colour True and whiten lam 0.85 (Hydrogen 03 line 207)."""
    c = dict(cfg)
    c["centre_colour"] = True
    c["whiten"] = {"lam": ENS_LAM, "lda": None}
    return c


def ensemble(Zs, weights):
    """NOTES.md 2: sum_m (w_m / W) Z_m in the given order, weight-0 members skipped (first term not added to 0)."""
    W = float(sum(w for w in weights if w))
    out = None
    for Z, w in zip(Zs, weights):
        if not w:
            continue
        t = (float(w) / W) * Z
        out = t if out is None else out + t
    return out


def score_mat(G, E_ens, z1, zm, zs, zr, ks, wE):
    """NOTES.md 4: prereg66 combine() with F = G + E_ens * w_E."""
    return P66.combine(G + E_ens * float(wE), z1, zm, zs, zr, np.asarray(ks, np.int64), P66.PRIMARY)


def pick_version(entries, present, cutoff_ok):
    """NOTES.md 6. entries: members.json dict; present[name]: files exist; cutoff_ok[name]: all mtimes <= cutoff.
    Returns (members, problems)."""
    problems = []
    use = list(FIXED)
    for nm in OPTIONAL:
        e = entries.get(nm)
        if e is None:
            if present.get(nm) and cutoff_ok.get(nm):
                problems.append("%s: vectors exist before the cutoff but check-member was not run" % nm)
            continue
        if e.get("pass") and cutoff_ok.get(nm) and e.get("files_unchanged", True):
            use.append(nm)
    members = [m for m in ORDER if m in use]
    return members, problems


def version_name(members):
    s = set(members)
    if s == set(ORDER):
        return "7"
    if s == set(FIXED) | {"p61"}:
        return "A"
    if s == set(FIXED):
        return "B"
    if s == set(FIXED) | {"X"}:
        return "6-X (p61 dropped)"
    raise SystemExit("unexpected member set %s" % sorted(s))


def e_uses_so_far(repo=REPO):
    nums = {}
    for f in sorted(glob.glob(os.path.join(repo, "results", "prereg*", "explore", "eval*.json"))):
        if os.path.abspath(f) == os.path.abspath(os.path.join(repo, "results", "prereg67", "explore", "eval.json")):
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "e_pools_use_number" in d:
            nums[os.path.relpath(f, repo).replace("\\", "/")] = int(d["e_pools_use_number"])
    mx = max(nums.values()) if nums else 0
    vals = sorted(nums.values())
    return {"files": nums, "max": mx, "count": mx,
            "consistent": len(set(vals)) == len(vals) and vals == list(range(4, mx + 1))}


# ================================================================ member files
def member_dir(name, entries=None):
    fz = frozen()["members"]
    if entries and name in entries:
        return os.path.join(REPO, entries[name]["dir"])
    return os.path.join(REPO, fz[name]["dir"])


def files_state(d):
    paths = [os.path.join(d, f) for f in FILES]
    present = all(os.path.exists(p) for p in paths)
    ok = present and all(datetime.datetime.fromtimestamp(os.path.getmtime(p)) <= CUTOFF for p in paths)
    return present, ok


def file_shas(d):
    return {f: L.sha256_file(os.path.join(d, f)) for f in FILES}


def check_files(name, d, want_sha=None, step=None):
    """NOTES.md 7 (b), (c). Returns a record; rec['ok'] False with problems listed."""
    c49d = member_dir("c49")
    shas = file_shas(d)
    problems = []
    if want_sha is not None and shas != want_sha:
        problems.append("file sha256 differs from the record")
    rec = {"dir": _rel(d), "sha256": shas,
           "mtime": {f: datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(d, f))).isoformat()
                     for f in FILES}}
    zero = {}
    for f in FILES:
        with np.load(os.path.join(d, f), allow_pickle=False) as z, \
                np.load(os.path.join(c49d, f), allow_pickle=False) as r:
            cfg = json.loads(str(z["embed_cfg"]))
            rcfg = json.loads(str(r["embed_cfg"]))
            zero[f] = int((np.asarray(z["count"]) == 0).sum())
            if z["emb"].shape != r["emb"].shape:
                problems.append("%s shape %s != c49's %s" % (f, z["emb"].shape, r["emb"].shape))
            if f == "store_E.npz" and not np.array_equal(z["store_row"], r["store_row"]):
                problems.append("store_E store_row differs from c49's")
            if cfg.get("tta") != 1 or cfg.get("weights") != "raw" or cfg.get("games") != rcfg.get("games"):
                problems.append("%s embed cfg tta %r weights %r games %r" % (f, cfg.get("tta"), cfg.get("weights"),
                                                                         cfg.get("games")))
            if step is not None and int(z["ckpt_step"]) != int(step):
                problems.append("%s checkpoint step %s != %d" % (f, z["ckpt_step"], step))
            if str(z["ckpt_sha256"]) != cfg.get("ckpt_sha256"):
                problems.append("%s ckpt sha inconsistent" % f)
            rec.setdefault("ckpt_sha256", str(z["ckpt_sha256"]))
            rec.setdefault("ckpt_step", int(z["ckpt_step"]))
    rec["zero_count_games"] = zero
    if any(zero.values()):
        problems.append("games without a sample: %s" % zero)
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def version_from_disk():
    entries = read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}
    present, cut = {}, {}
    for nm in OPTIONAL:
        present[nm], cut[nm] = files_state(member_dir(nm, entries))
    ent = {}
    for nm, e in entries.items():
        e = dict(e)
        if present.get(nm):
            e["files_unchanged"] = file_shas(member_dir(nm, entries)) == e["files"]["sha256"]
        else:
            e["files_unchanged"] = False
        ent[nm] = e
    members, problems = pick_version(ent, present, cut)
    return members, problems, {"present": present, "before_cutoff": cut,
                               "entries": {k: {"pass": v.get("pass"), "files_unchanged": v.get("files_unchanged"),
                                               "std10": v.get("std10_mean")} for k, v in ent.items()}}


def member_shas(name, entries):
    if name in OPTIONAL:
        return entries[name]["files"]["sha256"]
    return frozen()["members"][name]["sha256"]


# ================================================================ std10 pseudo pools (NOTES.md 8)
def pseudo_context():
    import p35_apply as PA
    import p52_pseudo as PS
    ps = BK.ps
    base = E45.v26()
    cfg = base["cfg"]
    w_G, _kG = EV.g_combo()
    models_L = P39E.fit_models(P39E.load_set("L"))
    P39E._MOCK.clear()
    if P39E.models_sha(models_L) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    mods = P63.load_multi(P66.p1multi_sha())
    sp = P66.load_sp(read_json(V28_SEAM)["sp128"]["sha256"])
    q09, s25 = P63.load_models()
    head = P66.load_head(P66.C49_HEAD)
    fit_st = sim.Store(L.FEATS, load=())
    resp_vocab = fit_st.csr("resp").vocab
    lda_cols, lm_fit = {}, {}
    for cch in ("B", "W"):
        ld = sim.LdaData(L.FEATS, cch, fit_st)
        lm_fit[cch] = ld.fit_excluding(fit_st, set())
        lda_cols[cch] = ld.cols
        del ld
    del fit_st
    gc.collect()
    pools, cols_l, pm_l, pc_l, pu_l, emb_l, cnt_l = [], [], [], [], [], [], []
    offset = 0
    for pool_kind, (pname, seed0) in PS.POOLS.items():
        items, _cand_ids, cand_rows, _ci = PS.read_candidates(pool_kind)
        n_cand = len(items)
        if n_cand != 6000:
            raise SystemExit("candidate games %d != 6000" % n_cand)
        cols_l.append(np.array([c for _s, c in items]))
        with np.load(os.path.join(member_dir("c49"), "pool_%s.npz" % pool_kind), allow_pickle=False) as zc:
            emb_l.append(np.asarray(zc["emb"][:n_cand], np.float32))
            cnt_l.append(np.asarray(zc["count"][:n_cand], np.int32))
        _g, mP, cP, uP = P63.load_p1_full(pname)
        P63._MOCK.clear()
        pm_l.append(np.asarray(mP[:n_cand], np.float32))
        pc_l.append(np.asarray(cP[:n_cand], np.int32))
        pu_l.append(np.asarray(uP[:n_cand], bool))
        del _g, mP, cP, uP
        pst = PA.PoolStore(items, resp_vocab)
        ldas = {cch: PA.PoolLda(pst, lda_cols[cch]) for cch in ("B", "W")}
        for s in range(5):
            ref_r, q_r, owner_r, ks_r = PS.split(cand_rows, seed0 + s)
            p_loc = {"cand_rows": ref_r, "q_rows": q_r, "owner": owner_r, "k": ks_r}
            T = sim.pool_terms(pst, p_loc, ldas, lm_fit, None, want=("O", "JO", "S2", "L", "PA"))
            pools.append({"cand_rows": [[r + offset for r in g] for g in ref_r],
                          "q_rows": [[r + offset for r in g] for g in q_r], "owner": owner_r, "k": ks_r,
                          "seed": seed0 + s, "pool_kind": pool_kind,
                          "G": st.fuse([T[n] for n in ps.TERMS], list(w_G))})
        del pst, ldas, items
        gc.collect()
        offset += n_cand
    cols = np.concatenate(cols_l)
    pm, pc, pu = np.concatenate(pm_l), np.concatenate(pc_l), np.concatenate(pu_l)
    c49_emb, c49_cnt = np.concatenate(emb_l), np.concatenate(cnt_l)
    del cols_l, pm_l, pc_l, pu_l, emb_l, cnt_l
    Zs_orig = P39E.project_all(pm[:, 0, :], models_L)
    Zp = P63.project_multi(pm, mods)
    Zsp = P66.project_sp128(pm, sp)
    lp_p1to, lp_cnn = P66.single_game_rank_logprobs(pm, pc, pu, c49_emb, c49_cnt, q09, head, P66.R["rk_cnn_tau"])
    ones = np.ones(len(cols), np.int32)
    R = P66.R
    for p in pools:
        cg, qg = p["cand_rows"], p["q_rows"]
        p["z1"], _h = P39E.p1_terms(pm[:, 0, :], pu, cols, cg, qg, models_L, "b4", Zs=Zs_orig)["joint"]
        p["zm"] = P63.z_multi_of(Zp, pu, cg, qg)
        p["zs"] = P66.pool_whiten_proto_z(Zsp, pu, cg, qg, R["c13_pool_lam"])
        p["zr"] = P66.tri_rank_compat_z(s25, q09, pm, pc, pu, lp_p1to, lp_cnn, ones, cg, qg, R["rk_w_s25"],
                                        R["rk_w_p1to"], R["rk_w_cnn"], R["rk_sigma"], R["rk_eps"])
        p["ks"] = np.asarray(p["k"], np.int64)
    del Zs_orig, Zp, Zsp, lp_p1to, lp_cnn, pm, pc, pu, c49_emb, c49_cnt
    gc.collect()
    return {"pools": pools, "cols": cols, "cfg": cfg}


def pseudo_emb(name, d, cols):
    el, cl = [], []
    for kind in ("public", "private"):
        with np.load(os.path.join(d, "pool_%s.npz" % kind), allow_pickle=False) as zz:
            el.append(np.asarray(zz["emb"][:6000], np.float32))
            cl.append(np.asarray(zz["count"][:6000], np.int32))
    return EV.Emb(np.concatenate(el), np.concatenate(cl), None, cols, "sha", name)


def pseudo_Z(ctx, emb, cfg_x):
    out = []
    for p in ctx["pools"]:
        E, ok = BK.e_term(p, emb, cfg_x)
        if not np.asarray(ok).all():
            raise SystemExit("a pseudo question without a vector")
        out.append(st.zscore_rows(E, ok))
    return out


def pseudo_single(ctx, d, name):
    """NOTES.md 7 (a): member alone in the v2-8 back end (cfg78, w_E 8) on std10."""
    cfg78 = P66.cfg_lam(ctx["cfg"], P66.R["cnn_whiten_lam"])
    Zs = pseudo_Z(ctx, pseudo_emb(name, d, ctx["cols"]), cfg78)
    per = []
    for p, Z in zip(ctx["pools"], Zs):
        S = score_mat(p["G"], ensemble([Z], [1.0]), p["z1"], p["zm"], p["zs"], p["zr"], p["ks"], WE_BASE)
        per.append(float(st.score(S, p["owner"])))
    return per


def recipe_check(ctx):
    """NOTES.md 9 (c): Hydrogen 5m_tilt2_085cc_wE14 on std10, float32 caches emulated (03 lines 218-222, 244-313)."""
    names = ["c49", "S", "B", "S1", "S2"]
    ce = cfg_ens(ctx["cfg"])
    Zs = {}
    for nm in names:
        d = member_dir(nm)
        if file_shas(d) != frozen()["members"][nm]["sha256"]:
            raise SystemExit("%s vectors differ from frozen.json" % nm)
        Zs[nm] = pseudo_Z(ctx, pseudo_emb(nm, d, ctx["cols"]), ce)
    w = [WEIGHTS[n] for n in names]
    wv = np.asarray(w, np.float64) / np.sum(w)
    f32 = lambda x: np.asarray(x, np.float32).astype(np.float64)  # noqa: E731
    emu, f64 = [], []
    for i, p in enumerate(ctx["pools"]):
        Zm = np.stack([f32(Zs[n][i]) for n in names])
        E_agg = np.tensordot(wv, Zm, axes=(0, 0))
        P1 = 5.5 * f32(p["z1"]) + 3.0 * f32(p["zm"]) + 3.0 * f32(p["zs"])
        zr = f32(p["zr"])
        F = f32(p["G"]) + 14.0 * E_agg + P1 + 1.8 * zr
        wp = np.array([P66.R["w_rk_post_by_k"][str(int(k))] for k in p["ks"]], np.float64)[:, None]
        emu.append(float(st.score(st.csls(F, 20) + wp * zr, p["owner"])))
        S = score_mat(p["G"], ensemble([Zs[n][i] for n in names], w), p["z1"], p["zm"], p["zs"], p["zr"], p["ks"],
                      WE_ENS)
        f64.append(float(st.score(S, p["owner"])))
    m = float(np.mean(emu))
    return {"emulated_float32_mean": m, "emulated_per_pool": emu, "want": RECIPE_STD10,
            "abs_diff": abs(m - RECIPE_STD10), "float64_mean": float(np.mean(f64)), "float64_per_pool": f64,
            "ok": abs(m - RECIPE_STD10) <= RECIPE_TOL}


# ================================================================ E (NOTES.md 4, 10)
def e_context():
    base = E45.v26()
    pools, z1s, _bper, vec, sets = P49.e_context(base)
    prep = read_json(V28_PREP)
    if L.sha256_file(P66.Z_E) != prep["sha256"]:
        raise SystemExit("data/prereg66/z_E.npz differs from prereg66 prep_e.json")
    w_G, _kG = EV.g_combo()
    with np.load(P66.Z_E) as z:
        for p, z1 in zip(pools, z1s):
            s = p["seed"]
            p["z1"] = z1
            p["zm"], p["zs"], p["zr"] = (np.asarray(z["%s_%03d" % (n, s)]) for n in ("multi", "sp", "rank"))
            p["ks"] = np.asarray(p["k"], np.int64)
            p["G"] = st.fuse([p["T"][n] for n in BK.ps.TERMS], list(w_G))
    return {"pools": pools, "cfg": base["cfg"], "sets": sets, "vec49": vec, "base": base}


def e_ensemble(ctx, members, entries, cfg_x):
    """E_ens per pool and ok counts; member vectors loaded one at a time, sha256 checked."""
    acc = [None] * len(ctx["pools"])
    okc = [np.ones(len(p["q_rows"]), bool) for p in ctx["pools"]]
    W = float(sum(WEIGHTS[m] for m in members)) if len(members) > 1 else 1.0
    used = {}
    for m in members:
        d = member_dir(m, entries)
        path = os.path.join(d, "store_E.npz")
        sha = L.sha256_file(path)
        if sha != member_shas(m, entries)["store_E.npz"]:
            raise SystemExit("%s store_E differs from its record" % m)
        used[m] = {"file": _rel(path), "sha256": sha}
        e = E43.load_emb(path)
        w = WEIGHTS[m] if len(members) > 1 else 1.0
        for i, p in enumerate(ctx["pools"]):
            E, ok = BK.e_term(p, e, cfg_x)
            okc[i] &= np.asarray(ok, bool)
            t = (float(w) / W) * st.zscore_rows(E, ok)
            acc[i] = t if acc[i] is None else acc[i] + t
        del e
        gc.collect()
    return acc, okc, used


def e_scores(ctx, E_ens, wE):
    per, byk = [], []
    for p, Ee in zip(ctx["pools"], E_ens):
        S = score_mat(p["G"], Ee, p["z1"], p["zm"], p["zs"], p["zr"], p["ks"], wE)
        per.append(float(st.score(S, p["owner"])))
        byk.append(EV.score_by_k(S, p))
    return per, byk


def base_v28_E(ctx):
    cfg78 = P66.cfg_lam(ctx["cfg"], P66.R["cnn_whiten_lam"])
    E_ens, okc, used = e_ensemble(ctx, ["c49"], {}, cfg78)
    per, byk = e_scores(ctx, E_ens, WE_BASE)
    want = read_json(V28_EVAL)["per_pool"]["A66"]
    return per, byk, {"identical": bool(per == want and float(np.mean(per)) == SEAM_E_MEAN),
                      "mean": float(np.mean(per)), "want": SEAM_E_MEAN, "vectors": used}


# ================================================================ commands
def cmd_seam(a):
    t0 = time.time()
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else {"format": "c10-prereg67 seam v1"}
    for nm in FIXED:
        if file_shas(member_dir(nm)) != frozen()["members"][nm]["sha256"]:
            raise SystemExit("%s vectors differ from frozen.json" % nm)
    if a.part == "pseudo":
        ctx = pseudo_context()
        per = pseudo_single(ctx, member_dir("c49"), "c49")
        want = read_json(hyd("01_prep_summary.json"))["models"]["c49"]["v28"]["per_pool"][:10]
        res = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": SEAM_PSEUDO_MEAN,
               "identical": bool(per == want and float(np.mean(per)) == SEAM_PSEUDO_MEAN),
               "max_abs_diff": max(abs(x - y) for x, y in zip(per, want)),
               "seeds": [p["seed"] for p in ctx["pools"]]}
        print(json.dumps({k: res[k] for k in ("mean", "identical", "max_abs_diff")}), flush=True)
        res["recipe_5m_tilt2_085cc_wE14"] = recipe_check(ctx)
        print(json.dumps({k: v for k, v in res["recipe_5m_tilt2_085cc_wE14"].items() if "per_pool" not in k}),
              flush=True)
        res["recorded_single_v28_std10"] = {nm: float(np.mean(pseudo_single(ctx, member_dir(nm), nm)))
                                            for nm in ("S1", "S2")}
        print(json.dumps(res["recorded_single_v28_std10"]), flush=True)
        res["ok"] = bool(res["identical"] and res["recipe_5m_tilt2_085cc_wE14"]["ok"])
        key = "pseudo"
    else:
        ctx = e_context()
        per, _byk, rec = base_v28_E(ctx)
        res = dict(rec, per_pool=per)
        res["ok"] = rec["identical"]
        key = "E"
        print(json.dumps({k: res[k] for k in ("mean", "identical")}), flush=True)
    res.update(git_head=L.git_head(), peak_rss_mb=P63.peak_mb(), seconds=round(time.time() - t0, 1))
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else seam
    seam[key] = res
    seam["ok"] = bool(seam.get("pseudo", {}).get("ok") and seam.get("E", {}).get("ok"))
    L.write_json(SEAM_JSON, seam)
    if not res["ok"]:
        raise SystemExit("seam %s failed: stop (do not judge)" % a.part)
    return 0


def cmd_check_member(a):
    t0 = time.time()
    seam = read_json(SEAM_JSON) if os.path.exists(SEAM_JSON) else {}
    if not seam.get("ok") or not L.git_committed(SEAM_JSON):
        raise SystemExit("seam.json is not ok / not committed")
    entries = read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}
    if a.name in entries:
        raise SystemExit("%s already checked (members.json): the check is made once" % a.name)
    d = os.path.abspath(a.dir) if a.dir else member_dir(a.name)
    present, before = files_state(d)
    if not present:
        raise SystemExit("%s: vectors missing in %s" % (a.name, _rel(d)))
    files = check_files(a.name, d, step=OPTIONAL[a.name])
    print(json.dumps({k: files[k] for k in ("ok", "problems", "zero_count_games", "ckpt_step")}), flush=True)
    rec = {"name": a.name, "dir": _rel(d), "files": files, "before_cutoff": before,
           "threshold": CHECK_MIN, "checked_at": datetime.datetime.now().isoformat(timespec="seconds")}
    if files["ok"]:
        ctx = pseudo_context()
        per = pseudo_single(ctx, d, a.name)
        rec["std10_per_pool"] = per
        rec["std10_mean"] = float(np.mean(per))
    else:
        rec["std10_mean"] = None
    rec["pass"] = bool(files["ok"] and rec["std10_mean"] is not None and rec["std10_mean"] >= CHECK_MIN)
    rec.update(git_head=L.git_head(), peak_rss_mb=P63.peak_mb(), seconds=round(time.time() - t0, 1))
    entries = read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}
    entries[a.name] = rec
    L.write_json(MEMBERS_JSON, entries)
    print(json.dumps({k: rec[k] for k in ("name", "std10_mean", "pass", "before_cutoff", "peak_rss_mb", "seconds")}))
    return 0


def cmd_version(a):
    members, problems, info = version_from_disk()
    print(json.dumps({"members": members, "version": version_name(members), "problems": problems, "info": info},
                     indent=1))
    return 0


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    for f in (PLAN, NOTES, FROZEN, os.path.abspath(__file__), SEAM_JSON):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))
    if os.path.exists(MEMBERS_JSON) and not L.git_committed(MEMBERS_JSON):
        raise SystemExit("members.json is not committed")
    if not read_json(SEAM_JSON).get("ok"):
        raise SystemExit("seam not ok")
    uses = e_uses_so_far()
    if not uses["consistent"]:
        raise SystemExit("E use counters disagree: %s" % uses)
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before" % (a.e_use, uses["count"]))
    members, problems, vinfo = version_from_disk()
    if problems:
        raise SystemExit("; ".join(problems))
    vname = version_name(members)
    entries = read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}
    print(json.dumps({"version": vname, "members": members}), flush=True)
    ctx = e_context()
    base_per, base_byk, brec = base_v28_E(ctx)
    if not brec["identical"]:
        raise SystemExit("seam: v2-8 rebuilt in the judgement differs from prereg66 eval.json")
    ce = cfg_ens(ctx["cfg"])
    E_ens, okc, used = e_ensemble(ctx, members, entries, ce)
    not_ok = int(sum(int((~o).sum()) for o in okc))
    if not_ok:
        raise SystemExit("%d E questions without a vector in some member (NOTES.md 3): stop" % not_ok)
    per, byk = e_scores(ctx, E_ens, WE_ENS)
    j = P63.judge_rec(per, base_per)
    bk_e, bk_b = EV.mean_by_k(byk), EV.mean_by_k(base_byk)
    need = int(np.ceil(0.9 * len(per)))
    rec = {"format": "c10-prereg67 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "version": vname, "members": members,
           "weights": {m: WEIGHTS[m] for m in members}, "w_E": WE_ENS, "cfg_ens": ce,
           "member_vectors": used, "member_files": {m: member_shas(m, entries) for m in members},
           "version_info": vinfo, "ok_ens_false_questions": not_ok, "seam_v28_E_identical": True,
           "seam_json_sha256": st.sha256_lf(SEAM_JSON),
           "members_json_sha256": st.sha256_lf(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else None,
           "git_head": L.git_head(), "means": {"ens": float(np.mean(per)), "v28": float(np.mean(base_per))},
           "per_pool": {"ens": per, "v28": base_per}, "by_k": {"ens": bk_e, "v28": bk_b},
           "by_k_diff": {k: bk_e[k] - bk_b[k] for k in bk_e},
           "ens_minus_v28": {x: j[x] for x in ("mean", "ci95", "pools_higher")},
           "rule": "adopt ens if mean(ens - v2-8) >= %.2f and ci95 low > 0 and higher in >= %d of %d pools (pool "
                   "bootstrap 10000, seed 43)" % (ADOPT_MEAN, need, len(per)),
           "adopt": bool(j["mean"] >= ADOPT_MEAN and j["ci95"][0] > 0 and j["pools_higher"] >= need),
           "note": "single-model E scores are not computed (PLAN); prereg66 F1 holds for both systems",
           "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("version", "means", "ens_minus_v28", "by_k_diff", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    t0 = time.time()
    ev = read_json(a.eval)
    if not ev["adopt"]:
        raise SystemExit("eval.json does not adopt the ensemble: keep v2-8")
    entries = read_json(MEMBERS_JSON) if os.path.exists(MEMBERS_JSON) else {}
    members = ev["members"]
    for m in members:
        if file_shas(member_dir(m, entries)) != ev["member_files"][m]:
            raise SystemExit("%s vectors differ from eval.json" % m)
    base = E45.v26()
    cfg = base["cfg"]
    cfg78 = P66.cfg_lam(cfg, P66.R["cnn_whiten_lam"])
    ce = cfg_ens(cfg)
    rec, items, cand_ids, qids = E43.official_rec(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    colour = np.array([c for _s, c in items])
    ks = np.array([len(g) for g in rec["q_rows"]], np.int64)
    if not set(ks.tolist()) <= {1, 2, 3, 5}:
        raise SystemExit("question game counts outside {1, 2, 3, 5}")
    # A66's non-CNN terms exactly as prereg66 apply (p66_eval.py lines 685-723)
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
    games, means, counts, usable = P63.load_p1_full(name)
    if len(games) != len(items) or any(g["color"] != c for g, (_s, c) in zip(games, items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if not (np.array_equal(usable, Pset["usable"]) and np.array_equal(means[:, 0, :], Pset["X"])):
        raise SystemExit("Carbon's P1 loader differs from the v2-7 P1 set on %s: stop" % name)
    del Pset
    seam66 = read_json(V28_SEAM)
    mods = P63.load_multi(seam66["p1multi_sha256"])
    sp = P66.load_sp(seam66["sp128"]["sha256"])
    head = P66.load_head(P66.C49_HEAD)
    q09, s25 = P63.load_models()
    cg, qg = rec["cand_rows"], rec["q_rows"]
    zm = P63.z_multi_of(P63.project_multi(means, mods), usable, cg, qg)
    zsp = P66.pool_whiten_proto_z(P66.project_sp128(means, sp), usable, cg, qg, P66.R["c13_pool_lam"])
    path49 = P49.c49_file(a.pool)
    vec49 = P49.check_c49_file(path49, a.pool, base["tune"]["c_star"])
    e49 = E43.pool_emb(path49, items, vec49["ckpt_sha256"])

    def emb_fn(rows):
        return e49.emb[rows], np.asarray(e49.count)[rows]

    zr = P66.rank_z_pool(cg, qg, means, counts, usable, emb_fn, q09, s25, head)
    w_G, _kG = EV.g_combo()
    G = st.fuse([rec["T"][n] for n in BK.ps.TERMS], list(w_G))
    # seam: v2-8 through the ensemble function with c49 alone
    E49, ok49 = BK.e_term(rec, e49, cfg78)
    del e49
    S28 = score_mat(G, ensemble([st.zscore_rows(E49, ok49)], [1.0]), z1, zm, zsp, zr, ks, WE_BASE)
    data28 = P39E.top5_bytes(qids, st.top5(S28, cand_ids))
    with open(os.path.join(V28_APPLY, a.pool, "top5.csv"), "rb") as f:
        if P63.lf(f.read()) != P63.lf(data28):
            raise SystemExit("seam: v2-8 rebuilt on %s differs from results/prereg66/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    Zs, okc, used = [], np.ones(len(qg), bool), {}
    for m in members:
        path = os.path.join(member_dir(m, entries), "pool_%s.npz" % a.pool)
        e = EV.Emb.load(path, colour=colour)
        if len(e.emb) != len(items):
            raise SystemExit("%s: %d vectors for %d games" % (m, len(e.emb), len(items)))
        E, ok = BK.e_term(rec, e, ce)
        okc &= np.asarray(ok, bool)
        Zs.append(st.zscore_rows(E, ok))
        used[m] = {"file": _rel(path), "sha256": L.sha256_file(path), "ckpt_sha256": e.sha}
        del e
    if not okc.all():
        raise SystemExit("%d questions without a vector in some member: stop" % int((~okc).sum()))
    S = score_mat(G, ensemble(Zs, [WEIGHTS[m] for m in members]), z1, zm, zsp, zr, ks, WE_ENS)
    data = P39E.top5_bytes(qids, st.top5(S, cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    r28 = [r.split(b",") for r in data28.split(b"\n")[1:] if r]
    r67 = [r.split(b",") for r in data.split(b"\n")[1:] if r]
    rep = {"format": "c10-prereg67 apply v1", "pool": a.pool, "version": ev["version"], "members": members,
           "weights": {m: WEIGHTS[m] for m in members}, "w_E": WE_ENS, "cfg_ens": ce, "member_vectors": used,
           "v28_top5_equals_prereg66_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "k_counts": {str(x): int((ks == x).sum()) for x in (1, 2, 3, 5)},
           "games_without_P1": int((~usable).sum()), "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": st.sha256_lf(a.eval), "top5_sha256": L.sha256_file(out),
           "rows_differ_from_v28": int(sum(x != y for x, y in zip(r67, r28))),
           "top1_differs_from_v28": int(sum(x[1] != y[1] for x, y in zip(r67, r28))),
           "peak_rss_mb": P63.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    rc = E43.cmd_assemble(a)
    path = os.path.join(a.out, "submission_player.csv")
    with open(path, "rb") as f:
        mine = P63.lf(f.read()).split(b"\n")
    with open(os.path.join(V28_APPLY, "submission_player.csv"), "rb") as f:
        v28 = P63.lf(f.read()).split(b"\n")
    orig = lambda rows: [r for r in rows if r.startswith((b"pub_", b"priv_"))]  # noqa: E731
    new = lambda rows: [r for r in rows if r.startswith((b"pub2_", b"priv2_"))]  # noqa: E731
    rep = read_json(os.path.join(a.out, "assemble_report.json"))
    rep["original_800_identical_to_v28"] = orig(mine) == orig(v28) and len(orig(mine)) == 800
    rep["header_identical_to_v28"] = mine[0] == v28[0]
    rep["rows_1600"] = rep["rows"] == 1600
    t1 = {r.split(b",")[0]: r.split(b",")[1] for r in new(v28)}
    rep["new_top1_changed_vs_v28"] = {
        p: sum(1 for r in new(mine) if r.startswith(p.encode()) and t1[r.split(b",")[0]] != r.split(b",")[1])
        for p in ("pub2_", "priv2_")}
    L.write_json(os.path.join(a.out, "assemble_report.json"), rep)
    print(json.dumps({x: rep[x] for x in ("ok", "rows", "sha256", "original_800_identical_to_v28",
                                          "new_top1_changed_vs_v28")}, indent=1))
    ok = rc == 0 and rep["ok"] and rep["original_800_identical_to_v28"] and rep["rows_1600"] \
        and rep["header_identical_to_v28"]
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seam")
    s.add_argument("--part", required=True, choices=["pseudo", "e"])
    c = sub.add_parser("check-member")
    c.add_argument("--name", required=True, choices=sorted(OPTIONAL))
    c.add_argument("--dir", default=None, help="vector folder if not the frozen.json default (recorded)")
    sub.add_parser("version")
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
    if a.cmd not in ("assemble", "version"):
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_MIN_MB)
        P63._WD["s"] = P63.rss_watchdog()
    return {"seam": cmd_seam, "check-member": cmd_check_member, "version": cmd_version, "judge": cmd_judge,
            "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
