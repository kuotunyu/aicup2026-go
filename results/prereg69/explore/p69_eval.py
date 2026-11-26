"""prereg69 (competition 2): the M8 gate simulation with 64's fusion form as the base (PLAN.md, NOTES.md 2-5).
CPU only, .venv (numpy / scipy), p58_lib.limits (4 cores, <= 3 GB). prereg58 / 59 / 64 / 65 and tools are imported
unchanged. No inner, no test file, never data/local, never tools/score.py.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe results/prereg69/explore/p69_eval.py mergesim  -> results/prereg69/explore/merge_sim.json

One pass over prereg65's pools (E n H, 100 pools per mode, seeds 9000 + sd) evaluates two bases on the same pools:
  p65  prereg65's own base (S25 + OPP4B_S0, section-2 weights) -> seam S: every number = prereg65's merge_sim.json
  p64  64's form: w_A log_softmax(lp_S25) + w_O OPP3_S0 with 64's frozen (w_A, w_O) -> the gate
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P65 = os.path.join(REPO, "results", "prereg65", "explore")
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg19", "explore"),
          os.path.join(REPO, "results", "prereg31", "explore"), os.path.join(REPO, "results", "prereg36", "explore"),
          os.path.join(REPO, "results", "prereg39", "explore"), os.path.join(REPO, "results", "prereg58", "explore"),
          os.path.join(REPO, "results", "prereg59", "explore"), os.path.join(REPO, "results", "prereg64", "explore"),
          P65, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p58_eval as E58  # noqa: E402
import p58_lib as K58  # noqa: E402
import p59_lib as K59  # noqa: E402
import p64_lib as K64  # noqa: E402
import p65_eval as E65  # noqa: E402
import p65_lib as K  # noqa: E402
import p69_lib as K69  # noqa: E402

log, git, rel, committed_clean, peak_gb = E58.log, E58.git, E58.rel, E58.committed_clean, E58.peak_gb
FROZEN = os.path.join(HERE, "frozen.json")
NOTES = os.path.join(HERE, "NOTES.md")
PLAN = os.path.join(HERE, "PLAN.md")
MERGE_SIM = os.path.join(HERE, "merge_sim.json")
REF65 = os.path.join(P65, "merge_sim.json")
MODES = (("zero_repeat_0pct", 0.0), ("moderate_repeat_35pct", 0.35), ("paired_repeat_100pct", 1.0))
T64 = (K64.TAU1, K64.TAU2, K64.BIAS)       # (50, 30, 1)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def code_state():
    mods = E58.loaded_repo_modules()
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prereg69_dirty": git("status", "--porcelain", "--", "results/prereg69"),
            "imported_dirty": git("status", "--porcelain", "--", *mods) if mods else "",
            "sha256_lf": {x: K58.sha_lf(os.path.join(REPO, *x.split("/"))) for x in mods}}


def require_clean(paths):
    for p in paths:
        if not committed_clean(p):
            raise SystemExit(f"{rel(p)} is not committed and clean")
    cs = code_state()
    if cs["prereg69_dirty"] or cs["imported_dirty"]:
        raise SystemExit(f"not committed and clean: {cs['prereg69_dirty']} {cs['imported_dirty']}")
    return cs


def games_and_terms(P1, Mp, log_):
    """prereg65 section 2 (p65_eval.neon_section2_3 lines 566-663) copied, plus the 64-proxy per-game values."""
    import p19_forward as FW19
    import p31_lib as L31
    with np.load(K58.guard(K58.FEATS_META), allow_pickle=False) as zm:
        meta = {k: np.asarray(zm[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}
    with np.load(K58.guard(K58.OPP_D), allow_pickle=False) as zd:
        s0_D, c0_D, sr0_D = np.asarray(zd["emb"], np.float32), np.asarray(zd["count"], np.int64), np.asarray(zd["store_row"], np.int64)
    with np.load(K58.guard(K58.OPP_E), allow_pickle=False) as ze:
        s0_E, c0_E, sr0_E = np.asarray(ze["emb"], np.float32), np.asarray(ze["count"], np.int64), np.asarray(ze["store_row"], np.int64)
    sr = np.concatenate([sr0_D, sr0_E])
    s0 = np.concatenate([s0_D, s0_E], axis=0)
    c0 = np.concatenate([c0_D, c0_E], axis=0)
    del s0_D, s0_E, c0_D, c0_E, sr0_D, sr0_E
    keep = np.array([int(x) in P1["row_to_p1"] for x in sr.tolist()], bool)
    sr, s0, c0 = sr[keep], s0[keep], c0[keep]
    idx = np.array([P1["row_to_p1"][int(x)] for x in sr.tolist()], np.int64)
    rk, gid, pid, col = meta["rank"][sr], meta["game_id"][sr], meta["player_id"][sr], meta["color"][sr]
    r_, c_ = K58.rank_index(rk), K58.colour_index(col)
    hd_raw, jr = L31.HdTable().join(rk, gid, pid, col)
    if not jr["ok"]:
        raise SystemExit("HD join failed")
    h_ = K58.hd_group(hd_raw)
    f_ = np.array([K58.player_fold(p) for p in pid.tolist()], np.int64)
    ok = P1["ok"][idx] & (c0 > 0) & (hd_raw <= 1)
    Wn0, wr0, _ = K58.load_head(K58.guard(K58.HEAD_NPZ))
    with np.load(K58.guard(K59.PROTO), allow_pickle=False) as zp:
        p0_ids, p0_proto, p0_rk = np.asarray(zp["player_id"]).astype(str), np.asarray(zp["proto"], np.float64), np.asarray(zp["rank"], np.int64)
    p0_map = {p: i for i, p in enumerate(p0_ids.tolist())}
    own0 = np.array([p0_map.get(p, -1) for p in pid.tolist()], np.int64)
    # one call with both tau sets: _rank_sums computes each tau from the same cosines independently
    pts = K59.class_parts(s0, Wn0, wr0, p0_proto, p0_rk, own=own0, tau1=(70, T64[0]), tau2=(50, T64[1]))
    del s0
    gc.collect()
    N_h0, N_p0 = np.bincount(wr0, minlength=K.L).astype(np.float64), np.bincount(p0_rk, minlength=K.L).astype(np.float64)
    SA0, mA0 = pts["A"][70]
    SB0, mB0 = pts["B"][50]
    m0 = np.maximum(mA0, mB0 - 1.0)
    q70 = (SA0 / N_h0[None, :] * N_h0.mean()) * np.exp(mA0 - m0)[:, None] + (SB0 / N_p0[None, :] * N_p0.mean()) * np.exp(mB0 - 1.0 - m0)[:, None]
    q70 /= np.maximum(q70.sum(axis=1, keepdims=True), 1e-300)
    q64 = K59.combine(pts, T64[0], T64[1], T64[2])
    del pts
    R_l25 = E65.oof_R(q70, r_, h_, c_, f_, ok, True, 0.25)
    R_p1 = K.game_R(P1["q"][idx], Mp, h_, c_)
    R_64 = E65.oof_R(q64, r_, h_, c_, f_, ok, False, 0.0)
    lg64 = np.log(R_64 + K.EPS)
    by_p = defaultdict(lambda: {0: [], 1: [], "r": 0, "f": 0, "is_H": False})
    for i in range(len(sr)):
        if ok[i]:
            p = pid[i]
            by_p[p][int(c_[i])].append(i)
            by_p[p]["r"] = int(r_[i])
            by_p[p]["f"] = int(f_[i])
            by_p[p]["is_H"] = (E65.holdout_bit_s25(f"{rk[i]}:{p}") == 1)
    rng = np.random.default_rng(20261123)
    grp, y, fo, kk = [], [], [], []
    for p in sorted(by_p):
        d = by_p[p]
        if not d["is_H"]:
            continue
        for ci in (0, 1):
            pool = d[ci]
            for kv in (5, 10, 15, 20):
                if len(pool) >= kv:
                    for _ in range(5):
                        grp.append(rng.choice(pool, size=kv, replace=False).tolist())
                        y.append(d["r"])
                        fo.append(d["f"])
                        kk.append(kv)
    y, fo, kk = np.asarray(y, np.int64), np.asarray(fo, np.int64), np.asarray(kk, np.int64)
    H_DE = [P1["H"][i][idx] for i in range(3)]
    _, _, lp_s25, _ = FW19.set_term(P1["s25"], H_DE, grp, len(sr), scale=4.0)
    E_s25 = ((kk / 12.0) ** 0.40)[:, None] * (4.0 * lp_s25)
    lg4b = K.lr_lg(q70, R_l25, R_p1, K.P_POW)
    E_4b = np.array([10.0 * lg4b[g].mean(axis=0) for g in grp])
    _, pred, w_eh, w_full = E65.fit_scalar_oof([E_s25, E_4b], y, fo, np.ones(len(y), bool))
    sec2 = {"n_players": int(sum(d["is_H"] for d in by_p.values())), "n_questions": int(len(y)),
            "OPP4B_S0_l25_p1.5": {"rb": K.rb_score(pred, y), "weights": w_eh, "weights_unrounded": w_full}}
    # record only: the 64-proxy base on the same section-2 questions with 64's frozen weights
    Z64 = K69.W_A_64 * K.log_softmax(lp_s25) + K69.W_O_64 * np.array([10.0 * lg64[g].mean(axis=0) for g in grp])
    sec2["record_64proxy_frozen_weights_rb"] = K.rb_score(K.decode(Z64), y)
    sec2["record_S25_alone_rb"] = K.rb_score(K.decode(lp_s25), y)
    log_(f"section 2: {json.dumps(sec2)}")
    return {"sr": sr, "rk": rk, "pid": pid, "col": col, "r_": r_, "ok": ok, "H_DE": H_DE, "lg4b": lg4b,
            "lg64": lg64, "w65": w_eh, "s25": P1["s25"]}, sec2


def section3(G, log_):
    """prereg65 section 3 (p65_eval.neon_section2_3 lines 664-718) copied; both bases on every pool."""
    import p19_forward as FW19
    sr, rk, pid, col, r_, ok = G["sr"], G["rk"], G["pid"], G["col"], G["r_"], G["ok"]
    with np.load(K58.guard(K64.OWN_E), allow_pickle=False) as ze:
        own_E, cnt_E, sr_E = np.asarray(ze["emb"], np.float32), np.asarray(ze["count"], np.int64), np.asarray(ze["store_row"], np.int64)
    sr_to = {int(x): i for i, x in enumerate(sr.tolist())}
    eh = defaultdict(lambda: {"B": [], "W": [], "r": 0})
    own_l2 = K64.l2(own_E)
    for i in range(len(sr_E)):
        if cnt_E[i] <= 0:
            continue
        di = sr_to.get(int(sr_E[i]), -1)
        if di < 0 or not ok[di]:
            continue
        if E65.holdout_bit_s25(f"{rk[di]}:{pid[di]}") != 1:
            continue
        eh[pid[di]][col[di]].append((i, di))
        eh[pid[di]]["r"] = int(r_[di])
    elig = sorted(p for p, d in eh.items() if len(d["B"]) >= 20 and len(d["W"]) >= 20)
    w = G["w65"]
    out = {b: {"n_elig_players": len(elig), "n_pools": 100, "modes": {}} for b in ("p65", "p64")}
    out["p65"]["base_weights"] = w
    out["p64"]["base_weights"] = [K69.W_A_64, K69.W_O_64]
    for mode, _frac in MODES:
        acc = {b: {"gains": [], "changed": [], "edges": [], "base": []} for b in ("p65", "p64")}
        for sd in range(100):
            rp = np.random.default_rng(9000 + sd)
            if mode == "zero_repeat_0pct":
                sel = [(p, rp.choice(["B", "W"]), int(rp.choice([5, 10, 15, 20]))) for p in elig]
            elif mode == "moderate_repeat_35pct":
                sel = []
                reps = set(rp.choice(elig, size=int(round(0.35 * len(elig))), replace=False).tolist())
                for p in elig:
                    c1 = str(rp.choice(["B", "W"]))
                    if p in reps:
                        sel.append((p, c1, int(rp.choice([5, 10]))))
                        sel.append((p, c1, int(rp.choice([5, 10]))))
                    else:
                        sel.append((p, c1, int(rp.choice([5, 10, 15, 20]))))
            else:
                sel = []
                for p in elig:
                    c1 = str(rp.choice(["B", "W"]))
                    sel.append((p, c1, 10))
                    sel.append((p, c1, 10))
            used = defaultdict(set)
            eg, g_own, g_de, qc, qy, qn = [], [], [], [], [], []
            cur = 0
            for p, cc, kv in sel:
                avail = [pp for pp in eh[p][cc] if pp[0] not in used[(p, cc)]]
                if len(avail) < kv:
                    continue
                ch_i = rp.choice(len(avail), size=kv, replace=False)
                chosen = [avail[j] for j in ch_i]
                for oi, _di in chosen:
                    used[(p, cc)].add(oi)
                eg.append(own_l2[[oi for oi, _ in chosen]])
                g_own.append(list(range(cur, cur + kv)))
                cur += kv
                g_de.append([di for _, di in chosen])
                qc.append(cc)
                qy.append(eh[p]["r"])
                qn.append(kv)
            S = K64.cc_whiten_sim(np.vstack(eg), g_own, qc)
            qy, qn = np.asarray(qy, np.int64), np.asarray(qn, np.int64)
            _, _, lp_p, _ = FW19.set_term(G["s25"], G["H_DE"], g_de, len(sr), scale=4.0)
            E_s = ((qn / 12.0) ** 0.40)[:, None] * (4.0 * lp_p)
            E_o = np.array([10.0 * G["lg4b"][g].mean(axis=0) for g in g_de])
            O64 = np.array([10.0 * G["lg64"][g].mean(axis=0) for g in g_de])
            bases = {"p65": K.log_softmax(w[0] * E_s + w[1] * E_o),
                     "p64": K.log_softmax(K69.W_A_64 * K.log_softmax(lp_p) + K69.W_O_64 * O64)}
            for b, lp_base in bases.items():
                pr_b = K.decode(lp_base)
                rb_b = K.rb_score(pr_b, qy)
                pr_m, _lpn, ned, _p0 = K.m8(lp_base, S, qc)
                acc[b]["base"].append(rb_b)
                acc[b]["gains"].append(K.rb_score(pr_m, qy) - rb_b)
                acc[b]["changed"].append(int((pr_m != pr_b).sum()))
                acc[b]["edges"].append(ned)
        for b in ("p65", "p64"):
            a = acc[b]
            out[b]["modes"][mode] = {"base_rb_mean": float(np.mean(a["base"])), "M8": {
                "gain_mean": float(np.mean(a["gains"])), "gain_min": float(np.min(a["gains"])),
                "gain_max": float(np.max(a["gains"])), "changed_mean": float(np.mean(a["changed"])),
                "changed_max": int(np.max(a["changed"])),
                "pools_with_any_change": int(sum(1 for x in a["changed"] if x > 0)),
                "edges_mean": float(np.mean(a["edges"]))}}
            log_(f"M8 {b} {mode}: {json.dumps(out[b]['modes'][mode])}")
    return out


def seam_s(sec2, s3_65, ref):
    """Every prereg65 number reproduced exactly (==)."""
    diffs = []
    r2 = ref["section2"]
    if sec2["n_players"] != r2["n_players"] or sec2["n_questions"] != r2["n_questions"]:
        diffs.append("section2 counts")
    for k in ("rb", "weights", "weights_unrounded"):
        if sec2["OPP4B_S0_l25_p1.5"][k] != r2["OPP4B_S0_l25_p1.5"][k]:
            diffs.append(f"section2 {k}")
    r3 = ref["section3"]
    if s3_65["n_elig_players"] != r3["n_elig_players"] or s3_65["base_weights"] != r3["base_weights"]:
        diffs.append("section3 header")
    for mode, _ in MODES:
        a, b = s3_65["modes"][mode], r3["modes"][mode]
        if a["base_rb_mean"] != b["base_rb_mean"]:
            diffs.append(f"{mode} base_rb_mean")
        for k, v in b["M8"].items():
            if a["M8"][k] != v:
                diffs.append(f"{mode} M8 {k}")
    return {"ok": not diffs, "differences": diffs}


def cmd_mergesim(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(MERGE_SIM):
        raise SystemExit("merge_sim.json exists")
    cs = require_clean([PLAN, FROZEN, NOTES, os.path.abspath(__file__), K69.__file__])
    ref = K58.read_json(REF65)
    _M, Mp, _k = E65.load_kernel()
    P1 = E65.neon_p1_compact()
    log(f"P1 compact {len(P1['rows'])} games ({time.time() - t0:.0f}s)")
    G, sec2 = games_and_terms(P1, Mp, log)
    del P1
    gc.collect()
    s3 = section3(G, log)
    seam = seam_s(sec2, s3["p65"], ref)
    z0, z35 = s3["p64"]["modes"]["zero_repeat_0pct"]["M8"], s3["p64"]["modes"]["moderate_repeat_35pct"]["M8"]
    gate = {"zero_changed_all_pools": bool(z0["changed_max"] == 0 and z0["pools_with_any_change"] == 0),
            "gain_35_mean": z35["gain_mean"], "gain_35_positive": bool(z35["gain_mean"] > 0)}
    gate["passed"] = bool(seam["ok"] and gate["zero_changed_all_pools"] and gate["gain_35_positive"])
    rep = {"format": "c10-prereg69 merge simulation v1", "created": now(), "code": cs,
           "what": "prereg65's M8 simulation (E n H, 100 pools per mode, seeds 9000 + sd) with 64's fusion form as the "
                   "base (NOTES 2: w_A log_softmax(lp_S25) + w_O OPP3_S0, 64's frozen weights); seam S = prereg65's "
                   "own base on the same pools reproduces its merge_sim.json exactly",
           "section2": sec2, "section3_p65_base": s3["p65"], "section3_p64_base": s3["p64"],
           "seam_S": seam, "gate": gate,
           "rule": "seam S ok AND (64-proxy base) 0% changed = 0 in 100/100 pools AND 35% mean gain > 0",
           "seconds": round(time.time() - t0, 1), "peak_working_set_gb": peak_gb(st)}
    K58.write_json(MERGE_SIM, rep)
    log(json.dumps({k: rep[k] for k in ("seam_S", "gate", "seconds", "peak_working_set_gb")}))
    return 0 if seam["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mergesim")
    a = ap.parse_args(argv)
    return {"mergesim": cmd_mergesim}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
