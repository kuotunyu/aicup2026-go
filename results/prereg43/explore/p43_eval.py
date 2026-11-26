"""prereg43 explore: seam, D selection, the one E judgement, official pools (PLAN.md 2-5 and revision 4). CPU only.

    PY results/prereg43/explore/p43_eval.py seam                    # -> seam.json (v2-5 rebuilt by this code path)
    PY results/prereg43/explore/p43_eval.py select                  # -> tune.json (commit it before any E vector)
    PY results/prereg43/explore/p43_eval.py judge --e-use 5         # once -> eval.json
    PY results/prereg43/explore/p43_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg43/explore/p43_eval.py assemble --v1 <v1 submission> --src <official tests dir>

Base (revision 4) = v2-5: the prereg39 system = M37 vectors + results/prereg37/explore/tune_D_B4_M37.json (prereg38
backend, fused score before its CSLS) + the KataGo P1 LDA term (joint / b4, wP 8, CSLS 10); its records are
results/prereg39/explore/tune.json + eval.json. A system here = (CNN vectors, prereg38 backend cfg, wP, k):
    S = CSLS_k( fuse_cfg(pool, e_term(pool, emb, cfg), cfg) + wP z(T_P1 joint/b4) ).
The P1 term does not depend on the CNN, so v2-5 and every candidate share it.

select (PLAN.md 2, 3, revision 4):
  1. per seed s in (0, 1): prereg42's quick screen Q on D (tune_D_B4 structure, wE {2,4,8,16} x CSLS {10,20,40}) of
     the raw and the EMA vectors of step 320000; EMA only if its Q is higher (tie -> raw);
  2. candidates in the fixed order S0, S1, S01, S0M, S01M (ensembles = p43_ens.py: per-model L2, concatenation,
     / sqrt(m); member order as written; M37 = data/prereg37/emb/step_160000); Q of each; c* = first maximum;
  3. c*: prereg38 retune_structure(kept ['B4'], the tune_D_B4 structure) on D (as prereg42's select), then the P1
     term with structure joint / b4 fixed and (wP, k) chosen on D over WP_GRID x K_GRID (first maximum) -> R;
  4. gate: R - D(v2-5) >= 0.01, D(v2-5) = 0.7244595631236577 (recomputed by this code and checked bit for bit).
judge: c* system vs v2-5 on the 20 E pools, once; adopt if mean >= +0.01, pool bootstrap (10,000, seed 43) 95% low
  > 0, higher in >= 18 of 20 pools. tune.json must be committed, the gate passed, and every new store_E.npz of
  c*'s members written after the commit of tune.json.
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P37 = os.path.join(REPO, "results", "prereg37", "explore")
P39 = os.path.join(REPO, "results", "prereg39", "explore")
P42 = os.path.join(REPO, "results", "prereg42", "explore")
for p in (P42, P39, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p39_eval as P39E  # noqa: E402   (P1 sets, LDA, b4 term, v2-5 records; unchanged)
import p42_eval as Q42  # noqa: E402    (quick screen, b4_cfg, bootstrap, check of players; unchanged)
import p43_ens as ENS  # noqa: E402     (concatenation ensemble; unchanged)

BK, EV, st, sim, L = P39E.BK, P39E.EV, P39E.st, P39E.sim, P39E.L
if Q42.BK is not BK:
    raise SystemExit("p42_eval and p39_eval import different p38_backend modules")

DATA = os.path.join(REPO, "data", "prereg43")
ENS_DIR = os.path.join(DATA, "ens")
M37_DIR = os.path.join(REPO, "data", "prereg37", "emb", "step_160000")
STEP = 320000
SEEDS = (0, 1)
WEIGHTS = ("raw", "ema")
CANDIDATES = (("S0", ("S0",)), ("S1", ("S1",)), ("S01", ("S0", "S1")), ("S0M", ("S0", "M37")),
              ("S01M", ("S0", "S1", "M37")))
V25_TUNE = os.path.join(P39, "tune.json")
V25_EVAL = os.path.join(P39, "eval.json")
V25_APPLY = os.path.join(REPO, "results", "prereg39", "apply")
M37_TUNE = os.path.join(P37, "tune_D_B4_M37.json")
D_V25 = 0.7244595631236577
E_V25 = 0.6592585013564308
P1_VARIANT, P1_PW = "joint", "b4"
WP_GRID = tuple(P39E.WP_GRID) + (16,)          # 0.25 ... 8, 16 (revision 4; contains v2-5's 8)
K_GRID = (10, 20, 40)                            # contains v2-5's 10
RETUNE_START = {"M": {"wE": 4, "csls_k": 20}}    # as prereg42's select (the start does not change the result)
GATE = 0.01
ADOPT_MEAN = 0.01
BOOT, BOOT_SEED = 10000, 43
COMP1_E_JUDGES = ("prereg36", "prereg37", "prereg38", "prereg39", "prereg42")   # competition 1 E-pool users
DROP_KEYS = ("games", "moves", "compare")
OUT_APPLY = os.path.join(REPO, "results", "prereg43", "apply")


# ================================================================ small helpers
def _abs(p):
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(REPO, p))


def _rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")
    except ValueError:                  # another drive
        return os.path.abspath(p).replace("\\", "/")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def first_max(items, key):
    """The first item with the largest key (ties -> the earlier one)."""
    best = None
    for it in items:
        if best is None or key(it) > key(best):
            best = it
    return best


def choose_weights(q_raw, q_ema):
    """PLAN.md 2: EMA only when its Q is higher; a tie keeps raw."""
    return "ema" if q_ema > q_raw else "raw"


def member_dir(member, weights=None):
    if member == "M37":
        return M37_DIR
    s = int(member[1:])
    return os.path.join(DATA, "emb_S%d" % s, "V2_step_%d_%s_tta1" % (STEP, weights))


def member_file(member, weights, which):
    """which: D, E (store vectors) or public, private (official pools)."""
    name = "store_%s.npz" % which if which in ("D", "E") else "pool_%s.npz" % which
    return os.path.join(member_dir(member, weights), name)


def cand_label_members(label):
    return dict(CANDIDATES)[label]


def ens_file(label, which):
    name = "store_%s.npz" % which if which in ("D", "E") else "pool_%s.npz" % which
    return os.path.join(ENS_DIR, label, name)


def cfg_key(embed_cfg):
    """embed_cfg without the per-game-list fields (games, moves, compare), also inside ensemble members."""
    if embed_cfg is None:
        return None
    out = {k: v for k, v in embed_cfg.items() if k not in DROP_KEYS}
    if "members" in out:
        out["members"] = [{k: (cfg_key(v) if k == "embed_cfg" else v) for k, v in m.items()} for m in out["members"]]
    return out


def embed_cfg_of(path):
    return Q42.embed_cfg_of(path)


def candidate_file(label, choice, which, build=True):
    """The vectors file of a candidate (a member's own file, or the ensemble built by p43_ens.py)."""
    members = cand_label_members(label)
    paths = [member_file(m, choice.get(m), which) for m in members]
    if len(paths) == 1:
        return paths[0]
    out = ens_file(label, which)
    if not os.path.exists(out):
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit("%s: member vectors missing: %s" % (label, [_rel(p) for p in missing]))
        if not build:
            raise SystemExit("%s not built" % _rel(out))
        args = []
        for p in paths:
            args += ["--in", p]
        ENS.main(args + ["--label", label, "--out", out])
    return out


def check_ensemble(path, label, choice, which):
    """An existing ensemble file must be made of exactly the member files of (label, choice) in that order."""
    members = cand_label_members(label)
    if len(members) == 1:
        return
    want = [cfg_key(embed_cfg_of(member_file(m, choice.get(m), which))) for m in members]
    got = embed_cfg_of(path) or {}
    got = [cfg_key(m.get("embed_cfg")) for m in got.get("members", [])]
    if got != want:
        raise SystemExit("%s is not the ensemble of %s (%s)" % (_rel(path), members, which))


_STORE = {}


def store():
    if "s" not in _STORE:
        _STORE["s"] = sim.Store(L.FEATS, load=())
    return _STORE["s"]


_SPLITS = {}


def splits():
    if "s" not in _SPLITS:
        _SPLITS["s"] = L.splits()
    return _SPLITS["s"]


def check_players(path, which):
    E, D, _T, _per = splits()
    want = L.sha256_ids({"D": D, "E": E}[which])
    with np.load(path, allow_pickle=False) as z:
        if "players_sha256_sorted_ids" not in z.files or str(z["players_sha256_sorted_ids"]) != want:
            raise SystemExit("%s is not the store_%s vectors of the prereg36 split" % (_rel(path), which))


def load_emb(path):
    e = EV.Emb.load(path, colour=store().color)
    e.path, e.embed_cfg = path, embed_cfg_of(path)
    return e


# ================================================================ the systems
def v25():
    """v2-5 exactly as prereg39 recorded and adopted it."""
    t, ev = read_json(V25_TUNE), read_json(V25_EVAL)
    if not ev.get("adopt") or ev.get("tune_sha256") != st.sha256_lf(V25_TUNE):
        raise SystemExit("prereg39 eval.json does not adopt prereg39 tune.json")
    base = P39E.base_from_record(t["base"])          # checks tune_D_B4_M37 / eval_backend have not changed
    v, pw, wP, k = P39E.m39_of(t)
    if (v, pw) != (P1_VARIANT, P1_PW):
        raise SystemExit("v2-5's P1 structure is not joint / b4")
    if wP not in WP_GRID or k not in K_GRID:
        raise SystemExit("v2-5's (wP, k) is not in the grid")
    return {"cfg": t["base"]["cfg"], "emb_dir": _abs(t["base"]["emb_dir"]),
            "ckpt_sha256": t["base"]["emb"]["ckpt_sha256"], "wP": wP, "k": k, "lda_sha256": t["lda"]["sha256"],
            "backend_k": base["k"], "backend_d_per_pool": base["d_seam"]["per_pool"],
            "backend_e_per_pool": list(base["e_per_pool"]), "e_per_pool": list(ev["per_pool"]["M39"]),
            "e_mean": ev["means"]["M39"], "d_mean": t["M39"]["tune_mean"],
            "record": {"tune": _rel(V25_TUNE), "eval": _rel(V25_EVAL), "tune_sha256_lf": st.sha256_lf(V25_TUNE),
                       "eval_sha256_lf": st.sha256_lf(V25_EVAL), "cfg": t["base"]["cfg"],
                       "emb_dir": t["base"]["emb_dir"], "emb": t["base"]["emb"], "M39": t["M39"],
                       "d_mean": t["M39"]["tune_mean"], "e_mean": ev["means"]["M39"]}}


def fused(pools, emb, cfg):
    """prereg38's backend score before its CSLS (= prereg39's F_base for a p38-format configuration)."""
    return [BK.fuse_cfg(p, BK.e_term(p, emb, cfg), cfg) for p in pools]


def scores(Fs, zs, wP, k, pools):
    return [float(st.score(st.csls(F + wP * z, k), p["owner"])) for F, z, p in zip(Fs, zs, pools)]


def p1_ctx(which):
    """The P1 LDA (fit on L, as prereg39) and the per-pool joint / b4 z term of the D or E (+E2) pools."""
    Lset = P39E.load_set("L")
    sets = [P39E.load_set("D")] if which == "D" else [P39E.load_set("E"), P39E.load_set("E2")]
    if any(s["meta"] != Lset["meta"] for s in sets):
        raise SystemExit("L and %s chunks have different metadata" % which)
    models = P39E.fit_models(Lset)
    X, usable, colour, pos = P39E.store_index(sets)
    Zs = P39E.project_all(X, models)
    names = ("D",) if which == "D" else ("E", "E2")
    return {"models": models, "lda_sha256": P39E.models_sha(models), "X": X, "usable": usable, "colour": colour,
            "pos": pos, "Zs": Zs, "sets": {n: {"games": len(s["games"]), "usable": int(s["usable"].sum()),
                                               "failed_fresh": s["failed"]} for n, s in zip(names, sets)}}


def p1_z(pool, ctx):
    """= P39E.pool_p1(...)[joint][0] with pw b4 (the joint part only: the same operations, the same values)."""
    usable = ctx["usable"]
    cg = [[ctx["pos"][int(r)] for r in g] for g in pool["cand_rows"]]
    qg = [[ctx["pos"][int(r)] for r in g] for g in pool["q_rows"]]
    cg = [[g for g in grp if usable[g]] for grp in cg]
    qg = [[g for g in grp if usable[g]] for grp in qg]
    T, has = P39E._term(ctx["Zs"][P1_VARIANT], cg, qg, P1_PW)
    return st.zscore_rows(T, has)


def p1_grid(Fs, zs, pools):
    rows = []
    for wP in WP_GRID:
        for k in K_GRID:
            per = scores(Fs, zs, wP, k, pools)
            rows.append({"mean": float(np.mean(per)), "wP": wP, "k": k, "per_pool": per})
    return rows, first_max(rows, key=lambda r: r["mean"])


def diff_rec(new, old):
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    lo, hi = Q42.bootstrap(d, BOOT, BOOT_SEED)
    return {"mean": float(d.mean()), "ci95": [lo, hi], "pools_higher": int((d > 0).sum())}


def judge_rule(new, old):
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    r = diff_rec(new, old)
    need = int(np.ceil(0.9 * len(d)))
    r["rule"] = ("adopt c* if mean difference >= %.2f and ci95 low > 0 and higher in >= %d of %d pools (pool "
                 "bootstrap, %d samples, seed %d)" % (ADOPT_MEAN, need, len(d), BOOT, BOOT_SEED))
    r["adopt"] = bool(r["mean"] >= ADOPT_MEAN and r["ci95"][0] > 0 and r["pools_higher"] >= need)
    return r


# ================================================================ seam
def seam_D(base, pools, ctx):
    e = load_emb(os.path.join(base["emb_dir"], "store_D.npz"))
    if e.sha != base["ckpt_sha256"]:
        raise SystemExit("M37 store_D is not v2-5's checkpoint")
    Fs = fused(pools, e, base["cfg"])
    backend = [float(st.score(st.csls(F, base["backend_k"]), p["owner"])) for F, p in zip(Fs, pools)]
    if backend != list(base["backend_d_per_pool"]):
        raise SystemExit("seam D: the M37 backend per-pool scores differ from tune_D_B4_M37.json")
    zs = [p1_z(p, ctx) for p in pools]
    per = scores(Fs, zs, base["wP"], base["k"], pools)
    if float(np.mean(per)) != D_V25 or base["d_mean"] != D_V25:
        raise SystemExit("seam D: v2-5 rebuilt = %r, recorded %r / %r" % (float(np.mean(per)), base["d_mean"], D_V25))
    return Fs, zs, per


def seam_E(base, pools, ctx):
    e = load_emb(os.path.join(base["emb_dir"], "store_E.npz"))
    if e.sha != base["ckpt_sha256"]:
        raise SystemExit("M37 store_E is not v2-5's checkpoint")
    Fs = fused(pools, e, base["cfg"])
    del e
    backend = [float(st.score(st.csls(F, base["backend_k"]), p["owner"])) for F, p in zip(Fs, pools)]
    if backend != base["backend_e_per_pool"]:
        raise SystemExit("seam E: the M37 backend per-pool scores differ from eval_backend.json")
    zs = [p1_z(p, ctx) for p in pools]
    per = scores(Fs, zs, base["wP"], base["k"], pools)
    if per != base["e_per_pool"] or float(np.mean(per)) != E_V25:
        raise SystemExit("seam E: v2-5 rebuilt per-pool scores differ from prereg39 eval.json")
    return zs, per


def cmd_seam(a):
    t0 = time.time()
    base = v25()
    rec = {"format": "c10-prereg43 seam v1", "base": base["record"], "git_head": L.git_head()}
    # Q path: prereg42's quick screen on C0 (M36) at wE 4, k 10 == tune_D_B4.json (prereg42 seam Q0)
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    c0 = load_emb(Q42.C0_EMB)
    cfg = BK.with_(Q42.b4_cfg(), wE=4, norm=["csls", 10])
    q0 = float(np.mean([st.score(BK.backend_scores(p, c0, cfg), p["owner"]) for p in pools]))
    want = read_json(Q42.P38_TUNE)["recommended"]["mean"]
    if q0 != want:
        raise SystemExit("seam Q0: %r != %r" % (q0, want))
    rec["Q0_C0_wE4_k10"] = {"value": q0, "tune_D_B4_mean": want, "identical": True}
    del c0
    # retune path: prereg38 retune_structure on M37 D (prereg42's call) == tune_D_B4_M37.json
    e = load_emb(os.path.join(base["emb_dir"], "store_D.npz"))
    bench = BK.Bench(pools, e, BK.base_cfg_of(RETUNE_START))
    c, r = BK.retune_structure(bench, ["B4"], Q42.b4_cfg())
    m37 = read_json(M37_TUNE)["recommended"]
    if c != m37["cfg"] or r["per_pool"] != m37["per_pool"] or r["mean"] != m37["mean"]:
        raise SystemExit("seam retune: %r %r != tune_D_B4_M37 %r %r" % (c, r["mean"], m37["cfg"], m37["mean"]))
    rec["retune_M37"] = {"cfg": c, "mean": r["mean"], "identical_to_tune_D_B4_M37": True}
    del e, bench
    # D: v2-5 rebuilt
    ctx = p1_ctx("D")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("seam: the P1 LDA refit differs from prereg39 tune.json")
    _Fs, _zs, per = seam_D(base, pools, ctx)
    rec["D"] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": D_V25, "identical": True,
                "lda_sha256": ctx["lda_sha256"], "sets": ctx["sets"]}
    del ctx, pools, _Fs, _zs
    # E: v2-5 rebuilt (the base's known scores only; no candidate is scored)
    if not a.skip_e:
        pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
        ctx = p1_ctx("E")
        _zs, per = seam_E(base, pools, ctx)
        rec["E"] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": E_V25, "identical": True,
                    "sets": ctx["sets"]}
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in rec if k not in ("base",)}, indent=1))
    return 0


# ================================================================ select (D)
def train_records(drop):
    out = {}
    for s in SEEDS:
        p = os.path.join(DATA, "S%d" % s, "train_record_S%d.json" % s)
        tr = read_json(p) if os.path.exists(p) else None
        done = bool(tr and tr.get("finished") and not tr.get("smoke") and tr.get("steps") == STEP
                    and tr["config"]["seed"] == s and tr["config"]["variant"] == "V2")
        if "S%d" % s in drop:
            if done:
                raise SystemExit("S%d finished: it cannot be dropped (PLAN.md 2)" % s)
            continue
        if not done:
            raise SystemExit("S%d has no finished 320,000-step train record (use --drop S%d only past its deadline, "
                             "PLAN.md 2 / revision 5)" % (s, s))
        out["S%d" % s] = {"file": _rel(p), "file_sha256": L.sha256_file(p), "steps": tr["steps"],
                          "device": tr.get("device"), "torch": tr.get("torch")}
    return out


def no_e_vectors_yet():
    """PLAN.md 3: no E vector of a prereg43 checkpoint before tune.json is committed."""
    found = [p for s in SEEDS for w in WEIGHTS for p in [member_file("S%d" % s, w, "E")] if os.path.exists(p)]
    found += glob.glob(os.path.join(ENS_DIR, "*", "store_E.npz"))
    if found:
        raise SystemExit("E vectors exist before select: %s" % [_rel(p) for p in found])


def quick(pools, path):
    check_players(path, "D")
    e = load_emb(path)
    rows = Q42.quick_grid(pools, e)
    b = Q42.best_row(rows)
    return e, {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt": e.name, "ckpt_sha256": e.sha,
               "embed_cfg": e.embed_cfg, "Q": b["mean"], "best": b,
               "grid": [{k: r[k] for k in ("mean", "wE", "k")} for r in rows]}


def cmd_select(a):
    t0 = time.time()
    drop = set(x for x in (a.drop or "").split(",") if x)
    if not drop <= {"S1"}:
        raise SystemExit("only S1 may be dropped (PLAN.md 2: seed 0 is finished)")
    trs = train_records(drop)
    no_e_vectors_yet()
    base = v25()
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    # D seam first: v2-5 rebuilt bit for bit, and the shared P1 term
    ctx = p1_ctx("D")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg39 tune.json")
    _Fb, zs, base_per = seam_D(base, pools, ctx)
    del _Fb
    # 1. raw / EMA per seed
    choice, seeds = {}, {}
    for s in SEEDS:
        m = "S%d" % s
        if m in drop:
            continue
        ck = os.path.join(DATA, m, "ckpt", "step_%d.pt" % STEP)
        cks = L.sha256_file(ck)
        q = {}
        for w in WEIGHTS:
            e, q[w] = quick(pools, member_file(m, w, "D"))
            if e.sha != cks or (e.embed_cfg or {}).get("weights") != w:
                raise SystemExit("%s %s vectors were not made from %s with weights %s" % (m, w, _rel(ck), w))
            del e
            print(json.dumps({"member": m, "weights": w, "Q": q[w]["Q"]}), flush=True)
        choice[m] = choose_weights(q["raw"]["Q"], q["ema"]["Q"])
        seeds[m] = {"ckpt": _rel(ck), "ckpt_sha256": cks, "raw": q["raw"], "ema": q["ema"], "chosen": choice[m]}
    # 2. candidates in the fixed order
    order = [lab for lab, mem in CANDIDATES if not set(mem) & drop]
    cands = {}
    for lab in order:
        members = cand_label_members(lab)
        if len(members) == 1:
            cands[lab] = dict(seeds[lab][choice[lab]])
        else:
            path = candidate_file(lab, choice, "D")
            check_ensemble(path, lab, choice, "D")
            _e, cands[lab] = quick(pools, path)
            del _e
        cands[lab]["members"] = [[m, choice.get(m)] for m in members]
        print(json.dumps({"candidate": lab, "Q": cands[lab]["Q"]}), flush=True)
    cstar = first_max(order, key=lambda x: cands[x]["Q"])
    # 3. retune c* (prereg38, kept B4) and choose the P1 weight / CSLS k on D
    e = load_emb(_abs(cands[cstar]["file"]))
    bench = BK.Bench(pools, e, BK.base_cfg_of(RETUNE_START))
    cfg, r = BK.retune_structure(bench, ["B4"], Q42.b4_cfg())
    Fs = fused(pools, e, cfg)
    del bench
    rows, top = p1_grid(Fs, zs, pools)
    gate = top["mean"] - D_V25
    rec = {"format": "c10-prereg43 tune v1", "plan_revision": 6, "d_seeds": EV.D_SEEDS,
           "train_records": trs, "dropped": sorted(drop), "seeds": seeds, "candidates_in_order": order,
           "candidates": cands, "Q": {lab: cands[lab]["Q"] for lab in order},
           "c_star": {"label": cstar, "members": cands[cstar]["members"], "file": cands[cstar]["file"],
                      "file_sha256": cands[cstar]["file_sha256"], "ckpt_sha256": cands[cstar]["ckpt_sha256"],
                      "embed_cfg": cands[cstar]["embed_cfg"],
                      "backend": {"cfg": cfg, "mean": r["mean"], "per_pool": r["per_pool"], "by_k": r["by_k"]},
                      "p1": {"variant": P1_VARIANT, "pw": P1_PW, "wP": top["wP"], "csls_k": top["k"],
                             "mean": top["mean"], "per_pool": top["per_pool"]}},
           "p1_grid": [{k: x[k] for k in ("mean", "wP", "k")} for x in rows],
           "base": base["record"], "base_D": {"per_pool": base_per, "mean": float(np.mean(base_per)),
                                              "identical_to_recorded_mean": True},
           "lda": {"sha256": ctx["lda_sha256"]}, "p1_sets": ctx["sets"],
           "R_minus_base_D": diff_rec(top["per_pool"], base_per),
           "gate": {"rule": "R(c* + P1) - D(v2-5) >= %.2f, D(v2-5) = %r" % (GATE, D_V25), "value": gate,
                    "pass": bool(gate >= GATE)},
           "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({"Q": rec["Q"], "seeds": {m: v["chosen"] for m, v in seeds.items()}, "c_star": cstar,
                      "backend": rec["c_star"]["backend"]["cfg"], "R_backend_only": r["mean"],
                      "p1": rec["c_star"]["p1"]["wP"], "k": top["k"], "R": top["mean"],
                      "R_minus_base_D": rec["R_minus_base_D"], "gate": rec["gate"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far(repo=REPO):
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def commit_time(path):
    rel = _rel(path)
    out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%ct", "--", rel], capture_output=True,
                         text=True).stdout.strip()
    if not out:
        raise SystemExit("%s has no commit" % rel)
    return int(out)


def made_after(paths, t_commit):
    """Every file must have been written after the commit time (seconds since the epoch)."""
    late = [p for p in paths if os.path.getmtime(p) <= t_commit]
    if late:
        raise SystemExit("E vectors older than the commit of tune.json: %s" % [_rel(p) for p in late])


def cstar_file(tune, which):
    """c*'s vectors for D / E / public / private, checked against tune.json (checkpoint hashes, embed settings)."""
    cs = tune["c_star"]
    choice = {m: w for m, w in cs["members"] if w}
    lab = cs["label"]
    path = candidate_file(lab, choice, which)
    check_ensemble(path, lab, choice, which)
    with np.load(path, allow_pickle=False) as z:
        sha = str(z["ckpt_sha256"])
    if sha != cs["ckpt_sha256"] or cfg_key(embed_cfg_of(path)) != cfg_key(cs["embed_cfg"]):
        raise SystemExit("%s was not made from c*'s checkpoint(s) with c*'s embed settings" % _rel(path))
    return path, choice


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    tune = read_json(a.tune)
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 3)")
    prior = e_judgements_so_far()
    if a.e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, len(prior), prior))
    base = v25()
    if base["record"]["tune_sha256_lf"] != tune["base"]["tune_sha256_lf"] or \
            base["record"]["eval_sha256_lf"] != tune["base"]["eval_sha256_lf"]:
        raise SystemExit("v2-5's records changed since tune.json")
    cs = tune["c_star"]
    choice = {m: w for m, w in cs["members"] if w}
    new = [member_file(m, choice.get(m), "E") for m, _w in cs["members"] if m != "M37"]
    for p in new:
        if not os.path.exists(p):
            raise SystemExit("missing %s (compute it on the 4090 after tune.json is committed)" % _rel(p))
    made_after(new, commit_time(a.tune))
    path, _choice = cstar_file(tune, "E")
    check_players(path, "E")
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    ctx = p1_ctx("E")
    if ctx["lda_sha256"] != tune["lda"]["sha256"]:
        raise SystemExit("the P1 LDA refit differs from tune.json")
    zs, bper = seam_E(base, pools, ctx)
    e = load_emb(path)
    cfg = cs["backend"]["cfg"]
    Fs = fused(pools, e, cfg)
    del e
    wP, k = cs["p1"]["wP"], cs["p1"]["csls_k"]
    per = scores(Fs, zs, wP, k, pools)
    byk = [EV.score_by_k(st.csls(F + wP * z, k), p) for F, z, p in zip(Fs, zs, pools)]
    backend_only = [float(st.score(st.csls(F, int(cfg["norm"][1])), p["owner"])) for F, p in zip(Fs, pools)]
    rec = {"format": "c10-prereg43 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "c_star": {x: cs[x] for x in ("label", "members", "ckpt_sha256")},
           "e_file": _rel(path), "e_file_sha256": L.sha256_file(path), "cfg": cfg, "p1": cs["p1"],
           "base": tune["base"], "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune),
           "git_head": L.git_head(), "p1_sets": ctx["sets"],
           "means": {"c_star": float(np.mean(per)), "base": float(np.mean(bper)),
                     "c_star_backend_only": float(np.mean(backend_only))},
           "per_pool": {"c_star": per, "base": bper, "c_star_backend_only": backend_only},
           "by_k": {"c_star": EV.mean_by_k(byk)}}
    j = judge_rule(per, bper)
    rec["c_star_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"], rec["adopt"] = j["rule"], j["adopt"]
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c_star_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def official_rec(src, pool):
    """The pool's games and the G terms, exactly as prereg39's apply."""
    import p35_apply as pa
    cands, qs = pa.read_pool(src, pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    fit = sim.Store(L.FEATS, load=())
    pst = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, lm = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        lm[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(pst, ld.cols)
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    rec["T"] = sim.pool_terms(pst, rec, ldas, lm, None, want=("O", "JO", "S2", "L", "PA"))
    return rec, items, cand_ids, qids


def pool_emb(path, items, sha):
    e = EV.Emb.load(path, colour=np.asarray([c for _s, c in items]))
    if e.sha != sha or len(e.emb) != len(items):
        raise SystemExit("%s does not match the checkpoint / the pool's games" % _rel(path))
    return e


def cmd_apply(a):
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c* (or belongs to another tune.json): keep v2-5")
    base = v25()
    rec, items, cand_ids, qids = official_rec(a.src, a.pool)
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
    if P39E.models_sha(models) != tune["lda"]["sha256"]:
        raise SystemExit("the LDA refit differs from tune.json")
    colour = np.array([c for _s, c in items])
    z, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                           P1_PW)[P1_VARIANT]
    # seam: v2-5 rebuilt on this pool == prereg39's uploaded top5
    eb = pool_emb(os.path.join(base["emb_dir"], "pool_%s.npz" % a.pool), items, base["ckpt_sha256"])
    Fb = BK.fuse_cfg(rec, BK.e_term(rec, eb, base["cfg"]), base["cfg"])
    data_b = P39E.top5_bytes(qids, st.top5(st.csls(Fb + base["wP"] * z, base["k"]), cand_ids))
    with open(os.path.join(V25_APPLY, a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: v2-5 rebuilt on %s differs from results/prereg39/apply/%s/top5.csv" % (a.pool,
                                                                                                         a.pool))
    del eb, Fb
    cs = tune["c_star"]
    path, _choice = cstar_file(tune, a.pool)
    e = pool_emb(path, items, cs["ckpt_sha256"])
    cfg = cs["backend"]["cfg"]
    F = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
    data = P39E.top5_bytes(qids, st.top5(st.csls(F + cs["p1"]["wP"] * z, cs["p1"]["csls_k"]), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg43 apply v1", "pool": a.pool, "c_star": cs["label"], "members": cs["members"],
           "vectors": _rel(path), "vectors_sha256": L.sha256_file(path), "cfg": cfg, "p1": cs["p1"],
           "v25_top5_equals_prereg39_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "tune_sha256": st.sha256_lf(a.tune), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    import p35_apply as pa
    return pa.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--skip-e", action="store_true")
    m.add_argument("--out", default=os.path.join(HERE, "seam.json"))
    s = sub.add_parser("select")
    s.add_argument("--drop", default=None, help="S1 only, past its deadline without a finished run (PLAN.md 2)")
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    p.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", required=True)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()                 # BELOW_NORMAL, 4 logical processors, 1 BLAS thread (prereg38's manners)
        BK.wait_for_ram(need_mb=8192)
    return {"seam": cmd_seam, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
