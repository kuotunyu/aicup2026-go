"""prereg54 (PLAN.md): score-level ensembles of strong members (c49, prereg51 L, prereg51 B). CPU only.

    PY results/prereg54/explore/p54_eval.py seam  [--base c49|L|B]   (L / B: prereg51's eval_<arm>.json, apply/<arm>)   # -> seam.json
    PY results/prereg54/explore/p54_eval.py dev   [base args]            # pseudo-pools once -> dev.json (commit it)
    PY results/prereg54/explore/p54_eval.py judge --e-use N [base args]  # E once -> eval.json
    PY results/prereg54/explore/p54_eval.py apply --src <official tests dir> --pool public|private [base args, --base-apply D]
    PY results/prereg54/explore/p54_eval.py assemble --src <official tests dir>

System(members) = CSLS_20( fuse_cfg(pool, ens_E([e_term(pool, emb_m, cfg_v26) for m]), cfg_v26) + 8 z(T_P1 joint/b4) ),
ens_E = prereg52's rule (one member -> its own matrix; else mean of row-z, ok = intersection).
Seams (bit for bit): the c49-only path == prereg52 pseudo.json per_pool A49 (pseudo-pools) and == prereg49 eval.json
per_pool c49 (E; the already-judged v2-7 score, nothing else is computed on E by `seam`); a base other than c49 must
also reproduce its prereg51 judgement's E per-pool scores.
Nothing here opens the validation-answer folder or runs tools/score.py; the pseudo-pools read the candidate files only.
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
for p in (os.path.join(REPO, "results", "prereg52", "explore"), os.path.join(REPO, "results", "prereg49", "explore"),
          HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p52_eval as P52  # noqa: E402     (ens_E, E45 / E43 handles; unchanged)
import p52_pseudo as PS  # noqa: E402    (pseudo-pool split, candidate reader; unchanged)
import p49_eval as P49  # noqa: E402     (E-use list, V1 submission; unchanged)

E45, E43, P39E, BK, EV, st, L = P52.E45, P52.E43, P52.P39E, P52.BK, P52.EV, P52.st, P52.L
_rel, read_json = E43._rel, E43.read_json

MEMBERS = {
    "c49": {"ckpt": "data/prereg49/S0/ckpt/step_320000.pt", "emb": "data/prereg49/emb/V2_step_320000_raw_tta1"},
    "L": {"ckpt": "data/prereg51/L/ckpt/step_480000.pt", "emb": "data/prereg51/emb_L/*_step_480000_raw_tta1"},
    "B": {"ckpt": "data/prereg51/B/ckpt/step_320000.pt", "emb": "data/prereg51/emb_B/*_step_320000_raw_tta1"},
    "S": {"ckpt": "data/prereg55/S/ckpt/step_320000.pt", "emb": "data/prereg55/emb_S/*_step_320000_raw_tta1"},   # rev 2: c49 recipe, seed 1
}
C49_SHA = "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"
# PLAN rev 2 (prereg55 seed-1 member S added; order = more members first, then the rev-1 order)
CANDIDATES = (("ENS_c49_S_L_B", ("c49", "S", "L", "B")), ("ENS_c49_L_B", ("c49", "L", "B")),
              ("ENS_c49_S_L", ("c49", "S", "L")), ("ENS_c49_S_B", ("c49", "S", "B")),
              ("ENS_c49_L", ("c49", "L")), ("ENS_c49_B", ("c49", "B")), ("ENS_c49_S", ("c49", "S")))
P52_PSEUDO = os.path.join(REPO, "results", "prereg52", "explore", "pseudo.json")
A49_PSEUDO_MEAN = 0.6188134125030416
V27_EVAL = os.path.join(REPO, "results", "prereg49", "explore", "eval.json")
V27_APPLY = os.path.join(REPO, "results", "prereg49", "apply")
V27_E = 0.7035676497986656
P51 = os.path.join(REPO, "results", "prereg51", "explore")
PLAN = os.path.join(HERE, "PLAN.md")
SEAM_JSON, DEV_JSON, EVAL_JSON = (os.path.join(HERE, x) for x in ("seam.json", "dev.json", "eval.json"))
OUT_APPLY = os.path.join(REPO, "results", "prereg54", "apply")
GATE_MEAN, GATE_POOLS = 0.01, 8
DEADLINE = datetime.datetime(2026, 11, 22, 12, 0)
RAM_NEED_MB = 9000


# ================================================================ members and vectors
_SHA = {}


def ckpt_sha(m):
    if m not in _SHA:
        p = os.path.join(REPO, MEMBERS[m]["ckpt"])
        if not os.path.exists(p):
            raise SystemExit("missing %s" % MEMBERS[m]["ckpt"])
        _SHA[m] = L.sha256_file(p)
    return _SHA[m]


def member_dir(m):
    hits = sorted(glob.glob(os.path.join(REPO, MEMBERS[m]["emb"])))
    if len(hits) > 1:
        raise SystemExit("%s: several vector folders %s" % (m, [_rel(h) for h in hits]))
    return hits[0] if hits else None


def member_file(m, which):
    d = member_dir(m)
    if d is None:
        return None
    p = os.path.join(d, "store_E.npz" if which == "E" else "pool_%s.npz" % which)
    return p if os.path.exists(p) else None


def check_member(m, which):
    """PLAN.md 1: checkpoint hash, raw / tta1, the same games as c49's file. -> (path, record)."""
    path, ref = member_file(m, which), member_file("c49", which)
    if path is None:
        raise SystemExit("%s has no %s vectors" % (m, which))
    sha = ckpt_sha(m)
    if m == "c49" and sha != C49_SHA:
        raise SystemExit("c49 checkpoint hash")
    cfg = E43.embed_cfg_of(path) or {}
    problems = []
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != sha:
            problems.append("ckpt_sha256 %s != %s" % (str(z["ckpt_sha256"]), sha))
        if len(z["emb"]) != len(r["emb"]) or not np.array_equal(z["count"], r["count"]):
            problems.append("games / count differ from c49's file")
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in z.files) != (k in r.files) or (k in z.files and not np.array_equal(z[k], r[k])):
                problems.append("%s differs from c49's file" % k)
    if cfg.get("weights") != "raw" or cfg.get("tta") != 1:
        problems.append("weights %r tta %r" % (cfg.get("weights"), cfg.get("tta")))
    if problems:
        raise SystemExit("%s: %s" % (_rel(path), "; ".join(problems)))
    return path, {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt": MEMBERS[m]["ckpt"],
                  "ckpt_sha256": sha}


class Emb32(EV.Emb):
    """EV.Emb keeping float32 storage (the v2-6 cfg is not 'plain': pool_games turns rows into float64, same values)."""

    def __init__(self, emb, count, rows=None, colour=None, sha=None, name=None):
        self.emb = np.asarray(emb, np.float32)
        self.count = np.asarray(count)
        rows = np.arange(len(self.emb)) if rows is None else np.asarray(rows)
        self.pos = {int(r): i for i, r in enumerate(rows.tolist())}
        self.colour, self.sha, self.name = colour, sha, name


def load_store32(path):
    with np.load(path, allow_pickle=False) as z:
        return Emb32(z["emb"], z["count"], z["store_row"], E43.store().color, str(z["ckpt_sha256"]), str(z["ckpt"]))


def available(which_list):
    """Members whose vectors exist for every file kind in which_list."""
    return [m for m in MEMBERS if all(member_file(m, w) is not None for w in which_list)]


# ================================================================ base
def p51_eval_file(arm):
    return os.path.join(P51, "eval_%s.json" % arm)


def adopted_base():
    """The single model adopted after prereg51 (format of prereg51's p51_eval.py): among eval_L.json / eval_B.json
    that adopt their arm, the one judged last (largest e_pools_use_number); none -> c49 (v2-7)."""
    best = None
    for arm in ("L", "B"):
        f = p51_eval_file(arm)
        if os.path.exists(f):
            ev = read_json(f)
            if ev.get("adopt") and (best is None or ev["e_pools_use_number"] > best[1]):
                best = (arm, ev["e_pools_use_number"])
    return "c49" if best is None else best[0]


def base_of(a):
    """-> {name, members, eval file, key, per-pool E, mean, apply dir}. PLAN.md 2. --base must be the adopted base;
    for L / B the record is results/prereg51/explore/eval_<arm>.json (key <arm>), apply results/prereg51/apply/<arm>."""
    want = adopted_base()
    if a.base != want:
        raise SystemExit("--base %s but prereg51's records make %s the adopted base" % (a.base, want))
    if a.base == "c49":
        ev = read_json(V27_EVAL)
        if not ev.get("adopt") or float(ev["means"]["c49"]) != V27_E:
            raise SystemExit("prereg49 eval.json does not adopt c49 at %r" % V27_E)
        return {"name": "c49", "members": ("c49",), "eval": _rel(V27_EVAL), "key": "c49",
                "e_per_pool": list(ev["per_pool"]["c49"]), "e_mean": V27_E, "apply": V27_APPLY}
    f = os.path.abspath(a.base_eval or p51_eval_file(a.base))
    key = a.base_key or a.base
    apply_dir = os.path.abspath(a.base_apply or os.path.join(REPO, "results", "prereg51", "apply", a.base))
    if f != os.path.abspath(p51_eval_file(a.base)) or key != a.base or             apply_dir != os.path.abspath(os.path.join(REPO, "results", "prereg51", "apply", a.base)):
        raise SystemExit("--base-eval / --base-key / --base-apply must be prereg51's eval_%s.json / %s / apply/%s"
                         % (a.base, a.base, a.base))
    if not L.git_committed(f):
        raise SystemExit("%s is not committed" % _rel(f))
    ev = read_json(f)
    if not ev.get("adopt") or ev.get("arm") != a.base:
        raise SystemExit("%s does not adopt arm %s" % (_rel(f), a.base))
    return {"name": a.base, "members": (a.base,), "eval": _rel(f), "key": key,
            "e_per_pool": list(ev["per_pool"][key]), "e_mean": float(ev["means"][key]), "apply": apply_dir}


def candidates(base, have):
    """Fixed order (PLAN.md 2); only those whose members all have vectors and that are not the base itself."""
    return [(lab, mem) for lab, mem in CANDIDATES if set(mem) <= set(have) and tuple(mem) != tuple(base["members"])]


# ================================================================ pseudo-pools
def g_store(items):
    """= p52_pseudo.g_store, with each colour's LDA data released after its fit (lower peak; same values)."""
    import p35_apply as pa
    fit = E43.sim.Store(L.FEATS, load=())
    pst = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, lm = {}, {}
    for c in ("B", "W"):
        ld = E43.sim.LdaData(L.FEATS, c, fit)
        lm[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(pst, ld.cols)
        del ld
        gc.collect()
    del fit
    gc.collect()
    return pst, ldas, lm


def pseudo_scores(members_sets):
    """{label: per-pseudo-pool scores, by_k} for every member set (label -> tuple of members), 10 pseudo-pools."""
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    Lset = P39E.load_set("L")
    models = P39E.fit_models(Lset)
    meta_L = Lset["meta"]
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    need = sorted({m for mem in members_sets.values() for m in mem})
    per = {lab: [] for lab in members_sets}
    byk = {lab: [] for lab in members_sets}
    inputs = {}
    for pool, (pname, seed0) in PS.POOLS.items():
        items, _ids, cand_rows, cinfo = PS.read_candidates(pool)
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
        pst, ldas, lm = g_store(items)
        pools, zs = [], []
        for s in PS.SPLITS:
            ref, q_rows, owner, ks = PS.split(cand_rows, seed0 + s)
            p = {"cand_rows": ref, "q_rows": q_rows, "owner": owner, "k": ks}
            p["T"] = E43.sim.pool_terms(pst, p, ldas, lm, None, want=("O", "JO", "S2", "L", "PA"))
            zs.append(P39E.p1_terms(X, usable, colour, ref, q_rows, models, E43.P1_PW, Zs=Zs)[E43.P1_VARIANT][0])
            pools.append(p)
        del pst, ldas, lm, X, Zs
        gc.collect()
        Eterms, vinfo = {}, {}
        for m in need:                                   # one member at a time
            path, vinfo[m] = check_member(m, pool)
            with np.load(path, allow_pickle=False) as z:
                if len(z["emb"]) != n_all:
                    raise SystemExit("%s rows" % _rel(path))
                e = Emb32(z["emb"][:n], z["count"][:n], None, colour, str(z["ckpt_sha256"]), m)
            Eterms[m] = [BK.e_term(p, e, cfg) for p in pools]
            del e
        inputs[pool] = {"candidates": cinfo, "vectors": vinfo, "games": n, "rows_in_files": n_all}
        for lab, mem in members_sets.items():
            for i, p in enumerate(pools):
                E = P52.ens_E([Eterms[m][i] for m in mem])
                S = st.csls(BK.fuse_cfg(p, E, cfg) + wP * zs[i], k)
                per[lab].append(float(st.score(S, p["owner"])))
                byk[lab].append(EV.score_by_k(S, p))
        print(json.dumps({"pool": pool, "peak_mb": E45.peak_mb()}), flush=True)
        del pools, zs, Eterms, items, cand_rows, usable, colour
        gc.collect()
    return per, {lab: EV.mean_by_k(v) for lab, v in byk.items()}, inputs


def pseudo_seam(per_c49):
    want = read_json(P52_PSEUDO)["per_pool"]["A49"]
    if per_c49 != want or float(np.mean(per_c49)) != A49_PSEUDO_MEAN:
        raise SystemExit("pseudo seam: c49-only = %r, prereg52 A49 %r" % (float(np.mean(per_c49)), A49_PSEUDO_MEAN))


# ================================================================ E
def e_context():
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    base = E45.v26()
    ctx = E43.p1_ctx("E")
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    gc.collect()
    return pools, zs, sets, base


def e_terms(pools, cfg, members):
    out, info = {}, {}
    for m in members:                                    # one member at a time
        path, info[m] = check_member(m, "E")
        E43.check_players(path, "E")
        e = load_store32(path)
        out[m] = [BK.e_term(p, e, cfg) for p in pools]
        del e
        gc.collect()
    return out, info


def e_system(pools, zs, base, Et, members):
    per, byk = [], []
    for i, (p, z) in enumerate(zip(pools, zs)):
        S = st.csls(BK.fuse_cfg(p, P52.ens_E([Et[m][i] for m in members]), base["cfg"]) + base["wP"] * z, base["k"])
        per.append(float(st.score(S, p["owner"])))
        byk.append(EV.score_by_k(S, p))
    return per, byk


def e_seams(pools, zs, v26, Et, b):
    """c49 alone == prereg49 eval.json (v2-7); the base (if not c49) == its judgement file."""
    per, _ = e_system(pools, zs, v26, Et, ("c49",))
    ev = read_json(V27_EVAL)
    if per != list(ev["per_pool"]["c49"]) or float(np.mean(per)) != V27_E:
        raise SystemExit("E seam: c49-only = %r, prereg49 eval.json %r" % (float(np.mean(per)), V27_E))
    bper = per
    if b["name"] != "c49":
        bper, _ = e_system(pools, zs, v26, Et, b["members"])
        if bper != b["e_per_pool"] or float(np.mean(bper)) != b["e_mean"]:
            raise SystemExit("E seam: base %s rebuilt differs from %s" % (b["name"], b["eval"]))
    return per, bper


# ================================================================ commands
def committed_or_stop(paths):
    for f in paths:
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))


def cmd_seam(a):
    t0 = time.time()
    committed_or_stop([PLAN, os.path.abspath(__file__)])
    b = base_of(a)
    per, _byk, _inp = pseudo_scores({"c49": ("c49",)})
    pseudo_seam(per["c49"])
    pools, zs, _sets, v26 = e_context()
    Et, info = e_terms(pools, v26["cfg"], sorted(set(("c49",) + tuple(b["members"]))))
    per_e, bper = e_seams(pools, zs, v26, Et, b)
    rec = {"format": "c10-prereg54 seam v1", "git_head": L.git_head(), "base": {k: b[k] for k in ("name", "eval", "key")},
           "pseudo_c49": {"mean": float(np.mean(per["c49"])), "identical_to_prereg52_A49": True},
           "E_c49": {"mean": float(np.mean(per_e)), "identical_to_prereg49_eval": True},
           "E_base_identical": True, "vectors_E": info,
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps(rec, indent=1))
    return 0


def cmd_dev(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the pseudo-pool stage is run once" % a.out)
    committed_or_stop([PLAN, os.path.abspath(__file__)])
    b = base_of(a)
    have = available(["public", "private"])
    if not set(b["members"]) <= set(have):
        raise SystemExit("the base %s has no official-pool vectors" % b["name"])
    cands = candidates(b, have)
    if not cands:
        raise SystemExit("no candidate has all its members' official-pool vectors (have %s)" % have)
    sets = {"c49": ("c49",), "BASE": tuple(b["members"])}
    sets.update(dict(cands))
    per, byk, inputs = pseudo_scores(sets)
    pseudo_seam(per["c49"])
    rows = {}
    for lab, mem in cands:
        r = E43.diff_rec(per[lab], per["BASE"])
        r["gate"] = bool(r["mean"] >= GATE_MEAN and r["ci95"][0] > 0 and r["pools_higher"] >= GATE_POOLS)
        rows[lab] = {"members": list(mem), "mean": float(np.mean(per[lab])), "per_pool": per[lab], "by_k": byk[lab],
                     "minus_base": r}
        print(json.dumps({"candidate": lab, "mean": rows[lab]["mean"], "minus_base": r}), flush=True)
    passed = [lab for lab, _m in cands if rows[lab]["minus_base"]["gate"]]
    chosen = None
    if passed:     # PLAN.md 4: more members first, then higher pseudo mean, then table order
        order = {lab: i for i, (lab, _m) in enumerate(CANDIDATES)}
        chosen = sorted(passed, key=lambda x: (-len(rows[x]["members"]), -rows[x]["mean"], order[x]))[0]
    rec = {"format": "c10-prereg54 dev v1", "git_head": L.git_head(), "base": {k: b[k] for k in ("name", "members",
                                                                                                "eval", "key")},
           "members_with_pool_vectors": have, "dropped": [lab for lab, _m in CANDIDATES if lab not in dict(cands)],
           "inputs": inputs, "c49_pseudo_identical_to_prereg52_A49": True,
           "base_pseudo": {"mean": float(np.mean(per["BASE"])), "per_pool": per["BASE"], "by_k": byk["BASE"]},
           "candidates": rows, "gate_rule": "mean - base >= %.2f, pseudo-pool bootstrap (10,000, seed 43) 95%% low > 0, "
                                            "higher in >= %d of 10" % (GATE_MEAN, GATE_POOLS),
           "passed": passed, "chosen": chosen, "chosen_members": rows[chosen]["members"] if chosen else None,
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("passed", "chosen", "chosen_members", "dropped")}, indent=1))
    return 0


def e_judgements_so_far():
    prior = P49.e_judgements_so_far()
    n51 = len(glob.glob(os.path.join(P51, "eval_*.json")))
    return prior, n51


def cmd_judge(a):
    t0 = time.time()
    if datetime.datetime.now() > DEADLINE:
        raise SystemExit("past the deadline %s (PLAN.md 5): keep the base" % DEADLINE)
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    committed_or_stop([PLAN, os.path.abspath(__file__), a.dev])
    dev = read_json(a.dev)
    if not dev.get("chosen"):
        raise SystemExit("no candidate passed the pseudo-pool gate: the E pools are not used (PLAN.md 4)")
    b = base_of(a)
    if dev["base"]["name"] != b["name"]:
        raise SystemExit("dev.json was made against base %s, not %s" % (dev["base"]["name"], b["name"]))
    prior, n51 = e_judgements_so_far()
    if a.e_use != len(prior) + n51 + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d + %d (prereg51) times before (%s)"
                         % (a.e_use, len(prior), n51, prior))
    mem = tuple(dev["chosen_members"])
    pools, zs, sets, v26 = e_context()
    Et, info = e_terms(pools, v26["cfg"], sorted(set(mem) | set(b["members"]) | {"c49"}))
    _c49, bper = e_seams(pools, zs, v26, Et, b)
    per, byk = e_system(pools, zs, v26, Et, mem)
    j = E43.judge_rule(per, bper)
    rec = {"format": "c10-prereg54 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "prereg51_e_judgements": n51, "dev_sha256": st.sha256_lf(a.dev),
           "chosen": dev["chosen"], "members": list(mem), "base": {k: b[k] for k in ("name", "members", "eval", "key")},
           "seam_c49_E_identical_to_prereg49": True, "seam_base_E_identical": True, "vectors_E": info, "p1_sets": sets,
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "git_head": L.git_head(),
           "means": {"chosen": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"chosen": per, "base": list(bper)}, "by_k": {"chosen": EV.mean_by_k(byk)},
           "chosen_minus_base": {x: j[x] for x in ("mean", "ci95", "pools_higher")},
           "rule": j["rule"].replace("c*", dev["chosen"]), "adopt": j["adopt"],
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "chosen_minus_base", "by_k", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    t0 = time.time()
    ev = read_json(a.eval)
    if not ev["adopt"]:
        raise SystemExit("eval.json does not adopt: keep the base")
    b = base_of(a)
    if ev["base"]["name"] != b["name"] or not b["apply"]:
        raise SystemExit("base mismatch or no --base-apply")
    v26 = E45.v26()
    cfg, wP, k = v26["cfg"], v26["wP"], v26["k"]
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
    if P39E.models_sha(models) != v26["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    colour = np.array([c for _s, c in items])
    z, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                           E43.P1_PW)[E43.P1_VARIANT]
    del models, Pset
    mem = tuple(ev["members"])
    Et, vinfo = {}, {}
    for m in sorted(set(mem) | set(b["members"])):
        path, vinfo[m] = check_member(m, a.pool)
        e = E43.pool_emb(path, items, vinfo[m]["ckpt_sha256"])
        Et[m] = BK.e_term(rec, e, cfg)
        del e

    def top5_of(members):
        F = BK.fuse_cfg(rec, P52.ens_E([Et[m] for m in members]), cfg)
        return P39E.top5_bytes(qids, st.top5(st.csls(F + wP * z, k), cand_ids))

    data_b = top5_of(b["members"])
    with open(os.path.join(b["apply"], a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: base rebuilt on %s differs from %s/%s/top5.csv (CRLF? re-checkout with "
                             "git -c core.autocrlf=false)" % (a.pool, _rel(b["apply"]), a.pool))
    data = top5_of(mem)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    rep = {"format": "c10-prereg54 apply v1", "pool": a.pool, "members": list(mem), "vectors": vinfo,
           "base": b["name"], "base_top5_equals_base_apply": True, "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": st.sha256_lf(a.eval), "top5_sha256": L.sha256_file(out),
           "differs_from_base_rows": int(sum(x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return E43.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def base_args(p):
        p.add_argument("--base", default="c49", choices=["c49", "L", "B"])
        p.add_argument("--base-eval", default=None, help="default results/prereg51/explore/eval_<base>.json (the only accepted value)")
        p.add_argument("--base-key", default=None, help="default <base> (the only accepted value)")
        p.add_argument("--base-apply", default=None, help="default results/prereg51/apply/<base> (the only accepted value)")

    s = sub.add_parser("seam")
    base_args(s)
    s.add_argument("--out", default=SEAM_JSON)
    d = sub.add_parser("dev")
    base_args(d)
    d.add_argument("--out", default=DEV_JSON)
    j = sub.add_parser("judge")
    base_args(j)
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--dev", default=DEV_JSON)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    base_args(p)
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--eval", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "dev": cmd_dev, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
