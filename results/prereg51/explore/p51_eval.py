"""prereg51 (PLAN.md 4-6): the one E judgement per arm (L, B) with v2-6's backend and P1 term unchanged; official
pools. CPU only.

    PY results/prereg51/explore/p51_eval.py seam                            # c49 through the judge path == v2-7 E
    PY results/prereg51/explore/p51_eval.py judge --arm L|B --e-use N       # once per arm -> eval_<arm>.json
    PY results/prereg51/explore/p51_eval.py apply --arm L|B --src <official tests dir> --pool public|private
    PY results/prereg51/explore/p51_eval.py assemble --arm L|B --src <official tests dir>

System(arm) = CSLS_k( fuse_cfg(pool, e_term(pool, emb_arm, cfg), cfg) + wP z(T_P1 joint/b4) ), cfg / wP / k = v2-6's
(results/prereg43/explore/tune.json c_star) - nothing is tuned.
Base (PLAN.md 4: the arm judged first is compared with v2-7; an adopted arm becomes the base of the later one):
  the other arm if results/prereg51/explore/eval_<other>.json exists and adopts it, else v2-7 = c49
  (results/prereg49/explore/eval.json per_pool.c49, mean 0.7035676497986656; apply results/prereg49/apply).
E-use number N = 7 + number of existing results/prereg51/explore/eval_*.json (prereg36 ... prereg49 used 6).
Arm vectors: data/prereg51/emb_<arm>/*_step_<S>_raw_tta1/<store_E | pool_public | pool_private>.npz; checkpoint sha256
= data/prereg51/<arm>/ckpt/step_<S>.pt; raw, tta1, the same games / counts / store rows as c49's file; the E file
written after the commits of PLAN.md and this file.
Output format (read by prereg54's p54_eval.py): eval_<arm>.json with adopt, means{<arm>, base}, per_pool{<arm>, base},
by_k{<arm>}, e_pools_use_number, base{name, members, eval, key, e_mean, apply}.
Nothing here opens the validation-answer folder or runs tools/score.py.
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
P49D = os.path.join(REPO, "results", "prereg49", "explore")
for p in (P49D, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p49_eval as P49  # noqa: E402   (E43 / E45 handles, E-use list, V1 submission; unchanged)

E43, E45, P39E = P49.E43, P49.E45, P49.P39E
BK, EV, st, L = P49.BK, P49.EV, P49.st, P49.L
_rel, read_json = E43._rel, E43.read_json

ARMS = {"L": {"step": 480000}, "B": {"step": 320000}}
C49_DIR = os.path.join(REPO, "data", "prereg49", "emb", "V2_step_320000_raw_tta1")
C49_SHA = "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"
V27_EVAL = os.path.join(REPO, "results", "prereg49", "explore", "eval.json")
V27_APPLY = os.path.join(REPO, "results", "prereg49", "apply")
V27_E = 0.7035676497986656
PLAN = os.path.join(HERE, "PLAN.md")
SEAM_JSON = os.path.join(HERE, "seam.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg51", "apply")
FIRST_E_USE = 7
LAST_JUDGE = datetime.datetime(2026, 11, 24, 23, 59, 59)     # PLAN.md 6
RAM_NEED_MB = 9000


# ================================================================ paths
def ckpt_path(arm, repo=REPO):
    return os.path.join(repo, "data", "prereg51", arm, "ckpt", "step_%d.pt" % ARMS[arm]["step"])


def arm_dir(arm, repo=REPO):
    hits = sorted(glob.glob(os.path.join(repo, "data", "prereg51", "emb_%s" % arm,
                                         "*_step_%d_raw_tta1" % ARMS[arm]["step"])))
    if len(hits) > 1:
        raise SystemExit("arm %s: several vector folders %s" % (arm, hits))
    return hits[0] if hits else None


def vec_name(which):
    return "store_E.npz" if which == "E" else "pool_%s.npz" % which


def arm_file(arm, which, repo=REPO):
    d = arm_dir(arm, repo)
    return None if d is None else os.path.join(d, vec_name(which))


def c49_file(which):
    return os.path.join(C49_DIR, vec_name(which))


def eval_path(arm, here=HERE):
    return os.path.join(here, "eval_%s.json" % arm)


def other(arm):
    return "B" if arm == "L" else "L"


# ================================================================ checks
def check_arm_file(path, ref, ckpt_sha):
    """Like p49_eval.check_c49_file: checkpoint hash, raw / tta1, the same games / counts / rows as c49's file."""
    if path is None or not os.path.exists(path):
        raise SystemExit("missing %s" % path)
    problems = []
    cfg = E43.embed_cfg_of(path) or {}
    rcfg = E43.embed_cfg_of(ref) or {}
    if cfg.get("tta") != 1 or cfg.get("weights") != "raw":
        problems.append("tta %r weights %r" % (cfg.get("tta"), cfg.get("weights")))
    if (cfg.get("games"), cfg.get("moves")) != (rcfg.get("games"), rcfg.get("moves")):
        problems.append("games / moves differ from c49's file")
    with np.load(path, allow_pickle=False) as z, np.load(ref, allow_pickle=False) as r:
        if str(z["ckpt_sha256"]) != ckpt_sha or cfg.get("ckpt_sha256", ckpt_sha) != ckpt_sha:
            problems.append("checkpoint sha %s != %s" % (str(z["ckpt_sha256"]), ckpt_sha))
        if len(z["emb"]) != len(r["emb"]) or not np.array_equal(z["count"], r["count"]):
            problems.append("games / count differ from c49's file")
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in z.files) != (k in r.files) or (k in z.files and not np.array_equal(z[k], r[k])):
                problems.append("%s differs from c49's file" % k)
    if problems:
        raise SystemExit("%s: %s" % (path, "; ".join(problems)))


_SHA = {}


def ckpt_sha(arm):
    if arm not in _SHA:
        p = ckpt_path(arm)
        if not os.path.exists(p):
            raise SystemExit("missing %s" % _rel(p))
        _SHA[arm] = L.sha256_file(p)
    return _SHA[arm]


def member_file(m, which):
    """(path, record) of member m (c49 / L / B) for E / public / private, checked."""
    if m == "c49":
        p = c49_file(which)
        with np.load(p, allow_pickle=False) as z:
            if str(z["ckpt_sha256"]) != C49_SHA:
                raise SystemExit("%s is not c49" % _rel(p))
        return p, {"file": _rel(p), "file_sha256": L.sha256_file(p), "ckpt_sha256": C49_SHA}
    p = arm_file(m, which)
    sha = ckpt_sha(m)
    check_arm_file(p, c49_file(which), sha)
    return p, {"file": _rel(p), "file_sha256": L.sha256_file(p), "ckpt": _rel(ckpt_path(m)), "ckpt_sha256": sha,
               "embed_cfg": E43.embed_cfg_of(p)}


def size_record(arm):
    """The trunk size actually trained (train_record_<arm>.json if present; informational)."""
    for p in (os.path.join(HERE, "train_record_%s.json" % arm),
              os.path.join(REPO, "data", "prereg51", arm, "train_record_%s.json" % arm)):
        if os.path.exists(p):
            tr = read_json(p)
            return {"file": _rel(p), "size": tr.get("size") or (tr.get("config") or {}).get("size"),
                    "steps": tr.get("steps"), "finished": tr.get("finished")}
    return None


# ================================================================ base and E-use
def resolve_base(arm, here=HERE):
    """PLAN.md 4: the other arm if its committed eval_<other>.json adopts it, else v2-7 (c49)."""
    o = other(arm)
    f = eval_path(o, here)
    if os.path.exists(f):
        ev = read_json(f)
        if ev.get("adopt"):
            return {"name": o, "members": [o], "eval": _rel(f), "key": o, "e_mean": float(ev["means"][o]),
                    "e_per_pool": list(ev["per_pool"][o]), "apply": _rel(os.path.join(OUT_APPLY, o)), "file": f}
    return v27_base()


def v27_base():
    ev = read_json(V27_EVAL)
    if not ev.get("adopt") or float(ev["means"]["c49"]) != V27_E:
        raise SystemExit("prereg49 eval.json does not adopt c49 at %r" % V27_E)
    return {"name": "c49", "members": ["c49"], "eval": _rel(V27_EVAL), "key": "c49", "e_mean": V27_E,
            "e_per_pool": list(ev["per_pool"]["c49"]), "apply": _rel(V27_APPLY), "file": V27_EVAL}


def existing_evals(here=HERE):
    return sorted(glob.glob(os.path.join(here, "eval_*.json")))


def expected_e_use(here=HERE):
    return FIRST_E_USE + len(existing_evals(here))


# ================================================================ E
class Emb32(EV.Emb):
    """EV.Emb with float32 storage (v2-6's cfg is not 'plain': pool_games turns rows into float64, same values)."""

    def __init__(self, emb, count, rows=None, colour=None, sha=None, name=None):
        self.emb = np.asarray(emb, np.float32)
        self.count = np.asarray(count)
        rows = np.arange(len(self.emb)) if rows is None else np.asarray(rows)
        self.pos = {int(r): i for i, r in enumerate(rows.tolist())}
        self.colour, self.sha, self.name = colour, sha, name


def e_context():
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    v26 = E45.v26()
    ctx = E43.p1_ctx("E")
    if ctx["lda_sha256"] != v26["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    gc.collect()
    return pools, zs, sets, v26


def e_scores(path, pools, zs, v26):
    E43.check_players(path, "E")
    with np.load(path, allow_pickle=False) as z:
        e = Emb32(z["emb"], z["count"], z["store_row"], E43.store().color, str(z["ckpt_sha256"]), str(z["ckpt"]))
    Fs = [BK.fuse_cfg(p, BK.e_term(p, e, v26["cfg"]), v26["cfg"]) for p in pools]
    del e
    gc.collect()
    per, byk = [], []
    for F, z, p in zip(Fs, zs, pools):
        S = st.csls(F + v26["wP"] * z, v26["k"])
        per.append(float(st.score(S, p["owner"])))
        byk.append(EV.score_by_k(S, p))
    return per, byk


def base_seam(base, pools, zs, v26):
    """The base rebuilt by this path == its judgement record, every pool bit for bit."""
    path, info = member_file(base["members"][0], "E")
    per, _byk = e_scores(path, pools, zs, v26)
    if per != base["e_per_pool"] or float(np.mean(per)) != base["e_mean"]:
        raise SystemExit("seam: base %s rebuilt = %r, recorded %r" % (base["name"], float(np.mean(per)), base["e_mean"]))
    return per, info


# ================================================================ commands
def committed(paths):
    for f in paths:
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))


def cmd_seam(a):
    t0 = time.time()
    pools, zs, _sets, v26 = e_context()
    per, info = base_seam(v27_base(), pools, zs, v26)
    rec = {"format": "c10-prereg51 seam v1", "rule": "the judge path on c49's E vectors == v2-7 per pool, bit for bit",
           "mean": float(np.mean(per)), "want_mean": V27_E, "identical": True, "vectors": info,
           "git_head": L.git_head(), "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    base = resolve_base(a.base_of) if a.base_of else None
    if base is not None and base["name"] != "c49":
        base_seam(base, pools, zs, v26)
        rec["base_" + base["name"] + "_identical"] = True
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("mean", "identical", "seconds", "peak_working_set_mb")}))
    return 0


def cmd_judge(a):
    t0 = time.time()
    if datetime.datetime.now() > LAST_JUDGE:
        raise SystemExit("past %s (PLAN.md 6)" % LAST_JUDGE)
    out = eval_path(a.arm)
    if os.path.exists(out):
        raise SystemExit("%s exists: one judgement per arm" % _rel(out))
    committed([PLAN, os.path.abspath(__file__)])
    prior = P49.e_judgements_so_far()
    if len(prior) + 1 != FIRST_E_USE:
        raise SystemExit("E was judged %d times before prereg51 (%s), expected %d" % (len(prior), prior,
                                                                                    FIRST_E_USE - 1))
    want = expected_e_use()
    if a.e_use != want:
        raise SystemExit("--e-use %d but it must be %d (7 + %d existing eval_*.json)" % (a.e_use, want,
                                                                                    len(existing_evals())))
    base = resolve_base(a.arm)
    if base["name"] != "c49":
        committed([base["file"]])
    path, vec = member_file(a.arm, "E")
    E43.made_after([path], max(E43.commit_time(PLAN), E43.commit_time(os.path.abspath(__file__))))
    pools, zs, sets, v26 = e_context()
    bper, binfo = base_seam(base, pools, zs, v26)
    per, byk = e_scores(path, pools, zs, v26)
    j = E43.judge_rule(per, bper)
    rec = {"format": "c10-prereg51 eval v1", "arm": a.arm, "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior + [os.path.basename(p) for p in existing_evals()],
           "vectors": vec, "train_size": size_record(a.arm), "base_vectors_E": binfo,
           "base": {k: base[k] for k in ("name", "members", "eval", "key", "e_mean", "apply")},
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "seam_base_E_identical": True,
           "git_head": L.git_head(), "p1_sets": sets,
           "means": {a.arm: float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {a.arm: per, "base": list(bper)}, "by_k": {a.arm: EV.mean_by_k(byk)}}
    rec["%s_minus_base" % a.arm] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "arm " + a.arm)
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "%s_minus_base" % a.arm, "by_k", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    t0 = time.time()
    f = eval_path(a.arm)
    ev = read_json(f)
    if not ev.get("adopt"):
        raise SystemExit("%s does not adopt arm %s: keep the base" % (_rel(f), a.arm))
    base = ev["base"]
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

    def top5_of(m):
        path, info = member_file(m, a.pool)
        e = E43.pool_emb(path, items, info["ckpt_sha256"])
        F = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
        return P39E.top5_bytes(qids, st.top5(st.csls(F + wP * z, k), cand_ids)), info, e

    data_b, _binfo, _eb = top5_of(base["members"][0])
    del _eb
    bfile = os.path.join(REPO, base["apply"], a.pool, "top5.csv")
    with open(bfile, "rb") as fh:
        if fh.read() != data_b:
            raise SystemExit("seam: base %s rebuilt on %s differs from %s (CRLF? re-checkout with "
                             "git -c core.autocrlf=false)" % (base["name"], a.pool, _rel(bfile)))
    data, info, e = top5_of(a.arm)
    od = os.path.join(a.out, a.arm, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as fh:
        fh.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg51 apply v1", "arm": a.arm, "pool": a.pool, "vectors": info, "cfg": cfg, "wP": wP,
           "k": k, "base": base["name"], "base_top5_equals": _rel(bfile), "questions": len(qids),
           "candidates": len(cand_ids), "games": len(items), "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": st.sha256_lf(f), "top5_sha256": L.sha256_file(out),
           "differs_from_base_rows": int(sum(x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    a.out = os.path.join(a.out, a.arm)
    return E43.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--base-of", default=None, choices=["L", "B"], help="also rebuild the base the given arm would face")
    m.add_argument("--out", default=SEAM_JSON)
    j = sub.add_parser("judge")
    j.add_argument("--arm", required=True, choices=["L", "B"])
    j.add_argument("--e-use", type=int, required=True)
    p = sub.add_parser("apply")
    p.add_argument("--arm", required=True, choices=["L", "B"])
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--arm", required=True, choices=["L", "B"])
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "judge": cmd_judge, "apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
