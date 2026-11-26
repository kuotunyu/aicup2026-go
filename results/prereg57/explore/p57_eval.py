"""prereg57 (PLAN.md Development / Judgement / Apply): seam, the D choice, the one E judgement, official pools. CPU only.

    PY results/prereg57/explore/p57_eval.py seam                         # unadapted bases through this path -> seam.json
    PY results/prereg57/explore/p57_eval.py dev                          # 3 runs x 2 D pools x 4 steps -> dev.json (commit)
    PY results/prereg57/explore/p57_eval.py judge --e-use N              # E pools 0..9, once -> eval.json
    PY results/prereg57/explore/p57_eval.py apply --pool public|private [--seam-only]
    PY results/prereg57/explore/p57_eval.py assemble

Every pool has its own vectors file (p57_ft.py embed: that pool's games only). Per-pool path:
    file -> p57_ft.read_vectors -> p36_eval.Emb (addressed by feature-store row) ->
    S = CSLS_k( fuse_cfg(pool, e_term(pool, emb, cfg), cfg) + wP z(T_P1 joint/b4) )      (p43_eval.fused / scores)
with cfg / wP / k = v2-6's (results/prereg43/explore/tune.json c_star; v2-7 uses them unchanged) - nothing is tuned.
The quick screen is prereg42's quick_grid (tune_D_B4 structure, wE {2,4,8,16} x CSLS {10,20,40}, no P1 term).

seam: (a) c49's existing E vectors restricted to each of E pools 0..9, written and read back in embed's layout ==
  results/prereg49/explore/eval.json per_pool.c49[0..9] bit for bit; (b) S0's existing D vectors restricted to D pools
  0, 1 == prereg43 tune.json: seeds.S0.raw.best.per_pool (quick screen at its recorded cell), c_star.backend.per_pool
  (v2-6 backend without P1), c_star.p1.per_pool (v2-6 with P1). The other 11 quick-screen cells have no per-pool
  record (tune.json holds their 10-pool means only) and are not compared.
dev (PLAN.md Development): reference = the unadapted S0 on D pools 0, 1 (checked against the records first); the 12
  settings' v2-6-style score (with P1) and quick screen, both pools averaged; choice = highest v2-6-style mean, ties ->
  fewer steps, then smaller share (then smaller lr). Gate (the smallest reading of "beats the reference by >= +0.01
  on both pools"): the CHOSEN setting is >= +0.01 above the reference on each of the two pools; whether any other
  setting does is recorded too. Refuses to run twice, or when a judge / apply vector of this method exists.
judge (PLAN.md Judgement): seam first (as (a)), then E pools 0..9 with their own adapted c49; adopt if mean(adapted -
  v2-7) >= +0.01, bootstrap low (10,000 resamples of the 10 pools, seed 43) > 0, higher in >= 9 of 10 (p43_eval's
  judge_rule). --e-use N must equal 1 + the E judgements made so far (prereg36..49: 6, + prereg51's eval_*.json +
  prereg54's eval*.json); refuses to run twice or without a committed, clean dev.json whose gate passed.
apply: v2-7 rebuilt on the pool == results/prereg49/apply/<pool>/top5.csv byte for byte, then the adapted vectors ->
  results/prereg57/apply/<pool>/top5.csv. assemble: p35_apply's assemble + the PLAN's checks (1,600 rows, ids, the
  original 800 rows byte-identical to v2-7's submission).
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import datetime
import gc
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P51D = os.path.join(REPO, "results", "prereg51", "explore")
P54D = os.path.join(REPO, "results", "prereg54", "explore")
for p in (P51D, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p51_eval as P51  # noqa: E402   (v2-7 records, E context, c49 files; unchanged)
import p57_ft as FT  # noqa: E402

P49, E43, E45, P39E = P51.P49, P51.E43, P51.E45, P51.P39E
BK, EV, st, L = P51.BK, P51.EV, P51.st, P51.L
Q42 = E43.Q42
_rel, read_json = E43._rel, E43.read_json
if FT.L is not L:
    raise SystemExit("p57_ft and p51_eval import different p36_lib modules")

PLAN = os.path.join(HERE, "PLAN.md")
SEAM_JSON, DEV_JSON, EVAL_JSON = (os.path.join(HERE, x) for x in ("seam.json", "dev.json", "eval.json"))
CODE = (PLAN, os.path.join(HERE, "p57_ft.py"), os.path.abspath(__file__))
OUT_APPLY = os.path.join(REPO, "results", "prereg57", "apply")
V27_APPLY, V27_E = P51.V27_APPLY, P51.V27_E
V27_SUBMISSION = os.path.join(V27_APPLY, "submission_player.csv")
GAIN = 0.01                                               # PLAN.md: dev gate and adoption mean
LAST = datetime.datetime(2026, 11, 23, 12, 0, 0)          # PLAN.md: hard stop
RAM_NEED_MB = 7000
RECIPE = {"warmup": FT.WARMUP, "batch_segments": FT.BATCH_SEGMENTS, "moves": FT.MOVES, "seed": FT.SEED,
          "weight_decay": FT.V.WD, "ema_decay": FT.V.EMA_DECAY, "smoke": False, "start_weights": "raw"}


# ================================================================ vectors
def base_store(base, kind):
    """The base's existing store_<D|E> vectors (raw, tta 1), addressable by feature-store row."""
    f = FT.base_vec_file(base, kind, None)
    E43.check_players(f, kind)
    with np.load(f, allow_pickle=False) as z:
        if str(z["ckpt_sha256"]) != FT.BASES[base]["sha256"]:
            raise SystemExit("%s is not %s" % (_rel(f), base))
        emb, count, rows = z["emb"], z["count"], z["store_row"]
    return {"file": f, "emb": emb, "count": count, "pos": {int(r): i for i, r in enumerate(rows.tolist())},
            "info": {"file": _rel(f), "file_sha256": L.sha256_file(f), "ckpt_sha256": FT.BASES[base]["sha256"]}}


def rows_of(pool):
    return FT.pool_rows(pool["cand_rows"], pool["q_rows"])


def restrict(bs, pool):
    rows = rows_of(pool)
    idx = np.asarray([bs["pos"][int(r)] for r in rows], np.int64)
    return bs["emb"][idx], bs["count"][idx], rows


def write_unadapted(path, base, kind, index, bs, pool):
    """The base's vectors of the pool's games in p57_ft embed's file layout (candidates' games, then questions')."""
    emb, count, rows = restrict(bs, pool)
    FT.write_vectors(path, emb, count, FT.BASES[base]["sha256"], "%s_unadapted" % base, 320000,
                     {"weights": "raw", "tta": 1, "base": base, "unadapted": True,
                      "pool": {"kind": kind, "name": int(index)}}, "unadapted", rows)


def load_emb(path, pool):
    """p57 vectors of a D / E pool -> (Emb by store row, file fields); the rows must be the pool's, in order."""
    v = FT.read_vectors(path)
    if v["store_row"] is None or not np.array_equal(v["store_row"], rows_of(pool)):
        raise SystemExit("%s does not hold the games of pool %s in the pool's order" % (_rel(path), pool["seed"]))
    return EV.Emb(v["emb"], v["count"], v["store_row"], E43.store().color, v["ckpt_sha256"], v["ckpt"]), v


def check_vectors(v, path, ckpt, base, kind, name, share, lr, n, ref_count, after):
    """A vectors file of this method: EMA / tta 1 / whole pool, the recipe, the setting, the checkpoint, the games."""
    cfg = v["embed_cfg"]
    c = cfg.get("p57") or {}
    bad = []
    if cfg.get("weights") != "ema" or cfg.get("tta") != 1 or cfg.get("limit"):
        bad.append("weights %r tta %r limit %r" % (cfg.get("weights"), cfg.get("tta"), cfg.get("limit")))
    for k, want in dict(RECIPE, base=base, pool_kind=kind, pool=name, share=float(share), lr=float(lr),
                        base_ckpt_sha256=FT.BASES[base]["sha256"], pool_identity=v["pool_identity"],
                        pool_segments=FT.split_segments(float(share))[0]).items():
        if c.get(k) != want:
            bad.append("%s %r != %r" % (k, c.get(k), want))
    if v["ckpt_step"] != int(n):
        bad.append("step %r != %r" % (v["ckpt_step"], n))
    if not os.path.exists(ckpt):
        bad.append("missing checkpoint %s" % _rel(ckpt))
    elif L.sha256_file(ckpt) != v["ckpt_sha256"]:
        bad.append("not made from %s" % _rel(ckpt))
    if len(v["count"]) != len(ref_count) or not np.array_equal(v["count"], ref_count):
        bad.append("games / counts differ from the base's vectors of the pool")
    if bad:
        raise SystemExit("%s: %s" % (_rel(path), "; ".join(bad)))
    E43.made_after([path], after)
    return {"file": _rel(path), "file_sha256": L.sha256_file(path), "ckpt": _rel(ckpt),
            "ckpt_sha256": v["ckpt_sha256"], "pool_identity": v["pool_identity"]}


# ================================================================ scores of one pool
def system_one(pool, e, z, v26):
    """(v2-6-style score with P1, by k, backend without P1) of one pool - p43_eval.fused / scores."""
    Fs = E43.fused([pool], e, v26["cfg"])
    per = E43.scores(Fs, [z], v26["wP"], v26["k"], [pool])[0]
    byk = EV.score_by_k(st.csls(Fs[0] + v26["wP"] * z, v26["k"]), pool)
    backend = float(st.score(st.csls(Fs[0], int(v26["cfg"]["norm"][1])), pool["owner"]))
    return per, byk, backend


def quick_one(pool, e):
    """prereg42's quick-screen grid of one pool: [{wE, k, score}] in grid order."""
    return [{"wE": r["wE"], "k": r["k"], "score": r["per_pool"][0]} for r in Q42.quick_grid([pool], e)]


def quick_combine(grids):
    """Grids of several pools -> cells [{wE, k, per_pool, mean}] and the quick screen Q = the first maximum."""
    cells = []
    for j, c in enumerate(grids[0]):
        per = [g[j]["score"] for g in grids]
        if any((g[j]["wE"], g[j]["k"]) != (c["wE"], c["k"]) for g in grids):
            raise ValueError("grids differ")
        cells.append({"wE": c["wE"], "k": c["k"], "per_pool": per, "mean": float(np.mean(per))})
    return cells, Q42.best_row(cells)


def cell_of(cells, wE, k):
    return [c for c in cells if (c["wE"], c["k"]) == (wE, k)][0]


# ================================================================ contexts
def d_context():
    seeds = [EV.D_SEEDS[i] for i in FT.DEV_POOLS]
    pools = EV.load_pools(EV.TERMS_DIR["D"], seeds)
    Ds = set(E43.splits()[1])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Ds:
            raise SystemExit("pool %d has players outside D" % p["seed"])
    v26 = E45.v26()
    ctx = E43.p1_ctx("D")
    if ctx["lda_sha256"] != v26["lda_sha256"]:
        raise SystemExit("the P1 LDA refit differs from prereg43 tune.json")
    zs = [E43.p1_z(p, ctx) for p in pools]
    sets = ctx["sets"]
    del ctx
    P39E._MOCK.clear()
    gc.collect()
    return pools, zs, sets, v26


def e_context():
    pools, zs, sets, v26 = P51.e_context()            # the 20 E pools of the v2-7 judgement, players checked
    idx = list(FT.JUDGE_POOLS)
    return [pools[i] for i in idx], [zs[i] for i in idx], sets, v26


def recorded_D(v26):
    t = v26["tune"]
    if t["c_star"]["members"] != [["S0", "raw"]]:
        raise SystemExit("prereg43's c* is not S0 raw")
    best = t["seeds"]["S0"]["raw"]["best"]
    i = list(FT.DEV_POOLS)
    return {"cell": (best["wE"], best["k"]), "quick": [best["per_pool"][j] for j in i],
            "backend": [v26["d_backend_per_pool"][j] for j in i], "v26": [v26["d_per_pool"][j] for j in i]}


def reference_D(tmp, pools, zs, v26):
    """The unadapted S0 on the dev pools through the per-pool path, checked against prereg43's records."""
    path, _choice = E43.cstar_file(v26["tune"], "D")
    if os.path.abspath(path) != os.path.abspath(FT.base_vec_file("S0", "D", None)):
        raise SystemExit("prereg43's c* D file is not S0's")
    bs = base_store("S0", "D")
    want = recorded_D(v26)
    grids, per, byk, backend, counts = [], [], [], [], []
    for i, p, z in zip(FT.DEV_POOLS, pools, zs):
        f = os.path.join(tmp, "D%d.npz" % i)
        write_unadapted(f, "S0", "D", i, bs, p)
        e, v = load_emb(f, p)
        grids.append(quick_one(p, e))
        a, b, c = system_one(p, e, z, v26)
        per.append(a)
        byk.append(b)
        backend.append(c)
        counts.append(np.asarray(v["count"]))
    cells, best = quick_combine(grids)
    got_cell = cell_of(cells, *want["cell"])["per_pool"]
    same = {"quick_at_recorded_cell": got_cell == want["quick"], "backend_only": backend == want["backend"],
            "v26_with_P1": per == want["v26"]}
    ref = {"base": "S0", "d_seeds": [p["seed"] for p in pools], "vectors": bs["info"],
           "v26": {"per_pool": per, "mean": float(np.mean(per)), "by_k": EV.mean_by_k(byk), "recorded": want["v26"]},
           "backend_only": {"per_pool": backend, "mean": float(np.mean(backend)), "recorded": want["backend"]},
           "quick": {"Q": best["mean"], "wE": best["wE"], "k": best["k"], "per_pool": best["per_pool"],
                     "at_recorded_cell": {"wE": want["cell"][0], "k": want["cell"][1], "per_pool": got_cell,
                                          "mean": float(np.mean(got_cell)), "recorded": want["quick"]},
                     "grid": [{k: c[k] for k in ("wE", "k", "mean")} for c in cells]},
           "identical": same, "all_identical": all(same.values())}
    return ref, counts


def seam_E(tmp, pools, zs, v26):
    """The unadapted c49 on E pools 0..9 through the per-pool path == v2-7's judgement record, bit for bit."""
    want = [P51.v27_base()["e_per_pool"][i] for i in FT.JUDGE_POOLS]
    bs = base_store("c49", "E")
    per, counts = [], []
    for i, p, z in zip(FT.JUDGE_POOLS, pools, zs):
        f = os.path.join(tmp, "E%d.npz" % i)
        write_unadapted(f, "c49", "E", i, bs, p)
        e, v = load_emb(f, p)
        per.append(system_one(p, e, z, v26)[0])
        counts.append(np.asarray(v["count"]))
        os.remove(f)
    info = bs["info"]
    del bs
    gc.collect()
    return {"base": "c49", "e_seeds": [p["seed"] for p in pools], "per_pool": per, "recorded": want,
            "identical": per == want, "mean_of_these_pools": float(np.mean(per)), "vectors": info}, counts


def committed(paths):
    for f in paths:
        if not L.git_committed(f):
            raise SystemExit("%s is not committed (or differs from HEAD)" % _rel(f))


def code_commit_time():
    return max(E43.commit_time(f) for f in CODE)


# ================================================================ seam
def cmd_seam(a):
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="p57_seam_")
    try:
        pools, zs, _sets, v26 = d_context()
        ref, _counts = reference_D(tmp, pools, zs, v26)
        del pools, zs
        gc.collect()
        pools, zs, _sets, v26 = e_context()
        se, _counts = seam_E(tmp, pools, zs, v26)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = bool(ref["all_identical"] and se["identical"])
    rec = {"format": "c10-prereg57 seam v1",
           "rule": "the unadapted base's existing vectors, restricted to one pool's games and written / read in "
                   "p57_ft embed's layout, through the per-pool scoring path == the recorded per-pool numbers, bit "
                   "for bit",
           "E": se, "D": ref, "identical": ok,
           "not_reproducible": ["quick screen: prereg43 tune.json records per-pool scores only at S0's best cell "
                                "(wE %d, k %d); the other 11 cells have 10-pool means only, so on 2 pools they are "
                                "computed but not compared" % recorded_D(v26)["cell"],
                                "D by-k values: recorded as 10-pool means only"],
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "git_head": L.git_head(),
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({"E_identical": se["identical"], "E_per_pool": se["per_pool"], "D_identical": ref["identical"],
                      "D_v26": ref["v26"]["per_pool"], "D_backend_only": ref["backend_only"]["per_pool"],
                      "D_quick_cell": ref["quick"]["at_recorded_cell"]["per_pool"],
                      "peak_working_set_mb": rec["peak_working_set_mb"], "seconds": rec["seconds"]}, indent=1))
    if not ok:
        raise SystemExit("seam failed")
    return 0


# ================================================================ dev (D)
def later_vectors(data=FT.DATA):
    return sorted(glob.glob(os.path.join(data, "judge", "*", "emb", "*.npz")) +
                  glob.glob(os.path.join(data, "apply", "*", "emb", "*.npz")))


def choose(settings):
    """PLAN.md: the highest v2-6-style D mean; ties -> fewer steps, then smaller share (then smaller lr)."""
    return min(settings, key=lambda s: (-s["v26"]["mean"], s["n"], float(s["share"]), float(s["lr"])))


def gate_of(settings, chosen):
    """Smallest reading: the chosen setting beats the reference by >= GAIN on each pool."""
    def beats(s):
        return bool(all(d >= GAIN for d in s["minus_reference"]["per_pool"]))
    others = [s["tag_n"] for s in settings if beats(s)]
    return {"rule": "the chosen setting's v2-6-style score is >= %.2f above the unadapted S0 on each of the %d D "
                    "pools (PLAN.md: otherwise stop, not adopted, E is not touched)" % (GAIN, len(FT.DEV_POOLS)),
            "pass": beats(chosen), "settings_beating_on_both": others,
            "any_setting_passes": bool(others),
            "note": None if beats(chosen) or not others else
            "the chosen (highest-mean) setting does not pass but another one does: PLAN.md's wording does not "
            "settle this case; this code stops - a PLAN revision is needed before E is touched"}


def cmd_dev(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the choice is made once" % _rel(a.out))
    committed(CODE)
    later = later_vectors()
    if later:
        raise SystemExit("judge / apply vectors exist before the choice: %s" % [_rel(p) for p in later])
    after = code_commit_time()
    tmp = tempfile.mkdtemp(prefix="p57_dev_")
    try:
        pools, zs, sets, v26 = d_context()
        ref, counts = reference_D(tmp, pools, zs, v26)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not ref["all_identical"]:
        raise SystemExit("seam: the unadapted S0 does not reproduce prereg43's D records: %s" % ref["identical"])
    cell = (ref["quick"]["at_recorded_cell"]["wE"], ref["quick"]["at_recorded_cell"]["k"])
    settings = []
    for share, lr in FT.GRID:
        tag = FT.run_tag(share, lr)
        for n in FT.DEV_STEPS:
            grids, per, byk, backend, vecs = [], [], [], [], []
            for i, p, z, cnt in zip(FT.DEV_POOLS, pools, zs, counts):
                d = FT.dev_dir(tag, i)
                path = FT.emb_file(d, n)
                if not os.path.exists(path):
                    raise SystemExit("missing %s" % _rel(path))
                e, v = load_emb(path, p)
                vecs.append(check_vectors(v, path, FT.ckpt_file(d, n), "S0", "D", i, share, lr, n, cnt, after))
                grids.append(quick_one(p, e))
                x, y, w = system_one(p, e, z, v26)
                per.append(x)
                byk.append(y)
                backend.append(w)
            cells, best = quick_combine(grids)
            diff = [x - r for x, r in zip(per, ref["v26"]["per_pool"])]
            settings.append({"share": share, "lr": lr, "n": n, "tag": tag, "tag_n": "%s_n%d" % (tag, n),
                             "v26": {"per_pool": per, "mean": float(np.mean(per)), "by_k": EV.mean_by_k(byk)},
                             "minus_reference": {"per_pool": diff, "mean": float(np.mean(per)) - ref["v26"]["mean"]},
                             "backend_only": {"per_pool": backend, "mean": float(np.mean(backend))},
                             "quick": {"Q": best["mean"], "wE": best["wE"], "k": best["k"],
                                       "per_pool": best["per_pool"],
                                       "at_recorded_cell": cell_of(cells, *cell)["per_pool"],
                                       "minus_reference_Q": best["mean"] - ref["quick"]["Q"]},
                             "vectors": vecs})
            print(json.dumps({"setting": settings[-1]["tag_n"], "v26_mean": settings[-1]["v26"]["mean"],
                              "minus_reference": diff}), flush=True)
    chosen = choose(settings)
    gate = gate_of(settings, chosen)
    rec = {"format": "c10-prereg57 dev v1", "d_seeds": ref["d_seeds"], "base": "S0", "reference": ref,
           "settings": settings,
           "rule": "highest v2-6-style D mean (with P1; cfg / wP / k of prereg43 tune.json unchanged) over the 12 "
                   "settings; ties -> fewer steps, then smaller share, then smaller lr",
           "chosen": {k: chosen[k] for k in ("share", "lr", "n", "tag", "tag_n", "v26", "minus_reference", "quick")},
           "gate": gate, "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "p1_sets": sets,
           "next": ("bash results/prereg57/explore/run_judge.sh %s %s %d" % (chosen["share"], chosen["lr"],
                                                                           chosen["n"])) if gate["pass"] else None,
           "git_head": L.git_head(), "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({"reference": ref["v26"], "chosen": rec["chosen"], "gate": gate, "next": rec["next"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_uses_so_far():
    prior = P49.e_judgements_so_far()
    p51 = [os.path.basename(p) for p in P51.existing_evals()]
    p54 = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(P54D, "eval*.json")))]
    return {"through_prereg49": prior, "prereg51": p51, "prereg54": p54,
            "count": len(prior) + len(p51) + len(p54)}


def read_dev(path=DEV_JSON):
    if not os.path.exists(path):
        raise SystemExit("missing %s" % _rel(path))
    committed([path])
    dev = read_json(path)
    if not dev["gate"]["pass"]:
        raise SystemExit("dev.json: the gate did not pass - not adopted, E is not touched (PLAN.md)")
    return dev


def cmd_judge(a):
    t0 = time.time()
    if datetime.datetime.now() > LAST:
        raise SystemExit("past %s (PLAN.md hard stop)" % LAST)
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % _rel(a.out))
    committed(CODE)
    dev = read_dev()
    ch = dev["chosen"]
    uses = e_uses_so_far()
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, uses["count"],
                                                                                         uses))
    after = max(code_commit_time(), E43.commit_time(DEV_JSON))
    paths = [FT.emb_file(FT.judge_dir(i), ch["n"]) for i in FT.JUDGE_POOLS]
    missing = [_rel(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing %s" % missing)
    pools, zs, sets, v26 = e_context()
    tmp = tempfile.mkdtemp(prefix="p57_judge_")
    try:
        se, counts = seam_E(tmp, pools, zs, v26)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not se["identical"]:
        raise SystemExit("seam: the unadapted c49 does not reproduce v2-7's per-pool scores")
    bper = se["per_pool"]
    per, byk, vecs = [], [], []
    for i, p, z, cnt, path in zip(FT.JUDGE_POOLS, pools, zs, counts, paths):
        e, v = load_emb(path, p)
        d = FT.judge_dir(i)
        vecs.append(check_vectors(v, path, FT.ckpt_file(d, ch["n"]), "c49", "E", i, ch["share"], ch["lr"], ch["n"],
                                  cnt, after))
        x, y, _w = system_one(p, e, z, v26)
        per.append(x)
        byk.append(y)
    j = E43.judge_rule(per, bper)
    rec = {"format": "c10-prereg57 eval v1", "e_seeds": [p["seed"] for p in pools], "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "e_reused": True,
           "plan_numbering_note": "PLAN.md writes '(7 + existing prereg51 / prereg54 judgements) + 1'; the number "
                                  "here is 1 + the judgements actually on record (6 through prereg49 + prereg51's + "
                                  "prereg54's), i.e. the PLAN's 7 already contains prereg51-B",
           "chosen": {k: ch[k] for k in ("share", "lr", "n", "tag")}, "dev_sha256": st.sha256_lf(DEV_JSON),
           "vectors": vecs, "base": {"name": "c49 (v2-7)", "eval": _rel(P51.V27_EVAL), "vectors_E": se["vectors"],
                                     "e_mean_20_pools": V27_E},
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"], "seam_base_E_identical": True,
           "git_head": L.git_head(), "p1_sets": sets,
           "means": {"p57": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"p57": per, "base": list(bper)}, "by_k": {"p57": EV.mean_by_k(byk)}}
    rec["p57_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "the adapted c49")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "p57_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    t0 = time.time()
    ch = None
    if not a.seam_only:
        committed([EVAL_JSON])
        ev = read_json(EVAL_JSON)
        if not ev.get("adopt"):
            raise SystemExit("eval.json does not adopt: keep v2-7")
        ch = read_dev()["chosen"]
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
    gc.collect()

    def top5_of(e):
        F = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
        return P39E.top5_bytes(qids, st.top5(st.csls(F + wP * z, k), cand_ids))

    bpath = P51.c49_file(a.pool)
    eb = E43.pool_emb(bpath, items, P51.C49_SHA)
    data_b = top5_of(eb)
    bcount = np.asarray(eb.count)
    del eb
    bfile = os.path.join(V27_APPLY, a.pool, "top5.csv")
    want_b, crlf = lf_bytes(bfile)
    if want_b != data_b:
        raise SystemExit("seam: v2-7 rebuilt on %s differs from %s" % (a.pool, _rel(bfile)))
    if a.seam_only:
        print(json.dumps({"pool": a.pool, "v27_top5_identical": True, "file": _rel(bfile),
                          "working_copy_had_crlf": crlf, "questions": len(qids),
                          "candidates": len(cand_ids), "games": len(items), "peak_working_set_mb": E45.peak_mb(),
                          "seconds": round(time.time() - t0, 1)}))
        return 0
    d = FT.apply_dir(a.pool)
    path = FT.emb_file(d, ch["n"])
    if not os.path.exists(path):
        raise SystemExit("missing %s" % _rel(path))
    v = FT.read_vectors(path)
    if v["store_row"] is not None:
        raise SystemExit("%s is not an official pool's vectors" % _rel(path))
    info = check_vectors(v, path, FT.ckpt_file(d, ch["n"]), "c49", "official", a.pool, ch["share"], ch["lr"], ch["n"],
                         bcount, max(code_commit_time(), E43.commit_time(DEV_JSON)))
    e = E43.pool_emb(path, items, v["ckpt_sha256"])
    data = top5_of(e)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as fh:
        fh.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg57 apply v1", "pool": a.pool, "chosen": {x: ch[x] for x in ("share", "lr", "n")},
           "vectors": info, "cfg": cfg, "wP": wP, "k": k, "base": "c49 (v2-7)", "base_top5_equals": _rel(bfile),
           "base_working_copy_had_crlf": crlf, "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "questions_without_P1": int((~np.asarray(has)).sum()),
           "eval_sha256": st.sha256_lf(EVAL_JSON), "top5_sha256": L.sha256_file(out),
           "differs_from_base_rows": int(sum(x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "peak_working_set_mb": E45.peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def lf_bytes(path):
    """(the file's bytes with CRLF -> LF, whether it had CR). The committed v2-7 files are LF; a checkout with
    core.autocrlf=true gives CRLF working copies of the same content (p06_style.sha256_lf's convention)."""
    with open(path, "rb") as f:
        data = f.read()
    return data.replace(b"\r\n", b"\n"), b"\r" in data


def original_rows(data):
    """The lines of the original 800 questions (ids pub_ / priv_) of a submission, as bytes."""
    return [x for x in data.split(b"\n")[1:] if x.startswith((b"pub_", b"priv_"))]


def assemble_check(new, old):
    """PLAN.md Apply: 1,600 rows, the original 800 rows byte-identical to v2-7's submission (LF bytes of both)."""
    rows = [x for x in new.split(b"\n")[1:] if x]
    a, b = original_rows(new), original_rows(old)
    chk = {"rows": len(rows), "original_rows": len(a), "new_rows": len(rows) - len(a),
           "header_identical_to_v2_7": new.split(b"\n")[0] == old.split(b"\n")[0],
           "original_rows_identical_to_v2_7": a == b}
    chk["ok"] = bool(chk["rows"] == 1600 and chk["original_rows"] == 800 and chk["new_rows"] == 800 and
                     chk["header_identical_to_v2_7"] and chk["original_rows_identical_to_v2_7"])
    return chk


def cmd_assemble(a):
    ev = read_json(EVAL_JSON)
    if not ev.get("adopt"):
        raise SystemExit("eval.json does not adopt: keep v2-7")
    rc = E43.cmd_assemble(a)          # p35_apply.cmd_assemble: ids, prefixes, top-5 of the pool's candidates
    with open(os.path.join(a.out, "submission_player.csv"), "rb") as f:
        new = f.read()
    old, crlf = lf_bytes(V27_SUBMISSION)
    chk = assemble_check(new, old)
    chk.update({"v2_7_working_copy_had_crlf": crlf,"v2_7_submission": _rel(V27_SUBMISSION), "v2_7_sha256": L.sha256_file(V27_SUBMISSION),
                "sha256": L.sha256_file(os.path.join(a.out, "submission_player.csv")), "assemble_rc": rc})
    chk["ok"] = bool(chk["ok"] and rc == 0)
    L.write_json(os.path.join(a.out, "assemble_check.json"), chk)
    print(json.dumps(chk, indent=1))
    return 0 if chk["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--out", default=SEAM_JSON)
    d = sub.add_parser("dev")
    d.add_argument("--out", default=DEV_JSON)
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    p.add_argument("--src", default=FT.SRC_OFFICIAL)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--seam-only", action="store_true", help="only rebuild v2-7 on the pool and compare (no eval.json)")
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", default=FT.SRC_OFFICIAL)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd in ("dev", "judge") and os.path.abspath(a.out) != {"dev": DEV_JSON, "judge": EVAL_JSON}[a.cmd]:
        raise SystemExit("--out is fixed for %s" % a.cmd)
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "dev": cmd_dev, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
