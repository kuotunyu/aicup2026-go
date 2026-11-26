"""prereg42 explore: D quick screen, selection, the one E judgement, official pools (PLAN.md 4, 5).

    PY results/prereg42/explore/p42_eval.py quick --emb data/prereg42/emb/<tag>/store_D.npz [--emb ...]
         -> results/prereg42/explore/quick_D.json (C0 = data/prereg36/emb/step_40000/store_D.npz is added first)
    PY results/prereg42/explore/p42_eval.py select [--base-tune T --base-eval E --base-emb-dir D] [--missing-ok V3]
         -> results/prereg42/explore/tune.json   (commit it before anything touches the E pools)
    PY results/prereg42/explore/p42_eval.py judge --e-use N           # once -> results/prereg42/explore/eval.json
    PY results/prereg42/explore/p42_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg42/explore/p42_eval.py assemble --v1 <v1 submission> --src <official tests dir>

Backend = prereg38's backend_scores (imported, unchanged). Quick screen Q: the structure and numbers of
results/prereg38/explore/tune_D_B4.json (gnorm, whitening lambda 0.02, LDA 32) with wE in {2, 4, 8, 16} x CSLS k in
{10, 20, 40}, best D mean. Seam Q0: C0 at wE 4, k 10 == tune_D_B4.json recommended.mean bit for bit. The full
retune of c* is prereg38's retune_structure with kept ['B4']. Base = the adopted system as a prereg38-format tune
file + its adopting judgement + its vectors directory (default v2-3: tune_D_B4.json, prereg38 eval.json,
data/prereg36/emb/step_40000). quick and select read the D pools only; nothing but judge loads the E pools.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p42_lib as V  # noqa: E402
import p38_backend as BK  # noqa: E402

L, pe, st, sim = V.L, BK.pe, BK.st, BK.sim
REPO = V.REPO
P38_TUNE = os.path.join(V.P38, "tune_D_B4.json")
P38_EVAL = os.path.join(V.P38, "eval.json")
C0_EMB = os.path.join(REPO, "data", "prereg36", "emb", "step_40000", "store_D.npz")
BASE_DEFAULT = {"tune": P38_TUNE, "eval": P38_EVAL, "emb_dir": os.path.dirname(C0_EMB)}
QUICK = os.path.join(HERE, "quick_D.json")
QUICK_WE = (2, 4, 8, 16)
QUICK_K = (10, 20, 40)
BOOT, BOOT_SEED = 10000, 41
MODULE_GAIN, MODULE_POOLS = 0.005, 7
GATE = 0.01
ADOPT_MEAN = 0.01
SEAM_TOL = 0.0
E_JUDGEMENTS = ("results/prereg36/explore/eval*.json", "results/prereg37/explore/eval*.json",
                "results/prereg38/explore/eval*.json", "results/prereg39/explore/eval*.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg42", "apply")


def _abs(p):
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(REPO, p))


def _rel(p):
    return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def b4_cfg():
    return read_json(P38_TUNE)["recommended"]["cfg"]


def embed_cfg_of(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["embed_cfg"])) if "embed_cfg" in z.files else None


def load_emb(path, store):
    e = pe.Emb.load(path, colour=store.color)
    e.path = path
    e.embed_cfg = embed_cfg_of(path)
    return e


def check_players(emb_path, which):
    """The npz must be the store vectors of exactly the D (or E) players of the prereg36 split."""
    E, D, _T, _per = L.splits()
    want = L.sha256_ids({"D": D, "E": E}[which])
    with np.load(emb_path, allow_pickle=False) as z:
        if "players_sha256_sorted_ids" not in z.files or str(z["players_sha256_sorted_ids"]) != want:
            raise SystemExit("%s is not the store_%s vectors of the prereg36 split" % (emb_path, which))


# ================================================================ quick screen (D)
def quick_grid(pools, emb, caches=None):
    """[(mean, wE, k, per_pool, by_k)] over QUICK_WE x QUICK_K with the B4 structure of tune_D_B4.json."""
    base = b4_cfg()
    caches = caches or [{} for _ in pools]
    rows = []
    for w in QUICK_WE:
        for k in QUICK_K:
            cfg = BK.with_(base, wE=w, norm=["csls", k])
            per, byk = [], []
            for p, c in zip(pools, caches):
                S = BK.backend_scores(p, emb, cfg, c)
                per.append(st.score(S, p["owner"]))
                byk.append(pe.score_by_k(S, p))
            rows.append({"mean": float(np.mean(per)), "wE": w, "k": k, "per_pool": [float(x) for x in per],
                         "by_k": pe.mean_by_k(byk)})
    return rows


def best_row(rows):
    return max(rows, key=lambda r: r["mean"])          # first maximum in grid order


def bootstrap(d, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, np.float64)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def diff_rec(per, ref):
    d = np.asarray(per) - np.asarray(ref)
    return {"mean": float(d.mean()), "ci95": bootstrap(d), "pools_higher": int((d > 0).sum())}


def cmd_quick(a):
    t0 = time.time()
    rec = read_json(a.out) if os.path.exists(a.out) else {"format": "c10-prereg42 quick D v1",
                                                          "d_seeds": pe.D_SEEDS, "rows": {}}
    store = sim.Store(L.FEATS, load=())
    pools = pe.load_pools(pe.TERMS_DIR["D"], pe.D_SEEDS)
    paths = list(a.emb)
    if "C0" not in rec["rows"]:
        paths = [C0_EMB] + paths
    for path in paths:
        path = _abs(path)
        check_players(path, "D")
        e = load_emb(path, store)
        label = "C0" if path == _abs(C0_EMB) else e.name
        if label in rec["rows"]:
            print("already screened: %s" % label)
            continue
        rows = quick_grid(pools, e)
        if label == "C0":
            seam = [r for r in rows if r["wE"] == 4 and r["k"] == 10][0]["mean"]
            want = read_json(P38_TUNE)["recommended"]["mean"]
            if seam != want:
                raise SystemExit("seam Q0 failed: C0 at wE 4, k 10 = %r, tune_D_B4.json = %r" % (seam, want))
            rec["seam_Q0"] = {"C0_wE4_k10": seam, "tune_D_B4_mean": want, "identical": True}
        b = best_row(rows)
        row = {"label": label, "file": _rel(path), "ckpt_sha256": e.sha, "embed_cfg": e.embed_cfg,
               "file_sha256": L.sha256_file(path), "best": b, "grid": [{k: r[k] for k in ("mean", "wE", "k")}
                                                                       for r in rows]}
        if label != "C0":
            row["vs_C0"] = diff_rec(b["per_pool"], rec["rows"]["C0"]["best"]["per_pool"])
        rec["rows"][label] = row
        L.write_json(a.out, rec)
        print(json.dumps({"label": label, "Q": b["mean"], "wE": b["wE"], "k": b["k"], "by_k": b["by_k"],
                          "vs_C0": row.get("vs_C0")}), flush=True)
    print("seconds %.1f" % (time.time() - t0))
    return 0


# ================================================================ base system
def base_info(tune_path=None, eval_path=None, emb_dir=None):
    tune_path = _abs(tune_path or BASE_DEFAULT["tune"])
    eval_path = _abs(eval_path or BASE_DEFAULT["eval"])
    emb_dir = _abs(emb_dir or BASE_DEFAULT["emb_dir"])
    t, ev = read_json(tune_path), read_json(eval_path)
    tsha, esha = st.sha256_lf(tune_path), st.sha256_lf(eval_path)
    if not ev.get("adopt") or ev.get("tune_sha256") != tsha:
        raise SystemExit("--base-eval does not adopt --base-tune (adopt false, or it judged another tune file)")
    cfg = t["recommended"]["cfg"]
    if cfg.get("aqe") or cfg.get("qsmooth") or cfg.get("norm", [None])[0] != "csls":
        raise SystemExit("base: only a prereg38 configuration ending in CSLS without aqe / qsmooth is supported")
    return {"tune": tune_path, "eval": eval_path, "emb_dir": emb_dir, "cfg": cfg, "ckpt_sha256": t["emb"]["ckpt_sha256"],
            "d_per_pool": t["recommended"]["per_pool"], "d_mean": t["recommended"]["mean"],
            "e_per_pool": ev["per_pool"]["backend"], "e_mean": ev["means"]["backend"],
            "record": {"tune": _rel(tune_path), "eval": _rel(eval_path), "emb_dir": _rel(emb_dir),
                       "tune_sha256_lf": tsha, "eval_sha256_lf": esha, "cfg": cfg, "emb": t["emb"],
                       "d_mean": t["recommended"]["mean"], "e_mean": ev["means"]["backend"],
                       "default": (tune_path, eval_path, emb_dir) == tuple(_abs(BASE_DEFAULT[k]) for k in
                                                                          ("tune", "eval", "emb_dir"))}}


def base_from_record(rec):
    b = base_info(rec["tune"], rec["eval"], rec["emb_dir"])
    if b["record"]["tune_sha256_lf"] != rec["tune_sha256_lf"] or b["record"]["eval_sha256_lf"] != rec["eval_sha256_lf"]:
        raise SystemExit("the base changed since tune.json")
    return b


def base_scores(base, pools, which, store):
    e = load_emb(os.path.join(base["emb_dir"], "store_%s.npz" % which), store)
    if e.sha != base["ckpt_sha256"]:
        raise SystemExit("base vectors (%s) were not made from the base checkpoint" % which)
    return [float(st.score(BK.backend_scores(p, e, base["cfg"]), p["owner"])) for p in pools]


# ================================================================ select (D)
def candidate_order(quick, train_records, base_tag_prefix):
    """The fixed candidate list of PLAN.md 4 that is present: T(base), V1..V4 raw / EMA (V3 x full / chunk24),
    T(best variant candidate). Returns (ordered labels, expected labels that are missing)."""
    labels = list(quick["rows"])
    order, missing = [], []
    tb = [x for x in labels if x.startswith(base_tag_prefix) and x.endswith("_tta8")]
    (order if tb else missing).extend(tb or ["T(base) %s*_tta8" % base_tag_prefix])
    for v in sorted(V.VARIANTS):
        tr = train_records.get(v)
        if tr is None:
            continue
        step = "step_%05d" % tr["steps"]
        modes = ("_full", "_chunk24") if V.is_tx(V.VARIANTS[v]) else ("",)
        for w in ("raw", "ema"):
            for m in modes:
                lab = "%s_%s_%s_tta1%s" % (v, step, w, m)
                (order if lab in labels else missing).append(lab)
    return order, missing


def cmd_select(a):
    t0 = time.time()
    quick = read_json(a.quick)
    if "seam_Q0" not in quick:
        raise SystemExit("quick_D.json has no C0 seam")
    trs = {}
    for f in sorted(glob.glob(os.path.join(HERE, "train_record_V*.json"))):
        tr = read_json(f)
        if tr.get("finished") and not tr.get("smoke"):
            trs[tr["config"]["variant"]] = tr
    missing_ok = set(x for x in (a.missing_ok or "").split(",") if x)
    for v in sorted(V.VARIANTS):
        if v not in trs and v not in missing_ok:
            raise SystemExit("no finished train_record_%s.json (list it in --missing-ok if it was not run)" % v)
    base = base_info(a.base_tune, a.base_eval, a.base_emb_dir)
    base_prefix = "base" + base["ckpt_sha256"][:8]
    order, missing = candidate_order(quick, trs, base_prefix)
    if missing:
        raise SystemExit("candidates not screened yet: %s" % missing)
    Q = {lab: quick["rows"][lab]["best"]["mean"] for lab in order}
    # T(best variant candidate) (PLAN.md 4): the best of the V candidates must also be screened with TTA-8
    vlabs = [x for x in order if x.split("_")[0] in V.VARIANTS]
    tbest = None
    if vlabs:
        bv = max(vlabs, key=lambda x: Q[x])
        tbest = bv.replace("_tta1", "_tta8")
        if tbest not in quick["rows"]:
            raise SystemExit("embed %s with --tta 8 and screen it (quick) first: %s" % (bv, tbest))
        order.append(tbest)
        Q[tbest] = quick["rows"][tbest]["best"]["mean"]
    cstar = max(order, key=lambda x: Q[x])             # first maximum in the fixed order
    # module verdicts (record only; stage 4)
    c0 = quick["rows"]["C0"]["best"]
    verdicts = {}
    for v in sorted(trs):
        labs = [x for x in order if x.startswith(v + "_")]
        b = max(labs, key=lambda x: Q[x])
        d = diff_rec(quick["rows"][b]["best"]["per_pool"], c0["per_pool"])
        verdicts[v] = {"best_candidate": b, "Q": Q[b], "vs_C0": d,
                       "effective": bool(d["mean"] >= MODULE_GAIN and d["pools_higher"] >= MODULE_POOLS)}
    for lab in order:
        if lab.endswith("_tta8"):
            raw = lab.replace("_tta8", "_tta1")
            ref = "C0" if lab.startswith(base_prefix) and base["record"]["default"] else raw
            if ref in quick["rows"]:
                d = diff_rec(quick["rows"][lab]["best"]["per_pool"], quick["rows"][ref]["best"]["per_pool"])
                verdicts["TTA8:" + lab] = {"vs": ref, "diff": d, "effective": bool(
                    d["mean"] >= MODULE_GAIN and d["pools_higher"] >= MODULE_POOLS)}
    # base D seam and the full retune of c*
    store = sim.Store(L.FEATS, load=())
    pools = pe.load_pools(pe.TERMS_DIR["D"], pe.D_SEEDS)
    bper = base_scores(base, pools, "D", store)
    if bper != list(base["d_per_pool"]):
        raise SystemExit("seam: the base's per-pool D scores %r differ from its tune file %r" % (bper, base["d_per_pool"]))
    cpath = _abs(quick["rows"][cstar]["file"])
    if L.sha256_file(cpath) != quick["rows"][cstar]["file_sha256"]:
        raise SystemExit("%s changed since it was screened" % cpath)
    e = load_emb(cpath, store)
    bench = BK.Bench(pools, e, BK.base_cfg_of({"M": {"wE": 4, "csls_k": 20}}))
    c, r = BK.retune_structure(bench, ["B4"], b4_cfg())
    gate_d = r["mean"] - base["d_mean"]
    rec = {"format": "c10-prereg42 tune v1", "d_seeds": pe.D_SEEDS, "quick_sha256_lf": st.sha256_lf(a.quick),
           "candidates_in_order": order, "Q": Q, "c_star": {
               "label": cstar, "file": quick["rows"][cstar]["file"], "file_sha256": quick["rows"][cstar]["file_sha256"],
               "ckpt_sha256": e.sha, "embed_cfg": e.embed_cfg, "quick": quick["rows"][cstar]["best"],
               "recommended": {"cfg": c, "mean": r["mean"], "per_pool": r["per_pool"], "by_k": r["by_k"]}},
           "base": base["record"], "base_D_per_pool_identical": True,
           "c_star_minus_base_D": diff_rec(r["per_pool"], base["d_per_pool"]),
           "gate": {"rule": "R(c*) - D_base >= %.2f" % GATE, "value": gate_d, "pass": bool(gate_d >= GATE)},
           "module_verdicts": verdicts, "missing_ok": sorted(missing_ok), "train_records": {
               v: {"steps": tr["steps"], "seconds": tr["seconds"], "device": tr["device"]} for v, tr in trs.items()},
           "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("Q", "gate", "c_star_minus_base_D")}, indent=1))
    print(json.dumps({"c_star": cstar, "R": r["mean"], "cfg": c}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far():
    return sorted(f for g in E_JUDGEMENTS for f in glob.glob(os.path.join(REPO, g)))


def judge_numbers(new, old):
    new, old = np.asarray(new), np.asarray(old)
    d = new - old
    lo, hi = bootstrap(d)
    need = int(np.ceil(0.9 * len(d)))
    return {"c_star_minus_base": {"mean": float(d.mean()), "ci95": [lo, hi], "pools_higher": int((d > 0).sum())},
            "rule": "adopt c* if mean difference >= %.2f and ci95 low > 0 and higher in >= %d of %d pools (pool "
                    "bootstrap, %d samples, seed %d)" % (ADOPT_MEAN, need, len(d), BOOT, BOOT_SEED),
            "adopt": bool(d.mean() >= ADOPT_MEAN and lo > 0 and int((d > 0).sum()) >= need)}


def cmd_judge(a):
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    tune = read_json(a.tune)
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 4)")
    prior = e_judgements_so_far()
    if a.e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but %d earlier E judgements are in the repo (%s): this would be use %d"
                         % (a.e_use, len(prior), [_rel(p) for p in prior], len(prior) + 1))
    base = base_from_record(tune["base"])
    cs = tune["c_star"]
    epath = os.path.join(os.path.dirname(_abs(cs["file"])), "store_E.npz")
    check_players(epath, "E")
    store = sim.Store(L.FEATS, load=())
    e = load_emb(epath, store)
    want = {k: v for k, v in (cs["embed_cfg"] or {}).items() if k not in ("games", "moves", "compare")}
    got = {k: v for k, v in (e.embed_cfg or {}).items() if k not in ("games", "moves", "compare")}
    if e.sha != cs["ckpt_sha256"] or got != want:
        raise SystemExit("store_E.npz was not made from c*'s checkpoint with c*'s embed settings")
    pools = pe.load_pools(pe.TERMS_DIR["E"], pe.E_SEEDS)
    Es = set(L.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in store.pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    bper = base_scores(base, pools, "E", store)
    if bper != list(base["e_per_pool"]):
        raise SystemExit("seam: the base's per-pool E scores differ from its eval file")
    cfg = cs["recommended"]["cfg"]
    per, byk, byk_b = [], [], []
    be = load_emb(os.path.join(base["emb_dir"], "store_E.npz"), store)
    for p in pools:
        S = BK.backend_scores(p, e, cfg)
        per.append(float(st.score(S, p["owner"])))
        byk.append(pe.score_by_k(S, p))
        byk_b.append(pe.score_by_k(BK.backend_scores(p, be, base["cfg"]), p))
    rec = {"format": "c10-prereg42 eval v1", "e_seeds": pe.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": [_rel(p) for p in prior], "c_star": {k: cs[k] for k in ("label", "ckpt_sha256",
                                                                                          "embed_cfg")},
           "cfg": cfg, "base": tune["base"], "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune),
           "git_head": L.git_head(), "means": {"c_star": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"c_star": per, "base": bper},
           "by_k": {"c_star": pe.mean_by_k(byk), "base": pe.mean_by_k(byk_b)}}
    rec.update(judge_numbers(per, bper))
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("means", "c_star_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    import csv
    import p35_apply as pa
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c* (or belongs to another tune.json): keep the base")
    cs = tune["c_star"]
    cfg = cs["recommended"]["cfg"]
    cands, qs = pa.read_pool(a.src, a.pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[a.pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    path = os.path.join(os.path.dirname(_abs(cs["file"])), "pool_%s.npz" % a.pool)
    e = pe.Emb.load(path, colour=np.asarray([c for _s, c in items]))
    ec = embed_cfg_of(path) or {}
    want = {k: v for k, v in (cs["embed_cfg"] or {}).items() if k not in ("games", "moves", "compare")}
    if e.sha != cs["ckpt_sha256"] or len(e.emb) != len(items) or \
            {k: v for k, v in ec.items() if k not in ("games", "moves", "compare")} != want:
        raise SystemExit("pool vectors do not match c*'s checkpoint / embed settings / the pool's games")
    fit = sim.Store(L.FEATS, load=())
    store = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, models = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        models[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(store, ld.cols)
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    rec["T"] = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
    S = BK.backend_scores(rec, e, cfg)
    tops = st.top5(S, cand_ids)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
        for q, t in zip(qids, tops):
            wr.writerow([q] + t)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg42 apply v1", "pool": a.pool, "cfg": cfg, "c_star": cs["label"],
           "ckpt_sha256": cs["ckpt_sha256"], "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
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
    q = sub.add_parser("quick")
    q.add_argument("--emb", action="append", default=[])
    q.add_argument("--out", default=QUICK)
    s = sub.add_parser("select")
    s.add_argument("--quick", default=QUICK)
    s.add_argument("--base-tune", default=None)
    s.add_argument("--base-eval", default=None)
    s.add_argument("--base-emb-dir", default=None)
    s.add_argument("--missing-ok", default=None, help="comma-separated variants that were not trained (recorded)")
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
    if a.cmd in ("quick", "select"):
        BK.polite()
    return {"quick": cmd_quick, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
