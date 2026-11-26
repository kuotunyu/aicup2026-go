"""prereg05 exploration B2, step 2: rate tables V0-V3 from the subsample counts, replay the 73,556 inner games, fuse
each Vi in place of Q02's O4 (5-fold OOF, rule B) and compare on the inner pseudo-questions.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg05/explore/B2_opp_cells_v2/b2_score.py --pct 25
    -> scratchpad/b2/tables_<pct>/*.npy, o_inner_<pct>.npz, oof_<pct>.npz
    -> results/prereg05/explore/B2_opp_cells_v2/b2_report.json (b2_report_<pct>.json for a non-25 % run)

Design-stage numbers on inner pseudo-questions of training players only; not validation results. Nothing under
data/local is opened (b2_lib guard); score.py is not run; no frozen file is written.
Seams (any failure stops): V0 table == tools/p04_opp.build_tables on the same counts; the replay's registered-O4
column == results/prereg04/opp/inner_o4.npz o4 (and its opponent move counts) on every inner game; u1_lib.run_gen
reproduces Q02 within 1e-12 (eval and single colour).
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # numpy + scipy with 24 OpenBLAS threads commit ~1.5 GB per process here; seam c
                                     # (Q02 reproduced within 1e-12) checks that the fusion is unaffected

import argparse  # noqa: E402
import json
import multiprocessing as mp
import os
import time

import b2_lib as B  # noqa: I100  (guard first)
import numpy as np

po = B.po
DS = os.path.join(B.REPO, "data", "prereg04", "design_scratch")
INNER_NPZ = os.path.join(DS, "moves_inner.npz")
INNER_NPY = os.path.join(DS, "moves_inner_moves.npy")
REG_COUNTS = os.path.join(B.REPO, "data", "prereg04", "counts_fit.npz")
INNER_REPORT = os.path.join(B.REPO, "results", "prereg04", "opp", "inner_report.json")
WORKERS = 4
TOL = 1e-12
Q02_EVAL, Q02_MONO = 0.6101211982700248, 0.6106683012815998
COLS = ("V0", "V1", "V2", "V3", "REG")
QBOOT, QSEED = 2000, 0
CHUNK_CELLS = 1 << 20


# ================================================================ tables
def write_rate_table(C, A, path, prior=None):
    """Registered formula into a file-backed [n, 10] float64 table, chunked over cells (elementwise identical to
    p04_opp.build_tables: integer counts summed in float64 are exact). prior(s, e) -> [10, e - s] replaces r(c)
    in the level step when given (V3)."""
    n = C.shape[1]
    Cp = np.zeros(n)
    Ap = np.zeros(n)
    for s in range(0, n, CHUNK_CELLS):
        e = min(n, s + CHUNK_CELLS)
        Cp[s:e] = np.asarray(C[:, s:e]).astype(np.float64).sum(0)
        Ap[s:e] = np.asarray(A[:, s:e]).astype(np.float64).sum(0)
    r0 = Cp.sum() / Ap.sum()
    r = (Cp + 1.0) / (Ap + 1.0 / r0)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n, 10))
    for s in range(0, n, CHUNK_CELLS):
        e = min(n, s + CHUNK_CELLS)
        c = np.asarray(C[:, s:e]).astype(np.float64)
        a = np.asarray(A[:, s:e]).astype(np.float64)
        base = r[None, s:e] if prior is None else prior(s, e, r[s:e])
        out[s:e] = ((c + B.ALPHA) / (a + B.ALPHA / base)).T
    out.flush()
    del out
    return r, float(r0)


def build_tables(pct, cdir, tdir):
    """-> {variant: path} and records. Counts are the file-backed int64 [10, N_BOTH] arrays of b2_count.py."""
    C = np.load(os.path.join(cdir, f"C_o_{pct}.npy"), mmap_mode="r")
    A = np.load(os.path.join(cdir, f"A_o_{pct}.npy"), mmap_mode="r")
    os.makedirs(tdir, exist_ok=True)
    paths, rec = {}, {}

    def marg(X, v):
        n = B.NCELLS_V[v]
        out = np.zeros((10, n), dtype=np.int64)
        for k in range(10):
            Y = np.asarray(X[k]).reshape(B.N_CELLS, B.N_LINE, B.N_LIB)
            out[k] = (Y.sum(axis=(1, 2)) if v == "V0" else Y.sum(axis=2).reshape(-1) if v == "V1"
                      else Y.sum(axis=1).reshape(-1))
        return out

    t0 = time.time()
    C0, A0 = marg(C, "V0"), marg(A, "V0")
    paths["V0"] = os.path.join(tdir, "V0.npy")
    rV0, r0 = write_rate_table(C0, A0, paths["V0"])
    tab0 = np.load(paths["V0"])
    ref, _info = po.build_tables({"C_t": C0, "A_t": A0, "C_o": C0, "A_o": A0})
    rec["V0_equal_to_p04_opp_build_tables"] = bool(np.array_equal(tab0, ref[1]))
    del ref
    rec["r0"] = r0
    for v in ("V1", "V2"):
        Cv, Av = marg(C, v), marg(A, v)
        paths[v] = os.path.join(tdir, f"{v}.npy")
        write_rate_table(Cv, Av, paths[v])
        del Cv, Av
    ratio = tab0 / rV0[:, None]                        # r_k^V0(c) / r^V0(c), [N_CELLS, 10]

    def prior(s, e, r_x):
        cells = np.arange(s, e) // (B.N_LINE * B.N_LIB)
        return r_x[None, :] * ratio[cells].T            # (r(x) * r_k^V0(cell)) / r^V0(cell) as r(x) * ratio

    paths["V3"] = os.path.join(tdir, "V3.npy")
    write_rate_table(C, A, paths["V3"], prior=prior)
    # registered O4 opponent table (100 % fit counts) for the scoring seam
    reg = po.load_counts(REG_COUNTS)
    tabs, _ = po.build_tables(reg)
    del reg
    trec = po.tables_record(tabs)
    want = json.load(open(INNER_REPORT, encoding="utf-8"))["tables_sha256"]
    rec["registered_tables_sha256"] = trec["tables_sha256"]
    rec["registered_tables_equal_inner_report"] = trec["tables_sha256"] == want
    paths["REG"] = os.path.join(tdir, "REG.npy")
    np.save(paths["REG"], np.ascontiguousarray(tabs[1]))
    del tabs
    rec["seconds"] = round(time.time() - t0, 1)
    rec["files"] = {k: {"path": p, "shape": list(np.load(p, mmap_mode="r").shape)} for k, p in paths.items()}
    return paths, rec


# ================================================================ inner replay
_S = {}


def _score_init(paths, npy, off, tcol):
    _S["tabs"] = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    _S["mv"] = np.load(npy, mmap_mode="r")
    _S["off"], _S["tcol"] = off, tcol


def score_one(m, tc, tabs):
    """-> ll [len(COLS), 3, 10], n_moves [3], anomalies: per phase sums of log P_k over the opponent's moves, the
    arithmetic of p04_opp.score_game (num - den, per phase sum) for each table."""
    ll = np.zeros((len(COLS), 3, 10))
    nm = np.zeros(3, dtype=np.int64)
    res, an = B.game_both(m, tc)
    if res is None:
        return ll, nm, an
    nb, te, nbp, phase, _ = res
    starts = np.flatnonzero(np.r_[True, te[1:] != te[:-1]])
    nm[:] = np.bincount(phase, minlength=3)
    for j, v in enumerate(COLS):
        vv = "V0" if v == "REG" else v
        T = tabs[v]
        num = np.log(T[B.variant_index(nbp, vv)])
        den = np.log(np.add.reduceat(T[B.variant_index(nb, vv)], starts, axis=0))
        per = num - den
        for ph in range(3):
            mk = phase == ph
            if mk.any():
                ll[j, ph] = per[mk].sum(0)
    return ll, nm, an


def _score_chunk(items):
    tabs, mv, off, tcol = _S["tabs"], _S["mv"], _S["off"], _S["tcol"]
    out = []
    for row, g in items:
        m = np.array(mv[off[g]:off[g + 1]])
        out.append((row,) + score_one(m, int(tcol[g]), tabs))
    return out


def replay_inner(paths, keys, limit=0, workers=WORKERS):
    z = np.load(INNER_NPZ, allow_pickle=False)
    inner = {k: z[k] for k in ("off", "rank", "game_id", "player_id", "color", "label")}
    z.close()
    idx = po.inner_rows(inner, keys)                         # inner_keys row -> moves_inner row (player / colour agree)
    tcol = np.where(inner["color"] == "B", 1, 2).astype(np.int64)
    n = keys["n"]
    rows = list(range(n if not limit else min(limit, n)))
    items = [(i, int(idx[i])) for i in rows]
    chunks = [items[i:i + 200] for i in range(0, len(items), 200)]
    ll = np.full((n, len(COLS), 3, 10), np.nan)
    nm = np.zeros((n, 3), dtype=np.int64)
    an = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    if workers == 0:                   # smoke: in-process (no extra worker while the count runs)
        _score_init(paths, INNER_NPY, inner["off"], tcol)
        for ch in chunks:
            for row, a, b, c in _score_chunk(ch):
                ll[row], nm[row], an[row] = a, b, c
    else:
        for v_ in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v_] = "1"       # workers only (spawned now); no BLAS in the replay; ~1.5 GB less commit each
        with mp.get_context("spawn").Pool(workers, initializer=_score_init,
                                          initargs=(paths, INNER_NPY, inner["off"], tcol)) as pool:
            for k, res in enumerate(pool.imap_unordered(_score_chunk, chunks)):
                for row, a, b, c in res:
                    ll[row], nm[row], an[row] = a, b, c
                if (k + 1) % 40 == 0:
                    print(f"{k + 1}/{len(chunks)} chunks ({time.time() - t0:.0f}s)", flush=True)
    labels_equal = bool(np.array_equal(inner["label"][idx].astype(np.int64), keys["label"]))
    return ll, nm, an, rows, round(time.time() - t0, 1), labels_equal


# ================================================================ comparison records
def qboot_diff(sa, sb, y, draws=QBOOT, seed=QSEED):
    d = np.asarray(sa, float) - np.asarray(sb, float)
    rng = np.random.default_rng(seed)
    per = []
    for i in range(10):
        di = d[y == i]
        if not len(di):
            continue
        acc = np.empty(draws)
        for s in range(0, draws, 200):
            e = min(draws, s + 200)
            acc[s:e] = di[rng.integers(0, len(di), size=(e - s, len(di)))].mean(1)
        per.append(acc)
    st = np.mean(per, axis=0)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def official_shape(sc, ms, mboot=None):
    """Single-colour pseudo-questions, each level 1/10 and each k 1/4 (official: 100 questions each of k = 5/10/15/20):
    point value, and the PlayerBootstrap draws (players within each level) when mboot is given."""
    y, ks = ms["y"], ms["k"]
    val = float(np.mean([B.u1.rank_balanced(sc, y, ks == k) for k in (5, 10, 15, 20)]))
    if mboot is None:
        return val, None
    per_k = []
    for k in (5, 10, 15, 20):
        mk = (ks == k).astype(np.float64)
        s = np.bincount(mboot.player, weights=sc * mk, minlength=mboot.n_players)
        c = np.bincount(mboot.player, weights=mk, minlength=mboot.n_players)
        lv = [s[d].sum(1) / c[d].sum(1) for d in mboot.draws]
        per_k.append(np.mean(lv, axis=0))
    return val, np.mean(per_k, axis=0)


def compare(name, A, Bn, R, inner):
    """A - B on the eval and single-colour pseudo-questions (R[x] = {'sc','scm','m','mm'})."""
    U = B.u1
    Xe, ms = inner["X_eval"], inner["ms"]
    a, b = R[A], R[Bn]
    boot = inner["boot"]
    d = boot.stats(a["sc"]) - boot.stats(b["sc"])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    delta = a["m"]["rank_balanced"] - b["m"]["rank_balanced"]
    mono = U.p3.mono_diff_record(ms, inner["mono_boot"], a["scm"], b["scm"])
    oa, da = official_shape(a["scm"], ms, inner["mono_boot"])
    ob, db = official_shape(b["scm"], ms, inner["mono_boot"])
    dd = da - db
    sev = U.p3.se_val_record(delta, a["sc"] - b["sc"], Xe.y)
    sev["adoption_probability_at_fraction_of_delta"] = U.p4.adoption(sev["delta"], sev["se_val"])
    return {
        "comparison": name, "a": A, "b": Bn,
        "eval_rank_balanced_delta": delta,
        "pboot_ci95": [lo, hi], "pboot_se": float(np.std(d, ddof=1)),
        "pboot": "paired PlayerBootstrap (players within each level, 1,000 draws, default_rng(0)); decision",
        "qboot_ci95_record_only": qboot_diff(a["sc"], b["sc"], Xe.y),
        "question_mean_delta": float(np.mean(a["sc"] - b["sc"])),
        "per_level_delta": {lv: a["m"]["per_level"][lv] - b["m"]["per_level"][lv] for lv in B.u1.LEVELS},
        "per_n_bin_delta": {k: a["m"]["per_n_bin"][k]["rank_balanced"] - b["m"]["per_n_bin"][k]["rank_balanced"]
                            for k in a["m"]["per_n_bin"]},
        "single_colour": {"rank_balanced": mono["rank_balanced"], "ci95": mono["ci95"], "se": mono["se"],
                          "question_mean": mono["question_mean"],
                          "black": mono["by_colour"]["B"], "white": mono["by_colour"]["W"],
                          "by_k": mono["by_k"], "n_questions": mono["n_questions"]},
        "official_shape_weighted": {"a": oa, "b": ob, "delta": oa - ob,
                                    "pboot_ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                    "definition": "single colour, mean over k = 5/10/15/20 of the rank-balanced mean"},
        "se_val_record": sev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", default="25")
    ap.add_argument("--limit", type=int, default=0, help="smoke: replay only the first N inner games, no fusion")
    ap.add_argument("--workers", type=int, default=WORKERS, help="0 = in-process (smoke)")
    a = ap.parse_args()
    T0 = time.time()
    import u1_lib as U                                   # the b2 guard is already installed (see b2_lib)
    B.u1 = U
    rep = {"what": "prereg05 exploration B2 step 2: finer opponent-move cells V0-V3 fused in place of Q02's O4 "
                   "(inner pseudo-questions of training players; design stage, NOT validation results)",
           "created": po.p3.now_iso(), "pct": a.pct, "guard": B.guard_selftest()}
    if not rep["guard"]["passed"]:
        raise SystemExit("guard self-test failed")
    cjson = os.path.join(B.HERE, f"b2_count_{a.pct}.json")
    crep = json.load(open(cjson, encoding="utf-8")) if os.path.exists(cjson) else \
        json.load(open(os.path.join(B.SCRATCH, f"smoke_b2_count_{a.pct}.json"), encoding="utf-8"))
    for name in ("C_o", "A_o"):
        p = os.path.join(B.SCRATCH, f"{name}_{a.pct}.npy")
        got = B.sha256_file(p)
        if got != crep["counts"][name]["sha256"]:
            raise SystemExit(f"{p}: sha256 differs from the count report")
    rep["counts"] = {"report": os.path.basename(cjson), "subsample": crep["subsample"], "count": crep["count"],
                     "sha256": {k: crep["counts"][k]["sha256"] for k in ("C_o", "A_o")}}
    sha = {"moves_inner.npz": B.sha256_file(INNER_NPZ), "moves_inner_moves.npy": B.sha256_file(INNER_NPY)}
    rep["inner_moves"] = {k: {"sha256": v, "manifest_sha256": B.manifest_sha(k), "equal": v == B.manifest_sha(k)}
                          for k, v in sha.items()}
    if not all(v["equal"] for v in rep["inner_moves"].values()):
        raise SystemExit("design inner moves differ from the manifest")

    # ---- P0 seam first (cheap to fail early): Q02 reproduced
    d = U.load_inputs()
    inner = d["inner"]
    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    R = {}

    def run(terms, tag):
        P, Pm, th = U.run_gen(d, terms)
        m, sc = U.metrics(U.rule_b_np(P), Xe.y, Xe.n, Xe.player)
        mm, scm = U.metrics(U.rule_b_np(Pm), Xm.y, Xm.n, Xm.player)
        R[tag] = {"P": P, "Pm": Pm, "theta": th, "m": m, "sc": sc, "mm": mm, "scm": scm}
        print(f"{tag}: eval {m['rank_balanced']:.6f}, single colour {mm['rank_balanced']:.6f} "
              f"({time.time() - T0:.0f}s)", flush=True)

    if not a.limit:
        run(["OP", "O4", "SWA", "SWB"], "Q02")
        q = R["Q02"]
        rep["seam_c"] = {"eval": q["m"]["rank_balanced"], "eval_recorded": Q02_EVAL,
                         "eval_abs_diff": abs(q["m"]["rank_balanced"] - Q02_EVAL),
                         "single_colour": q["mm"]["rank_balanced"], "single_colour_recorded": Q02_MONO,
                         "single_colour_abs_diff": abs(q["mm"]["rank_balanced"] - Q02_MONO)}
        rep["seam_c"]["passed"] = bool(rep["seam_c"]["eval_abs_diff"] <= TOL and
                                       rep["seam_c"]["single_colour_abs_diff"] <= TOL)
        print("seam c", rep["seam_c"], flush=True)
        if not rep["seam_c"]["passed"]:
            raise SystemExit("seam c failed")

    # ---- tables
    tdir = os.path.join(B.SCRATCH, f"tables_{a.pct}")
    paths, trec = build_tables(a.pct, B.SCRATCH, tdir)
    rep["tables"] = trec
    print("tables", {k: v for k, v in trec.items() if k != "files"}, flush=True)
    if not (trec["V0_equal_to_p04_opp_build_tables"] and trec["registered_tables_equal_inner_report"]):
        raise SystemExit("table seam failed")

    # ---- replay
    keys = inner["keys"]
    ll, nm, an, rows, secs, lab_eq = replay_inner(paths, keys, a.limit, a.workers)
    r = np.array(rows)
    o = {v: ll[:, j].sum(1) for j, v in enumerate(COLS)}            # phase sum as p04_opp.side_vectors
    reg_ref = d["meta"]["O4_raw"]["O4"]
    nm_ref = None
    with np.load(os.path.join(B.REPO, "results", "prereg04", "opp", "inner_o4.npz"), allow_pickle=False) as z:
        nm_ref = z["n_moves"][:, 1]
    seam = {"rows": len(rows), "seconds": secs, "anomalies": int(an.sum()), "labels_equal_inner_keys": lab_eq,
            "reg_max_abs_diff": float(np.max(np.abs(o["REG"][r] - reg_ref[r]))),
            "reg_bit_identical": bool(np.array_equal(o["REG"][r], reg_ref[r])),
            "opponent_moves_per_phase_equal": bool(np.array_equal(nm[r], nm_ref[r])),
            "opponent_moves": int(nm[r].sum())}
    seam["passed"] = bool(seam["reg_max_abs_diff"] <= TOL and seam["opponent_moves_per_phase_equal"]
                          and seam["anomalies"] == 0 and lab_eq)
    rep["seam_replay"] = seam
    print("seam replay", seam, flush=True)
    if not seam["passed"]:
        raise SystemExit("replay seam failed")
    lab = keys["label"]
    rep["per_game_top1_accuracy_record_only"] = {v: float(np.mean(o[v][r].argmax(1) == lab[r])) for v in COLS}
    rep["per_game_top1_accuracy_record_only"]["registered_O4_inner_report"] = json.load(
        open(INNER_REPORT, encoding="utf-8"))["per_game_argmax_accuracy_record_only"]["o4"]
    out_npz = os.path.join(B.SCRATCH, f"o_inner_{a.pct}.npz")
    np.savez(out_npz, **{f"o_{v}": o[v] for v in COLS}, n_moves_opp=nm, label=lab)
    rep["o_inner_file"] = {"path": out_npz, "sha256": B.sha256_file(out_npz)}
    if a.limit:
        rep["status"] = f"smoke (first {a.limit} inner games): seams only"
        with open(os.path.join(B.SCRATCH, f"smoke_b2_report_{a.pct}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
        print(rep["status"], f"({time.time() - T0:.0f}s)")
        return

    # ---- fusion: Q02 with O4 replaced by Vi
    for v in B.VARIANTS:
        d["lp"][v] = U.p4.log_softmax(o[v])
        run(["OP", v, "SWA", "SWB"], v)
    rep["scores"] = {k: {"eval_rank_balanced": x["m"]["rank_balanced"], "eval_question_mean": x["m"]["question_mean"],
                         "single_colour_rank_balanced": x["mm"]["rank_balanced"],
                         "official_shape_weighted": official_shape(x["scm"], ms)[0],
                         "per_level": x["m"]["per_level"],
                         "per_n_bin": {b: x["m"]["per_n_bin"][b]["rank_balanced"] for b in x["m"]["per_n_bin"]},
                         "fold_weight_of_opponent_term": [th[3] for th in x["theta"]]}
                     for k, x in R.items()}
    comps = [("V1 - V0", "V1", "V0"), ("V2 - V0", "V2", "V0"), ("V3 - V0", "V3", "V0"),
             ("V0 - Q02", "V0", "Q02"), ("V1 - Q02", "V1", "Q02"), ("V2 - Q02", "V2", "Q02"), ("V3 - Q02", "V3", "Q02")]
    rep["comparisons"] = {nm_: compare(nm_, x, y, R, inner) for nm_, x, y in comps}
    # ---- the success signal fixed before running (b2_variants.json)
    enter, direct = [], []
    for v in ("V1", "V2", "V3"):
        c0 = rep["comparisons"][f"{v} - V0"]
        cq = rep["comparisons"][f"{v} - Q02"]
        if (c0["eval_rank_balanced_delta"] >= 0.005 and c0["pboot_ci95"][0] > 0
                and c0["single_colour"]["black"]["rank_balanced"] > 0
                and c0["single_colour"]["white"]["rank_balanced"] > 0):
            enter.append(v)
        if cq["eval_rank_balanced_delta"] >= 0.008 and cq["pboot_ci95"][0] > 0:
            direct.append(v)
    rep["success_signal"] = {
        "enter_full_confirmation": enter, "direct_pass_25pct_vs_Q02": direct,
        "verdict": ("direct pass" if direct else "enter full confirmation" if enter else "B2 stops"),
        "rule": "fixed in b2_variants.json before running"}
    np.savez(os.path.join(B.SCRATCH, f"oof_{a.pct}.npz"),
             **{f"P_{k}": x["P"] for k, x in R.items()}, **{f"Pm_{k}": x["Pm"] for k, x in R.items()})
    rep["environment"] = U.p3.environment()
    rep["environment"]["blas_threads_env"] = {v_: os.environ.get(v_) for v_ in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    rep["git_head"] = U.p3.git_head()
    rep["seconds_total"] = round(time.time() - T0, 1)
    rep["status"] = "ok"
    name = "b2_report.json" if a.pct == "25" else f"b2_report_{a.pct}.json"
    with open(os.path.join(B.HERE, name), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("verdict", rep["success_signal"], f"({time.time() - T0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
