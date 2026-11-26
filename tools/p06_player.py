"""PREREG_06 (competition 1): design steps and the local-validation apply (docs/PREREG_06_PLAYER_STYLE.md,
results/prereg06/design/spec.md sections 0, 5 and 6). Run from the repository root with the CPU environment and
PYTHONDONTWRITEBYTECODE=1:

    PY tools/p06_player.py leakcheck    S0  fit vs the two validation game files (SHA-1 of the SGF text only)
    PY tools/p06_player.py build        S1  per-game features of the fit players with >= 105 games
    PY tools/p06_player.py lda          S2  LDA_B / LDA_W -> results/prereg06/player/lda_{B,W}.npz
    PY tools/p06_player.py selfcheck    S3  port check against the exploration functions (PREREG_06 2.8)
    (S4 = tools/p06_cnn_fit.py on the GPU)
    PY tools/p06_player.py tune         S5  PREREG_06 2.5 (E0 cap, V1)
    PY tools/p06_player.py switch       S6  PREREG_06 2.6 (k*, r*_X, usability, veto statistics)
    PY tools/p06_player.py dmix         S7  PREREG_06 2.8 D-mix
    PY tools/p06_player.py freeze       S8  results/prereg06/player/frozen_config.json
    PY tools/p06_player.py apply        S9  spec section 6 (only after the freeze commit)

Discipline (spec section 0):
  * an import-time guard makes every open / listdir / scandir / stat under data/local raise; only 'leakcheck' and
    'apply' open the two game files player_val_candidates.csv / player_val_questions.csv (never anything else
    there); any path whose name contains the answer-file marker is refused;
  * every design step reads fit players only (tools/c10data.load_rows(part="fit", local_dir=None), counts
    checked against EXPECTED);
  * tools/score.py is never imported or run; nothing under results/ is imported (selfcheck imports the
    exploration modules with bytecode writing off and checks that no __pycache__ appeared);
  * every command writes a JSON report with input SHA-256s, the p06 tools' SHA-256 (CRLF folded), git HEAD,
    versions, seconds and the mode. Real mode requires the p06 tools to be committed and clean; with
    C10_P06_PRECOMMIT=1 a design step may run on uncommitted tools: the report then says precommit: true and
    records the tools' SHA-256, and 'freeze' (real) accepts it only when those SHA-256 equal the committed tools.
    'apply' never runs in precommit mode;
  * at most 6 worker processes, each with OMP / MKL / OPENBLAS threads = --threads (default 4);
  * --smoke: synthetic inputs (--train-dir / --split-csv / --feat-dir / --design-dir / --player-dir all outside the
    repository), counts not checked; used by the tests.
"""
import argparse
import csv
import hashlib
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p06_style as st  # noqa: E402

st.install_guard()

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import p02_moves  # noqa: E402
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import p06_sim as sim  # noqa: E402
from c10data import EXPECTED, LEVELS, load_rows  # noqa: E402

REPO = st.REPO
DESIGN = os.path.join(REPO, "results", "prereg06", "design")
PLAYER = os.path.join(REPO, "results", "prereg06", "player")
FEATS = os.path.join(REPO, "data", "prereg06", "feats")
CNN_NPZ = os.path.join(REPO, "data", "prereg06", "cnn_fit", "black_ge105.npz")
CNN_REPORT = os.path.join(DESIGN, "cnn_fit_report.json")
EXPLORE = os.path.join(REPO, "results", "prereg06", "explore", "c1_headroom")
EXPLORE_CACHE = os.path.join(REPO, "data", "prereg06", "explore_cache")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
LOCAL = os.path.join(REPO, "data", "local")
P02_PLAYER = os.path.join(REPO, "results", "prereg02", "player")
MAX_WORKERS = 6

# registered numbers (PREREG_06; spec section 3)
EXPECT_SETS = {"black_ge105": (1082, 232165), "black_ge120": (926, 214767), "white_ge105": (1077, 229761),
               "both_ge105_players": 1937}
TUNE_SEEDS, TUNE_EVAL_SEEDS = tuple(range(300, 320)), tuple(range(400, 420))
SW_SEL_SEEDS, SW_EVAL_SEEDS = tuple(range(500, 530)), tuple(range(600, 620))
DMIX_SEEDS = tuple(range(700, 710))
SELFCHECK_TERM_SEEDS, SELFCHECK_SCORE_SEEDS = tuple(range(5)), tuple(range(20))
SELFCHECK_EXPECT = {"v2_nopost": 0.9819, "v2_csls10": 0.9869}
E0_CAP = 0.50
SELECT_TOL = 0.003
GATE_MIN = 0.005
TUNE_K = 10
GRID_L = (0.5, 1, 1.33, 2, 2.66, 4)
GRID_J = (0, 0.25, 0.5, 1)
GRID_S = (0, 0.25, 0.5, 1)
V1_MIN_GAIN = 0.003
FEAS_MEAN = -0.0005
FEAS_POOL = -0.005
O2O_OPEN_MIN = 0.8
REPORTS = ("leakcheck_report.json", "build_report.json", "lda_report.json", "selfcheck_report.json",
           "cnn_fit_report.json", "tune_report.json", "switch_report.json", "dmix_report.json")


class Stop(SystemExit):
    pass


# ================================================================ provenance
def mode_of(args):
    return "smoke" if getattr(args, "smoke", False) else "real"


def check_tools(args, allow_precommit=True):
    """Real mode: p06 tools committed and clean, else C10_P06_PRECOMMIT=1 (design steps only)."""
    ts = st.tools_state()
    if args.smoke:
        return ts, False
    if all(v["committed_and_clean"] for v in ts.values()):
        return ts, False
    if allow_precommit and os.environ.get("C10_P06_PRECOMMIT") == "1":
        return ts, True
    raise Stop("refused (exit 2): the p06 tools are not committed and clean: %s"
               % [k for k, v in ts.items() if not v["committed_and_clean"]])


def meta(args, cmd, t0, tools, precommit, inputs=None, extra_env=None):
    env = st.env_versions()
    if extra_env:
        env.update(extra_env)
    return {"command": cmd, "mode": mode_of(args), "precommit": bool(precommit), "created": st.now_iso(),
            "git_head": st.git("rev-parse", "HEAD"), "tools": tools, "env": env,
            "inputs": inputs or {}, "seconds": round(time.time() - t0, 1)}


def out_check(args, *paths):
    """Smoke runs never write into the repository; real runs never overwrite a report."""
    for p in paths:
        if args.smoke and st.inside_repo(p):
            raise Stop("--smoke refused: %s is inside the repository" % p)


def no_overwrite(path):
    if os.path.exists(path):
        raise Stop("refused (exit 2): %s exists; real design reports are never overwritten" % path)


def report_committed(path):
    return st.file_committed_clean(path)


def thread_env(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)
    return {k: str(n) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


def paths(args):
    return (args.design_dir or DESIGN, args.player_dir or PLAYER, args.feat_dir or FEATS)


def train_inputs(args):
    out = {"split_csv": st.sha256_file(args.split_csv)}
    for lv in LEVELS:
        f = os.path.join(args.train_dir, "train_%s.csv" % lv)
        if os.path.exists(f):
            out["train_%s.csv" % lv] = st.sha256_file(f)
    return out


def fit_rows(args):
    return load_rows(args.train_dir, args.split_csv, part="fit", local_dir=None, check_counts=not args.smoke)


# ================================================================ S0 leakcheck
def sha1_games(path):
    """SHA-1 digests of the SGF texts of one validation game file (sgf_content or sgf_1..num_games)."""
    csv.field_size_limit(10 ** 8)
    out = set()
    with st.allow_local():
        with open(st.refuse_marker(path), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "sgf_content" in r:
                    out.add(hashlib.sha1(r["sgf_content"].encode("utf-8")).digest())
                else:
                    for i in range(1, int(r["num_games"]) + 1):
                        out.add(hashlib.sha1(r["sgf_%d" % i].encode("utf-8")).digest())
    return out


def cmd_leakcheck(args):
    t0 = time.time()
    design, _pl, _fe = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "leakcheck_report.json")
    out_check(args, out)
    if not args.smoke:
        no_overwrite(out)
    rows = fit_rows(args)
    files = [os.path.join(args.local_dir, n) for n in st.LOCAL_ALLOWED]
    hashes = set()
    shas = {}
    for p in files:
        hashes |= sha1_games(p)
        with st.allow_local():
            shas[os.path.basename(p)] = st.sha256_file(p)
    overlap = sum(1 for r in rows if hashlib.sha1(r["sgf_content"].encode("utf-8")).digest() in hashes)
    rep = {"what": "PREREG_06 S0: fit games vs the two validation game files (SHA-1 of the SGF text; "
                   "VALIDATION rev 2 leak check)",
           "fit": {"players": len({r["player_id"] for r in rows}), "games": len(rows),
                   "expected": EXPECTED["fit"]},
           "validation_game_files": {"files": list(st.LOCAL_ALLOWED), "sha256": shas, "distinct_games": len(hashes)},
           "overlap": overlap, "passed": overlap == 0 and (args.smoke or {"players": len({r["player_id"] for r in rows}),
                                                                           "games": len(rows)} == EXPECTED["fit"]),
           "note": "only the two game files of competition 1 are read (rank_val_questions.csv, which "
                   "c10data._sgf_hashes_in_local would also read, is not opened)"}
    rep.update(meta(args, "leakcheck", t0, tools, pre, inputs=train_inputs(args)))
    st.write_json(out, rep)
    print("leakcheck: overlap %d, passed %s -> %s" % (overlap, rep["passed"], out))
    return 0 if rep["passed"] else 1


# ================================================================ S1 build
def _feat_chunk(items):
    """items: [(sgf, colour)] -> per-game arrays and three local CSR blocks (resp, jo, oraw)."""
    n = len(items)
    S2 = np.full((n, 66), np.nan)
    S34 = np.full((n, 34), np.nan)
    alt = np.zeros(n, bool)
    nm = np.zeros(n, np.int32)
    blocks = {k: ({}, [], [], [0]) for k in ("resp", "jo", "oraw")}

    def add(name, toks):
        voc, dat, ind, ptr = blocks[name]
        for t, c in Counter(toks).items():
            ind.append(voc.setdefault(t, len(voc)))
            dat.append(c)
        ptr.append(len(ind))

    for i, (sgf, colour) in enumerate(items):
        mv = st.parse_all(sgf)
        nm[i] = len(mv)
        add("oraw", po.game_tokens(colour, p02_moves.parse_moves(sgf), "raw"))
        if st.alternates(mv):
            alt[i] = True
            S2[i] = st.style2(mv, colour)
            S34[i] = st.style34(mv, colour)
            add("resp", st.resp_tokens(mv, colour))
            add("jo", st.jo_tokens(mv, colour))
        else:
            add("resp", [])
            add("jo", [])
    return S2, S34, alt, nm, {k: (list(v[0]), np.asarray(v[1], np.float64), np.asarray(v[2], np.int64),
                                  np.asarray(v[3], np.int64)) for k, v in blocks.items()}


def merge_blocks(parts):
    """Local CSR blocks -> one CSR with the vocabulary sorted."""
    gv = {}
    datas, inds, ptrs, off = [], [], [np.zeros(1, np.int64)], 0
    for voc, dat, ind, ptr in parts:
        m = np.asarray([gv.setdefault(t, len(gv)) for t in voc], np.int64)
        datas.append(dat)
        inds.append(m[ind] if len(ind) else ind)
        ptrs.append(ptr[1:] + off)
        off += len(ind)
    vocab = sorted(gv)
    perm = np.empty(len(gv), np.int64)
    for new, t in enumerate(vocab):
        perm[gv[t]] = new
    ind = np.concatenate(inds) if inds else np.zeros(0, np.int64)
    M = sp.csr_matrix((np.concatenate(datas) if datas else np.zeros(0), perm[ind] if len(ind) else ind,
                       np.concatenate(ptrs)), shape=(len(np.concatenate(ptrs)) - 1, len(vocab)))
    M.sort_indices()
    return M, vocab


def _init_worker(threads):
    thread_env(threads)
    sys.dont_write_bytecode = True


def cmd_build(args):
    t0 = time.time()
    design, _pl, feats = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "build_report.json")
    out_check(args, out, feats)
    if not args.smoke:
        no_overwrite(out)
    lk = os.path.join(design, "leakcheck_report.json")
    leak = None
    if os.path.exists(lk):
        leak = st.read_json(lk)
        if not leak.get("passed") or leak.get("overlap") != 0:
            raise Stop("refused (exit 2): leakcheck_report.json did not pass")
    if not args.smoke:
        if leak is None or not report_committed(lk):
            if not pre:
                raise Stop("refused (exit 2): leakcheck_report.json must be committed before build")
    rows = fit_rows(args)
    n_all = Counter(r["player_id"] for r in rows)
    n_b = Counter(r["player_id"] for r in rows if r["color"] == "B")
    n_w = Counter(r["player_id"] for r in rows if r["color"] == "W")
    keep = {p for p, n in n_all.items() if n >= sim.DMIX_MIN}
    sel = [r for r in rows if r["player_id"] in keep]
    del rows
    items = [(r["sgf_content"], r["color"]) for r in sel]
    chunk = max(1, min(2000, (len(items) + args.workers - 1) // args.workers))
    te = thread_env(1)                  # set before the workers are spawned (read when numpy is imported)
    with Pool(args.workers, initializer=_init_worker, initargs=(1,)) as pool:
        res = pool.map(_feat_chunk, [items[i:i + chunk] for i in range(0, len(items), chunk)])
    S2 = np.vstack([r[0] for r in res])
    S34 = np.vstack([r[1] for r in res])
    alt = np.concatenate([r[2] for r in res])
    nm = np.concatenate([r[3] for r in res])
    os.makedirs(feats, exist_ok=True)
    mats = {}
    for name in ("resp", "jo", "oraw"):
        M, vocab = merge_blocks([r[4][name] for r in res])
        sim.save_csr(os.path.join(feats, name), M, vocab)
        mats[name] = (M.shape, M.nnz, len(vocab))
    np.save(os.path.join(feats, "s2.npy"), S2)
    np.save(os.path.join(feats, "s34.npy"), S34)
    colour = np.array([r["color"] for r in sel])
    np.savez(os.path.join(feats, "meta.npz"), rank=np.array([r["rank"] for r in sel]),
             game_id=np.array([r["game_id"] for r in sel]), player_id=np.array([r["player_id"] for r in sel]),
             color=colour, alternating=alt, n_moves=nm)
    pids = np.array([r["player_id"] for r in sel])

    def set_counts(colour_, thr):
        ps = {p for p in keep if (n_b if colour_ == "B" else n_w)[p] >= thr}
        m = np.isin(pids, list(ps)) & (colour == colour_)
        return {"players": len(ps), "games": int(m.sum()), "not_alternating": int((m & ~alt).sum()),
                "no_move": int((m & (nm == 0)).sum())}

    sets = {"black_ge105": set_counts("B", 105), "black_ge120": set_counts("B", 120),
            "white_ge105": set_counts("W", 105),
            "both_ge105": {"players": len(keep), "games": len(sel), "not_alternating": int((~alt).sum()),
                           "not_alternating_black": int(((colour == "B") & ~alt).sum()),
                           "not_alternating_white": int(((colour == "W") & ~alt).sum()),
                           "no_move": int((nm == 0).sum())}}
    checks = {k: [sets[k]["players"], sets[k]["games"]] == list(v) for k, v in EXPECT_SETS.items()
              if k != "both_ge105_players"}
    checks["both_ge105_players"] = sets["both_ge105"]["players"] == EXPECT_SETS["both_ge105_players"]
    passed = args.smoke or all(checks.values())
    level = {}
    for r in sel:
        level[r["player_id"]] = r["rank"]
    by_level = {lv: sum(1 for p in keep if n_b[p] >= 120 and level[p] == lv) for lv in LEVELS}
    files = {}
    for root, _d, fs in os.walk(feats):
        for f in sorted(fs):
            p = os.path.join(root, f)
            files[os.path.relpath(p, feats).replace(os.sep, "/")] = st.sha256_file(p)
    rep = {"what": "PREREG_06 S1: per-game features (O_raw tokens, JO tokens, resp tokens, S2, style34) of every "
                   "game of the fit players with >= 105 games (both colours), file order",
           "sets": sets, "expected": EXPECT_SETS, "checks": checks, "passed": passed,
           "black_ge120_by_level": by_level,
           "matrices": {k: {"shape": list(v[0]), "nnz": v[1], "vocab": v[2]} for k, v in mats.items()},
           "leakcheck_report": {"exists": leak is not None,
                                "sha256": st.sha256_lf(lk) if leak is not None else None,
                                "committed": report_committed(lk) if leak is not None else False},
           "feat_dir": st.rel(feats) if not args.smoke else feats, "files_sha256": files,
           "workers": args.workers, "thread_env": te}
    rep.update(meta(args, "build", t0, tools, pre, inputs=train_inputs(args)))
    st.write_json(out, rep)
    print("build: %d games (%d players), not alternating %d; checks %s -> %s"
          % (len(sel), len(keep), sets["both_ge105"]["not_alternating"], checks, out))
    return 0 if passed else 1


# ================================================================ feature store verification
def feat_problems(build_rep, lda_rep, feats, need_lda=True):
    """Files of data/prereg06/feats whose SHA-256 differs from the one recorded by build (and lda)."""
    want = dict(build_rep.get("files_sha256") or {})
    if need_lda:
        for c in "BW":
            cc = (lda_rep or {}).get("colours", {}).get(c, {})
            for key, name in (("ldain_sha256", "ldain_%s.npy"), ("ldain_rows_sha256", "ldain_%s_rows.npy"),
                              ("ldain_cols_sha256", "ldain_%s_cols.npy")):
                want[name % c] = cc.get(key)
    bad = []
    for name, sha in sorted(want.items()):
        p = os.path.join(feats, *name.split("/"))
        if sha is None or not os.path.exists(p) or st.sha256_file(p) != sha:
            bad.append(name)
    return bad


def verify_feats(args, design, feats, need_lda=True):
    """Real mode: the feature store must be exactly what build (and lda) recorded; returns the check record."""
    if args.smoke:
        return {"checked": False, "note": "smoke"}
    bp, lp = os.path.join(design, "build_report.json"), os.path.join(design, "lda_report.json")
    if not os.path.exists(bp) or (need_lda and not os.path.exists(lp)):
        raise Stop("refused (exit 2): build_report.json / lda_report.json missing; cannot verify the feature store")
    bad = feat_problems(st.read_json(bp), st.read_json(lp) if need_lda else None, feats, need_lda)
    if bad:
        raise Stop("refused (exit 2): feature store differs from the build / lda reports: %s" % bad[:10])
    return {"checked": True, "build_report_sha256": st.sha256_lf(bp),
            "lda_report_sha256": st.sha256_lf(lp) if need_lda else None}


# ================================================================ S2 lda
def lda_training(store, colour, thr=105):
    """Rows of the colour's LDA training set: players with >= thr games of that colour (counted before the
    alternation check), their alternating games of that colour, file order."""
    col = store.color
    cnt = Counter(store.pid[col == colour].tolist())
    ps = {p for p, n in cnt.items() if n >= thr}
    m = np.isin(store.pid, list(ps)) & (col == colour)
    rows = np.flatnonzero(m & store.alt)
    return rows, len(ps), int(m.sum()), int((m & ~store.alt).sum())


def cmd_lda(args):
    t0 = time.time()
    design, player, feats = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "lda_report.json")
    out_check(args, out, player, feats)
    if not args.smoke:
        no_overwrite(out)
        for c in "BW":
            no_overwrite(os.path.join(player, "lda_%s.npz" % c))
    feat_check = verify_feats(args, design, feats, need_lda=False)
    store = sim.Store(feats, load=("s2",))
    R = store.csr("resp")
    rvocab = R.vocab
    os.makedirs(player, exist_ok=True)
    rep = {"what": "PREREG_06 S2: LDA64 per colour on the fit players with >= 105 games of that colour",
           "colours": {}}
    ok = True
    for c in "BW":
        rows, n_pl, n_g, n_bad = lda_training(store, c)
        Rc = R[rows]
        used = np.flatnonzero(np.asarray(Rc.sum(0)).ravel() > 0)
        vocab = [rvocab[j] for j in used]
        X = st.ldain_matrix(np.asarray(store.dense("s2")[rows]), Rc, used)
        m = st.lda_fit(X, store.pid[rows])
        path = os.path.join(player, "lda_%s.npz" % c)
        np.savez(path, W=m["W"], ev=m["ev"], mu=m["mu"], sd=m["sd"], vocab=np.array(vocab))
        np.save(os.path.join(feats, "ldain_%s.npy" % c), X)
        np.save(os.path.join(feats, "ldain_%s_rows.npy" % c), rows)
        np.save(os.path.join(feats, "ldain_%s_cols.npy" % c), used)
        finite = bool(np.isfinite(m["W"]).all() and np.isfinite(m["ev"]).all())
        dec = bool(np.all(np.diff(m["ev"]) <= 0))
        good = finite and dec and m["W"].shape[1] == st.LDA_DIM
        ok &= good
        rep["colours"][c] = {"players": n_pl, "games_of_colour": n_g, "not_alternating_excluded": n_bad,
                             "training_games": int(len(rows)), "vocab_size": len(vocab),
                             "dims": int(X.shape[1]), "directions": int(m["W"].shape[1]),
                             "eigenvalues_first10": [round(float(x), 6) for x in m["ev"][:10]],
                             "eigenvalue_64": float(m["ev"][-1]), "finite": finite, "decreasing": dec,
                             "passed": good, "file": st.rel(path) if not args.smoke else path,
                             "sha256": st.sha256_file(path),
                             "ldain_sha256": st.sha256_file(os.path.join(feats, "ldain_%s.npy" % c)),
                             "ldain_rows_sha256": st.sha256_file(os.path.join(feats, "ldain_%s_rows.npy" % c)),
                             "ldain_cols_sha256": st.sha256_file(os.path.join(feats, "ldain_%s_cols.npy" % c))}
    rep["passed"] = ok
    rep["feature_store_check"] = feat_check
    rep["inputs_note"] = "feature store of S1 (build_report.json files_sha256, verified before use)"
    bp = os.path.join(design, "build_report.json")
    rep.update(meta(args, "lda", t0, tools, pre,
                    inputs={"build_report.json": st.sha256_lf(bp) if os.path.exists(bp) else None}))
    st.write_json(out, rep)
    print("lda: B %s, W %s -> %s" % (rep["colours"]["B"]["passed"], rep["colours"]["W"]["passed"], out))
    return 0 if ok else 1


# ================================================================ simulation workers
_W = {}


def _sim_init(feat_dir, cnn_path, threads, colours):
    thread_env(threads)
    sys.dont_write_bytecode = True
    store = sim.Store(feat_dir)
    _W["store"] = store
    _W["lda"] = {c: sim.LdaData(feat_dir, c, store) for c in colours}
    _W["cnn"] = sim.Cnn(cnn_path, store) if cnn_path else None


def _lda_models(rec, colours=("B",)):
    store = _W["store"]
    return {c: _W["lda"][c].fit_excluding(store, set(rec["pool"])) for c in colours}


def _tune_seed(seed):
    t = time.time()
    store, cnn = _W["store"], _W["cnn"]
    rec = sim.make_pool(store.black_cache(), seed, "o2o")
    models = _lda_models(rec)
    T = sim.pool_terms(store, rec, _W["lda"], models, cnn)
    own = rec["owner"]
    out = {"seed": seed}
    src = ["PA"] + (["E0"] if cnn is not None else [])
    for s in src:
        out["%s_alone" % s] = st.score(T[s][0] if s == "E0" else st.zrows(T[s][0]), own)
        out["%s_q04" % s] = st.score(sim.combo_F(T, s, None), own)
        for name, w in (("V2", sim.V2_W), ("V3", sim.V3_W), ("V0", None)):
            out["%s_%s" % (s, name)] = st.score(st.csls(sim.combo_F(T, s, w), TUNE_K), own)
    if cnn is not None:
        out["E0_alone"] = st.score(st.zscore_rows(*T["E0"]), own)
        out["grid"] = {}
        for wl in GRID_L:
            for wj in GRID_J:
                for ws in GRID_S:
                    out["grid"]["%g|%g|%g" % (wl, wj, ws)] = st.score(st.csls(sim.combo_F(T, "E0", (wl, wj, ws)),
                                                                              TUNE_K), own)
    out["lda_train"] = [models["B"]["n_players"], models["B"]["n_games"]]
    out["seconds"] = round(time.time() - t, 1)
    return out


def _switch_seed(job):
    seed, combos, sources = job
    t = time.time()
    store, cnn = _W["store"], _W["cnn"]
    cache = store.black_cache()
    rec0 = sim.make_pool(cache, seed, "o2o")
    models = _lda_models(rec0)
    want = ("O", "JO", "S2", "L", "PA") + (("E0",) if "E0" in sources else ())
    pseudo = {}
    prs, _ids = sim.sr_pool(store, rec0, rounds=(0, 1))
    for pr in prs:
        Tp = sim.pool_terms(store, pr, _W["lda"], models, cnn if "E0" in sources else None, want=want)
        for name, w in combos:
            for s in sources:
                pseudo.setdefault("%s|%s" % (name, s), []).append(sim.forced_cost(sim.combo_F(Tp, s, w)))
            pseudo.setdefault("%s|open" % name, []).append(sim.forced_cost(sim.combo_F(Tp, None, w, open_only=True)))
    rows = []
    for scen in sim.STRUCTURES:
        rec = rec0 if scen == "o2o" else sim.make_pool(cache, seed, scen)
        T = sim.pool_terms(store, rec, _W["lda"], models, cnn if "E0" in sources else None, want=want)
        row = {"seed": seed, "scen": scen, "n_distinct": int(len(set(rec["owner"].tolist()))),
               "n_repairs": rec["n_repairs"], "res": {}}
        for name, w in combos:
            for s in sources:
                key = "%s|%s" % (name, s)
                r = sim.pool_scores(sim.combo_F(T, s, w), rec["owner"])
                r["r_forced_cost"] = r["forced_cost"] / max(float(np.mean(pseudo[key])), 1e-9)
                row["res"][key] = r
            key = "%s|open" % name
            F = sim.combo_F(T, None, w, open_only=True)
            r = {"forced_cost": sim.forced_cost(F)}
            r["r_forced_cost"] = r["forced_cost"] / max(float(np.mean(pseudo[key])), 1e-9)
            row["res"][key] = r
        rows.append(row)
    return {"seed": seed, "rows": rows, "pseudo": pseudo, "seconds": round(time.time() - t, 1)}


def _dmix_seed(seed):
    t = time.time()
    store = _W["store"]
    rec = sim.dmix_pool(store, seed)
    models = _lda_models(rec, ("B", "W"))
    own = rec["owner"]
    T = sim.pool_terms(store, rec, _W["lda"], models, None, want=("O", "JO", "S2", "L"))
    Tb = sim.pool_terms(store, rec, _W["lda"], models, None, want=("JO", "S2", "L"), colours=("B",))
    Tb["O"] = T["O"]
    base = st.score(st.csls(st.fuse([T["O"]], [1.0]), TUNE_K), own)
    main = st.score(st.csls(sim.combo_F(T, None, sim.V2_W, open_only=True), TUNE_K), own)
    blk = st.score(st.csls(sim.combo_F(Tb, None, sim.V2_W, open_only=True), TUNE_K), own)
    col = store.color
    cg = [r for g in rec["cand_rows"] for r in g]
    qg = [r for g in rec["q_rows"] for r in g]
    return {"seed": seed, "base_O_csls10": base, "main_O_v2_csls10": main, "blackonly_O_v2_csls10": blk,
            "candidate_black_share": float(np.mean(col[cg] == "B")), "question_black_share": float(np.mean(col[qg] == "B")),
            "not_alternating_games": int((~store.alt[cg + qg]).sum()),
            "lda_train": {c: [models[c]["n_players"], models[c]["n_games"]] for c in models},
            "seconds": round(time.time() - t, 1)}


def run_pool(args, fn, jobs, colours=("B",), cnn=True):
    _d, _p, feats = paths(args)
    cnn_path = (args.cnn_npz or CNN_NPZ) if cnn else None
    if cnn_path and not os.path.exists(cnn_path):
        cnn_path = None
    w = min(args.workers, MAX_WORKERS)
    # the BLAS thread count is read when numpy is imported: set it before the workers are spawned
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    thread_env(args.threads)
    try:
        pool = Pool(w, initializer=_sim_init, initargs=(feats, cnn_path, args.threads, colours))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    with pool:
        out = []
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            out.append(r)
            print("  seed %s done (%s s) %d/%d" % (r["seed"], r["seconds"], len(out), len(jobs)), flush=True)
    return sorted(out, key=lambda r: r["seed"])


def ci4(x):
    return [None if v is None else round(v, 5) for v in st.mean_ci(x)]


# ================================================================ S5 tune
def cnn_available(args):
    """(usable, reason): the fit-player CNN embedding exists, its report passed and records the same file."""
    rp = args.cnn_report or CNN_REPORT
    npz = args.cnn_npz or CNN_NPZ
    if not os.path.exists(rp):
        return False, "cnn_fit_report.json missing (K-CNN fallback: proxy A only)"
    r = st.read_json(rp)
    if r.get("mode") != ("smoke" if args.smoke else "real"):
        return False, "cnn_fit_report.json mode %s" % r.get("mode")
    if not os.path.exists(npz) or st.sha256_file(npz) != r["output"]["sha256"]:
        return False, "black_ge105.npz missing or its SHA-256 differs from cnn_fit_report.json"
    if r["data"]["games"] != EXPECT_SETS["black_ge105"][1] and not args.smoke:
        return False, "cnn_fit_report.json covers %s games" % r["data"]["games"]
    return True, "cnn_fit_report.json sha256 %s" % st.sha256_lf(rp)


def v1_decision(tune_means, eval_means, e0_mean, cnn_ok):
    """PREREG_06 2.5. tune_means / eval_means: {grid key: mean}; returns dict(v1 weights or None, reasons)."""
    v2k = "%g|%g|%g" % sim.V2_W
    reasons = []
    out = {"e0_alone_tune_mean": e0_mean, "e0_cap": E0_CAP}
    if not cnn_ok:
        out.update(v1=None, reasons=["no fit-player CNN embedding"], cnn_representative=None)
        return out
    rep = e0_mean <= E0_CAP
    out["cnn_representative"] = rep
    keys = list(tune_means)
    best = max(keys, key=lambda k: (tune_means[k], -keys.index(k)))
    out.update(best=best, best_tune=tune_means[best], v2_tune=tune_means[v2k],
               best_minus_v2_tune=tune_means[best] - tune_means[v2k],
               best_eval=eval_means[best], v2_eval=eval_means[v2k])
    c1 = rep
    c2 = tune_means[best] - tune_means[v2k] >= V1_MIN_GAIN - 1e-12
    c3 = eval_means[best] >= eval_means[v2k]
    out["conditions"] = {"cnn_representative": c1, "tune_gain_ge_0.003": c2, "eval_not_below_v2": c3}
    if not c1:
        reasons.append("E0 alone on the tuning pools %.4f > 0.50: CNN not representative" % e0_mean)
    if not c2:
        reasons.append("best grid point beats V2 by %.4f < 0.003 on the tuning pools"
                       % (tune_means[best] - tune_means[v2k]))
    if not c3:
        reasons.append("best grid point below V2 on the evaluation pools")
    out["v1"] = [float(x) for x in best.split("|")] if not reasons else None
    out["reasons"] = reasons
    return out


def cmd_tune(args):
    t0 = time.time()
    design, _pl, _fe = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "tune_report.json")
    out_check(args, out)
    if not args.smoke:
        no_overwrite(out)
    feat_check = verify_feats(args, design, paths(args)[2])
    cnn_ok, why = cnn_available(args)
    args_cnn = cnn_ok
    seeds_t = args.seeds_tune or TUNE_SEEDS
    seeds_e = args.seeds_eval or TUNE_EVAL_SEEDS
    res = run_pool(args, _tune_seed, list(seeds_t) + list(seeds_e), cnn=args_cnn)
    T = [r for r in res if r["seed"] in seeds_t]
    E = [r for r in res if r["seed"] in seeds_e]
    rep = {"what": "PREREG_06 S5 (2.5): E0 cap, V1 weights; official-shaped o2o pools of fit players",
           "seeds": {"tune": list(seeds_t), "eval": list(seeds_e)}, "cnn": {"available": cnn_ok, "basis": why},
           "csls_k_during_tune": TUNE_K}
    if cnn_ok:
        e0 = float(np.mean([r["E0_alone"] for r in T]))
        tm = {k: float(np.mean([r["grid"][k] for r in T])) for k in T[0]["grid"]}
        em = {k: float(np.mean([r["grid"][k] for r in E])) for k in E[0]["grid"]}
        dec = v1_decision(tm, em, e0, True)
        rep["grid_tune_means"] = {k: round(v, 6) for k, v in tm.items()}
        rep["grid_eval_means"] = {k: round(v, 6) for k, v in em.items()}
    else:
        dec = v1_decision({}, {}, None, False)
    rep["decision"] = dec
    rec = {}
    for s in ["PA"] + (["E0"] if cnn_ok else []):
        d = {"alone_tune": ci4([r["%s_alone" % s] for r in T]), "alone_eval": ci4([r["%s_alone" % s] for r in E]),
             "q04_eval": ci4([r["%s_q04" % s] for r in E])}
        for name in ("V2", "V3", "V0"):
            d["%s_eval" % name] = ci4([r["%s_%s" % (s, name)] for r in E])
            d["%s_minus_q04_eval" % name] = ci4([r["%s_%s" % (s, name)] - r["%s_q04" % s] for r in E])
        if s == "E0" and dec.get("best"):
            b = dec["best"]
            d["best_grid_eval"] = ci4([r["grid"][b] for r in E])
            d["best_grid_minus_q04_eval"] = ci4([r["grid"][b] - r["E0_q04"] for r in E])
        rec[s] = d
    rep["recorded_only"] = rec
    rep["note"] = ("the CNN is in-sample for fit players: the E0 numbers above are not an estimate of the 06 gain "
                   "(PREREG_06 2.5); 'alone' for E0 is z(E0) without CSLS = Q00")
    rep["per_seed"] = [{k: v for k, v in r.items() if k != "grid"} for r in res]
    rep["feature_store_check"] = feat_check
    rep["workers"] = min(args.workers, MAX_WORKERS)
    rep["thread_env_per_worker"] = args.threads
    inputs = {"build_report.json": _sha_or_none(os.path.join(design, "build_report.json")),
              "lda_report.json": _sha_or_none(os.path.join(design, "lda_report.json")),
              "cnn_fit_report.json": _sha_or_none(args.cnn_report or CNN_REPORT)}
    rep.update(meta(args, "tune", t0, tools, pre, inputs=inputs))
    st.write_json(out, rep)
    print("tune: E0 alone tune %s; V1 %s (%s) -> %s" % (dec.get("e0_alone_tune_mean"), dec["v1"], dec["reasons"], out))
    return 0


def _sha_or_none(p):
    return st.sha256_lf(p) if p and os.path.exists(p) else None


# ================================================================ S6 switch
def k_star(rows, key):
    """First k of 10, 20, 40 with mean (CSLSk - no correction) >= 0 in every structure; None otherwise."""
    detail = {}
    for k in sim.CSLS_KS:
        means = {}
        for sc in sim.STRUCTURES:
            rr = [r for r in rows if r["scen"] == sc]
            if rr:
                means[sc] = float(np.mean([r["res"][key]["off_%d" % k] - r["res"][key]["off_none"] for r in rr]))
        detail[str(k)] = means
        if means and min(means.values()) >= 0:
            return k, detail
    return None, detail


def select_threshold(rows, key, off):
    """PREREG_06 2.6 'selection': candidates = observed r_forced_cost values on the selection pools; feasible =
    every structure's mean (on - off) >= -0.0005 and no non-o2o pool below -0.005; among feasible, the largest
    mean over the structures of the mean gain; ties -> the smallest threshold. None when nothing feasible gains."""
    r = np.array([x["res"][key]["r_forced_cost"] for x in rows])
    gain = np.array([x["res"][key]["on"] - x["res"][key][off] for x in rows])
    scen = np.array([x["scen"] for x in rows])
    scs = [s for s in sim.STRUCTURES if (scen == s).any()]
    best, best_obj, n_feas = None, None, 0
    for thr in np.unique(r):
        on = r <= thr
        d = np.where(on, gain, 0.0)
        means = [float(d[scen == s].mean()) for s in scs]
        worst = float(d[scen != "o2o"].min()) if (scen != "o2o").any() else 0.0
        if min(means) < FEAS_MEAN or worst < FEAS_POOL:
            continue
        n_feas += 1
        obj = float(np.mean(means))
        if best_obj is None or obj > best_obj + 1e-15:
            best, best_obj = float(thr), obj
    if best is not None and best_obj <= 0:
        return None, {"feasible_thresholds": n_feas, "best_objective": best_obj,
                      "note": "no feasible threshold with a positive mean gain"}
    return best, {"feasible_thresholds": n_feas, "best_objective": best_obj}


def r_star_of(thrs):
    """r*_X = the smaller of the per-source thresholds; no switch when a source has none (PREREG_06 2.6)."""
    return None if not thrs or any(t is None for t in thrs) else min(thrs)


def usable_check(rows, key, open_key, off, thr):
    """PREREG_06 2.6 'usability' on the evaluation pools for one F source (key) and the opening+v2 variant."""
    out = {}
    if thr is None:
        return False, {"reason": "no threshold"}
    o2o = [x for x in rows if x["scen"] == "o2o"]
    share = float(np.mean([x["res"][key]["r_forced_cost"] <= thr for x in o2o])) if o2o else 0.0
    out["o2o_open_share"] = share
    false_open = sum(1 for x in rows if x["scen"] in sim.NON_O2O and x["res"][key]["r_forced_cost"] <= thr)
    out["non_o2o_pools"] = sum(1 for x in rows if x["scen"] in sim.NON_O2O)
    out["non_o2o_opens"] = false_open
    near = {}
    ok_near = True
    for sc in sim.NEAR_O2O:
        rr = [x for x in rows if x["scen"] == sc]
        if not rr:
            continue
        d = np.array([(x["res"][key]["on"] - x["res"][key][off]) if x["res"][key]["r_forced_cost"] <= thr else 0.0
                      for x in rr])
        near[sc] = {"mean": float(d.mean()), "min": float(d.min()), "opens": int(sum(
            x["res"][key]["r_forced_cost"] <= thr for x in rr))}
        ok_near &= d.mean() >= FEAS_MEAN and d.min() >= FEAS_POOL
    out["near_o2o"] = near
    open_false = sum(1 for x in rows if x["scen"] in sim.NON_O2O and x["res"][open_key]["r_forced_cost"] <= thr)
    out["opening_v2_only_non_o2o_opens"] = open_false
    conds = {"o2o_open_share_ge_0.8": share >= O2O_OPEN_MIN, "no_false_open": false_open == 0,
             "near_o2o_ok": bool(ok_near), "opening_v2_only_no_false_open": open_false == 0}
    out["conditions"] = conds
    return all(conds.values()), out


def veto_stats(rows, key, off, thr):
    """c1d_veto_check-style: per structure d = on - off over all pools and over the pools where the gate opens."""
    def summ(ds):
        ds = np.asarray(ds, np.float64)
        if not len(ds):
            return {"n": 0}
        o = {"n": int(len(ds)), "mean": float(ds.mean()), "min": float(ds.min()),
             "p05": float(np.percentile(ds, 5)), "median": float(np.median(ds)), "max": float(ds.max())}
        for c in (0.0, -0.001, -0.002, -0.005, -0.01):
            o["share_d_lt_%g" % c] = float((ds < c - 1e-12).mean())
        return o
    out = {}
    for sc in sim.STRUCTURES:
        rr = [x for x in rows if x["scen"] == sc]
        d = [x["res"][key]["on"] - x["res"][key][off] for x in rr]
        dg = [x["res"][key]["on"] - x["res"][key][off] for x in rr
              if thr is not None and x["res"][key]["r_forced_cost"] <= thr]
        out[sc] = {"if_switch_on": summ(d), "pools_where_gate_opens": summ(dg)}
    return out


def cmd_switch(args):
    t0 = time.time()
    design, _pl, _fe = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "switch_report.json")
    out_check(args, out)
    if not args.smoke:
        no_overwrite(out)
    feat_check = verify_feats(args, design, paths(args)[2])
    tp = os.path.join(design, "tune_report.json")
    if not os.path.exists(tp):
        raise Stop("refused (exit 2): tune_report.json missing (run tune first)")
    tune = st.read_json(tp)
    dec = tune["decision"]
    cnn_ok = bool(tune["cnn"]["available"]) and dec.get("cnn_representative") is True
    sources = ["E0", "PA"] if cnn_ok else ["PA"]
    combos = sim.combo_list(dec.get("v1"))
    sel_seeds = args.seeds_sel or SW_SEL_SEEDS
    ev_seeds = args.seeds_eval or SW_EVAL_SEEDS
    jobs = [(s, combos, sources) for s in list(sel_seeds) + list(ev_seeds)]
    res = run_pool(args, _switch_seed, jobs, cnn=cnn_ok)
    rows = [row for r in res for row in r["rows"]]
    sel = [r for r in rows if r["seed"] in sel_seeds]
    ev = [r for r in rows if r["seed"] in ev_seeds]
    first = "V2|%s" % sources[0]
    ks, kdet = k_star(sel, first)
    off = "off_%s" % ("none" if ks is None else ks)
    rep = {"what": "PREREG_06 S6 (2.6): k*, per-combination one-to-one switch thresholds r*_X and usability",
           "seeds": {"selection": list(sel_seeds), "evaluation": list(ev_seeds)}, "structures": list(sim.STRUCTURES),
           "f_sources": sources, "f_source_basis": ("real E0 and proxy A (CNN representative)" if cnn_ok else
                                                     "proxy A only: %s" % ("; ".join(dec.get("reasons", []))
                                                                            or "no CNN")),
           "combinations": [[n, w] for n, w in combos],
           "k_star": ks, "k_star_rule": "F = %s on the selection pools; first k of 10, 20, 40 with mean (CSLSk - none) "
                                        ">= 0 in every structure" % first, "k_star_detail": kdet,
           "off_rule": off, "per_combination": {}}
    for name, _w in combos:
        pc = {"thresholds": {}, "usability": {}}
        thrs = []
        for s in sources:
            key = "%s|%s" % (name, s)
            thr, det = select_threshold(sel, key, off)
            pc["thresholds"][s] = {"thr": thr, **det}
            thrs.append(thr)
        r_star = r_star_of(thrs)
        pc["r_star"] = r_star
        ok_all = r_star is not None
        for s in sources:
            key = "%s|%s" % (name, s)
            ok, det = usable_check(ev, key, "%s|open" % name, off, r_star)
            pc["usability"][s] = {"usable": ok, **det}
            ok_all &= ok
            pc.setdefault("veto_stats_eval", {})[s] = veto_stats(ev, key, off, r_star)
            pc.setdefault("eval_scores", {})[s] = {
                sc: {"off": ci4([x["res"][key][off] for x in ev if x["scen"] == sc]),
                     "on": ci4([x["res"][key]["on"] for x in ev if x["scen"] == sc]),
                     "none": ci4([x["res"][key]["off_none"] for x in ev if x["scen"] == sc])}
                for sc in sim.STRUCTURES}
        pc["usable"] = bool(ok_all)
        rep["per_combination"][name] = pc
    rep["switch_used_at_all"] = any(v["usable"] for v in rep["per_combination"].values())
    rep["feature_store_check"] = feat_check
    rep["rows"] = rows
    rep["pseudo_forced_cost"] = {r["seed"]: r["pseudo"] for r in res}
    rep["workers"] = min(args.workers, MAX_WORKERS)
    rep["thread_env_per_worker"] = args.threads
    rep["per_seed_seconds"] = {r["seed"]: r["seconds"] for r in res}
    rep.update(meta(args, "switch", t0, tools, pre, inputs={"tune_report.json": st.sha256_lf(tp)}))
    st.write_json(out, rep)
    print("switch: k* %s; %s -> %s" % (ks, {n: (v["r_star"], v["usable"]) for n, v in rep["per_combination"].items()},
                                       out))
    return 0


# ================================================================ S7 dmix
def cmd_dmix(args):
    t0 = time.time()
    design, _pl, _fe = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "dmix_report.json")
    out_check(args, out)
    if not args.smoke:
        no_overwrite(out)
    feat_check = verify_feats(args, design, paths(args)[2])
    seeds = args.seeds_eval or DMIX_SEEDS
    res = run_pool(args, _dmix_seed, list(seeds), colours=("B", "W"), cnn=False)
    d = [r["main_O_v2_csls10"] - r["base_O_csls10"] for r in res]
    db = [r["blackonly_O_v2_csls10"] - r["base_O_csls10"] for r in res]
    m = ci4(d)
    rep = {"what": "PREREG_06 S7 (2.8) D-mix: black-and-white pools of fit players; O_raw + v2 (per colour, V2 "
                   "weights) + CSLS10 vs O_raw + CSLS10; no CNN",
           "seeds": list(seeds), "main_diff": m, "passed": bool(m[0] > 0),
           "stop_rule": "main mean difference <= 0 -> 06 is not scored on the local validation set",
           "recorded_only_blackonly_diff": ci4(db),
           "scores": {"base": ci4([r["base_O_csls10"] for r in res]), "main": ci4([r["main_O_v2_csls10"] for r in res]),
                      "blackonly": ci4([r["blackonly_O_v2_csls10"] for r in res])},
           "per_seed": res, "workers": min(args.workers, MAX_WORKERS), "thread_env_per_worker": args.threads,
           "feature_store_check": feat_check}
    rep.update(meta(args, "dmix", t0, tools, pre,
                    inputs={"lda_report.json": _sha_or_none(os.path.join(design, "lda_report.json"))}))
    st.write_json(out, rep)
    print("dmix: main diff %s passed %s -> %s" % (m, rep["passed"], out))
    return 0 if rep["passed"] else 1


# ================================================================ S3 selfcheck
def _selfcheck_seed(seed):
    store = _W["store"]
    rec = sim.make_pool(store.black_cache(), seed, "o2o")
    models = _lda_models(rec)
    T = sim.pool_terms(store, rec, _W["lda"], models, None, want=("O", "JO", "S2", "L"))
    F = st.fuse([T["O"], T["L"], T["JO"], T["S2"]], [1.0, 2.66, 0.5, 0.5])
    return {"seed": seed, "v2_nopost": st.score(F, rec["owner"]), "v2_csls10": st.score(st.csls(F, 10), rec["owner"]),
            "seconds": 0}


def cmd_selfcheck(args):
    t0 = time.time()
    design, _pl, feats = paths(args)
    tools, pre = check_tools(args)
    out = os.path.join(design, "selfcheck_report.json")
    out_check(args, out)
    if not args.smoke:
        no_overwrite(out)
    feat_check = verify_feats(args, design, paths(args)[2])
    cache_dir = args.explore_cache or EXPLORE_CACHE
    os.environ["C10_C1H_SCRATCH"] = cache_dir
    before = _pycache_dirs(EXPLORE)
    sys.dont_write_bytecode = True
    if EXPLORE not in sys.path:
        sys.path.insert(0, EXPLORE)
    import c1h_lib as XL  # noqa: E402
    import c1h_fused as XF  # noqa: E402
    store = sim.Store(feats)
    lda = {"B": sim.LdaData(feats, "B", store)}
    xc = XL.load_cache()
    cache = store.black_cache()
    # exploration cache row r <-> store row (r-th black game of the >= 105 black players, file order)
    xrows = {}
    for p, rows in xc["by_player"].items():
        mine = cache["by_player"][p]
        if len(mine) != len(rows):
            raise Stop("player %s: %d vs %d cached black games" % (p, len(mine), len(rows)))
        for a, b in zip(rows, mine):
            xrows[a] = b
    gid_ok = all(str(store.gid[xrows[i]]) == str(xc["gid"][i]) for i in range(len(xc["gid"])))
    comp = []
    for seed in (args.seeds_tune or SELFCHECK_TERM_SEEDS):
        xrec = XL.make_pool(xc, seed, "o2o")
        rec = sim.make_pool(cache, seed, "o2o")
        same_pool = (rec["pool"] == xrec["pool"] and
                     all([xrows[a] for a in xa] == b for xa, b in zip(xrec["cand_rows"], rec["cand_rows"])) and
                     all([xrows[a] for a in xa] == b for xa, b in zip(xrec["q_rows"], rec["q_rows"])))
        G = {}
        X = {n: XF.term(n, xrec, xc, G) for n in ("base", "joseki_own", "style2", "lda64")}
        models = _lda_models_local(store, lda, rec)
        T = sim.pool_terms(store, rec, lda, models, None, want=("O", "JO", "S2", "L"))
        zl = float(np.max(np.abs(st.zrows(T["L"][0]) - st.zrows(X["lda64"]))))
        comp.append({"seed": seed, "same_pool": bool(same_pool),
                     "O_max_abs": float(np.max(np.abs(T["O"][0] - X["base"]))),
                     "JO_max_abs": float(np.max(np.abs(T["JO"][0] - X["joseki_own"]))),
                     "S2_max_abs": float(np.max(np.abs(sim.term_nb_raw(store, rec, "s2", "B") - X["style2"]))),
                     "S2_z_max_abs": float(np.max(np.abs(T["S2"][0] - st.zrows(X["style2"])))),
                     "L_z_max_abs": zl,
                     "not_alternating_games": int((~store.alt[[r for g in rec["cand_rows"] + rec["q_rows"]
                                                                for r in g]]).sum())})
        print("  selfcheck seed %d: %s" % (seed, comp[-1]), flush=True)
    del G, X
    args.workers = min(args.workers, MAX_WORKERS)
    res = run_pool(args, _selfcheck_seed, list(args.seeds_eval or SELFCHECK_SCORE_SEEDS), cnn=False)
    means = {k: float(np.mean([r[k] for r in res])) for k in ("v2_nopost", "v2_csls10")}
    after = _pycache_dirs(EXPLORE)
    checks = {
        "pools_identical": all(c["same_pool"] for c in comp) and gid_ok,
        "O_le_1e-9": all(c["O_max_abs"] <= 1e-9 for c in comp),
        "JO_le_1e-9": all(c["JO_max_abs"] <= 1e-9 for c in comp),
        "S2_le_1e-9": all(c["S2_max_abs"] <= 1e-9 for c in comp),
        "L_z_le_1e-6": all(c["L_z_max_abs"] <= 1e-6 for c in comp),
        "v2_nopost_rounds_to_0.9819": round(means["v2_nopost"], 4) == SELFCHECK_EXPECT["v2_nopost"],
        "v2_csls10_rounds_to_0.9869": round(means["v2_csls10"], 4) == SELFCHECK_EXPECT["v2_csls10"],
        "no_new_pycache_in_explore": after == before,
    }
    files = {}
    for f in sorted(os.listdir(cache_dir)):
        files[f] = st.sha256_file(os.path.join(cache_dir, f))
    rep = {"what": "PREREG_06 S3 (2.8): the p06 port vs the exploration functions on the exploration's "
                   "evaluation pools (black only)",
           "term_seeds": list(args.seeds_tune or SELFCHECK_TERM_SEEDS),
           "score_seeds": list(args.seeds_eval or SELFCHECK_SCORE_SEEDS), "term_comparison": comp,
           "v2_means": means, "v2_means_rounded": {k: round(v, 4) for k, v in means.items()},
           "expected": SELFCHECK_EXPECT, "checks": checks, "passed": all(checks.values()),
           "explore_cache": {"dir": st.rel(cache_dir) if not args.smoke else cache_dir, "sha256": files,
                             "bad_alternation_in_cache": xc.get("bad_alternation")},
           "per_seed": res, "workers": args.workers, "thread_env_per_worker": args.threads,
           "feature_store_check": feat_check}
    rep.update(meta(args, "selfcheck", t0, tools, pre,
                    inputs={"lda_report.json": _sha_or_none(os.path.join(design, "lda_report.json"))}))
    st.write_json(out, rep)
    print("selfcheck: %s means %s -> %s" % (checks, rep["v2_means_rounded"], out))
    return 0 if rep["passed"] else 1


def _lda_models_local(store, lda, rec):
    return {"B": lda["B"].fit_excluding(store, set(rec["pool"]))}


def _pycache_dirs(root):
    out = []
    for dp, dn, _fn in os.walk(root):
        out += [os.path.join(dp, d) for d in dn if d == "__pycache__"]
    return sorted(out)


# ================================================================ S8 freeze
DESIGN_REPORTS = ("build_report.json", "lda_report.json", "selfcheck_report.json", "tune_report.json",
                  "switch_report.json", "dmix_report.json")
DEP_FILES = ("tools/p06_cnn_fit.py", "tools/p02_player_fuse.py", "tools/p02_player_open.py", "tools/p02_moves.py",
             "tools/c10data.py")
FREEZE_LINKS = (("switch_report.json", "tune_report.json"), ("tune_report.json", "build_report.json"),
                ("tune_report.json", "lda_report.json"), ("tune_report.json", "cnn_fit_report.json"),
                ("dmix_report.json", "lda_report.json"), ("selfcheck_report.json", "lda_report.json"),
                ("lda_report.json", "build_report.json"))


def freeze_problems(args, design, player, feats, tools):
    """Every reason the real freeze must refuse (PREREG_06 section 9 steps 6-7)."""
    probs = []
    smoke = args.smoke
    want_mode = mode_of(args)

    def load(name):
        p = os.path.join(design, name)
        return (p, st.read_json(p)) if os.path.exists(p) else (p, None)

    def sha(p):
        return st.sha256_lf(p) if os.path.exists(p) else None

    # leakcheck: its tool SHAs are not compared with the current tools (it runs once, before build); the tools
    # only had to be committed and clean when it ran
    lkp, lk = load("leakcheck_report.json")
    if lk is None:
        probs.append("leakcheck_report.json missing")
    else:
        if lk.get("mode") != want_mode:
            probs.append("leakcheck_report.json: mode %s" % lk.get("mode"))
        if not smoke:
            lt = lk.get("tools") or {}
            if lk.get("precommit") or not lt or not all(v.get("committed_and_clean") for v in lt.values()):
                probs.append("leakcheck_report.json: tools were not committed and clean when it ran")
            if not report_committed(lkp):
                probs.append("leakcheck_report.json not committed")
        if lk.get("passed") is not True or lk.get("overlap") != 0:
            probs.append("leakcheck_report.json: passed %s, overlap %s" % (lk.get("passed"), lk.get("overlap")))
    R = {}
    for name in DESIGN_REPORTS:
        p, r = load(name)
        if r is None:
            probs.append("%s missing" % name)
            continue
        R[name] = (p, r)
        if r.get("mode") != want_mode:
            probs.append("%s: mode %s" % (name, r.get("mode")))
        if not smoke:
            if r.get("precommit"):
                probs.append("%s: precommit run" % name)
            if not report_committed(p):
                probs.append("%s not committed" % name)
        rt = r.get("tools") or {}
        for k, v in tools.items():
            if (rt.get(k) or {}).get("sha256_lf") != v["sha256_lf"]:
                probs.append("%s was made with a different %s" % (name, k))
    for name in ("build_report.json", "lda_report.json", "selfcheck_report.json"):
        if name in R and R[name][1].get("passed") is not True:
            probs.append("%s: passed %s" % (name, R[name][1].get("passed")))
    cp, cnn = load("cnn_fit_report.json")       # missing -> K-CNN fallback (allowed)
    if cnn is not None:
        if cnn.get("mode") != want_mode:
            probs.append("cnn_fit_report.json: mode %s" % cnn.get("mode"))
        if not smoke and not report_committed(cp):
            probs.append("cnn_fit_report.json not committed")
    # cross-report consistency
    for a, b in FREEZE_LINKS:
        if a in R:
            got = (R[a][1].get("inputs") or {}).get(b, "absent")
            exp = sha(os.path.join(design, b))
            if got != exp:
                probs.append("%s records %s %s, file is %s" % (a, b, got, exp))
    if "build_report.json" in R:
        bl = R["build_report.json"][1].get("leakcheck_report") or {}
        if bl.get("sha256") != sha(lkp):
            probs.append("build_report.json records leakcheck %s, file is %s" % (bl.get("sha256"), sha(lkp)))
    if "lda_report.json" in R:
        for c in "BW":
            f = os.path.join(player, "lda_%s.npz" % c)
            rec = R["lda_report.json"][1].get("colours", {}).get(c, {}).get("sha256")
            if not os.path.exists(f) or st.sha256_file(f) != rec:
                probs.append("lda_%s.npz differs from lda_report.json" % c)
            elif not smoke and not report_committed(f):
                probs.append("lda_%s.npz not committed" % c)
    if "build_report.json" in R and "lda_report.json" in R:
        bad = feat_problems(R["build_report.json"][1], R["lda_report.json"][1], feats)
        if bad:
            probs.append("feature store differs from build / lda: %s" % bad[:10])
    return probs


def cmd_freeze(args):
    t0 = time.time()
    design, player, feats = paths(args)
    tools, pre = check_tools(args)
    if args.smoke or pre:
        out = args.preview or os.path.join(REPO, "data", "prereg06", "frozen_config_precommit.json")
    else:
        out = os.path.join(player, "frozen_config.json")
        no_overwrite(out)
    out_check(args, out)
    problems = freeze_problems(args, design, player, feats, tools)
    if problems and not pre:
        raise Stop("refused (exit 2): %s" % problems)
    rd = {n: st.read_json(os.path.join(design, n)) for n in DESIGN_REPORTS}
    tune, sw, dm = rd["tune_report.json"], rd["switch_report.json"], rd["dmix_report.json"]
    lda, sc = rd["lda_report.json"], rd["selfcheck_report.json"]
    reps = {}
    for n in REPORTS:
        p = os.path.join(design, n)
        if os.path.exists(p):
            reps[n] = {"sha256": st.sha256_lf(p), "committed": report_committed(p)}
    cnn_rep = os.path.join(design, "cnn_fit_report.json")
    lk = os.path.join(design, "leakcheck_report.json")
    order = [n for n, _w in sim.combo_list(tune["decision"].get("v1"))]
    cfg = {
        "what": "PREREG_06 frozen configuration (competition 1); written by tools/p06_player.py freeze",
        "mode": mode_of(args), "precommit_preview": bool(pre),
        "combinations": {"order": order, "V2": list(sim.V2_W), "V1": tune["decision"].get("v1"),
                         "V1_reasons": tune["decision"].get("reasons"), "V3": list(sim.V3_W), "V0": None,
                         "base": "Q04 = z(E0) + 1 z(O_raw)"},
        "k_star": sw["k_star"], "k_switch_rest": st.CSLS_SWITCH_K,
        "switch": {n: {"r_star": v["r_star"], "usable": v["usable"],
                       "thresholds_by_source": {s_: t["thr"] for s_, t in v["thresholds"].items()}}
                   for n, v in sw["per_combination"].items()},
        "switch_used_at_all": sw["switch_used_at_all"], "switch_f_sources": sw["f_sources"],
        "lda": {c: {"file": "results/prereg06/player/lda_%s.npz" % c, "sha256": lda["colours"][c]["sha256"],
                    "vocab_size": lda["colours"][c]["vocab_size"]} for c in "BW"},
        "e0_cap": {"cap": E0_CAP, "value": tune["decision"].get("e0_alone_tune_mean"),
                   "cnn_representative": tune["decision"].get("cnn_representative")},
        "cnn_fit_report_sha256": _sha_or_none(cnn_rep), "cnn_available": tune["cnn"]["available"],
        "dmix": {"passed": dm["passed"], "main_diff": dm["main_diff"]},
        "selfcheck": {"passed": sc["passed"], "v2_means_rounded": sc["v2_means_rounded"]},
        "select_tolerance": SELECT_TOL, "gate_min_vs_V0": GATE_MIN,
        "seeds": {"tune": list(TUNE_SEEDS), "tune_eval": list(TUNE_EVAL_SEEDS), "switch_selection": list(SW_SEL_SEEDS),
                  "switch_eval": list(SW_EVAL_SEEDS), "dmix": list(DMIX_SEEDS)},
        "salts": {"sr": st.SR_SALT, "sr_k": st.SR_K_SALT}, "sr_rounds": st.SR_ROUNDS,
        "reports": reps, "leakcheck_report_sha256": _sha_or_none(lk),
        "dependency_sha256": {n: _sha_or_none(os.path.join(REPO, n)) for n in DEP_FILES},
        "prereg02_inputs": {k: st.sha256_file(os.path.join(P02_PLAYER, "dumps", k)) for k in pf.DUMP_FILES},
        "prereg02_top5_sha256": st.sha256_file(os.path.join(P02_PLAYER, "player_val_top5.csv")),
        "problems_at_freeze": problems,
    }
    cfg.update(meta(args, "freeze", t0, tools, pre))
    st.write_json(out, cfg)
    print("freeze: k* %s, V1 %s, switch %s, problems %s -> %s" % (cfg["k_star"], cfg["combinations"]["V1"],
                                                                 cfg["switch"], problems, out))
    return 0


# ================================================================ S9 apply (local validation)
class MemStore:
    """In-memory per-game features of one candidate / question pool (same interface as p06_sim.Store)."""

    def __init__(self, games, oraw, lda_models):
        """games: [(player_or_question_id, game_id, colour, sgf)]; oraw: CSR of the O_raw tokens (same rows)."""
        n = len(games)
        self.n = n
        self.pid = np.array([g[0] for g in games])
        self.gid = np.array([str(g[1]) for g in games])
        self.color = np.array([g[2] for g in games])
        self.alt = np.zeros(n, bool)
        self.nmov = np.zeros(n, np.int32)
        S2 = np.full((n, 66), np.nan)
        jo, resp = [], []
        for i, (_p, _g, c, sgf) in enumerate(games):
            mv = st.parse_all(sgf)
            self.nmov[i] = len(mv)
            if st.alternates(mv):
                self.alt[i] = True
                S2[i] = st.style2(mv, c)
                jo.append(Counter(st.jo_tokens(mv, c)))
                resp.append(Counter(st.resp_tokens(mv, c)))
            else:
                jo.append(Counter())
                resp.append(Counter())
        voc = {}
        r_, c_, v_ = [], [], []
        for i, cn in enumerate(jo):
            for t, k in cn.items():
                r_.append(i)
                c_.append(voc.setdefault(t, len(voc)))
                v_.append(k)
        self.mats = {"s2": S2, "oraw": oraw,
                     "jo": sp.csr_matrix((np.asarray(v_, np.float64), (np.asarray(r_, np.int64), np.asarray(c_, np.int64))),
                                         shape=(n, len(voc)))}
        self.oov = {}
        self.Z = {}
        for colour, m in lda_models.items():
            vi = {t: j for j, t in enumerate(m["vocab"])}
            rows = np.flatnonzero(self.alt & (self.color == colour))
            X = np.zeros((len(rows), 66 + len(vi)), np.float32)
            oov = 0
            for k, i in enumerate(rows):
                X[k] = st.ldain_from(S2[i], resp[i], vi)
                oov += sum(cnt for t, cnt in resp[i].items() if t not in vi)
            self.oov[colour] = oov
            self.Z[colour] = dict(zip(rows.tolist(), st.lda_project(X, m) if len(rows) else np.zeros((0, st.LDA_DIM))))

    def csr(self, name):
        return self.mats[name]

    def dense(self, name):
        return self.mats[name]


class FrozenLda:
    def __init__(self, store, colour):
        self.store, self.colour = store, colour

    def project(self, rows, _model):
        return np.vstack([self.store.Z[self.colour][int(r)] for r in rows]) if len(rows) else np.zeros((0, st.LDA_DIM))


def load_lda(path):
    with np.load(path, allow_pickle=False) as z:
        return {"W": z["W"], "ev": z["ev"], "mu": z["mu"], "sd": z["sd"], "vocab": [str(t) for t in z["vocab"].tolist()]}


def apply_terms(ms, models, cand_groups, q_groups, e_arrays):
    """Terms of a pool on the in-memory store; e_arrays = (cand arrays, query arrays) for E0 or None."""
    rec = {"cand_rows": cand_groups, "q_rows": q_groups}
    lda = {c: FrozenLda(ms, c) for c in models}
    T = sim.pool_terms(ms, rec, lda, models, None, want=("O", "JO", "S2", "L"))
    if e_arrays is not None:
        ca, cg2, qa, qg2 = e_arrays
        M, has, _ = pf.e_scores("E0", ca, cg2, qa, qg2)
        T["E0"] = (M, has)
    return T


def sr06_apply(ms, cand_ids, c_rows_by_pid, ks, cemb, models, combos, k_star, rounds=None, black_only=False):
    """sr06 on a candidate pool (PREREG_06 2.3). Returns ({combo: mean}, {combo: per-round F for rounds 0, 1})."""
    cg = {p: [str(ms.gid[r]) for r in c_rows_by_pid[p]] for p in cand_ids}
    row_of = {(p, str(ms.gid[r])): r for p in cand_ids for r in c_rows_by_pid[p]}
    qf = (lambda p, g: ms.color[row_of[(p, g)]] == "B") if black_only else None
    rr = st.sr06_rounds(cg, ks, query_filter=qf)
    pts = {n: [] for n, _w in combos}
    pts["R02"] = []
    F01 = {n: [] for n, _w in combos}
    for r, items in enumerate(rr):
        if rounds is not None and r not in rounds:
            continue
        cgr = [[row_of[(p, g)] for g in refs] for p, _q, refs in items]
        qsel = [(i, [row_of[(p, g)] for g in q]) for i, (p, q, _refs) in enumerate(items) if q]
        qi = np.array([i for i, _r in qsel], np.int64)
        qg = [rws for _i, rws in qsel]
        e = (cemb, cgr, cemb, qg)
        T = apply_terms(ms, models, cgr, qg, e)
        for n, w in combos:
            F = sim.combo_F(T, "E0", w)
            pts[n] += st.points(st.ranks(st.csls(F, k_star), qi)).tolist()
            if r in (0, 1):
                F01[n].append(F)
        pts["R02"] += st.points(st.ranks(sim.combo_F(T, "E0", None), qi)).tolist()
    return {n: float(np.mean(v)) if v else float("nan") for n, v in pts.items()}, F01


def choose_combo(means, order, tol=SELECT_TOL):
    """Best mean, then the earliest combination (list order) within tol of it (PREREG_06 2.4)."""
    best = max(means[n] for n in order)
    return next(n for n in order if best - means[n] <= tol + 1e-15)


def write_top5(path, qids, tops):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id"] + pf.TOP_COLS)
        for q, t in zip(qids, tops):
            w.writerow([q] + t)


def score_gate(chosen, srm, dmix_passed, q04_same, q04_06_same, struct, fmt):
    """PREREG_06 section 3 send gate: the list of reasons why score_allowed is false (empty = allowed)."""
    blocked = []
    if chosen == "V0":
        blocked.append("selected V0")
    if srm[chosen] - srm["V0"] < GATE_MIN - 1e-12:
        blocked.append("sr(selected) - sr(V0) = %.4f < +0.005" % (srm[chosen] - srm["V0"]))
    if dmix_passed is not True:
        blocked.append("D-mix did not pass")
    if not q04_same:
        blocked.append("Q04 reproduction (p02 functions) failed")
    if not q04_06_same:
        blocked.append("Q04 reproduction (06 path) failed")
    if not all(struct.values()):
        blocked.append("secondary file structure check failed")
    if not fmt or any(v["returncode"] != 0 for v in fmt.values()):
        blocked.append("format check failed")
    return blocked


APPLY_DEFAULTS = {"candidates": os.path.join(LOCAL, "player_val_candidates.csv"),
                  "questions": os.path.join(LOCAL, "player_val_questions.csv")}


def apply_real_overrides(args):
    """Real apply runs on the registered inputs only: no path overrides."""
    bad = [k for k, v in APPLY_DEFAULTS.items() if os.path.normcase(os.path.abspath(getattr(args, k)))
           != os.path.normcase(v)]
    bad += [k for k in ("p02_top5", "design_dir", "p02_dir", "player_dir", "feat_dir") if getattr(args, k)]
    return bad


def cmd_apply(args):
    t0 = time.time()
    _d, player, _fe = paths(args)
    frozen = os.path.join(player, "frozen_config.json")
    out_rep = os.path.join(player, "apply_report.json")
    if not args.smoke:
        bad = apply_real_overrides(args)
        if bad:
            raise Stop("refused (exit 2): real apply does not accept overrides: %s" % bad)
    if not os.path.exists(frozen):
        raise Stop("refused (exit 2): frozen_config.json missing; apply runs only after the freeze commit")
    if not args.smoke:
        if not report_committed(frozen):
            raise Stop("refused (exit 2): frozen_config.json is not committed and identical to HEAD")
        if os.path.exists(out_rep) and not (args.rerun and args.reason):
            raise Stop("refused (exit 2): apply_report.json exists (rerun needs --rerun --reason, recorded)")
    fz = st.read_json(frozen)
    if fz.get("precommit_preview"):
        raise Stop("refused (exit 2): frozen_config.json is a precommit preview")
    tools, _pre = check_tools(args, allow_precommit=False)
    changed = [k for k, v in tools.items() if fz["tools"].get(k, {}).get("sha256_lf") != v["sha256_lf"]]
    if changed and not args.smoke:
        raise Stop("refused (exit 2): tools changed since freeze: %s" % changed)
    out_check(args, out_rep)
    # frozen LDA files: SHA-256 as frozen, committed and clean
    for c in "BW":
        f = os.path.join(player, "lda_%s.npz" % c)
        want = (fz.get("lda") or {}).get(c, {}).get("sha256")
        if not os.path.exists(f) or st.sha256_file(f) != want:
            raise Stop("refused (exit 2): lda_%s.npz differs from frozen_config.json" % c)
        if not args.smoke and not report_committed(f):
            raise Stop("refused (exit 2): lda_%s.npz is not committed and clean" % c)
    ref = args.p02_top5 or os.path.join(P02_PLAYER, "player_val_top5.csv")
    ref_sha = st.sha256_file(ref)
    if not args.smoke and ref_sha != fz.get("prereg02_top5_sha256"):
        raise Stop("refused (exit 2): prereg02 player_val_top5.csv differs from the frozen SHA-256")
    p02 = args.p02_dir or P02_PLAYER
    for k in pf.DUMP_FILES:
        sha = st.sha256_file(os.path.join(p02, "dumps", k))
        if not args.smoke and sha != fz["prereg02_inputs"][k]:
            raise Stop("refused (exit 2): prereg02 dumps/%s differs from the frozen SHA-256" % k)
    prev = None
    if os.path.exists(out_rep):                 # --rerun: keep the previous report
        n = 1
        while os.path.exists(os.path.join(player, "apply_report.prev%d.json" % n)):
            n += 1
        prev = os.path.join(player, "apply_report.prev%d.json" % n)
        os.replace(out_rep, prev)
    with st.allow_local():
        d = pf.Data(args.candidates, p02, True, args.smoke, q_path=args.questions)
        cand_rows = pf.read_candidates(args.candidates)
        q_rows = pf.read_questions(args.questions)
    # step 3: Q04 with the p02 functions, byte-identical to prereg02's player_val_top5.csv
    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    gc, gq = d.G("raw")
    cosr, hasr = po.open_cos(gc, c_groups, gq, q_groups)
    c_arr, q_arr = (d.cemb, d.cnw, d.cok), (d.qemb, d.qnw, d.qok)
    ME, hasE, _ = pf.e_scores("E0", c_arr, c_groups, q_arr, q_groups)
    S04 = pf.fuse(pf.COMBO_BY_ID["Q04"], {"E0": (ME, hasE), "raw": (cosr, hasr)})
    os.makedirs(player, exist_ok=True)
    with open(ref, "rb") as f:
        ref_bytes = f.read()

    def same_as_ref(tops, name):
        tmp = os.path.join(player, name)
        write_top5(tmp, d.qids, tops)
        with open(tmp, "rb") as f:
            b = f.read()
        os.remove(tmp)
        return b == ref_bytes, hashlib.sha256(b).hexdigest()

    q04_same, q04_sha = same_as_ref(pf.top_ids(S04, d.cand_ids), "q04_repro.tmp.csv")
    if not q04_same:
        raise Stop("stopped: Q04 recomputed with the p02 functions differs from %s" % ref)
    # step 4: v2 features (one in-memory store: candidate games then question games, p02 row order)
    models = {c: load_lda(os.path.join(player, "lda_%s.npz" % c)) for c in "BW"}
    by_gid = {r[0]: r for r in cand_rows}
    games = [(g[1], g[0], g[2], by_gid[g[0]][3]) for g in d.cgames]
    qsgf = {(q, s): (c, x) for q, g in q_rows for s, c, x in g}
    games += [(g[0], "%s#%d" % (g[0], g[1]), g[2], qsgf[(g[0], g[1])][1]) for g in d.qgames]
    nc = len(d.cgames)
    oraw = sp.vstack([gc, gq]).tocsr()
    ms = MemStore(games, oraw, models)
    emb = np.vstack([d.cemb, d.qemb])
    nw = np.concatenate([d.cnw, d.qnw])
    ok = np.concatenate([d.cok, d.qok])
    earr = (emb, nw, ok)
    cgroups = [list(g) for g in c_groups]
    qgroups = [[nc + i for i in g] for g in q_groups]
    T = apply_terms(ms, models, cgroups, qgroups, (earr, cgroups, earr, qgroups))
    # Q04 once more through the 06 path (06 terms and fusion): must be byte-identical too
    q04_06_same, q04_06_sha = same_as_ref(st.top5(sim.combo_F(T, "E0", None), d.cand_ids), "q04_06.tmp.csv")
    if not q04_06_same:
        raise Stop("stopped: Q04 recomputed through the 06 path differs from %s" % ref)
    combos = sim.combo_list(fz["combinations"]["V1"])
    order = [n for n, _w in combos]
    ks = [len(g) for g in q_groups]
    kst = fz["k_star"]
    c_by_pid = {p: d.c_by_player[p] for p in d.cand_ids}
    srm, F01 = sr06_apply(ms, d.cand_ids, c_by_pid, ks, earr, models, combos, kst)
    chosen = choose_combo({n: srm[n] for n in order}, order)
    w = dict(combos)[chosen]
    F = sim.combo_F(T, "E0", w)
    main = st.top5(st.csls(F, kst), d.cand_ids)
    a = st.hungarian_top1(st.zrows(F))
    sw = st.switch_top5(F, d.cand_ids, a)
    p_main = os.path.join(player, "player_val_top5.csv")
    p_sw = os.path.join(player, "player_val_top5_switch.csv")
    write_top5(p_main, d.qids, main)
    write_top5(p_sw, d.qids, sw)
    # step 7: structure checks of the secondary file
    from scipy.optimize import linear_sum_assignment
    Z = st.zrows(F)
    r_, c_ = linear_sum_assignment(-Z)
    a2 = np.empty(len(r_), np.int64)
    a2[r_] = c_
    C3 = st.csls(F, st.CSLS_SWITCH_K)
    idx = {p: i for i, p in enumerate(d.cand_ids)}
    rest_ok = True
    for i, t in enumerate(sw):
        rest = [idx[p] for p in t[1:]]
        if d.cand_ids[a[i]] in t[1:]:
            rest_ok = False
        exp = [j for j in np.lexsort((np.arange(C3.shape[1]), -C3[i])) if j != a[i]][:4]
        rest_ok &= rest == list(exp)
    struct = {"rank1_distinct": len({t[0] for t in sw}) == len(sw),
              "rank1_equals_independent_lsa": bool(np.array_equal(a, a2)) or
              bool(np.isclose(Z[np.arange(len(a)), a].sum(), Z[np.arange(len(a2)), a2].sum(), atol=1e-9)),
              "ranks2to5_csls3_without_rank1": bool(rest_ok)}
    # step 8: detector (record only)
    det = st.detector(F, F01[chosen])
    swc = fz["switch"].get(chosen, {})
    r_star, usable = swc.get("r_star"), bool(swc.get("usable"))
    # step 9: black-only pseudo questions (record only)
    srb, _ = sr06_apply(ms, d.cand_ids, c_by_pid, ks, earr, models, combos, kst, black_only=True)
    # step 10: format checks
    fmt = {}
    for p in (p_main, p_sw):
        r = subprocess.run([sys.executable, os.path.join(HERE, "check_submission.py"), "player", p,
                            "--questions", args.questions, "--candidates", args.candidates],
                           capture_output=True, text=True)
        fmt[os.path.basename(p)] = {"returncode": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    if args.smoke and args.design_dir:
        dmix_passed = st.read_json(os.path.join(args.design_dir, "dmix_report.json")).get("passed")
    else:
        dmix_passed = (fz.get("dmix") or {}).get("passed")
    blocked = score_gate(chosen, srm, dmix_passed, q04_same, q04_06_same, struct, fmt)
    rep = {"what": "PREREG_06 apply (spec section 6) on the local validation pool",
           "q04_reproduction": {"identical": q04_same, "sha256": q04_sha, "identical_06_path": q04_06_same,
                                "sha256_06_path": q04_06_sha, "reference": st.rel(ref), "reference_sha256": ref_sha},
           "sr06": {n: round(v, 6) for n, v in srm.items()}, "selected": chosen, "order": order,
           "gate": {"sr_selected_minus_V0": srm[chosen] - srm["V0"], "min": GATE_MIN},
           "sr_selected_minus_R02": srm[chosen] - srm["R02"], "k_star": kst,
           "detector": {"r_forced_cost": det["r_forced_cost"], "forced_cost": det["forced_cost"],
                        "pseudo_forced_cost": det["pseudo_forced_cost"], "r_star": r_star, "usable": usable,
                        "would_open": bool(usable and r_star is not None and det["r_forced_cost"] <= r_star),
                        "changed_top1": int(sum(1 for x, y in zip(main, sw) if x[0] != y[0]))},
           "secondary_structure": struct,
           "mono_black_check": {"sr06_black_only": {n: round(v, 6) for n, v in srb.items()},
                                "selected_minus_V0": srb[chosen] - srb["V0"]},
           "games": {"candidate": {"B": int((ms.color[:nc] == "B").sum()), "W": int((ms.color[:nc] == "W").sum()),
                                   "not_alternating_B": int(((ms.color[:nc] == "B") & ~ms.alt[:nc]).sum()),
                                   "not_alternating_W": int(((ms.color[:nc] == "W") & ~ms.alt[:nc]).sum())},
                     "question": {"B": int((ms.color[nc:] == "B").sum()), "W": int((ms.color[nc:] == "W").sum()),
                                  "not_alternating_B": int(((ms.color[nc:] == "B") & ~ms.alt[nc:]).sum()),
                                  "not_alternating_W": int(((ms.color[nc:] == "W") & ~ms.alt[nc:]).sum())}},
           "out_of_vocabulary_tokens": ms.oov, "format_checks": fmt,
           "outputs": {"player_val_top5.csv": st.sha256_file(p_main),
                       "player_val_top5_switch.csv": st.sha256_file(p_sw)},
           "frozen_config_sha256": st.sha256_lf(frozen), "dmix_passed": dmix_passed,
           "lda_sha256": {c: fz["lda"][c]["sha256"] for c in "BW"},
           "score_allowed": not blocked, "score_blocked_reasons": blocked,
           "rerun": {"rerun": bool(args.rerun), "reason": args.reason,
                     "previous_report": st.rel(prev) if prev else None}}
    rep.update(meta(args, "apply", t0, tools, False, inputs=dict(d.shas)))
    st.write_json(out_rep, rep)
    print("apply: selected %s, sr %s, score_allowed %s %s" % (chosen, rep["sr06"], not blocked, blocked))
    return 0


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("leakcheck", "build", "lda", "selfcheck", "tune", "switch", "dmix", "freeze", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--train-dir", default=TRAIN_DIR)
        p.add_argument("--split-csv", default=SPLIT_CSV)
        p.add_argument("--local-dir", default=LOCAL)
        p.add_argument("--design-dir", default=None)
        p.add_argument("--player-dir", default=None)
        p.add_argument("--feat-dir", default=None)
        p.add_argument("--cnn-npz", default=None)
        p.add_argument("--cnn-report", default=None)
        p.add_argument("--explore-cache", default=None)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--seeds-tune", type=int, nargs="*", default=None)
        p.add_argument("--seeds-eval", type=int, nargs="*", default=None)
        p.add_argument("--seeds-sel", type=int, nargs="*", default=None)
        p.add_argument("--preview", default=None)
        p.add_argument("--candidates", default=os.path.join(LOCAL, "player_val_candidates.csv"))
        p.add_argument("--questions", default=os.path.join(LOCAL, "player_val_questions.csv"))
        p.add_argument("--p02-dir", default=None)
        p.add_argument("--p02-top5", default=None)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    args = ap.parse_args(argv)
    if args.workers > MAX_WORKERS:
        raise Stop("at most %d workers on this machine" % MAX_WORKERS)
    if args.smoke and args.seeds_tune is not None and args.cmd == "apply":
        pass
    if not args.smoke and any(getattr(args, k) for k in ("seeds_tune", "seeds_eval", "seeds_sel")):
        raise Stop("refused: seed overrides are for --smoke only (the seeds are registered)")
    fn = {"leakcheck": cmd_leakcheck, "build": cmd_build, "lda": cmd_lda, "selfcheck": cmd_selfcheck,
          "tune": cmd_tune, "switch": cmd_switch, "dmix": cmd_dmix, "freeze": cmd_freeze, "apply": cmd_apply}
    return fn[args.cmd](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
