"""prereg20 explore (competition 1, phase A): B'-ON (one-to-one on the 10B score, own recalibrated detector, public
veto) against the current full registered strategy, on fit-player mock pools. Rules: results/prereg20/explore/PLAN.md.

    PY = .venv/Scripts/python.exe  (CPU; Git Bash; repository root; PYTHONDONTWRITEBYTECODE=1; no thread variables)
    PY results/prereg20/explore/p20_sim.py seam  [--workers 2]   -> results/prereg20/explore/seam_report.json
    PY results/prereg20/explore/p20_sim.py dev   [--workers 2]   -> results/prereg20/explore/dev_report.json
    PY results/prereg20/explore/p20_sim.py eval  [--workers 2]   -> results/prereg20/explore/eval_report.json
    PY results/prereg20/explore/p20_sim.py smoke                 (timing and shapes only; prints no score)

Only imports the existing tools (p10_fuse, p06_sim, p06_style, p06_player); writes only results/prereg20/explore and
data/prereg20. Reads data/prereg06 (06 fit store, fit CNN) and data/prereg10 (mock sets, mock P1 side-0 means,
mock_points.json). Never data/local (p06_style's import guard is on); never tools/score.py; no GPU.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import p06_style as st  # noqa: E402

st.install_guard()

import numpy as np  # noqa: E402

import p06_sim as sim  # noqa: E402
import p10_fuse as pf10  # noqa: E402
from c10data import LEVELS  # noqa: E402

OUT = HERE
with open(os.path.abspath(__file__), "rb") as _f:
    CODE_SHA = hashlib.sha256(_f.read().replace(b"\r\n", b"\n")).hexdigest()
WORK = os.path.join(REPO, "data", "prereg20")
SETS = os.path.join(REPO, "data", "prereg10", "mock", "sets.json")
SIDE0 = os.path.join(REPO, "data", "prereg10", "mock_work", "p1_side0.npy")
USABLE = os.path.join(REPO, "data", "prereg10", "mock_work", "p1_usable.npy")
MOCK_POINTS = os.path.join(REPO, "data", "prereg10", "mock_work", "mock_points.json")
MOCK_REPORT = os.path.join(REPO, "results", "prereg10", "player", "mock_report.json")
FZ06 = os.path.join(REPO, "results", "prereg06", "player", "frozen_config.json")
SWITCH06 = os.path.join(REPO, "results", "prereg06", "design", "switch_report.json")
FEATS06 = pf10.FEATS06
CNN06 = pf10.CNN06

W_STAR = 16.0
K_STAR = 10
K_SWITCH06 = st.CSLS_SWITCH_K          # 3: 06-ON ranks 2-5
SPLIT_SALT = "c10-prereg20|split|"
N_POOL = 400
PER_LEVEL = 40
BASE_CAP = 20
DEV_SEEDS = tuple(range(1000, 1030))
EVAL_TRIALS = 40
PRIV_SEED0, PUB_SEED0 = 2000, 3000
DEV_STRUCTS = sim.STRUCTURES          # o2o u390 u375 u350 d80 u300 wr d1 d2 d3 d5
EVAL_SCEN = ("o2o", "d40", "d120", "nov4", "wr")
G1_MIN, G2_MIN, G3_MIN, FALSE_ON_MAX = 0.003, -0.002, -0.002, 0.01
BOOT_N, BOOT_SEED = 2000, 0
SEAM_SEEDS = tuple(range(20, 40))
SEAM_TOL = 1e-12
DET_TOL = 1e-9


def sha_raw(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(st.json_safe(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_head():
    import subprocess
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ================================================================ inputs
def verify_inputs():
    """PLAN section 1: the mock P1 side-0 file and mock_points.json are the ones the committed mock report bound."""
    rep = read_json(MOCK_REPORT)
    ib = rep["inputs_bound"]
    got = {"p1_side0.npy": sha_raw(SIDE0), "mock_points.json": sha_raw(MOCK_POINTS), "sets.json": sha_raw(SETS)}
    want = {"p1_side0.npy": ib["side0_npy_sha256"], "mock_points.json": ib["points_json_sha256"],
            "sets.json": ib["sets_sha256"]}
    bad = [k for k in got if got[k] != want[k]]
    u = np.load(USABLE)
    if bad or int(u.sum()) != rep["p1_games"]["usable"] or len(u) != rep["p1_games"]["total"]:
        raise SystemExit("inputs differ from the committed mock report: %s (usable %d)" % (bad, int(u.sum())))
    return {"sha256": got, "usable": int(u.sum()), "total": int(len(u))}


FZ10 = os.path.join(REPO, "results", "prereg10", "player", "frozen_config.json")


def check_constants():
    """Review fix 8: the hard-coded constants equal the frozen configurations."""
    fz6, fz10 = read_json(FZ06), read_json(FZ10)
    bad = []
    if fz6["k_star"] != K_STAR or fz6["k_switch_rest"] != K_SWITCH06 or fz6["combinations"]["V1"] is not None:
        bad.append("06 frozen k*, k_switch_rest or V1")
    if [n for n, _w in sim.combo_list(None)] != fz6["combinations"]["order"]:
        bad.append("06 combination order")
    if fz10["w_star"] != W_STAR or fz10["csls_k"] != K_STAR:
        bad.append("10B frozen w* / k")
    if bad:
        raise SystemExit("constants differ from the frozen configs: %s" % bad)


def committed_clean(path):
    import subprocess
    r = os.path.relpath(path, REPO).replace(os.sep, "/")
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", r], capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", r], capture_output=True,
                           text=True).stdout.strip()
    return tracked and not dirty


def split_players(ix):
    """PLAN section 1: per level by sha256(SPLIT_SALT + pid): eval = first max(40, ceil(n/2)), dev = the rest."""
    dev, ev = [], []
    for lv in LEVELS:
        pl = sorted((p for p in ix.players if ix.level[p] == lv),
                    key=lambda p: hashlib.sha256((SPLIT_SALT + p).encode()).hexdigest())
        ne = max(PER_LEVEL, -(-len(pl) // 2))
        ev += pl[:ne]
        dev += pl[ne:]
    return dev, ev


def _h(seed, p):
    return hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest()


def choose_eval(ix, players, seed):
    """40 per level from `players` by sha256('{seed}|{pid}') (p06_sim.choose_pool's rule on a subset)."""
    s = set(players)
    pool = []
    for lv in LEVELS:
        el = sorted((p for p in s if ix.level[p] == lv), key=lambda p: _h(seed, p))
        if len(el) < PER_LEVEL:
            raise RuntimeError("level %s has only %d players in this half" % (lv, len(el)))
        pool += el[:PER_LEVEL]
    return pool


def choose_dev(ix, players, seed):
    """PLAN section 3: per level min(40, n) by the seed hash, then filled to 400 from the rest of the dev players in
    the same hash order."""
    s = set(players)
    pool = []
    for lv in LEVELS:
        el = sorted((p for p in s if ix.level[p] == lv), key=lambda p: _h(seed, p))
        pool += el[:PER_LEVEL]
    rest = sorted((p for p in s if p not in set(pool)), key=lambda p: _h(seed, p))
    need = N_POOL - len(pool)
    if need < 0 or len(rest) < need:
        raise RuntimeError("dev half too small for a %d-player pool" % N_POOL)
    return pool + rest[:need]


# ================================================================ pools
def owners_for(struct, seed, n):
    """p06_sim.make_pool's owner rule (verbatim) for o2o / wr / d<m> / u<m>."""
    if struct in ("o2o", "base"):
        return list(range(n))
    r2 = random.Random("scen|%s|%d" % (struct, seed))
    if struct == "wr":
        return [r2.randrange(n) for _ in range(n)]
    if struct.startswith("d"):
        m = int(struct[1:])
        perm = list(range(n))
        r2.shuffle(perm)
        owner = perm[:n - m] + perm[n - 2 * m:n - m]
        r2.shuffle(owner)
        return owner
    if struct.startswith("u"):
        m = int(struct[1:])
        used = r2.sample(range(n), m)
        owner = used + [r2.choice(used) for _ in range(n - m)]
        r2.shuffle(owner)
        return owner
    raise ValueError(struct)


def repair(struct, seed, owner, ks, cap):
    """p10_fuse._wr_questions' capacity repair (verbatim rule, PREREG_10 rev 2 fallback) for any structure."""
    owner, ks = list(owner), list(ks)
    n = len(owner)
    n_rep = n_fb = 0
    r3 = random.Random("repair|%s|%d" % (struct, seed))
    for _ in range(10000):
        need = Counter()
        for q in range(n):
            need[owner[q]] += ks[q]
        bad = [i for i in need if need[i] > cap[i]]
        if not bad:
            break
        i = bad[0]
        qs = sorted([q for q in range(n) if owner[q] == i], key=lambda q: -ks[q])
        q = qs[0]
        cands = [q2 for q2 in range(n) if owner[q2] != i and ks[q2] < ks[q]
                 and need[owner[q2]] - ks[q2] + ks[q] <= cap[owner[q2]]]
        if cands:
            q2 = r3.choice(cands)
            ks[q], ks[q2] = ks[q2], ks[q]
        else:
            others = [j for j in set(owner) if j != i and need[j] + ks[q] <= cap[j]]
            if not others:
                others = [j for j in range(n) if j != i and need[j] + ks[q] <= cap[j]]
                n_fb += 1
            if not others:
                raise RuntimeError("seed %d %s: repair found no room" % (seed, struct))
            owner[q] = r3.choice(others)
        n_rep += 1
    else:
        raise RuntimeError("could not repair the k assignment")
    return owner, ks, n_rep, n_fb


def build_pool(ix, pool, seed, struct):
    """One pool (PLAN 3 / 4) on the mock sets: candidates = each player's first 100 sets games; questions from the
    base reserve (capacity repair) or, for nov4, p10_fuse's nov4 rule. -> p10_fuse.build_pool's dict."""
    n = len(pool)
    ks = pf10.k_list(n, sim.K_VALUES)
    random.Random("k|%d" % seed).shuffle(ks)
    cand = [ix.by[p]["cand"][:sim.N_CAND] for p in pool]
    if any(len(c) < sim.N_CAND for c in cand):
        raise RuntimeError("a player has fewer than 100 candidate games")
    if struct == "nov4":
        nov = [ix.by[p]["nov4"] for p in pool]
        q, owner, k_eff, n_rep = pf10._nov4_questions(seed, pool, ks, nov)
        info = {"n_replaced_owner": n_rep, "mean_k_eff": float(np.mean(k_eff)),
                "short_questions": int(sum(1 for a, b in zip(k_eff, ks) if a < b))}
        ks = k_eff
    else:
        owner0 = owners_for(struct, seed, n)
        cap = [len(ix.by[p]["base"]) for p in pool]
        owner, ks, n_rep, n_fb = repair(struct, seed, owner0, ks, cap)
        pos = [0] * n
        q = []
        for i in range(n):
            o = owner[i]
            q.append(ix.by[pool[o]]["base"][pos[o]:pos[o] + ks[i]])
            pos[o] += ks[i]
        info = {"n_repairs": n_rep, "n_fallback_owner_moves": n_fb}
    cnt = Counter(owner)
    info.update({"n_distinct_owner": len(cnt), "players_2plus": int(sum(1 for v in cnt.values() if v >= 2)),
                 "k_of_multi": [int(ks[i]) for i in range(n) if cnt[owner[i]] >= 2]})
    return {"pool": pool, "cand_idx": cand, "q_idx": q, "owner": np.asarray(owner, np.int64),
            "k": np.asarray(ks, np.int64), "scenario": struct, "seed": seed, "info": info}


# ================================================================ worker state
_W = {}


def _init(want_pa):
    sys.dont_write_bytecode = True
    st.install_guard()
    store = sim.Store(FEATS06)
    _W["store"] = store
    _W["lda06"] = {"B": sim.LdaData(FEATS06, "B", store)}
    _W["cnn"] = sim.Cnn(CNN06, store)
    _W["X"] = np.load(SIDE0, mmap_mode="r")
    _W["usable"] = np.load(USABLE)
    ix = pf10.SetsIndex(read_json(SETS))
    _W["ix"] = ix
    _W["row2set"] = {int(r): j for j, r in enumerate(ix.store_row.tolist())}
    _W["want_pa"] = want_pa
    fz = read_json(FZ06)
    _W["switch06"] = {n: (v["r_star"], bool(v["usable"]) and bool(fz["switch_used_at_all"]))
                      for n, v in fz["switch"].items()}


def fit_models(pool):
    """06 LDA without the pool's players; P1 LDA on the usable mock games of the players outside the pool (as the
    PREREG_10 mock) -> (m06, Z [all sets games x 64])."""
    ix, store, X, usable = _W["ix"], _W["store"], _W["X"], _W["usable"]
    pset = set(pool)
    m06 = {"B": _W["lda06"]["B"].fit_excluding(store, pset)}
    train = usable & ~np.isin(ix.pid, np.asarray(sorted(pset)))
    mp1 = pf10.fit_p1_lda(X, ix.pid, train)
    rows = np.flatnonzero(usable)
    Z = np.zeros((ix.n, st.LDA_DIM))
    Z[rows] = pf10.project_p1(X, mp1, rows)
    return m06, Z, [mp1["n_players"], mp1["n_games"]]


def terms(rec, m06, want):
    return sim.pool_terms(_W["store"], rec, _W["lda06"], m06, _W["cnn"], want=want, colours=("B",))


def set_groups(groups):
    r2s = _W["row2set"]
    return [[r2s[int(r)] for r in g] for g in groups]


def pts_of(S, owner):
    return st.points(st.ranks(S, owner))


def on_matrix(F, rest):
    """Rank 1 = Hungarian on z(F); ranks 2-5 by `rest` (06-ON: CSLS3(F); B'-ON: CSLS10(F_B))."""
    return st.with_top1(rest, st.hungarian_top1(st.zrows(F)))


def pseudo_rounds(rec, m06, Z, rounds, want):
    """sr06 pseudo pools of a pool (p06_sim.sr_pool) with their 06 terms and, when Z is given, z(T_P1) on the sets
    indices of the pseudo pools' store rows. Shared by pool_eval, the dev detector and seam S2."""
    prs, _ids = sim.sr_pool(_W["store"], rec, rounds=tuple(rounds))
    for p in prs:
        Tp = terms(p, m06, want)
        zTp = None if Z is None else pf10.z_p1(Z, set_groups(p["cand_rows"]), set_groups(p["q_rows"]),
                                               _W["usable"])
        yield p, Tp, zTp


# ================================================================ one pool, everything the strategies need
def pool_eval(pr, m06, Z, full=True):
    """PLAN section 2 on one pool. full=False: only V2 / 10B@16 (seam S1)."""
    ix, usable = _W["ix"], _W["usable"]
    rec = pf10.to_store_rec(ix, pr)
    own = pr["owner"]
    T = terms(rec, m06, ("O", "JO", "S2", "L", "E0"))
    zT = pf10.z_p1(Z, pr["cand_idx"], pr["q_idx"], usable)
    combos = sim.combo_list(None)
    Fc = {n: sim.combo_F(T, "E0", w) for n, w in combos}
    out = {"pts": {"V2": pts_of(st.csls(Fc["V2"], K_STAR), own),
                   "10B_V2": pts_of(pf10.fuse_10b(Fc["V2"], zT, W_STAR, K_STAR), own)}}
    if not full:
        return out
    t0 = time.time()
    sr = {n: [] for n, _w in combos}
    sr["R02"] = []
    sr10 = {n: [] for n, _w in combos}
    F01, FB01 = {n: [] for n, _w in combos}, {n: [] for n, _w in combos}
    for p, Tp, zTp in pseudo_rounds(rec, m06, Z, tuple(range(st.SR_ROUNDS)), ("O", "JO", "S2", "L", "E0")):
        qi = p["owner"]
        for n, w in combos:
            F = sim.combo_F(Tp, "E0", w)
            sr[n] += pts_of(st.csls(F, K_STAR), qi).tolist()
            sr10[n] += pts_of(pf10.fuse_10b(F, zTp, W_STAR, K_STAR), qi).tolist()
            if p["round"] in (0, 1):
                F01[n].append(F)
                FB01[n].append(np.asarray(F, np.float64) + W_STAR * zTp)
        sr["R02"] += pts_of(sim.combo_F(Tp, "E0", None), qi).tolist()
    srm = {n: float(np.mean(v)) for n, v in sr.items()}
    sr10m = {n: float(np.mean(v)) for n, v in sr10.items()}
    order = [n for n, _w in combos]
    best = max(srm[n] for n in order)
    chosen = next(n for n in order if best - srm[n] <= 0.003 + 1e-15)
    F = Fc[chosen]
    FB = np.asarray(F, np.float64) + W_STAR * zT
    off10 = pf10.fuse_10b(F, zT, W_STAR, K_STAR)
    det06 = st.detector(F, F01[chosen])
    detB = st.detector(FB, FB01[chosen])
    r_star06, usable06 = _W["switch06"][chosen]
    guard = pf10.rho_raw(_W["store"], rec)
    out["pts"].update({
        "Q04": pts_of(sim.combo_F(T, "E0", None), own),
        "E0": pts_of(st.zscore_rows(*T["E0"]), own),
        "off06": pts_of(st.csls(F, K_STAR), own),
        "on06": pts_of(on_matrix(F, st.csls(F, K_SWITCH06)), own),
        "off10": pts_of(off10, own),
        "onB": pts_of(on_matrix(FB, off10), own)})
    out.update({"chosen": chosen, "sr06": srm, "sr06_10B": sr10m,
                "gate06": bool(srm[chosen] - srm["R02"] >= 0), "gateB": bool(sr10m[chosen] - srm[chosen] >= 0),
                "level": int(guard["level"]), "rho": float(guard["rho"]),
                "r06": float(det06["r_forced_cost"]), "r_star06": r_star06, "usable06": usable06,
                "rB": float(detB["r_forced_cost"]), "sr_seconds": round(time.time() - t0, 1)})
    return out


# ================================================================ jobs
def _job_eval(job):
    """One eval seed (private or public role): all EVAL_SCEN pools on that seed's player set."""
    seed, role, players = job
    t0 = time.time()
    ix = _W["ix"]
    pool = choose_eval(ix, players, seed)
    m06, Z, lda_p1 = fit_models(pool)
    rows = []
    for scen in EVAL_SCEN:
        pr = build_pool(ix, pool, seed, scen)
        r = pool_eval(pr, m06, Z, full=True)
        r["pts"] = {k: v.tolist() for k, v in r["pts"].items()}
        r.update({"seed": seed, "role": role, "scen": scen, "info": pr["info"],
                  "owner_pid": [pool[o] for o in pr["owner"].tolist()]})
        rows.append(r)
    return {"seed": seed, "role": role, "rows": rows, "lda_p1": lda_p1, "seconds": round(time.time() - t0, 1)}


def _job_dev(seed_players):
    """One dev seed: the 11 structures; F_B = F_V2(E0 or PA) + 16 z(T_P1); pseudo forced cost from the o2o pool's
    sr06 rounds 0 and 1 (shared by the structures: same candidates)."""
    seed, players = seed_players
    t0 = time.time()
    ix, usable = _W["ix"], _W["usable"]
    pool = choose_dev(ix, players, seed)
    m06, Z, lda_p1 = fit_models(pool)
    want = ("O", "JO", "S2", "L", "PA", "E0")
    srcs = ("E0", "PA")
    pr0 = build_pool(ix, pool, seed, "o2o")
    rec0 = pf10.to_store_rec(ix, pr0)
    pseudo = {s: [] for s in srcs}
    for _p, Tp, zTp in pseudo_rounds(rec0, m06, Z, (0, 1), want):
        for s in srcs:
            pseudo[s].append(sim.forced_cost(np.asarray(sim.combo_F(Tp, s, sim.V2_W), np.float64) + W_STAR * zTp))
    rows = []
    for struct in DEV_STRUCTS:
        pr = pr0 if struct == "o2o" else build_pool(ix, pool, seed, struct)
        rec = pf10.to_store_rec(ix, pr)
        T = terms(rec, m06, want)
        zT = pf10.z_p1(Z, pr["cand_idx"], pr["q_idx"], usable)
        res = {}
        for s in srcs:
            FB = np.asarray(sim.combo_F(T, s, sim.V2_W), np.float64) + W_STAR * zT
            off = st.csls(FB, K_STAR)
            on = on_matrix(FB, off)
            fc = sim.forced_cost(FB)
            res["B|%s" % s] = {"off_10": st.score(off, pr["owner"]), "on": st.score(on, pr["owner"]),
                               "forced_cost": fc, "r_forced_cost": fc / max(float(np.mean(pseudo[s])), 1e-9)}
        rows.append({"seed": seed, "scen": struct, "res": res, "info": {k: v for k, v in pr["info"].items()
                                                                         if k != "k_of_multi"}})
    return {"seed": seed, "rows": rows, "pseudo": pseudo, "lda_p1": lda_p1, "seconds": round(time.time() - t0, 1)}


SEAM_SCEN = (("base", "o2o"), ("nov4", "nov4"), ("wr", "wr"))     # (mock row name, this program's structure)


def _job_seam(seed):
    """S1 (PLAN 6 + section 10 rev 1): the mock's 926-player pool of this seed built with THIS program's build_pool
    for base (o2o) / nov4 / wr; returns owner ids, k, V2 / 10B@16 points and the guard (rho, level)."""
    ix = _W["ix"]
    pool = pf10.choose_players(ix, seed, PER_LEVEL)
    m06, Z, _l = fit_models(pool)
    out = {"seed": seed, "pools": {}}
    for mname, struct in SEAM_SCEN:
        pr = build_pool(ix, pool, seed, struct)
        full = seed == SEAM_SEEDS[0] and struct == "o2o"
        r = pool_eval(pr, m06, Z, full=full)
        g = r if full else pf10.rho_raw(_W["store"], pf10.to_store_rec(ix, pr))
        rec = {"owner_pid": [pool[o] for o in pr["owner"].tolist()], "k": pr["k"].tolist(),
               "V2": r["pts"]["V2"].tolist(), "10B": r["pts"]["10B_V2"].tolist(),
               "rho": float(g["rho"]), "level": int(g["level"])}
        if full:
            rec["s3"] = {k: r[k] for k in ("chosen", "sr06", "sr06_10B", "gate06", "gateB", "level", "rho")}
            rec["s3"]["off06_equals_V2"] = bool(np.array_equal(r["pts"]["off06"], r["pts"]["V2"])) \
                if r["chosen"] == "V2" else None
        out["pools"][mname] = rec
    return out


def _job_seam_det(_x):
    """S2: PREREG_06 switch seed 500 o2o (p06_sim.make_pool on the 06 store): r_forced_cost and the two pseudo
    forced costs of V2|E0 and V2|PA, through pseudo_rounds (the function pool_eval and dev use)."""
    store = _W["store"]
    rec = sim.make_pool(store.black_cache(), 500, "o2o")
    m06 = {"B": _W["lda06"]["B"].fit_excluding(store, set(rec["pool"]))}
    want = ("O", "JO", "S2", "L", "PA", "E0")
    T = terms(rec, m06, want)
    Tps = [Tp for _p, Tp, _z in pseudo_rounds(rec, m06, None, (0, 1), want)]
    out = {}
    for s in ("E0", "PA"):
        det = st.detector(sim.combo_F(T, s, sim.V2_W), [sim.combo_F(Tp, s, sim.V2_W) for Tp in Tps])
        out["V2|%s" % s] = {"r": det["r_forced_cost"], "pseudo": det["pseudo_forced_cost"]}
    return out


def _save_job(jdir, r):
    if jdir is not None:
        p = os.path.join(jdir, "%s_%d.json" % (r["role"], r["seed"]))
        r = dict(r, code_sha256=CODE_SHA)
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            json.dump(st.json_safe(r), f)
        os.replace(p + ".tmp", p)


def run_jobs(fn, jobs, workers, want_pa=False, jdir=None):
    out = []
    if not jobs:
        return out
    if workers <= 1:
        _init(want_pa)
        for j in jobs:
            r = fn(j)
            _save_job(jdir, r)
            out.append(r)
        return out
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(want_pa,)) as pool:
        for r in pool.imap_unordered(fn, jobs):
            _save_job(jdir, r)
            out.append(r)
            print("  job done %d/%d" % (len(out), len(jobs)), flush=True)
    return out


# ================================================================ strategies (PLAN section 2)
def mean(x):
    return float(np.mean(x))


def strategies(priv, pub, r_star_b, ignore_guard=False):
    """-> dict of private per-question point arrays for (a) current, (b) B'-OFF, (c) candidate, (c0) candidate
    without the veto, plus the decision record."""
    def lvl_ok(r):
        return ignore_guard or r["level"] == 1

    def out06(r):
        if r["gate06"]:
            return r["pts"]["off06"]
        return r["pts"]["E0"] if (r["level"] == 3 and not ignore_guard) else r["pts"]["Q04"]

    def out10(r):
        return r["pts"]["off10"] if (r["gate06"] and r["gateB"]) else out06(r)

    P = {k: np.asarray(v) for k, v in priv["pts"].items()}
    U = {k: np.asarray(v) for k, v in pub["pts"].items()}
    priv = dict(priv, pts=P)
    pub = dict(pub, pts=U)
    b_off = out10(priv)
    # (a) current
    cond06 = priv["gate06"] and pub["gate06"] and lvl_ok(priv) and lvl_ok(pub)
    open06 = bool(priv["usable06"] and priv["r_star06"] is not None and priv["r06"] <= priv["r_star06"])
    d06 = mean(U["on06"]) - mean(U["off06"]) if pub["gate06"] else None
    on06_final = bool(cond06 and open06 and d06 is not None and d06 >= 0)
    cur = P["on06"] if on06_final else b_off
    # (c) candidate
    condB = (priv["gate06"] and priv["gateB"] and pub["gate06"] and pub["gateB"] and lvl_ok(priv) and lvl_ok(pub))
    # PLAN section 10 rev 1: r*_B is calibrated on F_B built on V2 only, so B' opens only when the private pool chose V2
    openB = bool(r_star_b is not None and priv["chosen"] == "V2" and priv["rB"] <= r_star_b)
    dB = mean(U["onB"]) - mean(U["off10"]) if (pub["gate06"] and pub["gateB"]) else None
    if condB:
        onB_final = bool(openB and dB >= 0)
        cand = P["onB"] if onB_final else P["off10"]
        cand0 = P["onB"] if openB else P["off10"]
    else:
        onB_final = False
        cand = cur
        cand0 = cur
    dec = {"cond06": bool(cond06), "open06": open06, "d06": d06, "on06_final": on06_final,
           "condB": bool(condB), "openB": openB, "dB": dB, "onB_final": onB_final,
           "priv_level": priv["level"], "pub_level": pub["level"], "priv_gateB": priv["gateB"],
           "pub_gateB": pub["gateB"], "priv_gate06": priv["gate06"], "priv_chosen": priv["chosen"],
           "a_06on_but_c_not_BON": bool(on06_final and condB and not onB_final)}
    return {"a": cur, "b": b_off, "c": cand, "c0": cand0}, dec


def boot(diffs, owners):
    _u, sums, counts = pf10.player_sums(diffs, owners)
    return pf10.player_boot(sums, counts, BOOT_N, BOOT_SEED)


# ================================================================ commands
def cmd_seam(a):
    t0 = time.time()
    inp = verify_inputs()
    check_constants()
    res = run_jobs(_job_seam, list(SEAM_SEEDS), a.workers)
    det = run_jobs(_job_seam_det, [0], 1)[0]
    mp_rows = {(r["seed"], r["scen"]): r for r in read_json(MOCK_POINTS)["rows"]}
    rep_all = read_json(MOCK_REPORT)["record_only"]["scores"]
    per = []
    means = {m: {"V2": [], "10B": []} for m, _s in SEAM_SCEN}
    for r in sorted(res, key=lambda x: x["seed"]):
        for mname, _s in SEAM_SCEN:
            mine, m = r["pools"][mname], mp_rows[(r["seed"], mname)]
            per.append({"seed": r["seed"], "scen": mname,
                        "owner_identical": mine["owner_pid"] == m["owner_pid"], "k_identical": mine["k"] == m["k"],
                        "V2_identical": mine["V2"] == m["pts"]["V2"], "10B_identical": mine["10B"] == m["pts"]["10B@16"],
                        "rho_abs_diff": abs(mine["rho"] - m["guard"]["rho"]),
                        "level_identical": mine["level"] == m["guard"]["level"]})
            means[mname]["V2"].append(np.mean(mine["V2"]))
            means[mname]["10B"].append(np.mean(mine["10B"]))
    mean_cmp = {m: {"V2": float(np.mean(v["V2"])), "mock_V2": rep_all[m]["eval"]["V2"],
                    "10B16": float(np.mean(v["10B"])), "mock_10B16": rep_all[m]["eval"]["10B@16"]}
                for m, v in means.items()}
    sw = read_json(SWITCH06)
    row = next(x for x in sw["rows"] if x["seed"] == 500 and x["scen"] == "o2o")
    pz = sw["pseudo_forced_cost"]["500"]
    s2 = {k: {"recomputed": det[k]["r"], "switch_report": row["res"][k]["r_forced_cost"],
              "abs_diff": abs(det[k]["r"] - row["res"][k]["r_forced_cost"]),
              "pseudo_recomputed": det[k]["pseudo"], "pseudo_switch_report": pz[k],
              "pseudo_abs_diff": max(abs(x - y) for x, y in zip(det[k]["pseudo"], pz[k]))} for k in det}
    s1_ok = all(p["owner_identical"] and p["k_identical"] and p["V2_identical"] and p["10B_identical"]
                and p["rho_abs_diff"] <= SEAM_TOL and p["level_identical"] for p in per) and \
        all(abs(v["V2"] - v["mock_V2"]) <= SEAM_TOL and abs(v["10B16"] - v["mock_10B16"]) <= SEAM_TOL
            for v in mean_cmp.values())
    s2_ok = all(v["abs_diff"] <= DET_TOL and v["pseudo_abs_diff"] <= DET_TOL for v in s2.values())
    s3 = next(r["pools"]["base"]["s3"] for r in res if "s3" in r["pools"]["base"])
    rep = {"what": "prereg20 seams (PLAN section 6, section 10 rev 1)", "inputs": inp,
           "S1": {"passed": bool(s1_ok), "per_pool": per, "means": mean_cmp},
           "S2": {"passed": bool(s2_ok), "values": s2},
           "S3_record_only": s3, "passed": bool(s1_ok and s2_ok), "git_head": git_head(),
           "env": st.env_versions(), "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(OUT, "seam_report.json"), rep)
    print(json.dumps({"S1": s1_ok, "S2": s2_ok, "S2_values": s2}, indent=1))
    return 0 if rep["passed"] else 1


def cmd_dev(a):
    import p06_player as P6
    t0 = time.time()
    inp = verify_inputs()
    check_constants()
    seamp = os.path.join(OUT, "seam_report.json")
    if not read_json(seamp).get("passed") or not committed_clean(seamp):
        raise SystemExit("seam_report.json did not pass or is not committed")
    if os.path.exists(os.path.join(OUT, "eval_report.json")):
        raise SystemExit("eval_report.json exists: dev cannot be rerun after the eval")
    ix = pf10.SetsIndex(read_json(SETS))
    dev, ev = split_players(ix)
    res = run_jobs(_job_dev, [(s, dev) for s in DEV_SEEDS], a.workers, want_pa=True)
    rows = [x for r in sorted(res, key=lambda r: r["seed"]) for x in r["rows"]]
    thr = {}
    for s in ("E0", "PA"):
        t, det = P6.select_threshold(rows, "B|%s" % s, "off_10")
        thr[s] = {"thr": t, **det}
    r_star = P6.r_star_of([thr[s]["thr"] for s in ("E0", "PA")])
    summ = {}
    for s in ("E0", "PA"):
        key = "B|%s" % s
        for sc in DEV_STRUCTS:
            R = [x for x in rows if x["scen"] == sc]
            g = [x["res"][key]["on"] - x["res"][key]["off_10"] for x in R]
            opens = [x["res"][key]["r_forced_cost"] <= r_star for x in R] if r_star is not None else [False] * len(R)
            summ.setdefault(key, {})[sc] = {
                "off_mean": mean([x["res"][key]["off_10"] for x in R]), "on_mean": mean([x["res"][key]["on"] for x in R]),
                "gain_if_on_mean": mean(g), "open_share_at_r_star": mean(opens),
                "gain_at_r_star_mean": mean([gi if o else 0.0 for gi, o in zip(g, opens)]),
                "r_median": float(np.median([x["res"][key]["r_forced_cost"] for x in R]))}
    rep = {"what": "prereg20 dev calibration of r*_B (PLAN section 3)", "inputs": inp,
           "split": {"dev_players": len(dev), "eval_players": len(ev),
                     "dev_by_level": dict(Counter(ix.level[p] for p in dev)),
                     "eval_by_level": dict(Counter(ix.level[p] for p in ev)), "salt": SPLIT_SALT},
           "seeds": list(DEV_SEEDS), "structures": list(DEV_STRUCTS), "thresholds": thr, "r_star_B": r_star,
           "summary": summ, "rows": rows, "pseudo_forced_cost": {r["seed"]: r["pseudo"] for r in res},
           "lda_p1_train": {r["seed"]: r["lda_p1"] for r in res}, "seconds_per_seed": {r["seed"]: r["seconds"] for r in res},
           "seam_report_sha256": sha_raw(os.path.join(OUT, "seam_report.json")), "git_head": git_head(),
           "env": st.env_versions(), "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(OUT, "dev_report.json"), rep)
    print(json.dumps({"r_star_B": r_star, "thresholds": thr}, indent=1))
    return 0


def judge(results, r_star_b):
    """PLAN sections 4-5 on the eval job results -> report dict."""
    by = {(x["seed"], x["role"], x["scen"]): x for r in results for x in r["rows"]}
    report = {"by_scenario": {}, "decisions": {}}
    for scen in EVAL_SCEN:
        for variant in (("registered", False), ("guard_ignored", True)) if scen == "nov4" else (("registered", False),):
            vname, ig = variant
            arrs = {k: [] for k in ("a", "b", "c", "c0")}
            owners, decs = [], []
            for i in range(EVAL_TRIALS):
                pv = by[(PRIV_SEED0 + i, "private", scen)]
                pb = by[(PUB_SEED0 + i, "public", scen)]
                S, dec = strategies(pv, pb, r_star_b, ignore_guard=ig)
                for k in arrs:
                    arrs[k].append(S[k])
                owners += pv["owner_pid"]
                dec["trial"] = i
                decs.append(dec)
            A = {k: np.concatenate(v) for k, v in arrs.items()}
            key = scen if scen != "nov4" else "nov4_%s" % vname
            rec = {"questions": int(len(A["a"])), "players": int(len(set(owners))),
                   "means": {k: float(A[k].mean()) for k in A},
                   "c_minus_a": boot(A["c"] - A["a"], owners), "b_minus_a": boot(A["b"] - A["a"], owners),
                   "c_minus_b": boot(A["c"] - A["b"], owners), "c0_minus_a": boot(A["c0"] - A["a"], owners),
                   "counts": {
                       "06ON_final": sum(d["on06_final"] for d in decs), "06_detector_open": sum(d["open06"] for d in decs),
                       "06_cond": sum(d["cond06"] for d in decs),
                       "06_vetoed": sum(1 for d in decs if d["cond06"] and d["open06"] and not d["on06_final"]),
                       "BON_final": sum(d["onB_final"] for d in decs), "B_detector_open": sum(d["openB"] for d in decs),
                       "B_cond": sum(d["condB"] for d in decs),
                       "B_vetoed": sum(1 for d in decs if d["condB"] and d["openB"] and not d["onB_final"]),
                       "priv_level": dict(Counter(d["priv_level"] for d in decs)),
                       "priv_gateB_false": sum(1 for d in decs if not d["priv_gateB"]),
                       "priv_gate06_false": sum(1 for d in decs if not d["priv_gate06"]),
                       "priv_chosen": dict(Counter(d["priv_chosen"] for d in decs)),
                       "a_06on_but_c_not_BON": sum(d["a_06on_but_c_not_BON"] for d in decs)},
                   "d06": [d["d06"] for d in decs], "dB": [d["dB"] for d in decs]}
            report["by_scenario"][key] = rec
            report["decisions"][key] = decs
    bs = report["by_scenario"]
    g1 = bs["o2o"]["c_minus_a"]
    g1_ok = g1[0] >= G1_MIN - 1e-15 and g1[1] > 0
    fo = [d["openB"] for s in ("d40", "d120") for d in report["decisions"][s]]
    false_on = float(np.mean(fo))
    g2_ok = all(bs[s]["c_minus_a"][0] >= G2_MIN - 1e-15 for s in ("d40", "d120")) and false_on <= FALSE_ON_MAX + 1e-15
    g3 = bs["nov4_guard_ignored"]["c_minus_a"]
    g3_ok = g3[0] >= G3_MIN - 1e-15
    report["gates"] = {
        "G1_o2o": {"passed": bool(g1_ok), "c_minus_a": g1, "rule": "mean >= +0.003 and 95% lower bound > 0"},
        "G2_non_o2o": {"passed": bool(g2_ok), "d40": bs["d40"]["c_minus_a"], "d120": bs["d120"]["c_minus_a"],
                       "B_detector_false_on_share": false_on, "B_detector_false_on_pools": int(sum(fo)),
                       "pools": len(fo), "rule": "d40 and d120 each mean >= -0.002; private B' detector opens in <= 1% "
                                                 "of the 80 non-o2o private pools"},
        "G3_nov4_guard_ignored": {"passed": bool(g3_ok), "c_minus_a": g3, "rule": "mean >= -0.002"},
        "passed": bool(g1_ok and g2_ok and g3_ok)}
    return report


def cmd_eval(a):
    t0 = time.time()
    inp = verify_inputs()
    check_constants()
    seamp = os.path.join(OUT, "seam_report.json")
    devp = os.path.join(OUT, "dev_report.json")
    if not read_json(seamp).get("passed") or not committed_clean(seamp):
        raise SystemExit("seam_report.json did not pass or is not committed")
    if not all(committed_clean(x) for x in (devp, os.path.join(HERE, "p20_sim.py"), os.path.join(HERE, "PLAN.md"))):
        raise SystemExit("dev_report.json, p20_sim.py and PLAN.md must be committed and clean before the eval")
    dev_rep = read_json(devp)
    r_star_b = dev_rep["r_star_B"]
    out_path = os.path.join(OUT, "eval_report.json")
    tag = ""
    if os.path.exists(out_path):
        if not (a.rerun and a.reason):
            raise SystemExit("eval_report.json exists: the eval is judged once (PLAN section 7); a rerun needs "
                             "--rerun --reason and a PLAN section 10 revision")
        n = 1
        while os.path.exists(os.path.join(OUT, "eval_report_rerun%d.json" % n)):
            n += 1
        tag = "_rerun%d" % n
        out_path = os.path.join(OUT, "eval_report%s.json" % tag)
    ix = pf10.SetsIndex(read_json(SETS))
    dev, ev = split_players(ix)
    if set(dev) & set(ev):
        raise SystemExit("dev and eval overlap")
    jobs = [(PRIV_SEED0 + i, "private", ev) for i in range(EVAL_TRIALS)] + \
        [(PUB_SEED0 + i, "public", ev) for i in range(EVAL_TRIALS)]
    # review fix 5: one checkpoint file per job (reused on a restart of the same run)
    jdir = os.path.join(WORK, "eval_jobs%s" % tag)
    os.makedirs(jdir, exist_ok=True)
    res, todo = [], []
    for j in jobs:
        jp = os.path.join(jdir, "%s_%d.json" % (j[1], j[0]))
        if os.path.exists(jp):
            cached = read_json(jp)
            if cached.get("code_sha256") != CODE_SHA or (cached.get("seed"), cached.get("role")) != (j[0], j[1]):
                raise SystemExit("%s was written by another p20_sim.py or job: delete eval_jobs%s first" % (jp, tag))
            res.append(cached)
        else:
            todo.append(j)
    print("eval: %d jobs cached, %d to run" % (len(res), len(todo)), flush=True)
    res += run_jobs(_job_eval, todo, a.workers, jdir=jdir)
    raw = os.path.join(WORK, "eval_points%s.json" % tag)
    with open(raw, "w", encoding="utf-8", newline="\n") as f:
        json.dump(st.json_safe({"results": sorted(res, key=lambda r: (r["role"], r["seed"]))}), f)
    rep = judge(res, r_star_b)
    rep.update({"what": "prereg20 eval (PLAN sections 4-5): (c) B'-ON/OFF switch vs (a) the current full registered "
                        "strategy; judged once", "r_star_B": r_star_b, "w_star": W_STAR, "k_star": K_STAR,
                "trials": EVAL_TRIALS, "scenarios": list(EVAL_SCEN),
                "pool_info": {"%s|%d|%s" % (x["role"], x["seed"], x["scen"]): {k: v for k, v in x["info"].items()
                                                                              if k != "k_of_multi"}
                              for r in res for x in r["rows"]},
                "multi_k_hist": {s: dict(Counter(k for r in res for x in r["rows"] if x["scen"] == s
                                                 for k in x["info"]["k_of_multi"])) for s in EVAL_SCEN},
                "inputs": inp, "dev_report_sha256": sha_raw(devp), "raw_points": os.path.relpath(raw, REPO),
                "raw_points_sha256": sha_raw(raw), "seconds_per_job": {"%s|%d" % (r["role"], r["seed"]): r["seconds"]
                                                                       for r in res},
                "git_head": git_head(), "env": st.env_versions(), "seconds": round(time.time() - t0, 1),
                "rerun": a.reason if tag else None})
    write_json(out_path, rep)
    print(json.dumps({"gates": rep["gates"]}, indent=1))
    return 0


def cmd_smoke(a):
    """Timing and shapes only (PLAN 7.3): one eval-shaped job on seed 9000 (not a judged seed); prints no score."""
    t0 = time.time()
    verify_inputs()
    ix = pf10.SetsIndex(read_json(SETS))
    dev, ev = split_players(ix)
    _init(False)
    seed = 9000
    pool = choose_eval(ix, ev, seed)
    t1 = time.time()
    m06, Z, lda = fit_models(pool)
    t2 = time.time()
    shapes = {}
    for scen in EVAL_SCEN:
        pr = build_pool(ix, pool, seed, scen)
        t3 = time.time()
        r = pool_eval(pr, m06, Z, full=True)
        shapes[scen] = {"q": len(pr["q_idx"]), "distinct_owner": pr["info"]["n_distinct_owner"],
                        "players_2plus": pr["info"]["players_2plus"], "seconds": round(time.time() - t3, 1),
                        "sr_seconds": r["sr_seconds"], "keys": sorted(r["pts"])}
    print(json.dumps({"dev": len(dev), "eval": len(ev), "fit_seconds": round(t2 - t1, 1), "lda_p1": lda,
                      "pools": shapes, "total": round(time.time() - t0, 1)}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("seam", "dev", "eval", "smoke"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(k):
            raise SystemExit("%s is set: unset it (as tools/p10_fuse.py mock)" % k)
    return {"seam": cmd_seam, "dev": cmd_dev, "eval": cmd_eval, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
