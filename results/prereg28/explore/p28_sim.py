"""prereg28 explore (competition 1): is the 06-ON detector with its frozen thresholds still reliable on N x N pools
(400 <= N <= 800)? Fit-player simulation pools only. Rules: results/prereg28/explore/PLAN.md.

    PY = .venv/Scripts/python.exe  (CPU; Git Bash; repository root; PYTHONDONTWRITEBYTECODE=1; no thread variables)
    PY results/prereg28/explore/p28_sim.py smoke              (timing and shapes only; prints no score and no r)
    PY results/prereg28/explore/p28_sim.py seam  [--workers 6] -> results/prereg28/explore/seam_report.json
    PY results/prereg28/explore/p28_sim.py eval  [--workers 6] -> results/prereg28/explore/eval_report.json

Only imports the existing tools (p06_player workers, p06_sim, p06_style); writes only results/prereg28/explore and
data/prereg28. Reads data/prereg06 (06 fit store, LDA rows, fit CNN) and results/prereg06/design/switch_report.json.
Never data/local (p06_style's import guard is on); never tools/score.py; no GPU.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

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

import p06_player as pp  # noqa: E402
import p06_sim as sim  # noqa: E402
from c10data import LEVELS  # noqa: E402

OUT = HERE
DATA = os.path.join(REPO, "data", "prereg28")
JOBS = os.path.join(DATA, "jobs")
SWITCH_REPORT = os.path.join(REPO, "results", "prereg06", "design", "switch_report.json")

SEEDS = tuple(range(600, 620))                  # PLAN 3: the PREREG_06 evaluation seed numbers
N_GRID = (500, 600, 700, 800)                   # PLAN 5: G1 / G2 sizes (400 = seam S1)
V2_M = (40, 100, 200)                           # PLAN 3.3: N = 400 + m
V2_KINDS = ("out", "rep", "mix")
N_ORIG = 400
R_STAR = {"V2": 2.2515761243137895, "V3": 2.2678004868728294, "V0": None}   # frozen; V0 not usable
COMBOS = (("V2", sim.V2_W), ("V3", sim.V3_W), ("V0", None))
SOURCES = ("E0", "PA")
OFF = "off_10"
SCALED = {"u390": 390, "u375": 375, "u350": 350, "u300": 300}
SEAM_TOL = 1e-12
BOOT_N, BOOT_SEED = 2000, 0
SMOKE_SEED = 9000
THREADS = 4
MIN_OUT = 105                                   # outsiders: fit players with >= 105 black games (CNN available)


# ================================================================ pools
def _h(seed, p, salt=None):
    s = ("%d|%s" % (seed, p)) if salt is None else ("%d|%s|%s" % (seed, salt, p))
    return hashlib.sha256(s.encode()).hexdigest()


def choose_pool_n(cache, seed, n):
    """PLAN 3.1. N = 400 -> p06_sim.choose_pool (40 per level, same order)."""
    if n % len(LEVELS) or n % 4:
        raise ValueError("N must be a multiple of 10 and of 4, got %d" % n)
    elig = defaultdict(list)
    for p in sorted(cache["n_all"]):
        if cache["n_black"][p] >= sim.MIN_BLACK:
            elig[cache["level"][p]].append(p)
    per = n // len(LEVELS)
    pool, rest = [], []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: _h(seed, p))
        pool += pl[:per]
        rest += pl[per:]
    need = n - len(pool)
    if need > 0:
        rest = sorted(rest, key=lambda p: _h(seed, p))
        if len(rest) < need:
            raise ValueError("only %d eligible players for N = %d" % (len(pool) + len(rest), n))
        pool += rest[:need]
    return pool


def scen_m(scenario, n):
    if scenario in SCALED:
        return int(round(n * SCALED[scenario] / 400.0))
    if scenario == "d80":
        return n // 5
    return int(scenario[1:])


def _draw(cache, seed, pool, n_cand=sim.N_CAND):
    rng = random.Random("pool|%d" % seed)
    queue, cand_rows = {}, []
    for p in pool:
        g = list(cache["by_player"][p])
        rng.shuffle(g)
        cand_rows.append(g[:n_cand])
        queue[p] = g[n_cand:]
    return cand_rows, queue


def _repair(owner, ks, cap, r3, allow_move=True):
    """p06_sim.make_pool's repair loop (verbatim); owner < 0 (outsiders) take no part. allow_move False: only swaps
    (the v2-like structures must keep their owners) -> RuntimeError when no swap is possible."""
    n = len(owner)
    n_rep = 0
    for _ in range(10000):
        need = Counter()
        for q in range(n):
            if owner[q] >= 0:
                need[owner[q]] += ks[q]
        bad = [i for i in need if need[i] > cap[i]]
        if not bad:
            break
        i = bad[0]
        qs = sorted([q for q in range(n) if owner[q] == i], key=lambda q: -ks[q])
        q = qs[0]
        cands = [q2 for q2 in range(n) if owner[q2] != i and owner[q2] >= 0 and ks[q2] < ks[q]
                 and need[owner[q2]] - ks[q2] + ks[q] <= cap[owner[q2]]]
        if cands:
            q2 = r3.choice(cands)
            ks[q], ks[q2] = ks[q2], ks[q]
        elif allow_move:
            others = [j for j in set(owner) if j != i and j >= 0 and need[j] + ks[q] <= cap[j]]
            owner[q] = r3.choice(others)
        else:
            raise RuntimeError("could not repair k assignment without moving a question")
        n_rep += 1
    else:
        raise RuntimeError("could not repair k assignment")
    return n_rep


def make_pool_n(cache, seed, scenario, n):
    """PLAN 3.2: p06_sim.make_pool with N players and the scaled structures; N = 400 -> identical to make_pool."""
    pool = choose_pool_n(cache, seed, n)
    ks = sim.k_list(n)
    random.Random("k|%d" % seed).shuffle(ks)
    cand_rows, queue = _draw(cache, seed, pool)
    if scenario == "o2o":
        owner = list(range(n))
    else:
        r2 = random.Random("scen|%s|%d" % (scenario, seed))
        if scenario == "wr":
            owner = [r2.randrange(n) for _ in range(n)]
        elif scenario.startswith("d"):
            m = scen_m(scenario, n)
            perm = list(range(n))
            r2.shuffle(perm)
            owner = perm[:n - m] + perm[n - 2 * m:n - m]
            r2.shuffle(owner)
        elif scenario.startswith("u"):
            m = scen_m(scenario, n)
            used = r2.sample(range(n), m)
            owner = used + [r2.choice(used) for _ in range(n - m)]
            r2.shuffle(owner)
        else:
            raise ValueError(scenario)
    ks = list(ks)
    cap = [len(queue[pool[i]]) for i in range(n)]
    n_rep = _repair(owner, ks, cap, random.Random("repair|%s|%d" % (scenario, seed)))
    q_rows = []
    pos = Counter()
    for q in range(n):
        p = pool[owner[q]]
        q_rows.append(queue[p][pos[owner[q]]:pos[owner[q]] + ks[q]])
        pos[owner[q]] += ks[q]
    return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner),
            "k": np.asarray(ks), "n_repairs": n_rep, "scenario": scenario, "seed": seed, "outsiders": []}


def make_v2_pool(cache, seed, kind, m):
    """PLAN 3.3: N = 400 + m candidates; 400 'original' candidates with one question each, m candidates without a
    question; m new questions: kind out (players outside the candidates), rep (m distinct originals ask a second
    question), mix (first m/2 out, the other m/2 rep). owner -1 - j = outsider j."""
    if kind not in V2_KINDS or m <= 0 or m % 2:
        raise ValueError((kind, m))
    n = N_ORIG + m
    pool = choose_pool_n(cache, seed, n)
    ks = sim.k_list(n)
    random.Random("k|%d" % seed).shuffle(ks)
    cand_rows, queue = _draw(cache, seed, pool)
    r2 = random.Random("v2|%s|%d|%d" % (kind, m, seed))
    orig = r2.sample(range(n), N_ORIG)
    n_out = m if kind == "out" else (m // 2 if kind == "mix" else 0)
    n_rep = m - n_out
    rep = r2.sample(orig, n_rep)
    inpool = set(pool)
    out_elig = [p for p in cache["by_player"] if p not in inpool and cache["n_black"][p] >= MIN_OUT]
    outsiders = sorted(out_elig, key=lambda p: _h(seed, p, "out"))[:n_out]
    if len(outsiders) < n_out:
        raise ValueError("only %d outsiders for m = %d" % (len(outsiders), n_out))
    owner = list(orig) + [-1 - j for j in range(n_out)] + list(rep)
    r2.shuffle(owner)
    ks = list(ks)
    cap = [len(queue[pool[i]]) for i in range(n)]
    n_rp = _repair(owner, ks, cap, random.Random("repair|v2|%s|%d|%d" % (kind, m, seed)), allow_move=False)
    q_rows = []
    pos = Counter()
    for q in range(n):
        o = owner[q]
        if o >= 0:
            p = pool[o]
            q_rows.append(queue[p][pos[o]:pos[o] + ks[q]])
            pos[o] += ks[q]
        else:
            p = outsiders[-1 - o]
            g = list(cache["by_player"][p])
            random.Random("out|%d|%s" % (seed, p)).shuffle(g)
            if ks[q] > len(g):
                raise RuntimeError("outsider %s has %d black games < k %d" % (p, len(g), ks[q]))
            q_rows.append(g[:ks[q]])
    return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner),
            "k": np.asarray(ks), "n_repairs": n_rp, "scenario": "v2%s%d" % (kind, m), "seed": seed,
            "outsiders": outsiders, "kind": kind, "m": m, "orig": sorted(orig), "rep": sorted(rep)}


# ================================================================ scores (owner < 0 -> 0 points)
def q_points(S, owner):
    owner = np.asarray(owner)
    pts = np.zeros(len(owner))
    ok = owner >= 0
    if ok.any():
        S = np.asarray(S, np.float64)
        pts[ok] = st.points(st.ranks(S[ok], owner[ok]))
    return pts


def pool_scores_q(F, owner):
    """p06_sim.pool_scores with per-question points kept (off_10, on) and outsider questions scoring 0."""
    out, per = {}, {}
    for k in (None,) + sim.CSLS_KS:
        name = "off_%s" % ("none" if k is None else k)
        p = q_points(st.csls(F, k), owner)
        out[name] = float(p.mean())
        if name == OFF:
            per[name] = p
    Z = st.zrows(F)
    a = st.hungarian_top1(Z)
    top = Z.argmax(1)
    n = Z.shape[0]
    p = q_points(st.with_top1(st.csls(F, st.CSLS_SWITCH_K), a), owner)
    out["on"] = float(p.mean())
    per["on"] = p
    out["forced_cost"] = float(np.mean(Z[np.arange(n), top] - Z[np.arange(n), a]))
    out["changed"] = float(np.mean(a != top))
    return out, per


# ================================================================ one job
def _lda(rec):
    store = pp._W["store"]
    excl = set(rec["pool"]) | set(rec.get("outsiders") or [])
    return {"B": pp._W["lda"]["B"].fit_excluding(store, excl)}


def _pseudo(rec, models):
    store, cnn = pp._W["store"], pp._W["cnn"]
    pseudo = {}
    prs, _ids = sim.sr_pool(store, rec, rounds=(0, 1))
    for pr in prs:
        Tp = sim.pool_terms(store, pr, pp._W["lda"], models, cnn)
        for name, w in COMBOS:
            for s in SOURCES:
                pseudo.setdefault("%s|%s" % (name, s), []).append(sim.forced_cost(sim.combo_F(Tp, s, w)))
            pseudo.setdefault("%s|open" % name, []).append(sim.forced_cost(sim.combo_F(Tp, None, w,
                                                                                          open_only=True)))
    return pseudo


def _row(rec, models, pseudo, keep_q):
    store, cnn = pp._W["store"], pp._W["cnn"]
    T = sim.pool_terms(store, rec, pp._W["lda"], models, cnn)
    own = rec["owner"]
    row = {"seed": rec["seed"], "scen": rec["scenario"], "n": int(len(own)),
           "n_distinct": int(len(set(own.tolist()))), "n_repairs": rec["n_repairs"], "res": {}}
    qd = {}
    for name, w in COMBOS:
        for s in SOURCES:
            key = "%s|%s" % (name, s)
            r, per = pool_scores_q(sim.combo_F(T, s, w), own)
            r["r_forced_cost"] = r["forced_cost"] / max(float(np.mean(pseudo[key])), 1e-9)
            row["res"][key] = r
            if keep_q and name in ("V2", "V3"):
                qd[key] = per
        key = "%s|open" % name
        r = {"forced_cost": sim.forced_cost(sim.combo_F(T, None, w, open_only=True))}
        r["r_forced_cost"] = r["forced_cost"] / max(float(np.mean(pseudo[key])), 1e-9)
        row["res"][key] = r
    if keep_q:
        row["q_owner_pid"] = [rec["pool"][o] if o >= 0 else None for o in own.tolist()]
    return row, qd


def _job(job):
    t = time.time()
    try:
        if job["type"] == "std":
            cache = pp._W["store"].black_cache()
            n, seed = job["n"], job["seed"]
            rec0 = make_pool_n(cache, seed, "o2o", n)
            models = _lda(rec0)
            pseudo = _pseudo(rec0, models)
            rows, qd = [], {}
            for scen in job["structures"]:
                rec = rec0 if scen == "o2o" else make_pool_n(cache, seed, scen, n)
                row, q = _row(rec, models, pseudo, keep_q=(scen == "o2o"))
                rows.append(row)
                if q:
                    qd = q
            lda_n = models["B"]["n_players"]
        else:
            cache = pp._W["store"].black_cache()
            rec = make_v2_pool(cache, job["seed"], job["kind"], job["m"])
            models = _lda(rec)
            pseudo = _pseudo(rec, models)
            row, qd = _row(rec, models, pseudo, keep_q=False)
            row["n_outsiders"] = len(rec["outsiders"])
            row["n_rep"] = len(rec["rep"])
            rows = [row]
            lda_n = models["B"]["n_players"]
        return {"job": job, "rows": rows, "pseudo": pseudo, "lda_players": lda_n,
                "q": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in qd.items()},
                "seconds": round(time.time() - t, 1)}
    except Exception as e:  # SystemExit would hang the pool
        raise RuntimeError("job %s failed: %r" % (job, e))


def job_name(job):
    if job["type"] == "std":
        return "std_n%d_s%d" % (job["n"], job["seed"])
    return "v2_%s_m%d_s%d" % (job["kind"], job["m"], job["seed"])


def code_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def run_jobs(jobs, workers, tag, cache_dir=JOBS):
    """Runs the jobs (largest first) with p06_player's worker initialiser (store, LDA rows, fit CNN); one file per
    job under data/prereg28/jobs/<tag>/ bound to this program's SHA (resume only with the same SHA)."""
    sha = code_sha()
    d = os.path.join(cache_dir, tag)
    os.makedirs(d, exist_ok=True)
    done, todo = {}, []
    for j in jobs:
        p = os.path.join(d, job_name(j) + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("code_sha") != sha or r["job"] != j:
                raise SystemExit("refused: %s was written by another program version or job; delete %s" % (p, d))
            done[job_name(j)] = r
        else:
            todo.append(j)
    todo.sort(key=lambda j: -(j.get("n") or N_ORIG + j.get("m", 0)))
    if todo:
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        pp.thread_env(THREADS)
        try:
            pool = Pool(workers, initializer=pp._sim_init, initargs=(pp.FEATS, pp.CNN_NPZ, THREADS, ("B",)))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        with pool:
            for r in pool.imap_unordered(_job, todo, chunksize=1):
                r["code_sha"] = sha
                with open(os.path.join(d, job_name(r["job"]) + ".json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(r, f)
                done[job_name(r["job"])] = r
                print("  %s done (%s s) %d/%d" % (job_name(r["job"]), r["seconds"], len(done), len(jobs)), flush=True)
    return [done[job_name(j)] for j in jobs]


# ================================================================ judging
def opens(row, key):
    name = key.split("|")[0]
    thr = R_STAR[name]
    return thr is not None and row["res"][key]["r_forced_cost"] <= thr


def usability(rows, name, src):
    """PREREG_06 2.6 'usability' (p06_player.usable_check) with the frozen r*_X for one F source."""
    key, okey, thr = "%s|%s" % (name, src), "%s|open" % name, R_STAR[name]
    if thr is None:
        return False, {"reason": "not usable in v1 (frozen)"}
    o2o = [x for x in rows if x["scen"] == "o2o"]
    share = float(np.mean([opens(x, key) for x in o2o])) if o2o else 0.0
    non = [x for x in rows if x["scen"] in sim.NON_O2O]
    false_open = sum(1 for x in non if opens(x, key))
    near, ok_near = {}, True
    for sc in sim.NEAR_O2O:
        rr = [x for x in rows if x["scen"] == sc]
        if not rr:
            continue
        d = np.array([(x["res"][key]["on"] - x["res"][key][OFF]) if opens(x, key) else 0.0 for x in rr])
        near[sc] = {"mean": float(d.mean()), "min": float(d.min()), "opens": int(sum(opens(x, key) for x in rr))}
        ok_near &= bool(d.mean() >= pp.FEAS_MEAN and d.min() >= pp.FEAS_POOL)
    open_false = sum(1 for x in non if opens(x, okey))
    conds = {"o2o_open_share_ge_0.8": share >= pp.O2O_OPEN_MIN, "no_false_open": false_open == 0,
             "near_o2o_ok": bool(ok_near), "opening_v2_only_no_false_open": open_false == 0}
    return all(conds.values()), {"o2o_open_share": share, "non_o2o_pools": len(non), "non_o2o_opens": false_open,
                                 "near_o2o": near, "opening_v2_only_non_o2o_opens": open_false, "conditions": conds}


def player_boot(sums, counts, n_boot=BOOT_N, seed=BOOT_SEED):
    """p10_fuse.player_boot (PREREG_10 rev 2): pooled per-question mean and the player-bootstrap 95 % interval."""
    sums, counts = np.asarray(sums, np.float64), np.asarray(counts, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))
    m = sums[idx].sum(1) / counts[idx].sum(1)
    return [float(sums.sum() / counts.sum()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def gain_boot(results, key):
    """'by the detector' (on if r <= r* else off) - off per question, over the o2o pools of the results; players
    = the true player of each question (all pools of this N)."""
    sums, counts = defaultdict(float), defaultdict(int)
    n_open = 0
    for r in results:
        row = [x for x in r["rows"] if x["scen"] == "o2o"][0]
        q = r["q"][key]
        on, off = np.asarray(q["on"]), np.asarray(q[OFF])
        o = opens(row, key)
        n_open += int(o)
        d = (on - off) if o else np.zeros(len(off))
        for pid, v in zip(row["q_owner_pid"], d.tolist()):
            sums[pid] += v
            counts[pid] += 1
    ids = sorted(sums)
    b = player_boot([sums[i] for i in ids], [counts[i] for i in ids])
    return {"mean": b[0], "ci95": b[1:], "o2o_pools_open": n_open, "o2o_pools": len(results),
            "players": len(ids), "questions": int(sum(counts.values()))}


def r_summary(rows, key):
    r = np.array([x["res"][key]["r_forced_cost"] for x in rows])
    return {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())} if len(r) else None


def judge(std, v2):
    """std: {N: [job results]}, v2: [job results]. PLAN 5."""
    out = {"G1": {}, "G2": {}, "G3": {}, "record_only": {}}
    g1_ok, g2_ok = True, True
    v3_usable_all = True
    for n in sorted(std):
        rows = [x for r in std[n] for x in r["rows"]]
        cnt = Counter(x["scen"] for x in rows)
        if len(std[n]) != len(SEEDS) or any(cnt[sc] != len(SEEDS) for sc in sim.STRUCTURES)                 or len(rows) != len(SEEDS) * len(sim.STRUCTURES):
            raise SystemExit("judge: N = %d has %d jobs / structure counts %s (PLAN 3: 20 seeds x 11 structures)"
                             % (n, len(std[n]), dict(cnt)))
        g1n = {}
        for name in ("V2", "V3"):
            for src in SOURCES:
                ok, det = usability(rows, name, src)
                g1n["%s|%s" % (name, src)] = {"usable": ok, **det}
        v2ok = g1n["V2|E0"]["usable"] and g1n["V2|PA"]["usable"]
        v3ok = g1n["V3|E0"]["usable"] and g1n["V3|PA"]["usable"]
        out["G1"][str(n)] = {"V2_passed": bool(v2ok), "V3_usable": bool(v3ok), "detail": g1n,
                             "counts_in_gate": n in N_GRID}
        g2 = gain_boot(std[n], "V2|E0")
        g2["passed"] = bool(g2["mean"] > 0 and g2["ci95"][0] > 0)
        g2["counts_in_gate"] = n in N_GRID
        out["G2"][str(n)] = g2
        if n in N_GRID:
            g1_ok &= bool(v2ok)
            g2_ok &= g2["passed"]
            v3_usable_all &= bool(v3ok)
        ro = {k: gain_boot(std[n], k) for k in ("V2|PA", "V3|E0", "V3|PA")}
        ro["scores"] = {sc: {k: {"off": float(np.mean([x["res"][k][OFF] for x in rows if x["scen"] == sc])),
                                 "on": float(np.mean([x["res"][k]["on"] for x in rows if x["scen"] == sc])),
                                 "open_share": float(np.mean([opens(x, k) for x in rows if x["scen"] == sc]))}
                             for k in ("V2|E0", "V2|PA", "V3|E0", "V3|PA")}
                        for sc in sim.STRUCTURES}
        ro["r"] = {sc: {k: r_summary([x for x in rows if x["scen"] == sc], k)
                        for k in ("V2|E0", "V2|PA", "V2|open", "V3|E0", "V3|PA", "V3|open")}
                   for sc in sim.STRUCTURES}
        ro["lda_players"] = sorted({r["lda_players"] for r in std[n]})
        ro["n_repairs"] = {sc: int(sum(x["n_repairs"] for x in rows if x["scen"] == sc)) for sc in sim.STRUCTURES}
        out["record_only"][str(n)] = ro
    keys = ("V2|E0", "V2|PA", "V2|open", "V3|E0", "V3|PA", "V3|open")
    v2rows = [x for r in v2 for x in r["rows"]]
    g3 = {}
    for kind in V2_KINDS:
        for m in V2_M:
            rr = [x for x in v2rows if x["scen"] == "v2%s%d" % (kind, m)]
            g3["%s_m%d" % (kind, m)] = {
                "pools": len(rr), "opens": {k: int(sum(opens(x, k) for x in rr)) for k in keys},
                "r": {k: r_summary(rr, k) for k in keys},
                "forced_on_minus_off": {k: float(np.mean([x["res"][k]["on"] - x["res"][k][OFF] for x in rr]))
                                        for k in ("V2|E0", "V3|E0")},
                "lda_players": sorted({r["lda_players"] for r in v2 if r["rows"][0]["scen"] == "v2%s%d" % (kind, m)})}
    total_open = sum(v for g in g3.values() for v in g["opens"].values())
    total_pools = sum(g["pools"] for g in g3.values())
    out["G3"] = {"passed": bool(total_open == 0 and total_pools == len(V2_KINDS) * len(V2_M) * len(SEEDS)),
                 "pools": total_pools, "opens_total": int(total_open), "by_structure": g3}
    out["G1_passed"] = bool(g1_ok)
    out["G2_passed"] = bool(g2_ok)
    out["V3_usable_at_every_N"] = bool(v3_usable_all)
    out["passed"] = bool(g1_ok and g2_ok and out["G3"]["passed"])
    out["verdict"] = "PASS: register PREREG_28" if out["passed"] else "FAIL: not used, the 400 x 400 literal stays"
    return out


def pool_overlap():
    """Review M2 (record only): how many candidates two seeds' pools share, per N (min / mean over seed pairs)."""
    cache = sim.Store(pp.FEATS).black_cache()
    out = {}
    for n in (400,) + N_GRID + tuple(N_ORIG + m for m in V2_M if N_ORIG + m not in N_GRID):
        pools = [set(choose_pool_n(cache, s, n)) for s in SEEDS]
        sh = [len(pools[i] & pools[j]) for i in range(len(pools)) for j in range(i + 1, len(pools))]
        out[str(n)] = {"min_shared": int(min(sh)), "mean_shared": float(np.mean(sh)), "n": n}
    return out


# ================================================================ commands
def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True).stdout.strip()


def committed_clean(rel):
    return git("status", "--porcelain", "--", rel) == "" and git("ls-files", "--", rel) != ""


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=False)
        f.write("\n")


def std_jobs(n, seeds=SEEDS, structures=sim.STRUCTURES):
    return [{"type": "std", "n": n, "seed": s, "structures": list(structures)} for s in seeds]


def v2_jobs(seeds=SEEDS):
    return [{"type": "v2", "kind": k, "m": m, "seed": s} for k in V2_KINDS for m in V2_M for s in seeds]


def cmd_smoke(a):
    t0 = time.time()
    store = sim.Store(pp.FEATS)
    cache = store.black_cache()
    for n in (400, 800):
        rec = make_pool_n(cache, SMOKE_SEED, "d80", n)
        print("std N=%d: pool %d, questions %d, distinct %d, repairs %d" % (
            n, len(rec["pool"]), len(rec["q_rows"]), len(set(rec["owner"].tolist())), rec["n_repairs"]))
    for kind in V2_KINDS:
        rec = make_v2_pool(cache, SMOKE_SEED, kind, 200)
        print("v2 %s m=200: pool %d, questions %d, outsiders %d, rep %d, repairs %d" % (
            kind, len(rec["pool"]), len(rec["q_rows"]), len(rec["outsiders"]), len(rec["rep"]), rec["n_repairs"]))
    jobs = [{"type": "std", "n": 800, "seed": SMOKE_SEED, "structures": ["o2o", "d80"]},
            {"type": "v2", "kind": "mix", "m": 200, "seed": SMOKE_SEED}]
    res = run_jobs(jobs, min(a.workers, 2), "smoke")
    for r in res:
        print("smoke job %s: %s s, rows %d, LDA players %d" % (job_name(r["job"]), r["seconds"], len(r["rows"]),
                                                               r["lda_players"]))
    print("smoke total %.1f s (no score or r printed)" % (time.time() - t0))
    return 0


def cmd_seam(a):
    t0 = time.time()
    out = os.path.join(OUT, "seam_report.json")
    if os.path.exists(out) and not a.rerun:
        raise SystemExit("refused: seam_report.json exists")
    ref = json.load(open(SWITCH_REPORT, encoding="utf-8"))
    res = run_jobs(std_jobs(400), a.workers, "seam400")
    rows = [x for r in res for x in r["rows"]]
    refrows = {(x["seed"], x["scen"]): x for x in ref["rows"] if x["seed"] in SEEDS}
    worst, bad = 0.0, []
    for x in rows:
        y = refrows.get((x["seed"], x["scen"]))
        if y is None:
            bad.append(("missing", x["seed"], x["scen"]))
            continue
        for f in ("n_distinct", "n_repairs"):
            if x[f] != y[f]:
                bad.append((f, x["seed"], x["scen"], x[f], y[f]))
        if set(x["res"]) != set(y["res"]):
            bad.append(("keys", x["seed"], x["scen"]))
            continue
        for k in y["res"]:
            if set(x["res"][k]) != set(y["res"][k]):
                bad.append(("fields", x["seed"], x["scen"], k))
                continue
            for f, v in y["res"][k].items():
                dv = abs(x["res"][k][f] - v)
                worst = max(worst, dv)
                if dv > SEAM_TOL:
                    bad.append(("value", x["seed"], x["scen"], k, f, x["res"][k][f], v))
    if len(rows) != len(refrows):
        bad.append(("row count", len(rows), len(refrows)))
    pworst = 0.0
    for r in res:
        rp = ref["pseudo_forced_cost"][str(r["job"]["seed"])]
        for k, v in rp.items():
            for p, q in zip(r["pseudo"][k], v):
                pworst = max(pworst, abs(p - q))
            if len(r["pseudo"][k]) != len(v):
                bad.append(("pseudo len", r["job"]["seed"], k))
    if pworst > SEAM_TOL:
        bad.append(("pseudo", pworst))
    usab, ubad = {}, []
    for name in ("V2", "V3"):
        for src in SOURCES:
            ok, det = usability(rows, name, src)
            refu = ref["per_combination"][name]["usability"][src]
            usab["%s|%s" % (name, src)] = {"usable": ok, **det}
            for f in ("o2o_open_share", "non_o2o_pools", "non_o2o_opens", "opening_v2_only_non_o2o_opens"):
                if abs(det[f] - refu[f]) > SEAM_TOL:
                    ubad.append((name, src, f, det[f], refu[f]))
            if ok != refu["usable"] or det["conditions"] != refu["conditions"]:
                ubad.append((name, src, "usable/conditions"))
            for sc, v in refu["near_o2o"].items():
                for f in ("mean", "min", "opens"):
                    if abs(det["near_o2o"][sc][f] - v[f]) > SEAM_TOL:
                        ubad.append((name, src, sc, f, det["near_o2o"][sc][f], v[f]))
    rep = {"what": "prereg28 seam S1 (PLAN 4): N = 400, seeds 600-619, 11 structures, this program vs "
                   "results/prereg06/design/switch_report.json",
           "passed": not bad and not ubad, "rows": len(rows), "max_abs_diff_rows": worst,
           "max_abs_diff_pseudo": pworst, "tolerance": SEAM_TOL, "problems": [list(map(str, b)) for b in bad][:50],
           "usability_problems": [list(map(str, b)) for b in ubad], "usability_recomputed": usab,
           "switch_report_sha256": st.sha256_lf(SWITCH_REPORT), "code_sha256": code_sha(),
           "per_job_seconds": {job_name(r["job"]): r["seconds"] for r in res},
           "lda_players": sorted({r["lda_players"] for r in res}), "seconds": round(time.time() - t0, 1),
           "git_head": git("rev-parse", "HEAD")}
    write_json(out, rep)
    print("seam: passed %s; max diff rows %.3g pseudo %.3g; problems %d / %d -> %s" % (
        rep["passed"], worst, pworst, len(bad), len(ubad), out))
    return 0 if rep["passed"] else 1


def cmd_eval(a):
    t0 = time.time()
    out = os.path.join(OUT, "eval_report.json")
    if os.path.exists(out):
        if not (a.rerun and a.reason):
            raise SystemExit("refused: eval_report.json exists (PLAN 6: a rerun needs --rerun --reason and is "
                             "written as eval_report_rerunN.json)")
        i = 1
        while os.path.exists(os.path.join(OUT, "eval_report_rerun%d.json" % i)):
            i += 1
        out = os.path.join(OUT, "eval_report_rerun%d.json" % i)
    for rel in ("results/prereg28/explore/PLAN.md", "results/prereg28/explore/p28_sim.py",
                "results/prereg28/explore/test_p28.py", "results/prereg28/explore/seam_report.json"):
        if not committed_clean(rel):
            raise SystemExit("refused: %s is not committed and clean" % rel)
    seam = json.load(open(os.path.join(OUT, "seam_report.json"), encoding="utf-8"))
    if not seam.get("passed") or seam.get("code_sha256") != code_sha():
        raise SystemExit("refused: the seam did not pass with this program")
    seam_res = run_jobs(std_jobs(400), a.workers, "seam400")
    jobs = [j for n in N_GRID for j in std_jobs(n)] + v2_jobs()
    res = run_jobs(jobs, a.workers, "eval")
    std = {400: seam_res}
    for n in N_GRID:
        std[n] = [r for r in res if r["job"]["type"] == "std" and r["job"]["n"] == n]
    v2 = [r for r in res if r["job"]["type"] == "v2"]
    verdict = judge(std, v2)
    verdict["record_only_pool_overlap"] = pool_overlap()
    rep = {"what": "prereg28 eval (PLAN 5): frozen r*_X on N x N fit-player pools; judged once",
           "r_star": R_STAR, "n_grid": list(N_GRID), "v2_m": list(V2_M), "v2_kinds": list(V2_KINDS),
           "seeds": list(SEEDS), "judgement": verdict, "code_sha256": code_sha(), "git_head": git("rev-parse", "HEAD"),
           "reason": a.reason, "seconds": round(time.time() - t0, 1),
           "per_job_seconds": {job_name(r["job"]): r["seconds"] for r in res},
           "rows": {"std": {str(n): [x for r in std[n] for x in r["rows"]] for n in std},
                    "v2": [x for r in v2 for x in r["rows"]]}}
    for part in rep["rows"]["std"].values():
        for x in part:
            x.pop("q_owner_pid", None)
    write_json(out, rep)
    print("eval: %s (G1 %s, G2 %s, G3 %s) -> %s" % (verdict["verdict"], verdict["G1_passed"], verdict["G2_passed"],
                                                    verdict["G3"]["passed"], out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "seam", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"smoke": cmd_smoke, "seam": cmd_seam, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
