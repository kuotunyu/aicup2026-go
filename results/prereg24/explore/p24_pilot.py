"""prereg24 phase 0 (competition 2): the CPU scaling pilot of PLAN.md 2-3 (is S19 data-limited?).

    Git Bash, repository root (a subst drive when the worktree path contains 'AppData\\Local'); PYTHONDONTWRITEBYTECODE=1;
    .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg24/explore
    $S/p24_pilot.py train --jobs R100:0,P25:0,...   -> data/prereg24/pilot/models/<arm>_seed<s>.npz + <arm>_seed<s>.json
    $S/p24_pilot.py report                          -> $S/pilot_report.json (run once; judges GO / NO-GO by PLAN.md 3)
    --smoke (both commands): P1 features replaced by default_rng(12345) normal noise, 300 steps, held-out 5 per cell,
    outputs in data/prereg24/smoke/ (pipeline and timing only, never a result; every arm incl. R100 is trained).

Training, the model, the sampler, the step choice and the numpy forward are imported unchanged from
results/prereg19/explore/p19_train.py and p19_lib.py; only the A rows differ per arm (PLAN.md 2.2). Probe players only:
no inner array, no validation data, never data/local, no tools/score.py, no GPU.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P19 = os.path.join(REPO, "results", "prereg19", "explore")
for p in (os.path.join(REPO, "tools"), P19, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p19_train as T  # noqa: E402  (also puts tools/ and the prereg12 finetune folder on sys.path)
import p19_lib as K  # noqa: E402
import p09_forward as FW  # noqa: E402
import p02_rank_fuse as R2  # noqa: E402  (rule_b_np)

r = T.r
guard_path = T.guard_path
ARMS = ("P25", "P50", "R33", "R56", "R100")
AXES = {"P": ("P25", "P50", "R100"), "R": ("R33", "R56", "R100")}
X_GO = 0.004
PLAYER_SALT = "c10-prereg24|players|"
GAME_SALT = "c10-prereg24|games|"
BOOT = 1000
BOOT_SEED = 24
DATA = os.path.join(REPO, "data", "prereg24", "pilot")
SMOKE_DIR = os.path.join(REPO, "data", "prereg24", "smoke")
REPORT = os.path.join(HERE, "pilot_report.json")
REPORT_STARTED = os.path.join(DATA, "report_started.json")
S19_FROZEN = os.path.join(REPO, "results", "prereg19", "rank")
S19_RECORD = os.path.join(P19, "train_record.json")
OWN_FILES = ("PLAN.md", "p24_pilot.py")
FORMAT = "c10-prereg24 explore pilot v1"
SEAM_H_TOL = 1e-4


def log(*x):
    print(*x, flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES]
    rel = sorted(set(rel) | T.loaded_repo_modules())
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def quarter(key):
    return int(hashlib.sha256((PLAYER_SALT + key).encode("utf-8")).hexdigest()[:8], 16) % 4


# ---------------------------------------------------------------- data (PLAN.md 2.1-2.2)
class Data:
    def __init__(self, smoke):
        self.smoke = smoke
        self.hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
        mp = FW.load_map(T.Q09_MAP)
        if r.sha_raw(T.Q09_MAP) != r.read_json(T.Q09_FROZEN)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
        want = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
        got = T.sha_file(os.path.join(T.PROBE_PREP, "probe_X.npy"))
        self.seam_p = {"want": want.get("probe_X.npy"), "got": got, "ok": bool(want.get("probe_X.npy") == got)}
        if not self.seam_p["ok"]:
            raise SystemExit(f"seam P failed: {self.seam_p}")
        prep = T.ProbeOnly(T.PROBE_PREP)
        if set(np.unique(prep.src)) != set(r.PROBE_SETS):
            raise SystemExit("probe rows of another set")
        self.prep = prep
        self.col = T.probe_colours(prep)
        self.keys = np.asarray(prep.players).astype(str)
        bit = {k: K.holdout_bit(k) for k in set(self.keys.tolist())}
        q = {k: quarter(k) for k in bit}
        self.rows_a = np.array([i for i, k in enumerate(self.keys) if bit[k] == 0], dtype=np.int64)
        self.rows_h = np.array([i for i, k in enumerate(self.keys) if bit[k] == 1], dtype=np.int64)
        src = np.asarray(prep.src).astype(str)
        games = {}
        for s_, pth in T.SETS_JSON.items():
            with open(guard_path(pth), encoding="utf-8") as f:
                games[s_] = json.load(f)["probe"]["games"]
        gkey = [f"{games[str(s_)][int(j)]['rank']}:{games[str(s_)][int(j)]['game_id']}" for s_, j in zip(src, prep.ref)]
        half = np.array([int(hashlib.sha256((GAME_SALT + g).encode("utf-8")).hexdigest()[:8], 16) % 2 == 0
                         for g in gkey])
        ha = half[self.rows_a]
        qa = np.array([q[self.keys[i]] for i in self.rows_a])
        sa = src[self.rows_a]
        self.arm_rows = {"P25": self.rows_a[qa == 0], "P50": self.rows_a[qa <= 1],
                         "R33": self.rows_a[(sa == "p1_probe") | ((sa == "p2_probe") & ha)],
                         "R56": self.rows_a[(sa == "p1_probe") | (sa == "p2_probe")], "R100": self.rows_a}
        self.pool_h = K.PlayerPool(self.keys, prep.y, self.col, self.rows_h)
        self.hold, self.hold_stats = K.holdout_set(self.pool_h, self.hold_per, K.HOLD_SEED)
        t = time.time()
        self.X32 = T.standardised_probe(prep, mp["mu_x"], mp["sd_x"], smoke)
        log(f"standardised probe X ({time.time() - t:.0f}s)")

    def nested_check(self):
        s = {a: set(self.arm_rows[a].tolist()) for a in ARMS}
        hset = set(self.rows_h.tolist())
        return {"P25_in_P50": s["P25"] <= s["P50"], "P50_in_A": s["P50"] <= s["R100"],
                "R33_in_R56": s["R33"] <= s["R56"], "R56_in_A": s["R56"] <= s["R100"],
                "no_H_rows": all(not (s[a] & hset) for a in ARMS)}

    def pool(self, arm):
        return K.PlayerPool(self.keys, self.prep.y, self.col, self.arm_rows[arm])

    def arm_counts(self, arm, pool):
        c = pool.counts()
        return {"empty_cells_n_level": [[n, k] for n in c for k in range(K.L) if sum(c[n][k]) == 0],"rows": int(len(self.arm_rows[arm])), "players": len(pool.players),
                "players_per_level": np.bincount(pool.level, minlength=K.L).tolist(),
                "eligible_n_level_colour": pool.counts()}


def out_dir(smoke):
    return SMOKE_DIR if smoke else DATA


def model_path(smoke, arm, seed):
    return os.path.join(out_dir(smoke), "models", f"{arm}_seed{seed}.npz")


def record_path(smoke, arm, seed):
    return os.path.join(out_dir(smoke), "models", f"{arm}_seed{seed}.json")


# ---------------------------------------------------------------- train
def cmd_train(a):
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    jobs = []
    for j in a.jobs.split(","):
        arm, s = j.split(":")
        if arm not in ARMS or int(s) not in K.SEEDS:
            raise SystemExit(f"bad job {j}")
        jobs.append((arm, int(s)))
    cs = code_state()
    if not smoke and cs["dirty"]:
        raise SystemExit(f"the prereg24 files or the imported code are not committed and clean: {cs['dirty']}")
    for arm, s in jobs:
        if os.path.exists(model_path(smoke, arm, s)) and not smoke and not (a.rerun and a.reason):
            raise SystemExit(f"{model_path(smoke, arm, s)} exists: run once (--rerun --reason after a commit that "
                             f"names a concrete program error)")
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    if os.path.exists(T.STOP):
        raise SystemExit(f"STOP file {T.STOP} exists")
    d = Data(smoke)
    nest = d.nested_check()
    if not all(nest.values()):
        raise SystemExit(f"nesting check failed {nest}")
    import torch
    for arm, s in jobs:
        pool = d.pool(arm)
        cnt = pool.counts()
        dead = [n for n in cnt if all(sum(cnt[n][k]) == 0 for k in range(K.L))]
        if dead:
            raise SystemExit(f"arm {arm}: no eligible player at all for n in {dead} (the S19 sampler would loop)")
        log(f"arm {arm} seed {s}: {len(d.arm_rows[arm])} rows, {len(pool.players)} players")
        best, curve, stats, n_count, secs = T.train_seed(s, d.X32, pool, d.hold, steps, eval_every,
                                                         f"{'smoke' if smoke else 'pilot'} {arm}")
        pc = T.port_check(best["params"], d.X32, d.hold)
        msha = T.save_model(model_path(smoke, arm, s), best["params"], s, best, smoke)
        rec = {"format": FORMAT + " seed", "created": r.now_iso(), "smoke": smoke, "arm": arm, "seed": s,
               "code": cs, "rerun_reason": a.reason if a.rerun else None,
               "best_step": best["step"], "best_holdout_nll": best["nll"], "curve": curve, "sampling": stats,
               "batches_per_n": n_count, "seconds": secs, "port_check": pc, "model_sha256": msha,
               "arm_counts": d.arm_counts(arm, pool), "nested": nest,
               "settings": {"steps": steps, "eval_every": eval_every, "hold_per": d.hold_per, "batch": K.BATCH,
                            "sampler_base": K.SAMPLER_BASE, "hold_seed": K.HOLD_SEED, "threads": K.THREADS},
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}}
        T.write_json(record_path(smoke, arm, s), rec)
        log(f"arm {arm} seed {s}: best step {best['step']} NLL {best['nll']:.5f} ({secs:.0f}s); port {pc['ok']}")
        if not pc["ok"]:
            raise SystemExit(f"port check failed {pc}")
    return 0


# ---------------------------------------------------------------- evaluation (PLAN.md 2.3)
def hold_logp(par, X32, hold):
    out = {}
    for n, (idx, y) in hold.items():
        if len(y) == 0:
            continue
        Xg = X32[idx.ravel()].astype(np.float64).reshape(len(y), n, K.D_IN)
        out[n] = K.set_logp_np(par, Xg)
    return out


def flat(hold, per_n):
    ns = [n for n in K.KS if n in per_n]
    lp = np.concatenate([per_n[n] for n in ns])
    y = np.concatenate([hold[n][1] for n in ns])
    nn = np.concatenate([np.full(len(hold[n][1]), n) for n in ns])
    return lp, y, nn


def scores(lp, y):
    P = np.exp(lp - lp.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return K.official_score(R2.rule_b_np(P), y)


def summary(lp, y, nn):
    sc = scores(lp, y)
    return {"S": K.rank_balanced(sc, y), "nll": float(-lp[np.arange(len(y)), y].mean()),
            "per_n": {str(n): K.rank_balanced(sc[nn == n], y[nn == n]) for n in K.KS if (nn == n).any()},
            "per_level": K.per_level(sc, y)}, sc


def paired_boot(sc_a, sc_b, y, rng):
    """95% interval of rank_balanced(sc_b) - rank_balanced(sc_a), questions resampled within each level."""
    lv_idx = [np.flatnonzero(y == k) for k in range(K.L) if (y == k).any()]
    d = sc_b - sc_a
    out = np.empty(BOOT)
    for b in range(BOOT):
        out[b] = np.mean([d[ix[rng.integers(len(ix), size=len(ix))]].mean() for ix in lv_idx])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def load_frozen_s19():
    rec = r.read_json(S19_RECORD)
    out, shas = {}, {}
    for s in K.SEEDS:
        p = os.path.join(S19_FROZEN, f"s19_seed{s}.npz")
        shas[str(s)] = T.sha_file(p)
        out[s] = T.load_model(p)[0]
    ok = shas == {str(s): rec["models_sha256"][str(s)] for s in K.SEEDS}
    return out, {"want": rec["models_sha256"], "got": shas, "ok": bool(ok)}, rec


def cmd_report(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"not committed and clean: {cs['dirty']}")
        if os.path.exists(REPORT) or os.path.exists(REPORT_STARTED):
            if not (a.rerun and a.reason):
                raise SystemExit("the report was already run once (--rerun --reason after a commit naming a program "
                                 "error)")
    d = Data(smoke)
    seams = {"P": d.seam_p, "nested": d.nested_check()}
    # ---- models
    models, recs = {}, {}
    for arm in ARMS:
        for s in K.SEEDS:
            p = model_path(smoke, arm, s)
            if os.path.exists(p):
                models[(arm, s)] = T.load_model(p)[0]
                recs[(arm, s)] = r.read_json(record_path(smoke, arm, s))
                if recs[(arm, s)]["model_sha256"] != T.sha_file(p):
                    raise SystemExit(f"{p}: SHA differs from its record")
    need = [(arm, s) for arm in ("P25", "P50", "R33", "R56") for s in K.SEEDS]
    missing = [x for x in need if x not in models]
    if missing:
        raise SystemExit(f"missing models {missing}")
    s19_rec = r.read_json(S19_RECORD)
    if smoke:
        if any(("R100", s) not in models for s in K.SEEDS):
            raise SystemExit("smoke: train R100 seeds too")
        hundred = {s: models[("R100", s)] for s in K.SEEDS}
        seams["note"] = "smoke: R100 is the smoke-trained model; seams F, H, R skipped"
        hundred_source = "smoke R100"
    else:
        frozen, seam_f, s19_rec = load_frozen_s19()
        seams["F"] = seam_f
        pl = s19_rec["players"]
        hq = {str(n): int(len(v[1])) for n, v in d.hold.items()}
        seam_h = {"H_players": [len(d.pool_h.players), pl["H_players"]], "holdout_questions": [hq,
                  pl["holdout_questions"]], "nll": {}}
        ok_h = len(d.pool_h.players) == pl["H_players"] and hq == pl["holdout_questions"]
        for s in K.SEEDS:
            lp, y, _nn = flat(d.hold, hold_logp(K.p64(frozen[s]), d.X32, d.hold))
            nll = float(-lp[np.arange(len(y)), y].mean())
            want = s19_rec["seeds"][str(s)]["best_holdout_nll"]
            seam_h["nll"][str(s)] = {"numpy": nll, "record": want, "diff": abs(nll - want)}
            ok_h = ok_h and abs(nll - want) <= SEAM_H_TOL
        seam_h["ok"] = bool(ok_h)
        seams["H"] = seam_h
        if ("R100", 0) not in models:
            raise SystemExit("seam R: train R100:0 first")
        same = all(np.array_equal(np.asarray(models[("R100", 0)][k]), np.asarray(frozen[0][k])) for k in K.PARAM_NAMES)
        seams["R"] = {"retrained_R100_seed0_equals_s19_seed0": bool(same),
                      "retrained_sha256": recs[("R100", 0)]["model_sha256"],
                      "frozen_sha256": seam_f["got"]["0"]}
        if not (seam_f["ok"] and seam_h["ok"]):
            raise SystemExit(f"seam failed: {json.dumps(seams, default=str)[:2000]}")
        if same:
            hundred, hundred_source = frozen, "frozen S19 (seam R bit-exact)"
        else:
            if any(("R100", s) not in models for s in K.SEEDS):
                raise SystemExit("seam R not bit-exact: PLAN.md 2.4 item 4 says train R100:1,R100:2 and use them")
            hundred, hundred_source = {s: models[("R100", s)] for s in K.SEEDS}, "retrained R100 (seam R failed)"
        T.write_json(REPORT_STARTED, {"format": FORMAT + " report started", "created": r.now_iso(), "code": cs,
                                      "rerun_reason": a.reason if a.rerun else None})
    # ---- evaluate every arm on the S19 held-out questions
    per_arm = {}
    sc_arm = {}
    y_ref = None
    for arm in ARMS:
        pars = hundred if arm == "R100" else {s: models[(arm, s)] for s in K.SEEDS}
        lps_seed = {s: hold_logp(K.p64(pars[s]), d.X32, d.hold) for s in K.SEEDS}
        flats = {s: flat(d.hold, lps_seed[s]) for s in K.SEEDS}
        y, nn = flats[0][1], flats[0][2]
        if y_ref is None:
            y_ref = y
        lp_ens = FW.lse_mean([flats[s][0] for s in K.SEEDS])
        summ, sc = summary(lp_ens, y, nn)
        summ["per_seed"] = {str(s): {k: v for k, v in summary(flats[s][0], y, nn)[0].items()
                                     if k in ("S", "nll")} for s in K.SEEDS}
        if arm != "R100" or smoke or hundred_source.startswith("retrained"):
            summ["train"] = {str(s): {k: recs[(arm, s)][k] for k in ("best_step", "best_holdout_nll", "seconds")}
                             for s in K.SEEDS}
            summ["port_check_max"] = max(recs[(arm, s)]["port_check"]["max_abs_prob_diff"] for s in K.SEEDS)
            summ["counts"] = recs[(arm, 0)]["arm_counts"]
        else:
            summ["train"] = {str(s): {"best_step": s19_rec["seeds"][str(s)]["best_step"],
                                      "best_holdout_nll": s19_rec["seeds"][str(s)]["best_holdout_nll"]}
                             for s in K.SEEDS}
            summ["counts"] = {"rows": int(len(d.arm_rows["R100"])), "players": s19_rec["players"]["A_players"]}
        per_arm[arm] = summ
        sc_arm[arm] = sc
        log(f"{arm}: S {summ['S']:.5f} NLL {summ['nll']:.5f}")
    rng = np.random.default_rng(BOOT_SEED)
    diffs = {}
    for ax, (a1, a2, a3) in AXES.items():
        for lo, hi in ((a1, a2), (a2, a3)):
            diffs[f"{lo}->{hi}"] = {"diff": per_arm[hi]["S"] - per_arm[lo]["S"],
                                    "ci95_question_boot": paired_boot(sc_arm[lo], sc_arm[hi], y_ref, rng),
                                    "nll_diff": per_arm[hi]["nll"] - per_arm[lo]["nll"]}

    def rule(ax):
        a1, a2, a3 = AXES[ax]
        s1, s2, s3 = (per_arm[x]["S"] for x in (a1, a2, a3))
        return {"gain_50_to_100": s3 - s2, "X": X_GO, "a_gain_ge_X": bool(s3 - s2 >= X_GO),
                "b_monotone": bool(s1 < s2 < s3), "holds": bool(s3 - s2 >= X_GO and s1 < s2 < s3)}

    rr, rp = rule("R"), rule("P")
    verdict = "GO" if rr["holds"] else "NO-GO"
    rep = {"format": FORMAT + " report", "created": r.now_iso(), "smoke": smoke,
           "smoke_note": "noise features, 300 steps: never a result" if smoke else None,
           "code": cs, "seconds": round(time.time() - t0, 1), "hundred_source": hundred_source, "seams": seams,
           "holdout": {"questions": {str(n): int(len(v[1])) for n, v in d.hold.items()}, "sampling": d.hold_stats},
           "arms": per_arm, "diffs": diffs,
           "rule_R_decisive": rr, "rule_P_record_only": rp, "verdict": verdict,
           "what": "PLAN.md 2-3: S19 retrained on nested subsets of the A players' probe data, scored on S19's "
                   "held-out H questions (rank-balanced official score, rule B on the 3-seed ensemble)"}
    path = os.path.join(SMOKE_DIR, "pilot_report.json") if smoke else REPORT
    T.write_json(path, rep)
    log(json.dumps({"rule_R": rr, "rule_P": rp, "verdict": verdict}, indent=1))
    log(f"-> {path} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--jobs", required=True)
    for q in (p, sub.add_parser("report")):
        q.add_argument("--smoke", action="store_true")
        q.add_argument("--rerun", action="store_true")
        q.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"train": cmd_train, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
