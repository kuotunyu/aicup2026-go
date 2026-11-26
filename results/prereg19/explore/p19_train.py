"""prereg19 exploration (competition 2): train the question-level set model S19 of PLAN.md 3 on probe players only
(CPU, .venv torch). Reads no inner data, no validation data, never data/local; computes no candidate number.

    Git Bash, repository root (a subst drive when the worktree path contains 'AppData\\Local'); PYTHONDONTWRITEBYTECODE=1;
    .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg19/explore
    $S/p19_train.py train            -> data/prereg19/models/s19_seed{0,1,2}.npz + $S/train_record.json (run once; the
                                        code must be committed and clean)
    $S/p19_train.py train --smoke    -> data/prereg19/smoke/ (P1 features replaced by default_rng(12345) normal noise,
                                        300 steps per seed: pipeline and timing only, never a result)
    --resume keeps a finished seed (its model file and seed record) after an interruption (seeds are independent).
    A file data/prereg19/STOP makes the run stop at the next held-out evaluation (delete it by hand to restart).
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
FT_DIR = os.path.join(REPO, "results", "prereg12", "explore", "finetune")
for p in (TOOLS, FT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402
import p09_forward as FW  # noqa: E402
import ft_eval  # noqa: E402  (guard_path)
import p19_lib as K  # noqa: E402

guard_path = ft_eval.guard_path
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
SETS_JSON = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
             "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "sets.json"),
             "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "sets.json")}
DATA = os.path.join(REPO, "data", "prereg19")
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p19_lib.py", "p19_train.py")
IMPORTED = ("tools/p09_forward.py", "tools/p09_rank_fuse.py", "tools/p09_map.py",
            "results/prereg12/explore/finetune/ft_eval.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/frozen_config.json",
            "results/prereg09/rank/check_report.json")
FORMAT = "c10-prereg19 explore train v1"
MODEL_FORMAT = "c10-prereg19 s19 model v1"
NOISE_SEED = 12345
BLOCK = 8192
STATE_KEYS = {"h.1.weight": "W1", "h.1.bias": "b1", "h.4.weight": "W2", "h.4.bias": "b2", "rho.0.weight": "R1",
              "rho.0.bias": "c1", "rho.2.weight": "R2", "rho.2.bias": "c2", "t0": "t0", "dt": "dt"}


def log(*x):
    print(*x, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(guard_path(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def loaded_repo_modules():
    """Repository files of every imported module (code_state coverage)."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(REPO.lower() + os.sep):
            out.add(os.path.relpath(os.path.abspath(f), REPO).replace("\\", "/"))
    return out


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    rel = sorted(set(rel) | loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


# ---------------------------------------------------------------- torch model (PLAN.md 3.1)
def torch_model(seed):
    import torch

    class S19(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h = torch.nn.Sequential(torch.nn.Dropout(K.DROP_IN), torch.nn.Linear(K.D_IN, K.HIDDEN),
                                         torch.nn.GELU(), torch.nn.Dropout(K.DROP),
                                         torch.nn.Linear(K.HIDDEN, K.H_DIM), torch.nn.GELU())
            self.rho = torch.nn.Sequential(torch.nn.Linear(K.H_DIM, K.RHO), torch.nn.GELU(),
                                           torch.nn.Linear(K.RHO, 1))
            self.t0 = torch.nn.Parameter(torch.tensor(-4.0))
            self.dt = torch.nn.Parameter(torch.full((K.L - 2,), math.log(math.e - 1)))

        def forward(self, x):
            """x [Q, n, 1540] -> log P(y = k) [Q, 10]."""
            v = self.h(x).mean(dim=1)
            f = self.rho(v)
            th = torch.cat([self.t0.view(1), self.t0 + torch.cumsum(torch.nn.functional.softplus(self.dt), 0)])
            c = torch.sigmoid(th.view(1, -1) - f)
            c = torch.cat([torch.zeros_like(f), c, torch.ones_like(f)], 1)
            return torch.log((c[:, 1:] - c[:, :-1]).clamp_min(1e-12))

    torch.manual_seed(seed)
    return S19()


def export_params(model):
    sd = model.state_dict()
    if set(sd) != set(STATE_KEYS):
        raise SystemExit(f"unexpected state_dict keys {sorted(sd)}")
    return {STATE_KEYS[k]: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in sd.items()}


# ---------------------------------------------------------------- data
class ProbeOnly:
    """The probe part of 09's prep only (probe_X.npy mmap, probe_meta.npz); the inner arrays are never opened."""

    def __init__(self, d):
        self.probe_X = np.load(guard_path(os.path.join(d, "probe_X.npy")), mmap_mode="r")
        with np.load(guard_path(os.path.join(d, "probe_meta.npz")), allow_pickle=False) as z:
            self.y = np.asarray(z["y"]).astype(np.int64)
            self.players = np.asarray(z["players"]).astype(str)
            self.src = np.asarray(z["src"]).astype(str)
            self.ref = np.asarray(z["ref"]).astype(np.int64)
        if self.probe_X.shape != (len(self.y), FW.X_DIM):
            raise SystemExit("probe_X.npy and probe_meta.npz disagree")


def probe_colours(prep):
    """Probe colours from the sets.json files (player cross-checked, as prereg18) -> colour [N]."""
    games = {}
    for s, p in SETS_JSON.items():
        with open(guard_path(p), encoding="utf-8") as f:
            games[s] = json.load(f)["probe"]["games"]
    col = np.empty(len(prep.ref), dtype="<U1")
    for i, (s, j, pk) in enumerate(zip(prep.src, prep.ref, prep.players)):
        g = games[str(s)][int(j)]
        if f"{g['rank']}:{g['player_id']}" != str(pk):
            raise SystemExit(f"probe row {i}: sets.json player differs from probe_meta")
        col[i] = g["color"]
    return col


def standardised_probe(prep, mu, sd, smoke):
    n = prep.probe_X.shape[0]
    X = np.empty((n, K.D_IN), dtype=np.float32)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        if smoke:
            X[s:e] = np.random.default_rng([NOISE_SEED, 0, s]).standard_normal((e - s, K.D_IN)).astype(np.float32)
        else:
            X[s:e] = FW.standardise_x(np.asarray(prep.probe_X[s:e], dtype=np.float64), mu, sd).astype(np.float32)
    return X


# ---------------------------------------------------------------- training (PLAN.md 3.2-3.5)
def holdout_nll(model, X32, hold):
    import torch
    model.eval()
    tot, cnt = 0.0, 0
    lps = {}
    with torch.no_grad():
        for n, (idx, y) in hold.items():
            if len(y) == 0:
                continue
            xb = torch.from_numpy(X32[idx.ravel()]).view(len(y), n, K.D_IN)
            lp = model(xb).double().numpy()
            lps[n] = lp
            tot += float(-lp[np.arange(len(y)), y].sum())
            cnt += len(y)
    model.train()
    return tot / cnt, lps


def train_seed(seed, X32, pool_a, hold, steps, eval_every, log_prefix):
    import torch
    torch.set_num_threads(K.THREADS)
    model = torch_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=K.LR, weight_decay=K.WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=steps, pct_start=K.PCT_START)
    rng = np.random.default_rng([K.SAMPLER_BASE, seed])
    stats = {"colour_switched": 0, "level_redrawn": 0, "questions": 0}
    n_count = {str(n): 0 for n in K.KS}
    curve, best = [], None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        idx, y, n = K.sample_batch(rng, pool_a, K.BATCH, stats)
        n_count[str(n)] += 1
        xb = torch.from_numpy(X32[idx.ravel()]).view(K.BATCH, n, K.D_IN)
        lp = model(xb)
        loss = -lp[torch.arange(K.BATCH), torch.from_numpy(y)].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % eval_every == 0:
            nll, _ = holdout_nll(model, X32, hold)
            curve.append({"step": step, "holdout_nll": nll, "train_batch_loss": float(loss.item()),
                          "seconds": round(time.time() - t0, 1)})
            if best is None or nll < best["nll"]:
                best = {"step": step, "nll": nll, "params": export_params(model)}
            log(f"{log_prefix} seed {seed} step {step}: holdout NLL {nll:.5f} (best step {best['step']}) "
                f"{time.time() - t0:.0f}s")
            if os.path.exists(STOP):
                raise SystemExit(f"STOP file {STOP}: stopped at step {step}")
    return best, curve, stats, n_count, round(time.time() - t0, 1)


def port_check(params, X32, hold):
    """torch float32 vs numpy float64 forward of the same weights on the held-out questions (PLAN.md 3.6)."""
    import torch
    model = torch_model(0)
    sd = {k: torch.from_numpy(np.asarray(params[v])) for k, v in STATE_KEYS.items()}
    model.load_state_dict(sd)
    _nll, lps_t = holdout_nll(model, X32, hold)
    par = K.p64(params)
    worst_p, worst_lp = 0.0, 0.0
    for n, (idx, y) in hold.items():
        if len(y) == 0:
            continue
        Xg = X32[idx.ravel()].astype(np.float64).reshape(len(y), n, K.D_IN)
        lp_np = K.set_logp_np(par, Xg)
        worst_p = max(worst_p, float(np.max(np.abs(np.exp(lp_np) - np.exp(lps_t[n])))))
        worst_lp = max(worst_lp, float(np.max(np.abs(lp_np - lps_t[n]))))
    return {"max_abs_prob_diff": worst_p, "max_abs_logp_diff_record": worst_lp, "tol": K.TORCH_TOL,
            "ok": bool(worst_p <= K.TORCH_TOL)}


def save_model(path, params, seed, best, smoke):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(MODEL_FORMAT), seed=np.array(seed), best_step=np.array(best["step"]),
             smoke=np.array(bool(smoke)), **{f"p_{k}": params[k] for k in K.PARAM_NAMES})
    os.replace(tmp, path)
    return sha_file(path)


def load_model(path):
    with np.load(guard_path(path), allow_pickle=False) as z:
        if str(z["format"]) != MODEL_FORMAT:
            raise SystemExit(f"{path}: not an S19 model file")
        return {k: np.asarray(z[f"p_{k}"]) for k in K.PARAM_NAMES}, int(z["seed"]), int(z["best_step"]), \
            bool(z["smoke"])


def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_dir = SMOKE_DIR if smoke else DATA
    model_dir = os.path.join(out_dir, "models")
    rec_path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else RECORD
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg19 files or the imported tools are not committed and clean: {cs['dirty']}")
        prior = [p for p in (rec_path, STARTED) if os.path.exists(p)]
        if prior and not a.resume and not (a.rerun and a.reason):
            raise SystemExit(f"{prior} exist: run once (--resume after an interruption; --rerun --reason only after "
                             f"a commit that names a concrete program error)")
        if a.rerun and prior:
            heads = set()
            for p in prior:
                with open(p, encoding="utf-8") as f:
                    heads.add((json.load(f).get("code") or {}).get("head"))
            if cs["head"] in heads:
                raise SystemExit("--rerun needs a newer commit than the earlier run's")
        if a.resume and os.path.exists(STARTED):
            stc = r.read_json(STARTED).get("code") or {}
            if stc.get("sha256_lf") != cs["sha256_lf"] or stc.get("head") != cs["head"]:
                raise SystemExit("--resume needs the same commit and code as train_started.json")
        else:
            write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    if os.path.exists(STOP):
        raise SystemExit(f"STOP file {STOP} exists: delete it by hand to start")
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    # ---- inputs
    mp = FW.load_map(Q09_MAP)
    map_sha = r.sha_raw(Q09_MAP)
    if map_sha != r.read_json(Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    want = (r.read_json(Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = sha_file(os.path.join(PROBE_PREP, "probe_X.npy"))
    seam_p = {"want": want.get("probe_X.npy"), "got": got, "ok": bool(want.get("probe_X.npy") == got)}
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    prep = ProbeOnly(PROBE_PREP)
    if set(np.unique(prep.src)) != set(r.PROBE_SETS):
        raise SystemExit("probe rows of another set")
    col = probe_colours(prep)
    keys = np.asarray(prep.players).astype(str)
    bit = {k: K.holdout_bit(k) for k in set(keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K.PlayerPool(keys, prep.y, col, rows_a)
    pool_h = K.PlayerPool(keys, prep.y, col, rows_h)
    seam_ah = len(set(pool_a.players) & set(pool_h.players))
    if seam_ah:
        raise SystemExit("A and H share players")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    players_rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": seam_ah,
                   "A_players_per_level": np.bincount(pool_a.level, minlength=K.L).tolist(),
                   "H_players_per_level": np.bincount(pool_h.level, minlength=K.L).tolist(),
                   "A_eligible_n_level_colour": pool_a.counts(), "H_eligible_n_level_colour": pool_h.counts(),
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats}
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games')})}")
    t = time.time()
    X32 = standardised_probe(prep, mp["mu_x"], mp["sd_x"], smoke)
    log(f"standardised probe X ({time.time() - t:.0f}s)")
    import torch
    seeds_rec = {}
    for seed in K.SEEDS:
        mpath = os.path.join(model_dir, f"s19_seed{seed}.npz")
        spath = os.path.join(out_dir, f"seed{seed}_record.json")
        if a.resume and os.path.exists(mpath) and os.path.exists(spath):
            seeds_rec[str(seed)] = r.read_json(spath)
            if seeds_rec[str(seed)].get("model_sha256") != sha_file(mpath):
                raise SystemExit(f"{mpath}: SHA differs from its seed record")
            if not (seeds_rec[str(seed)].get("port_check") or {}).get("ok"):
                raise SystemExit(f"seed {seed}: kept record without a passed port check")
            log(f"seed {seed}: kept (resume)")
            continue
        best, curve, stats, n_count, secs = train_seed(seed, X32, pool_a, hold, steps, eval_every,
                                                       "smoke" if smoke else "train")
        pc = port_check(best["params"], X32, hold)
        msha = save_model(mpath, best["params"], seed, best, smoke)
        sr = {"seed": seed, "best_step": best["step"], "best_holdout_nll": best["nll"], "curve": curve,
              "sampling": stats, "batches_per_n": n_count, "seconds": secs, "port_check": pc,
              "model_file": os.path.relpath(mpath, REPO).replace("\\", "/"), "model_sha256": msha}
        write_json(spath, sr)
        seeds_rec[str(seed)] = sr
        log(f"seed {seed}: best step {best['step']} NLL {best['nll']:.5f}; port check {pc}")
        if not pc["ok"]:
            raise SystemExit(f"seed {seed}: torch / numpy port check failed {pc}")
    rec = {"format": FORMAT, "created": r.now_iso(), "smoke": smoke,
           "smoke_note": "P1 features replaced by default_rng(12345) normal noise, 300 steps: never a result"
           if smoke else None, "seconds": round(time.time() - t_all, 1), "code": cs,
           "what": "PLAN.md 3: S19 (per-game h -> mean -> rho -> cumulative link) trained on probe players A, step "
                   "chosen on held-out probe players H; no inner data read",
           "settings": {"steps": steps, "eval_every": eval_every, "hold_per": hold_per, "batch": K.BATCH,
                        "ks": list(K.KS), "seeds": list(K.SEEDS), "lr": K.LR, "wd": K.WD, "pct_start": K.PCT_START,
                        "hidden": K.HIDDEN, "h_dim": K.H_DIM, "rho": K.RHO, "drop_in": K.DROP_IN, "drop": K.DROP,
                        "threads": K.THREADS, "hold_seed": K.HOLD_SEED, "sampler_base": K.SAMPLER_BASE},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "q09_map_sha256": map_sha, "seam_p": seam_p, "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in K.SEEDS}}
    write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
