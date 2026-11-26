"""prereg29 exploration (competition 2): shared parts (no torch at import time, no GPU, never the answers folder).

- the seeds of S29 (PLAN.md 1): seeds 0-2 = the frozen S25 files, seeds 3-9 new (same seed rule as S25);
- paths of the frozen S25 files and of this exploration's outputs;
- loaded_repo_modules with an absolute-path filter (as prereg26: torch._classes / torch._ops report relative names);
- the frozen S25 models (results/prereg25/rank, SHA = prereg25 train_record.json) and the judged Q25 reference.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P25 = os.path.join(REPO, "results", "prereg25", "explore")
for p in (os.path.join(REPO, "tools"), P25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p25_lib as L25  # noqa: E402  (guard, holdout_bit / SALT)

guard = L25.guard

S25_SEEDS = (0, 1, 2)                  # reused byte-identically (results/prereg25/rank)
NEW_SEEDS = (3, 4, 5, 6, 7, 8, 9)      # trained here, same rule as S25: torch.manual_seed(s), default_rng([19, s])
ALL_SEEDS = S25_SEEDS + NEW_SEEDS      # ensemble order of S29
SEAM_R_SEED = 0                        # retrained here; must equal S25 seed 0 byte for byte
SUBSETS = (3, 5, 10)                   # record-only seed-count curve: seeds 0 .. k-1

DATA = os.path.join(REPO, "data", "prereg29")
MODELS = os.path.join(DATA, "models")
SMOKE_DIR = os.path.join(DATA, "smoke")
S25_DIR = os.path.join(REPO, "results", "prereg25", "rank")
S25_FORMAT = "c10-prereg25 s25 model v1"   # S29 files keep S25's format string (seam R compares bytes)
P25_TRAIN = os.path.join(P25, "train_record.json")
P25_EVAL = os.path.join(P25, "eval_report.json")
P25_AUDIT = os.path.join(P25, "audit.json")
V_PREP = os.path.join(REPO, "data", "prereg25", "prep")


def model_name(seed, retrain=False):
    return f"s29_seed{seed}_retrain.npz" if retrain else f"s29_seed{seed}.npz"


def loaded_repo_modules(repo=REPO):
    """Repository files of every imported module (normalised prefix; non-absolute __file__ skipped, as prereg26)."""
    root = os.path.join(os.path.abspath(repo), "").lower()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.isabs(f) and os.path.abspath(f).lower().startswith(root):
            out.add(os.path.relpath(os.path.abspath(f), repo).replace("\\", "/"))
    return out


def read_json(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)


def sha_file(path):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def s25_frozen_sha():
    """seed -> SHA of the frozen S25 file as recorded by prereg25 train_record.json."""
    tr = read_json(P25_TRAIN)
    return {int(k): v for k, v in tr["models_sha256"].items()}


def s25_models(K19, load_npz_model):
    """The frozen S25 models (results/prereg25/rank) with SHA = prereg25 train_record.json (seam F)."""
    tr = read_json(P25_TRAIN)
    pars, seam = [], {"models": {}, "ok": True}
    for s in S25_SEEDS:
        path = os.path.join(S25_DIR, f"s25_seed{s}.npz")
        sha = sha_file(path)
        par, seed, best_step, msmoke = load_npz_model(path, S25_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] and seed == s and not msmoke
                  and best_step == tr["seeds"][str(s)]["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam


def refs25():
    e = read_json(P25_EVAL)
    if e.get("smoke") or not e["gates"]["passed"] or e["candidate"]["name"] != "Q25":
        raise SystemExit("results/prereg25/explore/eval_report.json is not the judged Q25 report")
    return {"Q25": e["candidate"]["inner_rank_balanced"], "Q25_mono": e["candidate"]["single_colour_rank_balanced"]}


def registered_settings(K19):
    """The S25 training settings of PLAN.md 1 (= prereg25 train_record.json settings except 'seeds')."""
    return {"steps": K19.STEPS, "eval_every": K19.EVAL_EVERY, "hold_per": K19.HOLD_PER, "batch": K19.BATCH,
            "ks": list(K19.KS), "lr": K19.LR, "wd": K19.WD, "pct_start": K19.PCT_START, "hidden": K19.HIDDEN,
            "h_dim": K19.H_DIM, "rho": K19.RHO, "drop_in": K19.DROP_IN, "drop": K19.DROP, "threads": K19.THREADS,
            "hold_seed": K19.HOLD_SEED, "sampler_base": K19.SAMPLER_BASE, "salt": L25.SALT}


def subset_seeds(k):
    if k not in SUBSETS:
        raise ValueError(k)
    return ALL_SEEDS[:k]


def models_equal_bytes(a, b):
    with open(guard(a), "rb") as x, open(guard(b), "rb") as y:
        return x.read() == y.read()


def finite_params(par):
    return bool(all(np.all(np.isfinite(np.asarray(v))) for v in par.values()))
