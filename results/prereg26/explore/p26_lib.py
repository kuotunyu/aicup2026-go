"""prereg26 exploration (competition 2): shared parts (no torch at import time, no GPU, never data/local).

- paths of the inputs (probe arrays of 09, V arrays of prereg25, the V EX array of this exploration);
- the training rows of P1TO_E26: probe rows (09's order) followed by the V rows (v_meta order), loaded into memory;
- a duck type of tools/p09_rank_fuse.Prep for Prep.moments (so the moments are computed by 09's own function);
- the frozen S25 models (results/prereg25/rank, SHA = prereg25 train_record.json).
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

import p25_lib as L25  # noqa: E402  (guard, loaded_repo_modules)

guard = L25.guard


def loaded_repo_modules(repo=REPO):
    """Repository files of every imported module (normalised prefix, as p25_lib). Modules whose __file__ is not an
    absolute path (torch._classes / torch._ops report '_classes.py' / '_ops.py') are skipped: abspath would put them in
    the working directory, which is the repository root."""
    root = os.path.join(os.path.abspath(repo), "").lower()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.isabs(f) and os.path.abspath(f).lower().startswith(root):
            out.add(os.path.relpath(os.path.abspath(f), repo).replace("\\", "/"))
    return out

PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")          # = 09's prep (SHA = check_report.json)
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")
V_PREP = os.path.join(REPO, "data", "prereg25", "prep")              # v_X.npy, v_meta.npz, prep_record.json
V_DIR = os.path.join(REPO, "data", "prereg25", "v")                  # V chunks and sets.json
DATA = os.path.join(REPO, "data", "prereg26")
V_EX = os.path.join(DATA, "prep", "v_ex.npy")
V_EX_RECORD = os.path.join(HERE, "v_ex_record.json")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
Q09_TRAIN = os.path.join(REPO, "results", "prereg09", "rank", "train_report.json")
P25_TRAIN = os.path.join(P25, "train_record.json")
P25_EVAL = os.path.join(P25, "eval_report.json")
S25_DIR = os.path.join(REPO, "results", "prereg25", "rank")
S25_FORMAT = "c10-prereg25 s25 model v1"
SEEDS = (0, 1, 2)
PROBE_FILES = ("probe_X.npy", "probe_ex.npy", "probe_meta.npz")
INNER_FILES = ("inner_X.npy", "inner_ex.npy", "inner_meta.npz")
SMOKE_ROWS = (6000, 2000)        # smoke: first probe rows, first V rows
SMOKE_EPOCHS = 2
NOISE_SEED = 12345
PORT_ROWS = 8192                 # port check: first PORT_ROWS probe rows + first PORT_ROWS V rows


def read_json(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path):
    with open(guard(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


class MomentsInput:
    """Duck type of p09_rank_fuse.Prep for Prep.moments: only probe_X / probe_ex are read."""

    def __init__(self, X, EX):
        self.probe_X, self.probe_ex = X, EX


def moments(r, X, EX, idx):
    """09's own Prep.moments (two streaming passes, fix_x / fix_ex repairs) on the given arrays."""
    return r.Prep.moments(MomentsInput(X, EX), np.asarray(idx, dtype=np.int64))


def moments_equal(a, b):
    """Bit-exact equality of the four moment vectors (NaN where numpy gives NaN counts as equal)."""
    return {k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True))
            for k in ("mu_x", "sd_x", "mu_ex", "sd_ex")}


def load_probe(r):
    """09's probe rows: raw float32 X, EX (memmaps), y, players, src (storage order P1, P2, P3)."""
    X = np.load(guard(os.path.join(PROBE_PREP, "probe_X.npy")), mmap_mode="r")
    EX = np.load(guard(os.path.join(PROBE_PREP, "probe_ex.npy")), mmap_mode="r")
    with np.load(guard(os.path.join(PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        players = np.asarray(z["players"]).astype(str)
        src = np.asarray(z["src"]).astype(str)
    n = len(y)
    if X.shape != (n, r.FW.X_DIM) or EX.shape != (n, r.FW.NEX):
        raise SystemExit("probe arrays and probe_meta.npz disagree")
    idx = np.nonzero(np.isin(src, np.array(list(r.PROBE_SETS))))[0]
    if not np.array_equal(idx, np.arange(n)):
        raise SystemExit("probe rows are not exactly the P1 + P2 + P3 rows in storage order")
    return {"X": X, "EX": EX, "y": y, "players": players, "src": src, "n": n}


def load_v(r):
    """V rows of prereg25 (v_X, v_meta) and the EX of this exploration (v_ex)."""
    X = np.load(guard(os.path.join(V_PREP, "v_X.npy")), mmap_mode="r")
    EX = np.load(guard(V_EX), mmap_mode="r")
    with np.load(guard(os.path.join(V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        y = np.asarray(z["y"]).astype(np.int64)
        players = np.asarray(z["players"]).astype(str)
        ref = np.asarray(z["ref"]).astype(np.int64)
    n = len(y)
    if X.shape != (n, r.FW.X_DIM) or EX.shape != (n, r.FW.NEX):
        raise SystemExit("v_X / v_ex and v_meta.npz disagree")
    return {"X": X, "EX": EX, "y": y, "players": players, "ref": ref, "n": n}


def player_ids(keys):
    return {k.split(":", 1)[1] for k in keys}


def s25_models(K19, load_npz_model):
    """The frozen S25 models (results/prereg25/rank, committed) with SHA = prereg25 train_record.json."""
    tr = read_json(P25_TRAIN)
    pars, seam = [], {"models": {}, "ok": True}
    for s in SEEDS:
        path = os.path.join(S25_DIR, f"s25_seed{s}.npz")
        sha = sha_file(path)
        par, seed, best_step, msmoke = load_npz_model(path, S25_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] and seed == s and not msmoke
                  and best_step == tr["seeds"][str(s)]["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam
