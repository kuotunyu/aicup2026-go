"""prereg40 exploration (competition 2): shared parts of PLAN.md (no torch).

S40 = prereg23's SET2 (model, features, pooling, training settings: imported unchanged from p23_lib / p23_train)
trained on prereg25's rows (probe + V) with prereg25's A / H split and S25's held-out questions (p19_lib.holdout_set,
HOLD_PER 100, HOLD_SEED 19019). This module builds the row table (keys, levels, colours), the standardised [X, EX]
features of selected rows, and the stage-0b held-out NLL. Never the validation answers, never inner rows.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P19 = os.path.join(REPO, "results", "prereg19", "explore")
P23 = os.path.join(REPO, "results", "prereg23", "explore")
P25 = os.path.join(REPO, "results", "prereg25", "explore")
for p in (os.path.join(REPO, "tools"), P19, P23, P25):
    if p not in sys.path:
        sys.path.insert(0, p)

import p09_forward as FW  # noqa: E402
import p19_lib as K19  # noqa: E402
import p23_lib as K23  # noqa: E402
import p25_lib as L25  # noqa: E402

L = K23.L
DATA = os.path.join(REPO, "data", "prereg40")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
V_PREP = os.path.join(REPO, "data", "prereg25", "prep")
V_EX = os.path.join(REPO, "data", "prereg26", "prep", "v_ex.npy")
V_EX_RECORD = os.path.join(REPO, "results", "prereg26", "explore", "v_ex_record.json")
S25_TRAIN_RECORD = os.path.join(P25, "train_record.json")
S25_EVAL_REPORT = os.path.join(P25, "eval_report.json")
S25_AUDIT = os.path.join(P25, "audit.json")
S25_MODEL_FORMAT = "c10-prereg25 s25 model v1"
S25_SEEDS = (0, 1, 2)
MODEL_FORMAT = "c10-prereg40 s40 model v1"

# PLAN.md 2.2: S25's split and held-out questions
SALT = L25.SALT
holdout_bit = L25.holdout_bit
HOLD_PER = K19.HOLD_PER
HOLD_SEED = K19.HOLD_SEED
# PLAN.md 2.1: SET2's settings (p23_lib, unchanged)
SEEDS = K23.SEEDS
D_IN = K23.D_IN
# PLAN.md 3 / 4
REF_Q25 = 0.7456519871546352
REF_Q25_MONO = 0.7482450052195216
SEAM_H_TOL = 1e-3          # PLAN.md rev 1 (torch float32 vs numpy float64 near the 1e-12 floor)
GATE_0B = -0.005
FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers")


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"refused path {p}")
    return p


# ---------------------------------------------------------------- rows (PLAN.md 2.2)
class Rows:
    """probe rows (data/prereg17/prep, as S25) followed by the V rows; keys '<rank>:<player_id>', levels, colours.
    Feature arrays are memory-mapped; raw X / EX of chosen rows through raw()."""

    def __init__(self, probe_colours):
        self.probe_X = np.load(guard(os.path.join(PROBE_PREP, "probe_X.npy")), mmap_mode="r")
        self.probe_ex = np.load(guard(os.path.join(PROBE_PREP, "probe_ex.npy")), mmap_mode="r")
        with np.load(guard(os.path.join(PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
            py = np.asarray(z["y"]).astype(np.int64)
            pk = np.asarray(z["players"]).astype(str)
            self.probe_src = np.asarray(z["src"]).astype(str)
        self.v_X = np.load(guard(os.path.join(V_PREP, "v_X.npy")), mmap_mode="r")
        self.v_ex = np.load(guard(V_EX), mmap_mode="r")
        with np.load(guard(os.path.join(V_PREP, "v_meta.npz")), allow_pickle=False) as z:
            vy = np.asarray(z["y"]).astype(np.int64)
            vk = np.asarray(z["players"]).astype(str)
            vc = np.asarray(z["color"]).astype(str)
        if self.probe_X.shape != (len(py), FW.X_DIM) or self.probe_ex.shape != (len(py), FW.NEX):
            raise SystemExit("probe_X / probe_ex and probe_meta.npz disagree")
        if self.v_X.shape != (len(vy), FW.X_DIM) or self.v_ex.shape != (len(vy), FW.NEX):
            raise SystemExit("v_X / v_ex and v_meta.npz disagree")
        pc = np.asarray(probe_colours).astype(str)
        if len(pc) != len(py):
            raise SystemExit("probe colours and probe_meta.npz disagree")
        self.n_probe, self.n_v = len(py), len(vy)
        self.keys = np.concatenate([pk, vk])
        self.y = np.concatenate([py, vy])
        self.col = np.concatenate([pc, vc])
        pks, vks = set(pk.tolist()), set(vk.tolist())
        self.overlap = {"probe_and_V_keys": len(pks & vks),
                        "probe_and_V_player_id": len(player_ids(pks) & player_ids(vks))}
        if self.overlap["probe_and_V_keys"] or self.overlap["probe_and_V_player_id"]:
            raise SystemExit(f"probe and V share players {self.overlap}")

    def __len__(self):
        return self.n_probe + self.n_v

    def raw(self, idx):
        """Raw (X float64, EX float32) of the given global row indices (any order, repeats allowed)."""
        return gather_raw(idx, self.n_probe, (self.probe_X, self.probe_ex), (self.v_X, self.v_ex))


def player_ids(keys):
    return {k.split(":", 1)[1] for k in keys}


def gather_raw(idx, n_first, first, second):
    """Rows idx of the concatenation [first; second] of two (X, EX) array pairs (memory-mapped reads in sorted
    order) -> (X float64, EX float32)."""
    idx = np.asarray(idx, dtype=np.int64).ravel()
    X = np.empty((len(idx), first[0].shape[1]), dtype=np.float64)
    EX = np.empty((len(idx), first[1].shape[1]), dtype=np.float32)
    for mask, (Xa, Ea), off in ((idx < n_first, first, 0), (idx >= n_first, second, n_first)):
        if not mask.any():
            continue
        pos = np.flatnonzero(mask)
        src = idx[mask] - off
        o = np.argsort(src, kind="stable")
        X[pos[o]] = np.asarray(Xa[src[o]], dtype=np.float64)
        EX[pos[o]] = np.asarray(Ea[src[o]], dtype=np.float32)
    return X, EX


def features32(Xraw, EXraw, mp):
    """= p23_lib.features (standardise X, standardise EX with non-finite -> 0) as float32 [N, 1584], the dtype the
    SET2 training sees. Element-wise, so a row's value does not depend on its block."""
    if np.isinf(np.asarray(EXraw)).any():
        raise SystemExit("raw EX has inf values: standardise_ex would map them to +-1.8e308")
    X = K23.features(Xraw, EXraw, mp).astype(np.float32)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite standardised features")
    return X


def split_pools(rows):
    """A / H pools of prereg25's salt (PLAN.md 2.2)."""
    bit = {k: holdout_bit(k) for k in set(rows.keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(rows.keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(rows.keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K19.PlayerPool(rows.keys, rows.y, rows.col, rows_a)
    pool_h = K19.PlayerPool(rows.keys, rows.y, rows.col, rows_h)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    return rows_a, rows_h, pool_a, pool_h


# ---------------------------------------------------------------- held-out NLL in numpy (seam H, stage 0b)
def holdout_lp(fwd, par_list, hold, Xfun):
    """{n: [lp of each model, Q x 10]} on the held-out questions; Xfun(flat idx) -> features [len(idx), d]."""
    out = {}
    for n, (idx, y) in hold.items():
        if len(y) == 0:
            continue
        Xg = np.asarray(Xfun(idx.ravel()), dtype=np.float64)
        Xg = Xg.reshape(len(y), n, Xg.shape[1])
        out[n] = [fwd(par, Xg) for par in par_list]
    return out


def mean_nll(lp_by_n, hold, members=None):
    """Mean NLL over all held-out questions of the lse-mean ensemble of the chosen members (all when None)
    -> (mean, {n: mean})."""
    tot, cnt = 0.0, 0
    per_n = {}
    for n, lps in lp_by_n.items():
        y = np.asarray(hold[n][1])
        sel = lps if members is None else [lps[i] for i in members]
        lp = FW.lse_mean(sel) if len(sel) > 1 else sel[0]
        v = -lp[np.arange(len(y)), y]
        per_n[str(n)] = float(v.mean())
        tot += float(v.sum())
        cnt += len(y)
    return tot / cnt, per_n


def stage0b_verdict(nll_s40, nll_s25):
    d = nll_s40 - nll_s25
    return {"delta": d, "threshold": GATE_0B, "passed": bool(d <= GATE_0B)}


def loaded_repo_modules(repo=REPO):
    out = set()
    root = os.path.abspath(repo).lower() + os.sep
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(root):
            out.add(os.path.relpath(os.path.abspath(f), repo).replace("\\", "/"))
    return out
