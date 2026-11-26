"""PREREG_09 competition 2: the features and the numpy float64 forward pass of P1TO_E (ens3fam), shared by
tools/p09_map.py, tools/p09_rank_fuse.py (tune, apply) and later the test-day tool (docs/PREREG_09_MAPPING_ENSEMBLE.md
rev 1, section 1.3).

Pure numpy / scipy: this module never imports torch, u1_lib, hs_lib, pilot_* or p1_* (tests/test_p09_hygiene.py).

Features of one game (PREREG_09 1.3 item 1)
    X   1,540: target-side means (770) next to opponent-side means (770) of the P1 chunk (= p1_fuse.variant_X(.., "TO");
        the opponent half is NaN when the opponent has no move), standardised with the probe's mu / sd exactly as
        p1_fuse.standardise ((x - mu) / sd, every non-finite value -> 0).
    EX  44: results/prereg09/explore/mapping/prep.py extra_features (per side: std, 10/25/50/75/90 % quantiles and the
        turn-bucket means [0,30) [30,80) [80,150) [150,inf) of log p and of the entropy, the fraction of log p < log 0.05,
        log1p(positions)), from the chunk's per-position pos_game / pos_side / pos_turn / pos_logp / pos_entropy; the
        raw value is rounded to float32 (as prep.py stored it), then standardised with the probe's nanmean / nanstd
        (non-finite or 0 sd -> 1) and NaN -> 0 (np.nan_to_num).
Canonical forward input: float64 X and EX computed this way from the float32 chunk values (no float32 rounding of the
standardised values). Training feeds float32(X), float32(EX) (= what prep.py stored); the difference is covered by
the registered torch-vs-numpy tolerance 1e-4 (PREREG_09 2.3 / 2.6).

Members (PREREG_09 1.3 item 2; torch layouts, weights stored float32 and read as float64)
    mlpclm  Dropout -> Linear(1540, 256) W1 b1 -> GELU (erf) -> Dropout -> Linear(256, 1) W2 b2 = strength f;
            theta_1 = t0, theta_k = t0 + sum softplus(dt); P(y <= k) = sigmoid(theta_k - f); adjacent differences,
            floor 1e-12, log.
    mlp     Dropout -> Linear(1540, 512) -> GELU -> Dropout -> Linear(512, 10) -> log_softmax.
    lowrank Linear(1584, 3, no bias) A -> Linear(3, 10) B b -> log_softmax, input [X, EX].
    ensemble lp = log(max(mean over members of exp(lp_member), 1e-300)) (= combos.lse_mean, member order fixed).
"""

import json
import math

import numpy as np
from scipy import special

L = 10
F_DIM = 770
X_DIM = 2 * F_DIM
TURN_EDGES = [0, 30, 80, 150, 10 ** 6]
QS = [0.1, 0.25, 0.5, 0.75, 0.9]
LOWP = np.log(0.05)
MAP_FORMAT = "c10-prereg09 p1e_map v1"
SCALE = 4.0
LSE_FLOOR = 1e-300
CLM_FLOOR = 1e-12
SOFTPLUS_THRESHOLD = 20.0          # torch.nn.functional.softplus default (beta 1)

# the six members of ens3fam, in the registered order (= the exploration's ensemble order)
MEMBERS = (
    {"name": "mlpclm256", "kind": "mlpclm", "seed": 0},
    {"name": "mlpclm256_s1", "kind": "mlpclm", "seed": 1},
    {"name": "mlpclm256_s2", "kind": "mlpclm", "seed": 2},
    {"name": "mlp512", "kind": "mlp", "seed": 0},
    {"name": "mlp512_s1", "kind": "mlp", "seed": 1},
    {"name": "lowrank3_ex", "kind": "lowrank", "seed": 0},
)
# PREREG_09 1.3 item 2 (= train.py V["mlpclm256"], V["mlp512"], V["lowrank3_ex"])
CONFIGS = {
    "mlpclm": {"kind": "mlpclm", "hidden": [256], "drop_in": 0.1, "drop": 0.3, "wd": 1e-2, "epochs": 24, "lr": 1e-3},
    "mlp": {"kind": "mlp", "hidden": [512], "drop_in": 0.2, "drop": 0.4, "wd": 1e-2, "epochs": 24, "lr": 1e-3},
    "lowrank": {"kind": "lowrank", "rank": 3, "lam": 1e-4, "epochs": 30, "lr": 2e-3, "ex": True},
}
PARAMS = {"mlpclm": ("W1", "b1", "W2", "b2", "t0", "dt"), "mlp": ("W1", "b1", "W2", "b2"),
          "lowrank": ("A", "B", "b")}
# torch state_dict key -> stored name
STATE_KEYS = {"mlpclm": {"f.1.weight": "W1", "f.1.bias": "b1", "f.4.weight": "W2", "f.4.bias": "b2", "t0": "t0",
                         "dt": "dt"},
              "mlp": {"f.1.weight": "W1", "f.1.bias": "b1", "f.4.weight": "W2", "f.4.bias": "b2"},
              "lowrank": {"a.weight": "A", "b.weight": "B", "b.bias": "b"}}


class ForwardError(RuntimeError):
    pass


def member_names():
    return [m["name"] for m in MEMBERS]


def uses_ex(kind):
    return bool(CONFIGS[kind].get("ex", False))


# ================================================================ features
def ex_names():
    out = []
    for side in ("t", "o"):
        for q in ("logp", "ent"):
            out += [f"{side}_{q}_std"] + [f"{side}_{q}_q{int(100 * p)}" for p in QS]
            out += [f"{side}_{q}_turn{TURN_EDGES[i]}_{TURN_EDGES[i + 1]}" for i in range(len(TURN_EDGES) - 1)]
        out += [f"{side}_frac_lowp", f"{side}_log1p_count"]
    return out


NEX = len(ex_names())


def extra_features(z, G):
    """prep.py extra_features, unchanged: per game and side std / quantiles of log p and entropy, turn-bucket means,
    low-probability fraction, log1p count -> float64 [G, 44] (NaN where a side has no position)."""
    ex = np.full((G, NEX), np.nan, dtype=np.float64)
    pg, ps, pt = z["pos_game"], z["pos_side"], z["pos_turn"]
    lp, en = z["pos_logp"].astype(np.float64), z["pos_entropy"].astype(np.float64)
    order = np.lexsort((ps, pg))
    pg, ps, pt, lp, en = pg[order], ps[order], pt[order], lp[order], en[order]
    key = pg.astype(np.int64) * 2 + ps
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key)]])
    per_side = NEX // 2
    for s0, e0 in zip(starts, ends):
        if e0 <= s0:
            continue
        g, s = int(pg[s0]), int(ps[s0])
        row = []
        for v in (lp[s0:e0], en[s0:e0]):
            row.append(v.std())
            row += list(np.quantile(v, QS))
            t = pt[s0:e0]
            for i in range(len(TURN_EDGES) - 1):
                m = (t >= TURN_EDGES[i]) & (t < TURN_EDGES[i + 1])
                row.append(v[m].mean() if m.any() else np.nan)
        row.append(float((lp[s0:e0] < LOWP).mean()))
        row.append(np.log1p(e0 - s0))
        ex[g, s * per_side:(s + 1) * per_side] = row
    return ex


def ex_raw32(z, G):
    """The stored (float32) raw extra features of a chunk's G games."""
    return extra_features(z, G).astype(np.float32)


def usable_games(failed, counts):
    """P1 usable: not failed and >= 1 target move (PREREG_08 1.6 / p1_fuse.variant_X)."""
    return (~np.asarray(failed, dtype=bool)) & (np.asarray(counts)[:, 0] > 0)


def x_raw(means, counts):
    """TO features in p1_fuse.variant_X's layout: float64 [G, 1540], opponent half NaN without an opponent move."""
    means = np.asarray(means)
    T = means[:, 0, :].astype(np.float64)
    O = means[:, 1, :].astype(np.float64)
    O[np.asarray(counts)[:, 1] == 0] = np.nan
    return np.concatenate([T, O], axis=1)


def x_raw32(means, counts):
    """The stored (float32, lossless: the chunk means are float32) raw TO features."""
    return x_raw(means, counts).astype(np.float32)


def standardise_x(X, mu, sd):
    """p1_fuse.standardise with given mu / sd: (X - mu) / sd, non-finite -> 0 (float64)."""
    Xs = (np.asarray(X, dtype=np.float64) - mu) / sd
    Xs[~np.isfinite(Xs)] = 0.0
    return Xs


def standardise_ex(EX, mu, sd):
    """prep.py / train.py load(): nan_to_num((ex - mu) / sd) on the float32-rounded raw values (float64)."""
    return np.nan_to_num((np.asarray(EX, dtype=np.float32).astype(np.float64) - mu) / sd)


def nan_moments(blocks):
    """Streaming np.nanmean / np.nanstd over axis 0, bit for bit equal to the in-memory calls on the row-wise
    concatenation (two passes, rows in order). blocks: a callable returning an iterator of float64 [n, d] blocks.
    -> (mean, std) with NaN where a column has no value (as numpy)."""
    acc = cnt = None
    for b in blocks():
        b = np.asarray(b, dtype=np.float64)
        m = np.isnan(b)
        v = np.where(m, 0.0, b)
        if acc is None:
            acc = np.zeros(b.shape[1])
            cnt = np.zeros(b.shape[1], dtype=np.intp)
        acc = np.sum(np.concatenate([acc[None, :], v]), axis=0)
        cnt = cnt + (~m).sum(axis=0)
    if acc is None:
        raise ForwardError("no rows for the probe moments")
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = acc / cnt
    acc2 = np.zeros_like(acc)
    for b in blocks():
        b = np.asarray(b, dtype=np.float64)
        m = np.isnan(b)
        d = np.where(m, 0.0, b) - mu
        d[m] = 0.0
        d = d * d
        acc2 = np.sum(np.concatenate([acc2[None, :], d]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(acc2 / cnt)
    return mu, sd


def fix_x_moments(mu, sd):
    """p1_fuse.standardise's repairs (sd non-finite or 0 -> 1, mu non-finite -> 0)."""
    mu, sd = mu.copy(), sd.copy()
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return mu, sd


def fix_ex_moments(mu, sd):
    """train.py load(): only sd is repaired (non-finite or 0 -> 1)."""
    sd = sd.copy()
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return mu.copy(), sd


def player_folds(players, seed=0, k=5):
    """= results/prereg08/p1/p1_fuse.py player_folds (tests compare them)."""
    uniq = np.array(sorted(set(players)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {p: int(i % k) for i, p in zip(range(len(uniq)), uniq[perm])}
    return np.array([fold_of[p] for p in players])


# ================================================================ forward (numpy float64)
def gelu(x):
    return 0.5 * x * (1.0 + special.erf(x / math.sqrt(2.0)))


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > SOFTPLUS_THRESHOLD, x, np.log1p(np.exp(np.minimum(x, SOFTPLUS_THRESHOLD))))


def log_softmax(z):
    m = np.max(z, axis=-1, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))


def clm_logp(f, t0, dt):
    """f [N, 1] -> log P(y = k) [N, 10] of the cumulative link."""
    t0 = float(np.asarray(t0, dtype=np.float64).reshape(()))
    th = np.concatenate([[t0], t0 + np.cumsum(softplus(np.asarray(dt, dtype=np.float64)))])
    c = special.expit(th[None, :] - f)
    c = np.concatenate([np.zeros_like(f), c, np.ones_like(f)], axis=1)
    return np.log(np.maximum(c[:, 1:] - c[:, :-1], CLM_FLOOR))


def _p64(params):
    return {k: np.asarray(v).astype(np.float64) for k, v in params.items()}


def member_logp(kind, params, X, EX=None):
    """lp [N, 10] of one member (eval mode: dropout off) on float64 X [N, 1540] (and EX [N, 44] for lowrank)."""
    p = _p64(params)
    X = np.asarray(X, dtype=np.float64)
    if kind == "mlpclm":
        h = gelu(X @ p["W1"].T + p["b1"])
        f = h @ p["W2"].T + p["b2"]
        return clm_logp(f, p["t0"], p["dt"])
    if kind == "mlp":
        h = gelu(X @ p["W1"].T + p["b1"])
        return log_softmax(h @ p["W2"].T + p["b2"])
    if kind == "lowrank":
        if EX is None:
            raise ForwardError("lowrank member needs EX")
        Z = np.concatenate([X, np.asarray(EX, dtype=np.float64)], axis=1)
        return log_softmax((Z @ p["A"].T) @ p["B"].T + p["b"])
    raise ForwardError(f"member kind {kind!r}")


def lse_mean(lps):
    """combos.lse_mean: log(clip(mean of exp(lp), 1e-300))."""
    m = np.mean([np.exp(x) for x in lps], 0)
    return np.log(np.clip(m, LSE_FLOOR, None))


def ensemble_logp(members, X, EX):
    """members: [(kind, params)] in the registered order -> (ensemble lp [N, 10], [member lp])."""
    lps = [member_logp(k, p, X, EX if uses_ex(k) else None) for k, p in members]
    return lse_mean(lps), lps


# ================================================================ the map file
def save_map(path, members, mu_x, sd_x, mu_ex, sd_ex, record):
    """members: [{"name", "kind", "seed", "e_star", "params": {..}}] in MEMBERS order; record: json-able dict
    (probe_used, games, players, sets_sha256, cv, ...). Weights float32; numpy-loadable without pickle."""
    names = [m["name"] for m in members]
    if names != member_names():
        raise ForwardError(f"members {names} are not the registered ens3fam order {member_names()}")
    arrays = {"format": np.array(MAP_FORMAT), "members": np.array(names),
              "kinds": np.array([m["kind"] for m in members]), "seeds": np.array([int(m["seed"]) for m in members]),
              "e_star": np.array([int(m["e_star"]) for m in members]), "scale": np.array(SCALE),
              "mu_x": np.asarray(mu_x, dtype=np.float64), "sd_x": np.asarray(sd_x, dtype=np.float64),
              "mu_ex": np.asarray(mu_ex, dtype=np.float64), "sd_ex": np.asarray(sd_ex, dtype=np.float64),
              "ex_names": np.array(ex_names()), "features": np.array(X_DIM), "ex_features": np.array(NEX),
              "record": np.array(json.dumps(record, sort_keys=True))}
    for i, m in enumerate(members):
        want = PARAMS[m["kind"]]
        if sorted(m["params"]) != sorted(want):
            raise ForwardError(f"{m['name']}: params {sorted(m['params'])} != {sorted(want)}")
        for k in want:
            arrays[f"m{i}_{k}"] = np.asarray(m["params"][k], dtype=np.float32)
    np.savez(path, **arrays)


def load_map(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if str(d.get("format")) != MAP_FORMAT:
        raise ForwardError(f"{path}: not a {MAP_FORMAT} file")
    names = [str(x) for x in d["members"]]
    kinds = [str(x) for x in d["kinds"]]
    if names != member_names() or kinds != [m["kind"] for m in MEMBERS]:
        raise ForwardError(f"{path}: members {names} / kinds {kinds} differ from ens3fam")
    if float(d["scale"]) != SCALE or int(d["features"]) != X_DIM or int(d["ex_features"]) != NEX:
        raise ForwardError(f"{path}: scale / feature sizes differ")
    if [str(x) for x in d["ex_names"]] != ex_names():
        raise ForwardError(f"{path}: extra feature names differ")
    members = []
    for i, (n, k) in enumerate(zip(names, kinds)):
        members.append({"name": n, "kind": k, "seed": int(d["seeds"][i]), "e_star": int(d["e_star"][i]),
                        "params": {p: d[f"m{i}_{p}"] for p in PARAMS[k]}})
    return {"members": members, "mu_x": d["mu_x"], "sd_x": d["sd_x"], "mu_ex": d["mu_ex"], "sd_ex": d["sd_ex"],
            "scale": float(d["scale"]), "record": json.loads(str(d["record"]))}


def map_logp(mp, Xraw, EXraw32, block=8192, with_members=False):
    """lp_P1TO_E of raw features (Xraw float64/float32 [N, 1540] with NaN opponent halves; EXraw32 the float32 raw
    extra features) with a loaded map -> lp [N, 10] (and the member lps)."""
    n = len(Xraw)
    out = np.zeros((n, L))
    mem = [np.zeros((n, L)) for _ in mp["members"]] if with_members else None
    ms = [(m["kind"], m["params"]) for m in mp["members"]]
    for s in range(0, n, block):
        X = standardise_x(np.asarray(Xraw[s:s + block], dtype=np.float64), mp["mu_x"], mp["sd_x"])
        EX = standardise_ex(EXraw32[s:s + block], mp["mu_ex"], mp["sd_ex"])
        lp, lps = ensemble_logp(ms, X, EX)
        out[s:s + block] = lp
        if with_members:
            for j, x in enumerate(lps):
                mem[j][s:s + block] = x
    return (out, mem) if with_members else out


def lr_logp(W, b, Xs):
    """multinomial LR log-probabilities (p1_fuse.logp_of = hs_lib.log_softmax_np(Xs @ W + b))."""
    return log_softmax(Xs @ W + b)
