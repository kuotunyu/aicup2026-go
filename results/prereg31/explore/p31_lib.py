"""prereg31 exploration (competition 2): shared parts (no torch at import time, no GPU, never the answers folder).

- stage 0a hand-over: the committed hd_stats.json (PASS, decided encoding, table SHA) and the HD table of p31_hd;
- the join of training / inner rows to the HD table by (rank, game_id), with the player / colour checks (seam A);
- the 2 per-game HD inputs (PLAN.md 1.4): x_B = flag * [target plays Black], x_W = flag * [target plays White],
  their mu / sigma from the role-A training rows, and the value of an even game;
- input_width: the context manager that sets p19_lib.D_IN to 1542 around the registered calls (PLAN.md 1.5);
  with HD off it reads, sets and restores nothing;
- the S31 model file (format "c10-prereg31 s31 model v1": the 10 parameter arrays + hd_mu, hd_sd, encoding);
- stage 0b: the control NLLs (prereg25 seeds 0-2, prereg29 seeds 3-9, read from the committed records) and the
  GO rule (PLAN.md 3) as a pure function;
- the HD-only naive Bayes of PLAN.md revision 1 (record-only);
- the ten no-HD models of Q29 and the judged Q29 reference (prereg29).

Nothing here reads data/training_v1.1 (only p31_hd.py does) and nothing here reads inner data.
"""

import contextlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P25 = os.path.join(REPO, "results", "prereg25", "explore")
P29 = os.path.join(REPO, "results", "prereg29", "explore")
for p in (os.path.join(REPO, "tools"), P25, P29, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p25_lib as L25  # noqa: E402  (guard, holdout_bit / SALT)
import p29_lib as L29  # noqa: E402  (frozen S25 files, registered settings, prereg25 references)
import p31_hd as HD  # noqa: E402  (table format, RANKS, flag_of; standard library + numpy only)

guard = L25.guard
read_json = L29.read_json
sha_file = L29.sha_file
loaded_repo_modules = L29.loaded_repo_modules
finite_params = L29.finite_params
registered_settings = L29.registered_settings

SEEDS = tuple(range(10))               # S31 seeds, same rule as S25 / S29 (PLAN.md 1.5)
SEAM_R_SEED = 0                        # retrained with HD off; must equal S25 seed 0 byte for byte
SUBSETS = (3, 5, 10)                   # record-only seed-count curve: seeds 0 .. k-1
ENCODINGS = ("Z", "C1")                # PLAN.md 1.3: Z = [HD == 1], C1 = [HD >= 1]
D_IN_HD = 1542                         # 1540 + x_B + x_W
N_HD = 2
GO_MEAN = -0.010                       # PLAN.md 3 condition 1: mean best held-out NLL, S31 - control <= this
GO_PAIRS = 8                           # PLAN.md 3 condition 2: S31 lower in at least this many of the 10 pairs
N_PROBE = 270000                       # PLAN.md 1.5 / 4.2 seam A
N_V = 74946
MODEL_FORMAT = "c10-prereg31 s31 model v1"
TRAIN_FORMAT = "c10-prereg31 explore train v1"
# PLAN.md 4.2 seam 1, as written in the registration (the committed files must agree with these)
REF_PLAN = {"Q09": 0.7114420614163589, "Q09_mono": 0.7145431684624712,
            "Q19": 0.7345843051712626, "Q19_mono": 0.7430172362615272,
            "Q25": 0.7456519871546352, "Q25_mono": 0.7482450052195216,
            "Q29": 0.7445162616450341, "Q29_mono": 0.7494986278839273}

DATA = os.path.join(REPO, "data", "prereg31")
MODELS = os.path.join(DATA, "models")
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
HD_TABLE = os.path.join(DATA, "hd_table.npz")
HD_STATS = os.path.join(HERE, "hd_stats.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
STAGE0B = os.path.join(HERE, "stage0b.json")
HD_NB = os.path.join(HERE, "hd_nb.json")
P29_TRAIN = os.path.join(P29, "train_record.json")
P29_EVAL = os.path.join(P29, "eval_report.json")
S29_MODELS = os.path.join(REPO, "data", "prereg29", "models")
V_SETS = os.path.join(REPO, "data", "prereg25", "v", "sets.json")
V_META = os.path.join(L29.V_PREP, "v_meta.npz")

_RANK_IDX = {r: i for i, r in enumerate(HD.RANKS)}
_PREFIX = np.array([f"{i}:".encode("ascii") for i in range(len(HD.RANKS))], dtype="S2")


class P31Error(RuntimeError):
    pass


def model_name(seed, hd_off_retrain=False):
    return f"s31_seed{seed}_hdoff_retrain.npz" if hd_off_retrain else f"s31_seed{seed}.npz"


def subset_seeds(k):
    if k not in SUBSETS:
        raise ValueError(k)
    return SEEDS[:k]


# ---------------------------------------------------------------- stage 0a hand-over
def stage0a(stats_path=None, table_path=None):
    """The committed stage 0a result and the table it describes. ok: PASS, sanity and guard hold, seam A of the
    build holds, a registered encoding was decided and the table on disk is the one hd_stats.json describes."""
    stats_path = stats_path or HD_STATS
    table_path = table_path or HD_TABLE
    st = read_json(stats_path)
    tab = st.get("table") or {}
    rec = {"hd_stats_sha256_lf": HD.sha_lf(guard(stats_path)), "stage0a": st.get("stage0a"),
           "stop_reasons": st.get("stop_reasons"), "encoding": st.get("encoding"),
           "sanity_ok": st.get("sanity_ok"), "guard_ok": st.get("guard_ok"),
           "seam_a_ok": (st.get("seam_a") or {}).get("ok"), "table_rows": tab.get("rows"),
           "table_unique_keys": tab.get("unique_keys"), "table_sha256_hd_stats": tab.get("sha256"),
           "table_sha256": sha_file(table_path)}
    rec["table_is_hd_stats"] = bool(rec["table_sha256"] == rec["table_sha256_hd_stats"])
    rec["ok"] = bool(rec["stage0a"] == "PASS" and rec["sanity_ok"] is True and rec["guard_ok"] is True
                     and rec["seam_a_ok"] is True and rec["encoding"] in ENCODINGS and rec["table_is_hd_stats"]
                     and rec["table_rows"] is not None and rec["table_rows"] == rec["table_unique_keys"])
    return rec


# ---------------------------------------------------------------- the HD table and the join (seam A)
def rank_index(rank):
    try:
        return np.array([_RANK_IDX[str(x)] for x in rank], dtype=np.int64)
    except KeyError as e:
        raise P31Error(f"unknown rank {e}") from None


def colour_index(col):
    """B -> 0, W -> 1 (the order of p31_hd.COLOURS and of p19_lib.COLOURS)."""
    col = np.asarray(col).astype(str)
    out = np.full(len(col), -1, dtype=np.int64)
    for i, c in enumerate(HD.COLOURS):
        out[col == c] = i
    if (out < 0).any():
        raise P31Error(f"colour that is not B / W: {sorted(set(col[out < 0].tolist()))[:5]}")
    return out


def _ascii(xs):
    try:
        return np.array([str(x).encode("ascii") for x in xs], dtype="S")
    except UnicodeEncodeError as e:
        raise P31Error(f"non-ASCII id: {e}") from None


def _compose(rank_idx, game_id):
    if len(rank_idx) == 0:
        return np.array([], dtype="S3")
    return np.char.add(_PREFIX[rank_idx], game_id)


class HdTable:
    """The HD table of p31_hd (one row per training game), searchable by (rank, game_id)."""

    def __init__(self, path=None):
        path = guard(path or HD_TABLE)
        self.sha256 = sha_file(path)
        t = HD.load_table(path)
        self.n = int(len(t["hd"]))
        key = _compose(t["rank"].astype(np.int64), t["game_id"])
        order = np.argsort(key, kind="stable")
        ks = key[order]
        if self.n > 1 and bool(np.any(ks[1:] == ks[:-1])):
            raise P31Error("HD table: (rank, game_id) is not unique")
        self._key, self._order, self._t = ks, order, t

    def join(self, rank, game_id, player_id, colour):
        """HD of every row by (rank, game_id) -> (hd int8, -1 where the game is not in the table; record).
        record.ok: every row found and its player_id and colour equal the table (PLAN.md 1.5, seam A)."""
        n = len(rank)
        if not (len(game_id) == len(player_id) == len(colour) == n):
            raise P31Error("join: columns of different lengths")
        hd = np.full(n, -1, dtype=np.int8)
        rec = {"rows": int(n), "joined": 0, "missing": 0, "player_mismatch": 0, "colour_mismatch": 0}
        if n and self.n:
            q = _compose(rank_index(rank), _ascii(game_id))
            pos = np.minimum(np.searchsorted(self._key, q), self.n - 1)
            found = self._key[pos] == q
            row = self._order[pos]
            hd[found] = self._t["hd"][row[found]]
            rec["player_mismatch"] = int(np.sum(found & (self._t["player_id"][row] != _ascii(player_id))))
            rec["colour_mismatch"] = int(np.sum(found & (self._t["color"][row].astype(np.int64)
                                                         != colour_index(colour))))
            rec["joined"] = int(found.sum())
        rec["missing"] = int(n - rec["joined"])
        rec["hd_groups"] = [int(np.sum(hd == 0)), int(np.sum(hd == 1)), int(np.sum(hd >= 2))]
        rec["ok"] = bool(rec["missing"] == 0 and rec["player_mismatch"] == 0 and rec["colour_mismatch"] == 0)
        return hd, rec


def _load_games(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)["probe"]["games"]


def probe_games(src, ref, players, sets_json):
    """rank / game_id / player_id / color of every probe row: probe_meta.npz (src, ref) -> the three sets.json
    game lists (the walk of p19_train.probe_colours). The player key of the row must be the one of the game."""
    games = {s: _load_games(p) for s, p in sets_json.items()}
    out = {"rank": [], "game_id": [], "player_id": [], "color": []}
    for i, (s, j, pk) in enumerate(zip(src, ref, players)):
        g = games[str(s)][int(j)]
        if f"{g['rank']}:{g['player_id']}" != str(pk):
            raise P31Error(f"probe row {i}: sets.json player differs from probe_meta")
        for k in out:
            out[k].append(str(g[k]))
    return out


def v_games(v_meta_path, v_sets_path, sets_sha256):
    """rank / game_id / player_id / color of every V row: v_meta.npz (ref) -> the game list of prereg25
    (data/prereg25/v/sets.json, raw SHA = prep_record.json sets_sha256). Player key and colour must be the row."""
    got = sha_file(v_sets_path)
    if got != sets_sha256:
        raise P31Error(f"{v_sets_path}: SHA {got} is not prereg25 prep_record.json sets_sha256")
    games = _load_games(v_sets_path)
    with np.load(guard(v_meta_path), allow_pickle=False) as z:
        ref, players, colour = z["ref"], z["players"], z["color"]
    out = {"rank": [], "game_id": [], "player_id": [], "color": []}
    for i, (j, pk, c) in enumerate(zip(ref, players, colour)):
        g = games[int(j)]
        if f"{g['rank']}:{g['player_id']}" != str(pk) or str(g["color"]) != str(c):
            raise P31Error(f"V row {i}: sets.json player / colour differs from v_meta")
        for k in out:
            out[k].append(str(g[k]))
    return out


def concat_games(a, b):
    return {k: list(a[k]) + list(b[k]) for k in ("rank", "game_id", "player_id", "color")}


def rows_agree(games, keys, levels, colours):
    """The walked games against the row arrays the training uses (p25_train.Rows): same player key, level and
    colour, row by row."""
    n = len(games["rank"])
    if not (len(keys) == len(levels) == len(colours) == n):
        return {"rows": int(n), "length_mismatch": True, "ok": False}
    key = np.array([f"{r}:{p}" for r, p in zip(games["rank"], games["player_id"])])
    rec = {"rows": int(n),
           "key_mismatch": int(np.sum(key != np.asarray(keys).astype(str))),
           "level_mismatch": int(np.sum(rank_index(games["rank"]) != np.asarray(levels, dtype=np.int64))),
           "colour_mismatch": int(np.sum(np.array(games["color"]) != np.asarray(colours).astype(str)))}
    rec["ok"] = bool(rec["key_mismatch"] == 0 and rec["level_mismatch"] == 0 and rec["colour_mismatch"] == 0)
    return rec


# ---------------------------------------------------------------- the 2 HD inputs (PLAN.md 1.4)
def flag_array(hd, encoding):
    """= p31_hd.flag_of per element: Z = [HD == 1], C1 = [HD >= 1]."""
    hd = np.asarray(hd)
    if encoding == "Z":
        return hd == 1
    if encoding == "C1":
        return hd >= 1
    raise ValueError(encoding)


def hd_columns(hd, col_idx, encoding):
    """[n, 2] float64: x_B = flag * [target plays Black], x_W = flag * [target plays White]."""
    hd = np.asarray(hd)
    ci = np.asarray(col_idx)
    if hd.shape != ci.shape or hd.ndim != 1:
        raise P31Error("hd_columns: hd and colour of different shapes")
    if (hd < 0).any():
        raise P31Error("hd_columns: a row without HD")
    if not np.isin(ci, (0, 1)).all():
        raise P31Error("hd_columns: colour index that is not 0 / 1")
    f = flag_array(hd, encoding)
    out = np.zeros((len(hd), N_HD), dtype=np.float64)
    out[:, 0] = f & (ci == 0)
    out[:, 1] = f & (ci == 1)
    return out


def hd_moments(cols, rows_a):
    """mu, sigma (2 numbers each) of the HD inputs over the role-A training rows; sigma must be > 0."""
    a = np.asarray(cols, dtype=np.float64)[np.asarray(rows_a, dtype=np.int64)]
    if a.ndim != 2 or a.shape[1] != N_HD or len(a) == 0:
        raise P31Error("hd_moments: need [n, 2] role-A rows")
    mu, sd = a.mean(axis=0), a.std(axis=0)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sd)) and np.all(sd > 0)):
        raise P31Error(f"hd_moments: sigma must be > 0 (mu {mu.tolist()}, sigma {sd.tolist()})")
    return mu, sd


def _check_moments(mu, sd):
    mu, sd = np.asarray(mu, dtype=np.float64), np.asarray(sd, dtype=np.float64)
    if mu.shape != (N_HD,) or sd.shape != (N_HD,) or not np.all(np.isfinite(mu)) \
            or not np.all(np.isfinite(sd)) or not np.all(sd > 0):
        raise P31Error("HD mu / sigma: need 2 finite numbers each, sigma > 0")
    return mu, sd


def hd_standardise(cols, mu, sd):
    mu, sd = _check_moments(mu, sd)
    return (np.asarray(cols, dtype=np.float64) - mu) / sd


def even_value(mu, sd):
    """The standardised inputs of an even game (flag 0): what HD inputs set to the even-game value means."""
    mu, sd = _check_moments(mu, sd)
    return (0.0 - mu) / sd


@contextlib.contextmanager
def input_width(K, width):
    """Set K.D_IN to `width` for the duration of the block (p19_train reads K.D_IN at call time).
    width None = HD off: nothing is read, set or restored, the code path is the one of S25."""
    if width is None:
        yield
        return
    old = K.D_IN
    K.D_IN = width
    try:
        yield
    finally:
        K.D_IN = old


# ---------------------------------------------------------------- the S31 model file
def save_model(path, params, seed, best_step, smoke, hd_mu, hd_sd, encoding, names):
    if encoding not in ENCODINGS:
        raise ValueError(encoding)
    mu, sd = _check_moments(hd_mu, hd_sd)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(MODEL_FORMAT), seed=np.array(seed), best_step=np.array(best_step),
             smoke=np.array(bool(smoke)), hd_mu=mu, hd_sd=sd, encoding=np.array(encoding),
             **{f"p_{k}": params[k] for k in names})
    os.replace(tmp, path)
    return sha_file(path)


def load_model(path, names):
    """(params, meta) of an S31 file; the first layer must be 1542 wide."""
    with np.load(guard(path), allow_pickle=False) as z:
        if str(z["format"]) != MODEL_FORMAT:
            raise SystemExit(f"{path}: not a {MODEL_FORMAT} file")
        par = {k: np.asarray(z[f"p_{k}"]) for k in names}
        meta = {"seed": int(z["seed"]), "best_step": int(z["best_step"]), "smoke": bool(z["smoke"]),
                "hd_mu": np.asarray(z["hd_mu"], dtype=np.float64),
                "hd_sd": np.asarray(z["hd_sd"], dtype=np.float64), "encoding": str(z["encoding"])}
    if par["W1"].ndim != 2 or par["W1"].shape[1] != D_IN_HD:
        raise SystemExit(f"{path}: first layer is not {D_IN_HD} wide")
    if meta["encoding"] not in ENCODINGS:
        raise SystemExit(f"{path}: encoding {meta['encoding']!r}")
    _check_moments(meta["hd_mu"], meta["hd_sd"])
    return par, meta


# ---------------------------------------------------------------- stage 0b (PLAN.md 3)
def train_record_checks(tr):
    """What stage 0b (and seam M of the eval) need from train_record.json -> {name: bool}."""
    seeds = tr.get("seeds") or {}
    return {"format": tr.get("format") == TRAIN_FORMAT, "not_smoke": tr.get("smoke") is False,
            "code_clean": (tr.get("code") or {}).get("dirty") == "",
            "stage0a_ok": (tr.get("stage0a") or {}).get("ok") is True,
            "seam_a_ok": (tr.get("seam_a") or {}).get("ok") is True,
            "seam_r_ok": (tr.get("seam_r") or {}).get("ok") is True
            and (tr.get("seam_r") or {}).get("applicable") is True,
            "seeds_0_to_9": sorted(seeds) == sorted(str(s) for s in SEEDS),
            "port_checks_ok": bool(seeds) and all((v.get("port_check") or {}).get("ok") is True
                                                  for v in seeds.values()),
            "same_players_as_prereg25": (tr.get("players") or {}).get("same_as_prereg25") is True}


def control_nll(p25_train=None, p29_train=None):
    """Best held-out NLL of the ten no-HD models of the same seeds, from the committed records:
    seeds 0-2 prereg25 train_record.json, seeds 3-9 prereg29 train_record.json -> ({seed: nll}, sources)."""
    p25_train = p25_train or L29.P25_TRAIN
    p29_train = p29_train or P29_TRAIN
    tr25, tr29 = read_json(p25_train), read_json(p29_train)
    if tr25.get("smoke") or tr29.get("smoke"):
        raise SystemExit("control records: a smoke record")
    if not (tr29.get("seam_r") or {}).get("ok") or (tr29.get("players") or {}).get("same_as_prereg25") is not True:
        raise SystemExit("prereg29 train_record.json: seam R / same players as prereg25 not recorded")
    out = {}
    for s in L29.S25_SEEDS:
        out[s] = float(tr25["seeds"][str(s)]["best_holdout_nll"])
    for s in L29.NEW_SEEDS:
        out[s] = float(tr29["seeds"][str(s)]["best_holdout_nll"])
    if tuple(sorted(out)) != SEEDS:
        raise SystemExit("control records: not seeds 0-9")
    src = {"prereg25_train_record_sha256_lf": HD.sha_lf(guard(p25_train)), "prereg25_seeds": list(L29.S25_SEEDS),
           "prereg29_train_record_sha256_lf": HD.sha_lf(guard(p29_train)), "prereg29_seeds": list(L29.NEW_SEEDS),
           "holdout_questions_prereg25": tr25["players"]["holdout_questions"],
           "holdout_questions_prereg29": tr29["players"]["holdout_questions"]}
    return out, src


def stage0b_decision(s31, control):
    """PLAN.md 3: GO iff (1) mean over the 10 seeds of the best held-out NLL, S31 - control <= -0.010 and
    (2) S31 is lower in at least 8 of the 10 same-seed pairs. No tolerance."""
    if tuple(sorted(s31)) != SEEDS or tuple(sorted(control)) != SEEDS:
        raise P31Error("stage 0b needs seeds 0-9 on both sides")
    a = [float(s31[s]) for s in SEEDS]
    b = [float(control[s]) for s in SEEDS]
    if not all(math.isfinite(x) for x in a + b):
        raise P31Error("stage 0b: a non-finite NLL")
    mean_a, mean_b = math.fsum(a) / len(a), math.fsum(b) / len(b)
    diff = mean_a - mean_b
    lower = int(sum(x < y for x, y in zip(a, b)))
    c1, c2 = bool(diff <= GO_MEAN), bool(lower >= GO_PAIRS)
    return {"seeds": list(SEEDS), "s31_best_holdout_nll": a, "control_best_holdout_nll": b,
            "pair_diff_s31_minus_control": [x - y for x, y in zip(a, b)],
            "mean_s31": mean_a, "mean_control": mean_b, "mean_diff_s31_minus_control": diff,
            "threshold_mean_diff": GO_MEAN, "condition_mean": c1,
            "pairs_s31_lower": lower, "threshold_pairs": GO_PAIRS, "condition_pairs": c2,
            "decision": "GO" if (c1 and c2) else "NO-GO"}


# ---------------------------------------------------------------- HD-only naive Bayes (PLAN.md revision 1 item 1)
def nb_table(flag, level, col_idx, rows_a, n_levels=None):
    """P(flag = 1 | level r, colour c) from the role-A rows with add-1 smoothing -> (P [L, 2], n, n1)."""
    n_levels = n_levels or len(HD.RANKS)
    rows_a = np.asarray(rows_a, dtype=np.int64)
    lv = np.asarray(level, dtype=np.int64)[rows_a]
    ci = np.asarray(col_idx, dtype=np.int64)[rows_a]
    z = np.asarray(flag).astype(np.int64)[rows_a]
    n = np.zeros((n_levels, 2), dtype=np.int64)
    n1 = np.zeros((n_levels, 2), dtype=np.int64)
    np.add.at(n, (lv, ci), 1)
    np.add.at(n1, (lv, ci), z)
    return (n1 + 1.0) / (n + 2.0), n, n1


def nb_nll(P, hold, flag, col_idx):
    """Mean NLL of the true level on the held-out questions under posterior(r) ~ prod_k P(z_k | r, c_k), uniform
    prior over levels. hold: {n: (idx [Q, n], y [Q])} as p19_lib.holdout_set."""
    P = np.asarray(P, dtype=np.float64)
    log1, log0 = np.log(P).T, np.log1p(-P).T              # [2, L]
    flag = np.asarray(flag).astype(bool)
    ci = np.asarray(col_idx, dtype=np.int64)
    per_n, tot, cnt = {}, [], 0
    for n, (idx, y) in hold.items():
        if len(y) == 0:
            continue
        z = flag[idx][:, :, None]                         # [Q, n, 1]
        s = np.where(z, log1[ci[idx]], log0[ci[idx]]).sum(axis=1)       # [Q, L]
        m = s.max(axis=1, keepdims=True)
        lp = s - (m + np.log(np.exp(s - m).sum(axis=1, keepdims=True)))
        q = -lp[np.arange(len(y)), np.asarray(y, dtype=np.int64)]
        per_n[str(n)] = {"questions": int(len(y)), "mean_nll": float(q.mean())}
        tot.extend(q.tolist())
        cnt += len(y)
    if cnt == 0:
        raise P31Error("naive Bayes: no held-out question")
    mean = math.fsum(tot) / cnt
    uni = math.log(P.shape[0])
    return {"questions": int(cnt), "mean_nll": mean, "uniform_nll": uni, "mean_nll_minus_uniform": mean - uni,
            "per_n": per_n}


# ---------------------------------------------------------------- Q29 (prereg29): reference and the ten models
def refs29():
    e = read_json(P29_EVAL)
    if e.get("smoke") or e["candidate"]["name"] != "Q29" or list(e["candidate"]["seeds"]) != list(SEEDS):
        raise SystemExit("results/prereg29/explore/eval_report.json is not the judged Q29 report")
    return {"Q29": e["candidate"]["inner_rank_balanced"], "Q29_mono": e["candidate"]["single_colour_rank_balanced"]}


def s29_models(K19, load_npz_model):
    """The ten no-HD models of Q29 in seed order: the frozen S25 files (seeds 0-2, SHA = prereg25
    train_record.json) and data/prereg29/models/s29_seed{3..9}.npz (SHA = prereg29 train_record.json) (seam F)."""
    pars, seam25 = L29.s25_models(K19, load_npz_model)
    tr = read_json(P29_TRAIN)
    seam = {"s25": seam25, "models": {}, "ok": bool(seam25["ok"])}
    for s in L29.NEW_SEEDS:
        path = os.path.join(S29_MODELS, L29.model_name(s))
        sha = sha_file(path)
        par, seed, best_step, msmoke = load_npz_model(path, L29.S25_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] == tr["seeds"][str(s)]["model_sha256"] and seed == s
                  and not msmoke and best_step == tr["seeds"][str(s)]["best_step"] and finite_params(par))
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam
