"""prereg57 (PLAN.md "Method"): closed-set adaptation of a base model to one pool's candidates; per-game vectors.

    GPUPY results/prereg57/explore/p57_ft.py adapt --base S0|c49 --pool-kind D|E|official --pool <index | public|private>
          --share F --lr F --steps N --ckpt-at 1000,2000,4000,6000 --out DIR [--device cuda|cpu] [--workers N]
          [--resume | --auto-resume] [--smoke]
    GPUPY results/prereg57/explore/p57_ft.py embed --base ... --pool-kind ... --pool ... --ckpt FILE --weights ema|raw
          --out NPZ [--device cuda|cpu] [--workers N] [--limit N] [--compare-base]

One model per pool, pools never mixed. Nothing of prereg36 / 42 / 43 / 49 / 50 is edited: the trunk, the head, the
loss (p42_train.loss_of), the base sampler (p42_lib.GameBatches with the base's own class set), the EMA (p42_lib.Ema)
and the per-game vectors (p42_embed.GameSet / game_vectors / load_net, tta 1) are imported.

adapt (PLAN.md Method 1, 2):
  * pool: D / E = pool <index> of p36_eval.load_pools (the candidate / question rows the eval code scores), games from
    the prereg36 move cache through p36_embed.store_games; official = the CSVs through p36_lib.pool_items /
    p36_embed.pool_games. Questions are never used for training; a simulation pool's owners are never read.
  * head = the base's class weights + one row per candidate = the L2-normalised mean of the base's game vectors (raw
    weights, tta 1, the existing files) of that candidate (= p36_eval.Emb.group_vecs).
  * network = the base's raw weights (the weights v2-6 / v2-7 use); a new EMA (0.999) starts from them.
  * batch s = (1 - share) x 96 segments of batch (base step + s) of the base's sampler (its stream continues where the
    base's training ended; the first segments of that batch) + share x 96 segments of the pool's candidates drawn from
    numpy default_rng([0, s, 57]): candidate uniform, one of its games uniform, <= 24 of the candidate's own moves,
    one board symmetry - the base sampler's recipe. Label of a candidate = number of base classes + its index.
  * AdamW (weight decay 1e-4), lr = `lr` x min(1, (s + 1) / 200), fp16 autocast on the GPU, seed 0, new optimiser.
  * checkpoints <out>/ckpt/step_NNNNNN.pt at the listed steps (net = raw, ema, head, optimiser ...); --resume picks
    the highest step number. STOP file (<out>/STOP or data/prereg57/STOP) or low RAM -> save, exit code 4.
  * refuses to start on the GPU when nvidia-smi lists another python compute process (p36_train's check).
  * pool kinds E and official need the committed results/prereg57/explore/dev.json (gate passed) and its chosen
    share / lr / n: no E vector of this method before the choice is committed (PLAN.md Development).
embed: vectors of exactly that pool's games - D / E: candidates' games then questions' games (store rows in
  `store_row`); official: p36_lib.pool_items order (= pool_<name>.npz of the base) - in p42_embed's layout.

Nothing here opens the validation-answer folder or runs tools/score.py; no score of any kind is computed here.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(REPO, "results", r, "explore") for r in ("prereg36", "prereg42", "prereg43", "prereg49",
                                                                 "prereg50")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p42_lib as V  # noqa: E402

L = V.L
DATA = os.path.join(REPO, "data", "prereg57")
STOP_FILE = os.path.join(DATA, "STOP")
PLAN = os.path.join(HERE, "PLAN.md")
DEV_JSON = os.path.join(HERE, "dev.json")
SRC_OFFICIAL = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
VARIANT = "V2"
SEED = 0
BATCH_SEGMENTS = L.BATCH_GAMES          # 96
MOVES = L.SEG_MOVES                     # 24
WARMUP = 200
POOL_STREAM = 57
GAME_TRIES = 1000
STOP_EVERY = 100
LOG_EVERY = 100
SMOKE_CANDS, SMOKE_SEGMENTS, SMOKE_MOVES, SMOKE_STEPS = 4, 8, 6, 3
# PLAN.md Development / Judgement
GRID = (("0.25", "2e-4"), ("0.5", "2e-4"), ("0.5", "5e-4"))      # (share, lr) as written on the command line
DEV_STEPS = (1000, 2000, 4000, 6000)
DEV_POOLS = (0, 1)
JUDGE_POOLS = tuple(range(10))
BASES = {
    "S0": {"ckpt": os.path.join(REPO, "data", "prereg43", "S0", "ckpt", "step_320000.pt"),
           "sha256": "49c1c8f10c927922ca9f3ab3f7d2aaa37a24ba852d6acf77d3042ce1fd495939",
           "vec_dir": os.path.join(REPO, "data", "prereg43", "emb_S0", "V2_step_320000_raw_tta1"),
           "players": "T36", "n_classes": 3685},
    "c49": {"ckpt": os.path.join(REPO, "data", "prereg49", "S0", "ckpt", "step_320000.pt"),
            "sha256": "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12",
            "vec_dir": os.path.join(REPO, "data", "prereg49", "emb", "V2_step_320000_raw_tta1"),
            "players": "T49", "n_classes": 5771},
}
ALLOWED = (("S0", "D"), ("c49", "E"), ("c49", "official"))      # PLAN.md: dev, judgement, apply
CODE_FILES = (os.path.join(HERE, "p57_ft.py"), os.path.join(V.HERE, "p42_lib.py"),
              os.path.join(V.HERE, "p42_train.py"), os.path.join(V.HERE, "p42_embed.py"),
              os.path.join(V.P36, "p36_lib.py"))


# ================================================================ paths (shared with p57_eval.py and the chains)
def run_tag(share, lr):
    """Folder name of a dev run; share / lr are the command-line strings of GRID."""
    return "sh%s_lr%s" % (share, lr)


def dev_dir(tag, index, data=DATA):
    return os.path.join(data, "dev", tag, "D%d" % int(index))


def judge_dir(index, data=DATA):
    return os.path.join(data, "judge", "E%d" % int(index))


def apply_dir(pool, data=DATA):
    return os.path.join(data, "apply", pool)


def ckpt_file(run_dir, step):
    return os.path.join(run_dir, "ckpt", "step_%06d.pt" % int(step))


def emb_file(run_dir, step, weights="ema"):
    return os.path.join(run_dir, "emb", "step_%d_%s.npz" % (int(step), weights))


def last_ckpt(d):
    """The checkpoint with the highest step number (p50_run's rule, not the file-name order)."""
    import p50_run
    return p50_run.last_checkpoint_by_step(d)


def code_sha():
    out = {}
    for f in CODE_FILES:
        with open(f, "rb") as fh:
            out[os.path.basename(f)] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def _rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


# ================================================================ the pool
def pool_rows(cand_rows, q_rows):
    """Game order of a simulation pool's vectors: every candidate's games (candidate order), then every question's."""
    return np.asarray([r for g in cand_rows for r in g] + [r for g in q_rows for r in g], np.int64)


def sim_pool(kind, index):
    """(seed, cand_rows, q_rows) of pool <index> of the D / E pools the eval code uses (feature-store rows)."""
    import p36_eval as EV
    seeds = {"D": EV.D_SEEDS, "E": EV.E_SEEDS}[kind]
    index = int(index)
    if not 0 <= index < len(seeds):
        raise SystemExit("%s pool index must be in 0..%d" % (kind, len(seeds) - 1))
    p = EV.load_pools(EV.TERMS_DIR[kind], [seeds[index]])[0]
    return int(seeds[index]), p["cand_rows"], p["q_rows"]


_STORE_GAMES = {}


def store_games(kind):
    """p36_embed.store_games of the D / E players on the prereg36 cache, addressable by feature-store row."""
    if kind not in _STORE_GAMES:
        import p36_embed
        cache = L.Cache()
        E, D, _T, _per = L.splits(cache)
        rows, codes, start, end, is_white = p36_embed.store_games({"D": D, "E": E}[kind], cache)
        with np.load(os.path.join(L.FEATS, "meta.npz"), allow_pickle=False) as z:
            pid = z["player_id"][rows]
        _STORE_GAMES[kind] = {"pos": {int(r): i for i, r in enumerate(rows.tolist())}, "codes": codes,
                              "start": np.asarray(start, np.int64), "end": np.asarray(end, np.int64),
                              "is_white": np.asarray(is_white, bool), "pid": pid}
    return _STORE_GAMES[kind]


def games_sha(codes, start, end, is_white):
    c = np.load(codes, mmap_mode="r") if isinstance(codes, str) else codes
    h = hashlib.sha256()
    for g in range(len(is_white)):
        h.update(b"W" if is_white[g] else b"B")
        h.update(np.asarray(c[start[g]:end[g]], "<u2").tobytes())
        h.update(b"|")
    return h.hexdigest()


def identity_of(kind, name, cand_groups, q_groups, store_row, cand_names, g_sha):
    head = {"kind": kind, "name": name, "cand_groups": cand_groups, "q_groups": q_groups,
            "store_row": None if store_row is None else [int(r) for r in store_row], "cand_names": cand_names,
            "games_sha256": g_sha}
    return hashlib.sha256(json.dumps(head, sort_keys=True).encode("utf-8")).hexdigest()


def build_pool(kind, name, src=SRC_OFFICIAL):
    """The pool's games (codes, start, end, is_white per game, in the vectors' order), the candidates' and the
    questions' games as lists of positions in that order, candidate names, store rows (D / E) and an identity
    hash of all of it. No label of a question is read."""
    if kind in ("D", "E"):
        seed, cand_rows, q_rows = sim_pool(kind, name)
        rows = pool_rows(cand_rows, q_rows)
        if len(set(rows.tolist())) != len(rows):
            raise SystemExit("a game appears twice in pool %s %s" % (kind, name))
        sg = store_games(kind)
        j = np.asarray([sg["pos"][int(r)] for r in rows], np.int64)
        local = {int(r): i for i, r in enumerate(rows.tolist())}
        cand_groups = [[local[int(r)] for r in g] for g in cand_rows]
        q_groups = [[local[int(r)] for r in g] for g in q_rows]
        pid = sg["pid"][j]
        cand_names = []
        for g in cand_groups:
            who = {str(x) for x in pid[g].tolist()}
            if len(who) != 1:
                raise SystemExit("a candidate of pool %s %s has games of %d players" % (kind, name, len(who)))
            cand_names.append(who.pop())
        if len(set(cand_names)) != len(cand_names):
            raise SystemExit("a player is two candidates in pool %s %s" % (kind, name))
        pool = {"kind": kind, "name": int(name), "seed": seed, "codes": sg["codes"], "start": sg["start"][j],
                "end": sg["end"][j], "is_white": sg["is_white"][j], "store_row": rows, "cand_groups": cand_groups,
                "q_groups": q_groups, "cand_names": cand_names,
                "players": sorted({str(x) for x in pid.tolist()})}
    elif kind == "official":
        if name not in ("public", "private"):
            raise SystemExit("--pool must be public or private for --pool-kind official")
        import p35_apply as pa
        import p36_embed
        cands, qs = pa.read_pool(src, name)
        items, cand_ids, cand_rows, q_rows, _qids = L.pool_items(cands, qs)
        if not all(c.startswith(pa.PREFIX[name] + "p_") for c in cand_ids):
            raise SystemExit("candidate id prefix")
        codes, start, end, is_white = p36_embed.pool_games(src, name)
        if len(is_white) != len(items):
            raise SystemExit("pool_games and pool_items disagree")
        pool = {"kind": kind, "name": name, "seed": None, "codes": np.asarray(codes), "start": np.asarray(start),
                "end": np.asarray(end), "is_white": np.asarray(is_white, bool), "store_row": None,
                "cand_groups": [list(map(int, g)) for g in cand_rows], "q_groups": [list(map(int, g)) for g in q_rows],
                "cand_names": list(cand_ids), "players": None,
                "files": {f: L.sha256_file(os.path.join(src, f)) for f in
                          ("player_identification_candidates2_%s.csv" % name,
                           "player_identification_test2_%s.csv" % name)}}
    else:
        raise SystemExit("--pool-kind must be D, E or official")
    pool["games_sha256"] = games_sha(pool["codes"], pool["start"], pool["end"], pool["is_white"])
    pool["identity"] = identity_of(kind, pool["name"], pool["cand_groups"], pool["q_groups"], pool["store_row"],
                                   pool["cand_names"], pool["games_sha256"])
    return pool


def pool_stats(pool):
    cg = [len(g) for g in pool["cand_groups"]]
    return {"kind": pool["kind"], "name": pool["name"], "seed": pool["seed"], "identity": pool["identity"],
            "games": int(len(pool["is_white"])), "candidates": len(cg), "candidate_games": int(sum(cg)),
            "candidate_games_min_max": [int(min(cg)), int(max(cg))], "questions": len(pool["q_groups"]),
            "question_games": int(sum(len(g) for g in pool["q_groups"])),
            "candidate_games_white": int(sum(int(pool["is_white"][j]) for g in pool["cand_groups"] for j in g)),
            "moves": int((pool["end"] - pool["start"]).sum()), "files": pool.get("files")}


# ================================================================ the base
def base_vec_file(base, kind, name):
    fn = "store_%s.npz" % kind if kind in ("D", "E") else "pool_%s.npz" % name
    return os.path.join(BASES[base]["vec_dir"], fn)


def base_vectors(base, pool):
    """The base's game vectors (raw weights, tta 1; the files v2-6 / v2-7 score) of the pool's games, in order."""
    f = base_vec_file(base, pool["kind"], pool["name"])
    if not os.path.exists(f):
        raise SystemExit("missing %s" % _rel(f))
    with np.load(f, allow_pickle=False) as z:
        if str(z["ckpt_sha256"]) != BASES[base]["sha256"]:
            raise SystemExit("%s was not made from %s" % (_rel(f), base))
        cfg = json.loads(str(z["embed_cfg"])) if "embed_cfg" in z.files else {}
        if cfg.get("weights") != "raw" or cfg.get("tta") != 1:
            raise SystemExit("%s is not raw / tta 1" % _rel(f))
        if pool["store_row"] is not None:
            pos = {int(r): i for i, r in enumerate(z["store_row"].tolist())}
            idx = np.asarray([pos[int(r)] for r in pool["store_row"]], np.int64)
            return np.asarray(z["emb"][idx], np.float32), np.asarray(z["count"][idx])
        if len(z["emb"]) != len(pool["is_white"]):
            raise SystemExit("%s does not hold the pool's games" % _rel(f))
        return np.asarray(z["emb"], np.float32), np.asarray(z["count"])


def head_init_rows(emb, count, groups):
    """PLAN.md Method 1: per candidate the L2-normalised mean of its game vectors (games with a sample) - the
    arithmetic of p36_eval.Emb.group_vecs. Returns (float64 rows, has); has False -> a zero row."""
    emb = np.asarray(emb)
    out = np.zeros((len(groups), emb.shape[1]), np.float64)
    has = np.zeros(len(groups), bool)
    for i, g in enumerate(groups):
        idx = [int(j) for j in g if count[int(j)] > 0]
        if idx:
            v = np.asarray(emb[idx], np.float64).mean(0)
            nv = np.linalg.norm(v)
            if nv > 0:
                out[i] = v / nv
                has[i] = True
    return out, has


def extended_head_weight(base_weight, rows):
    """The base's class weights followed by one row per candidate (float32)."""
    base_weight, rows = np.asarray(base_weight, np.float32), np.asarray(rows, np.float32)
    if base_weight.ndim != 2 or rows.ndim != 2 or base_weight.shape[1] != rows.shape[1]:
        raise ValueError("head shapes %s %s" % (base_weight.shape, rows.shape))
    return np.concatenate([base_weight, rows])


def base_index(base):
    """(class ids, games grouped by class, offsets, split hashes, cache) of the base's own training sampler."""
    if BASES[base]["players"] == "T36":
        cache = L.Cache()
        T, order, t_off, ids = V.t_index(cache)
    else:
        import p49_lib as P
        import p49_run as R49
        cache = P.Cache49()
        T, order, t_off, ids = R49.t_index49(cache)
    return T, order, t_off, ids, cache


# ================================================================ the mixed sampler (PLAN.md Method 2)
def split_segments(share, batch_segments=BATCH_SEGMENTS):
    """(pool segments, base segments) of a batch; share x batch must be a whole number in 1..batch."""
    x = float(share) * batch_segments
    n_pool = int(round(x))
    if abs(x - n_pool) > 1e-9 or not 1 <= n_pool <= batch_segments:
        raise SystemExit("share %r: share x %d segments must be a whole number in 1..%d" % (share, batch_segments,
                                                                                         batch_segments))
    return n_pool, batch_segments - n_pool


class MixedBatches:
    """Item s = one training batch (x uint8 (P, 17, 19, 19), seg, y, t - p42_lib.GameBatches' layout): first the
    n_base segments of batch (base_offset + s) of the base's sampler (`base`, a GameBatches with batch_games =
    n_base; labels = base classes), then n_pool segments of the pool's candidates (labels n_base_classes + c)."""

    def __init__(self, base, base_offset, n_base_classes, codes, start, end, is_white, cand_groups, n_pool,
                 seed=SEED, moves=MOVES):
        self.base, self.base_offset, self.n_base_classes = base, int(base_offset), int(n_base_classes)
        self.codes, self.start, self.end, self.is_white = codes, np.asarray(start), np.asarray(end), \
            np.asarray(is_white, bool)
        self.cand_games = np.asarray([j for g in cand_groups for j in g], np.int64)
        self.cand_off = np.concatenate([[0], np.cumsum([len(g) for g in cand_groups])]).astype(np.int64)
        if (np.diff(self.cand_off) < 1).any():
            raise ValueError("a candidate without a game")
        self.n_pool, self.seed, self.moves = int(n_pool), seed, int(moves)
        self.n_base = 0 if base is None else int(base.batch_games)

    def __len__(self):
        return 1 << 62

    def _codes(self):
        if isinstance(self.codes, str):
            self.codes = np.load(self.codes, mmap_mode="r")
        return self.codes

    def draw_pool(self, s):
        """[(candidate, game (position in the pool's order), planes, move indices)] x n_pool for batch s: candidate
        uniform, one of its games uniform, <= moves of the candidate's own moves, one symmetry. A game without a
        sample (illegal move, too short) is drawn again for the SAME candidate, so the candidates stay uniform
        (PLAN.md Method 2); every candidate has a game with a sample (it has a base vector)."""
        codes_all = self._codes()
        rng = np.random.default_rng([self.seed, int(s), POOL_STREAM])
        n_c = len(self.cand_off) - 1
        out = []
        while len(out) < self.n_pool:
            c = int(rng.integers(n_c))
            lo, hi = int(self.cand_off[c]), int(self.cand_off[c + 1])
            for _try in range(GAME_TRIES):
                g = int(self.cand_games[lo + int(rng.integers(hi - lo))])
                codes = np.asarray(codes_all[self.start[g]:self.end[g]])
                try:
                    x, t = L.planes_for_codes(codes, "w" if self.is_white[g] else "b", self.moves, rng)
                except ValueError:
                    continue
                if len(x):
                    out.append((c, g, L.sym(x, int(rng.integers(8))), np.asarray(t, np.int64)))
                    break
            else:
                raise RuntimeError("candidate %d gives no sample" % c)
        return out

    def batch(self, s):
        xs, segs, ys, ts = [], [], [], []
        n = 0
        if self.base is not None:
            b = self.base.batch(self.base_offset + int(s))
            xs.append(b["x"])
            segs.append(b["seg"])
            ys.append(b["y"])
            ts.append(b["t"])
            n = len(b["y"])
        for c, _g, x, t in self.draw_pool(s):
            xs.append(x)
            segs.append(np.full(len(x), n, np.int64))
            ys.append(np.asarray([self.n_base_classes + c], np.int64))
            ts.append(t)
            n += 1
        return {"x": np.ascontiguousarray(np.concatenate(xs)), "seg": np.concatenate(segs),
                "y": np.concatenate(ys), "t": np.concatenate(ts)}

    def __getitem__(self, s):
        import torch
        return {k: torch.from_numpy(v) for k, v in self.batch(s).items()}


# ================================================================ guards
def require_dev_choice(kind, share=None, lr=None, step=None, dev_json=DEV_JSON):
    """E / official: the committed dev.json with a passed gate and exactly its chosen share / lr / n."""
    if kind == "D":
        return None
    if not os.path.exists(dev_json) or not L.git_committed(dev_json):
        raise SystemExit("pool kind %s needs the committed %s (PLAN.md: the choice is committed before any E vector "
                         "of this method exists)" % (kind, _rel(dev_json)))
    with open(dev_json, encoding="utf-8") as f:
        dev = json.load(f)
    if not dev["gate"]["pass"]:
        raise SystemExit("dev.json: the gate did not pass - E and the official pools are not touched")
    ch = dev["chosen"]
    for name, got, want in (("share", share, float(ch["share"])), ("lr", lr, float(ch["lr"])),
                            ("step", step, int(ch["n"]))):
        if got is not None and got != want:
            raise SystemExit("%s %r is not dev.json's choice %r" % (name, got, want))
    return ch


def stop_requested(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def read_log(path, upto):
    """The loss curve of train_log.jsonl: one row per logged step <= upto (a later line replaces an earlier one)."""
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r["step"] <= upto:
                        rows[r["step"]] = r
    return [rows[k] for k in sorted(rows)]


# ================================================================ adapt
def cmd_adapt(a):
    import p42_train as T42
    if (a.base, a.pool_kind) not in ALLOWED:
        raise SystemExit("PLAN.md allows %s only" % " / ".join("%s on %s" % x for x in ALLOWED))
    cuda = a.device == "cuda"
    if not cuda and not a.smoke:
        raise SystemExit("--device cpu is for --smoke only")
    if a.smoke:
        a.steps = SMOKE_STEPS if a.steps is None else a.steps
        if a.steps > SMOKE_STEPS:
            raise SystemExit("--smoke runs at most %d steps" % SMOKE_STEPS)
        a.ckpt_at = a.ckpt_at or "%d" % a.steps
    if a.steps is None or a.steps < 1:
        raise SystemExit("--steps is required")
    ckpt_at = sorted({int(x) for x in (a.ckpt_at or "").split(",") if x} | {a.steps})
    if ckpt_at[0] < 1 or ckpt_at[-1] > a.steps:
        raise SystemExit("--ckpt-at must lie in 1..--steps")
    share, lr, end = float(a.share), float(a.lr), int(a.steps)
    if not lr > 0:
        raise SystemExit("--lr must be positive")
    segments, moves = (SMOKE_SEGMENTS, SMOKE_MOVES) if a.smoke else (BATCH_SEGMENTS, MOVES)
    n_pool, n_base = split_segments(share, segments)
    require_dev_choice(a.pool_kind, share, lr, None if a.smoke else end)
    out = os.path.abspath(a.out)
    ckpt_dir, log_path, rec_path = os.path.join(out, "ckpt"), os.path.join(out, "train_log.jsonl"), \
        os.path.join(out, "adapt_record.json")
    stops = (os.path.join(out, "STOP"), a.stop_file)
    hit = stop_requested(stops)
    if hit:
        print("STOP file present: %s" % hit, flush=True)
        return 4
    if cuda:
        others = T42.other_python_on_gpu()
        if others is None:
            raise SystemExit("nvidia-smi could not be queried; not starting")
        if others:
            raise SystemExit("another python compute process is on the GPU; not starting:\n  " + "\n  ".join(others))

    import torch
    from torch.utils.data import DataLoader

    dev = torch.device(a.device)
    torch.manual_seed(SEED)
    if not cuda:
        torch.set_num_threads(2)
    base = BASES[a.base]
    base_sha = L.sha256_file(base["ckpt"])
    if base_sha != base["sha256"]:
        raise SystemExit("%s is not the %s checkpoint" % (_rel(base["ckpt"]), a.base))
    ck0 = torch.load(base["ckpt"], map_location="cpu", weights_only=False)
    bcfg = ck0["config"]
    spec = V.spec_of(VARIANT)
    want = {"variant": VARIANT, "spec": spec, "blocks": L.BLOCKS, "channels": L.CHANNELS, "emb_dim": L.EMB_DIM,
            "scale": L.SCALE, "weight_decay": V.WD, "ema_decay": V.EMA_DECAY, "n_classes": base["n_classes"],
            "game": {"batch_games": BATCH_SEGMENTS, "moves": MOVES}, "seed": SEED}
    bad = {k: (bcfg.get(k), v) for k, v in want.items() if bcfg.get(k) != v}
    if bad or list(bcfg["gpse"]["gpool_blocks"]) != list(V.GPOOL_BLOCKS):
        raise SystemExit("the base checkpoint's recipe is not the one this code builds: %s" % bad)
    base_step = int(ck0["step"])

    pool = build_pool(a.pool_kind, a.pool, a.src)
    groups = pool["cand_groups"][:SMOKE_CANDS] if a.smoke else pool["cand_groups"]
    emb0, count0 = base_vectors(a.base, pool)
    rows, has = head_init_rows(emb0, count0, groups)
    del emb0
    # PLAN.md revision 1: the new rows get the median length of the base's class rows (same step size under AdamW)
    rows = rows * float(np.median(np.linalg.norm(ck0["head"]["weight"].detach().cpu().numpy().astype(np.float64), axis=1)))
    if not has.all():
        raise SystemExit("%d candidates have no base vector: PLAN.md gives no initial row for them"
                         % int((~has).sum()))
    T, order, t_off, ids, cache = base_index(a.base)
    if len(T) != base["n_classes"] or ids["T"] != bcfg["T_sha256_sorted_ids"] or \
            cache.meta["codes_sha256"] != bcfg["cache_codes_sha256"]:
        raise SystemExit("the base sampler's classes / cache are not the base checkpoint's")
    if pool["players"] is not None and set(pool["players"]) & set(T):
        raise SystemExit("the pool's players are among the base's training players")
    gb = None
    if n_base:
        gb = V.GameBatches(cache.dir, cache.off, cache.color == "W", order, t_off, SEED, batch_games=n_base,
                           moves=moves)
    ds = MixedBatches(gb, base_step, len(T), pool["codes"], pool["start"], pool["end"], pool["is_white"], groups,
                      n_pool, SEED, moves)
    n_cand = len(groups)
    config = {"variant": VARIANT, "spec": spec,
              "p57": {"base": a.base, "base_ckpt_sha256": base_sha, "base_step": base_step,
                      "pool_kind": pool["kind"], "pool": pool["name"], "pool_identity": pool["identity"],
                      "n_base_classes": len(T), "n_candidates": n_cand, "share": share, "batch_segments": segments,
                      "pool_segments": n_pool, "base_segments": n_base, "moves": moves, "lr": lr, "warmup": WARMUP,
                      "weight_decay": V.WD, "ema_decay": V.EMA_DECAY, "seed": SEED, "scale": L.SCALE, "fp16": cuda,
                      "base_sampler": {"players": base["players"], "T_sha256_sorted_ids": ids["T"],
                                       "cache_codes_sha256": cache.meta["codes_sha256"], "first_batch": base_step},
                      "pool_stream": [SEED, "s", POOL_STREAM],
                      "head_init": "L2-normalised mean of the base's game vectors (raw, tta 1) per candidate x median base row length (rev 1)",
                      "start_weights": "raw", "smoke": bool(a.smoke)}}
    net, head = V.build(spec, len(T) + n_cand)
    net.load_state_dict(ck0["net"])
    w0 = extended_head_weight(ck0["head"]["weight"].float().numpy(), rows)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(w0))
    del ck0, w0
    net, head = net.to(dev), head.to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=lr, weight_decay=V.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / float(WARMUP)))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = V.Ema(net)

    start, seconds, positions = 0, 0.0, 0
    os.makedirs(ckpt_dir, exist_ok=True)
    last = last_ckpt(ckpt_dir)
    if a.auto_resume and last is not None:
        a.resume = True
    if a.resume:
        if last is None:
            raise SystemExit("--resume: no checkpoint in %s" % ckpt_dir)
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["config"] != config:
            raise SystemExit("--resume: the checkpoint's config differs:\n%s\n%s" % (ck["config"], config))
        if ck["step"] > end:
            raise SystemExit("--resume: %s is past --steps %d" % (last, end))
        net.load_state_dict(ck["net"])
        head.load_state_dict(ck["head"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, seconds, positions = int(ck["step"]), ck["seconds"], ck["positions"]
        del ck
        print("resumed from %s (step %d)" % (last, start), flush=True)
    elif last is not None:
        raise SystemExit("checkpoints already exist in %s; use --resume" % ckpt_dir)
    t0 = time.time()

    def save(step):
        path = ckpt_file(out, step)
        torch.save({"net": net.state_dict(), "head": head.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "config": config, "seconds": seconds + time.time() - t0, "positions": positions,
                    "git_head": L.git_head(), "code_sha256": code_sha()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print("saved %s" % path, flush=True)

    loader = DataLoader(ds, batch_size=None, sampler=range(start, end), num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None, persistent_workers=False)
    net.train()
    head.train()
    tw, lw, nw = time.time(), 0.0, 0
    step, why = start, None
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for s, b in zip(range(start, end), loader):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = T42.loss_of(b, net, head, spec, cuda, s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(net)
            step = s + 1
            positions += len(b["seg"])
            lv = float(loss.detach())
            if not np.isfinite(lv):
                raise SystemExit("loss is not finite at step %d" % step)
            lw += lv
            nw += 1
            if step % LOG_EVERY == 0 or step == end:
                dt = time.time() - tw
                log.write(json.dumps({"step": step, "loss": round(lw / nw, 5), "steps_per_s": round(nw / dt, 3),
                                      "lr": opt.param_groups[0]["lr"],
                                      "elapsed_s": round(seconds + time.time() - t0, 1)}) + "\n")
                log.flush()
                tw, lw, nw = time.time(), 0.0, 0
            if step in ckpt_at:
                save(step)
            if step % STOP_EVERY == 0 and step < end:
                fr = T42.free_ram_mb() if a.min_free_ram_mb > 0 else None
                hit = stop_requested(stops)
                if hit:
                    why = "STOP file %s" % hit
                elif fr is not None and fr < a.min_free_ram_mb:
                    why = "free RAM %.0f MB < %.0f MB" % (fr, a.min_free_ram_mb)
                if why:
                    if step not in ckpt_at:
                        save(step)
                    break
    seconds += time.time() - t0
    rec = {"format": "c10-prereg57 adapt v1", "config": config, "base_ckpt": _rel(base["ckpt"]),
           "base_ckpt_sha256": base_sha, "pool": pool_stats(pool), "pool_identity": pool["identity"],
           "candidates_trained": n_cand, "steps": step, "target_steps": end, "finished": step == end,
           "stopped": why, "checkpoint_steps": ckpt_at, "seconds": round(seconds, 1),
           "positions_seen": int(positions), "loss_curve": read_log(log_path, step),
           "checkpoints": {os.path.basename(ckpt_file(out, n)): L.sha256_file(ckpt_file(out, n))
                           for n in ckpt_at if n <= step and os.path.exists(ckpt_file(out, n))},
           "device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers,
           "torch": torch.__version__, "git_head": L.git_head(), "code_sha256": code_sha()}
    L.write_json(rec_path, rec)
    if why:
        print("%s: saved at step %d, exiting (resume with --resume)" % (why, step), flush=True)
        return 4
    print(json.dumps({k: rec[k] for k in ("finished", "steps", "seconds", "candidates_trained")}), flush=True)
    return 0


# ================================================================ embed
def write_vectors(path, emb, count, ckpt_sha, tag, step, embed_cfg, identity, store_row=None):
    """p42_embed's layout (emb, count, ckpt_sha256, ckpt, ckpt_step, embed_cfg [, store_row]) + pool_identity."""
    extra = {} if store_row is None else {"store_row": np.asarray(store_row, np.int64)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, emb=np.asarray(emb, np.float32), count=np.asarray(count, np.int32), ckpt_sha256=np.array(ckpt_sha),
             ckpt=np.array(tag), ckpt_step=np.array(int(step)),
             embed_cfg=np.array(json.dumps(embed_cfg, sort_keys=True)), pool_identity=np.array(identity), **extra)
    os.replace(tmp, path)


def read_vectors(path):
    with np.load(path, allow_pickle=False) as z:
        return {"emb": z["emb"], "count": z["count"],
                "store_row": z["store_row"] if "store_row" in z.files else None,
                "ckpt_sha256": str(z["ckpt_sha256"]), "ckpt": str(z["ckpt"]), "ckpt_step": int(z["ckpt_step"]),
                "embed_cfg": json.loads(str(z["embed_cfg"])), "pool_identity": str(z["pool_identity"])}


def cmd_embed(a):
    t0 = time.time()
    if (a.base, a.pool_kind) not in ALLOWED:
        raise SystemExit("PLAN.md allows %s only" % " / ".join("%s on %s" % x for x in ALLOWED))
    cuda = a.device == "cuda"
    if not cuda and not a.limit:
        raise SystemExit("--device cpu is for --limit N (smoke) only")
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        print("exists: %s" % out)
        return 0
    import torch
    from torch.utils.data import DataLoader
    import p42_embed as EM
    dev = torch.device(a.device)
    if not cuda:
        torch.set_num_threads(2)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c57 = ck["config"].get("p57")
    ck_step = int(ck["step"])
    del ck
    ck_sha = L.sha256_file(a.ckpt)
    pool = build_pool(a.pool_kind, a.pool, a.src)
    if c57 is None:
        if ck_sha != BASES[a.base]["sha256"]:
            raise SystemExit("%s is neither a prereg57 checkpoint nor the %s base" % (a.ckpt, a.base))
    else:
        if c57["base"] != a.base or c57["pool_identity"] != pool["identity"]:
            raise SystemExit("the checkpoint was adapted to another pool / base: pools are never mixed")
        if c57["smoke"] and not a.limit:
            raise SystemExit("a smoke checkpoint is embedded with --limit only")
        require_dev_choice(a.pool_kind, c57["share"], c57["lr"], None if c57["smoke"] else ck_step)
        if a.pool_kind != "D" and a.weights != "ema":
            raise SystemExit("PLAN.md Method 3: the vectors come from the EMA weights")
    net, spec, _run, step = EM.load_net(a.ckpt, a.weights, dev)
    n = len(pool["is_white"]) if not a.limit else min(int(a.limit), len(pool["is_white"]))
    start, end, is_white = pool["start"][:n], pool["end"][:n], pool["is_white"][:n]
    ds = EM.GameSet(pool["codes"], start, end, is_white)
    loader = DataLoader(ds, batch_size=None, sampler=range(len(ds)), num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None)
    embs, cnts, n_pos = {}, {}, 0
    with torch.no_grad():
        for i, x, seg, t in loader:
            i = int(i)
            g = min(n, (i + 1) * EM.CHUNK) - i * EM.CHUNK
            v, c = EM.game_vectors(net, spec, x, seg, t, g, 1, None, dev, cuda)
            embs[i] = v.cpu().numpy()
            cnts[i] = c.cpu().numpy().astype(np.int32)
            n_pos += len(seg)
    emb = np.concatenate([embs[i] for i in range(len(ds))]).astype(np.float32)
    cnt = np.concatenate([cnts[i] for i in range(len(ds))])
    assert len(emb) == n == len(cnt)
    tag = "p57_%s_%s%s_step_%d_%s_tta1" % (a.base, pool["kind"], pool["name"], step, a.weights)
    embed_cfg = {"ckpt_sha256": ck_sha, "weights": a.weights, "tta": 1, "mode": None, "spec": spec, "games": int(n),
                 "moves": int((end - start).sum()), "chunk": EM.CHUNK, "limit": a.limit, "p57": c57,
                 "pool": {"kind": pool["kind"], "name": pool["name"], "identity": pool["identity"]},
                 "base": a.base, "ckpt": _rel(a.ckpt)}
    rep = {"out": out, "tag": tag, "games": int(n), "games_without_sample": int((cnt == 0).sum()),
           "positions": int(n_pos), "seconds": None}
    if a.compare_base:
        if c57 is not None:
            raise SystemExit("--compare-base is for the unadapted base checkpoint (a seam of the embed path)")
        ref, rc = base_vectors(a.base, pool)
        ref, rc = ref[:n], rc[:n]
        rep["compare_base"] = {"file": _rel(base_vec_file(a.base, pool["kind"], pool["name"])),
                               "count_equal": bool((rc == cnt).all()),
                               "max_abs_diff": float(np.abs(ref - emb).max()) if n else 0.0,
                               "largest_abs_value": float(np.abs(ref).max()) if n else 0.0}
        embed_cfg["compare_base"] = rep["compare_base"]
    write_vectors(out, emb, cnt, ck_sha, tag, step, embed_cfg, pool["identity"],
                  None if pool["store_row"] is None else pool["store_row"][:n])
    rep["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rep), flush=True)
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("adapt", "embed"):
        s = sub.add_parser(name)
        s.add_argument("--base", required=True, choices=sorted(BASES))
        s.add_argument("--pool-kind", required=True, choices=["D", "E", "official"])
        s.add_argument("--pool", required=True, help="D / E: pool index; official: public | private")
        s.add_argument("--src", default=SRC_OFFICIAL, help="official tests folder")
        s.add_argument("--out", required=True)
        s.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        s.add_argument("--workers", type=int, default=4)
        if name == "adapt":
            s.add_argument("--share", required=True)
            s.add_argument("--lr", required=True)
            s.add_argument("--steps", type=int, default=None)
            s.add_argument("--ckpt-at", default=None, help="comma-separated step numbers; --steps is always saved")
            s.add_argument("--resume", action="store_true")
            s.add_argument("--auto-resume", action="store_true", help="resume if a checkpoint exists, else start")
            s.add_argument("--smoke", action="store_true",
                           help="<= %d steps, the first %d candidates, %d segments x %d moves; CPU allowed"
                                % (SMOKE_STEPS, SMOKE_CANDS, SMOKE_SEGMENTS, SMOKE_MOVES))
            s.add_argument("--stop-file", default=STOP_FILE)
            s.add_argument("--min-free-ram-mb", type=float, default=4096)
        else:
            s.add_argument("--ckpt", required=True)
            s.add_argument("--weights", required=True, choices=["ema", "raw"])
            s.add_argument("--limit", type=int, default=None, help="smoke / seam only: the first N games")
            s.add_argument("--compare-base", action="store_true",
                           help="unadapted base checkpoint only: largest difference to the base's existing vectors")
    a = ap.parse_args(argv)
    if a.pool_kind in ("D", "E"):
        a.pool = int(a.pool)
    return {"adapt": cmd_adapt, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
