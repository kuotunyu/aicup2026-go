"""prereg12 exploration (competition 2): HumanSL fine-tuning pilot, data / GPU steps (PLAN.md sections 3-4).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; CPU .venv/Scripts/python.exe, GPU .venv-gpu/Scripts/python.exe
    S=results/prereg12/explore/finetune
    CPU  $S/ft_pilot.py sets                          -> data/prereg12/sets.json + $S/sets_record.json
    CPU  $S/ft_pilot.py cache --set train|holdout|bank|mono [--workers 16]   -> data/prereg12/cache/<set>/chunk_*.npz
    GPU  $S/ft_pilot.py bench   --device cuda         -> data/prereg12/train/bench.json
    GPU  $S/ft_pilot.py train   --device cuda         -> data/prereg12/train/{ckpt_last.pt, best.pt, train_done.json}
    GPU  $S/ft_pilot.py extract --set bank|mono --device cuda   -> data/prereg12/ft/<set>/chunk_*.npz
    CPU  $S/ft_pilot.py status | gpu-free
    The detached launcher is data/prereg12/run_pilot.sh (-> $S/run_pilot.sh). Inner evaluation: $S/ft_eval.py.

Rules: fit players only for training and checkpoint selection (sets: tools/c10data part 'fit'); bank / mono are inner
games (training players), featurised exactly as P1 (p1_lib.featurize_game). Never data/local (hs_lib guard), never
validation features, never tools/score.py. Every step is resumable: chunk files are atomic and skipped when present;
train keeps ckpt_last.pt. Create data/prereg12/STOP to stop after the current chunk / step block; delete it and rerun.
GPU steps refuse to start while nvidia-smi lists another compute process (exit 3) and never call CUDA before that check.
Smoke options (--out-root, --max-games, --tiny, --max-steps) write under the given root only.
Exit codes: 0 done (or stopped by STOP), 2 error, 3 GPU busy.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import ft_lib as F  # noqa: E402

GPU_BUSY = 3


class Paths:
    def __init__(self, root=None):
        self.root = os.path.abspath(root) if root else F.DATA
        self.sets = os.path.join(self.root, "sets.json")
        self.cache = os.path.join(self.root, "cache")
        self.train = os.path.join(self.root, "train")
        self.ft = os.path.join(self.root, "ft")
        self.log = os.path.join(self.root, "run_log.jsonl")
        self.stop = os.path.join(self.root, "STOP")

    def is_real(self):
        return os.path.normcase(self.root) == os.path.normcase(F.DATA)


def L(paths, msg, **kv):
    F.log(msg, log_path=paths.log, **kv)


# ================================================================ sets
def round_robin(by_player, players, quota, rng):
    pools = {p: sorted(by_player[p], key=lambda r: r["game_id"]) for p in players}
    out = []
    while len(out) < quota:
        moved = False
        for p in players:
            if len(out) >= quota:
                break
            if pools[p]:
                out.append(pools[p].pop(rng.randrange(len(pools[p]))))
                moved = True
        if not moved:
            break
    return out


def cmd_sets(a):
    from c10data import load_rows
    paths = Paths(a.out_root)
    if os.path.exists(paths.sets) and not a.force:
        raise F.FTError(f"{paths.sets} exists (its SHA-256 is recorded); not rebuilt")
    t0 = time.time()
    fit = load_rows(F.TRAIN_CSV_DIR, F.SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    inner = load_rows(F.TRAIN_CSV_DIR, F.SPLIT_CSV, part="inner", local_dir=None, check_counts=True)
    inner_players = {r["player_id"] for r in inner}
    inner_keys = {F.move_key(r["sgf_content"]) for r in inner}
    rng = random.Random(F.SETS_SEED)
    train, hold, rec_lv = [], [], {}
    dup_excluded = 0
    for lv in F.LEVELS:
        rows = [r for r in fit if r["rank"] == lv]
        keep = []
        for r in rows:
            if F.move_key(r["sgf_content"]) in inner_keys:
                dup_excluded += 1
            else:
                keep.append(r)
        by_player = {}
        for r in keep:
            by_player.setdefault(r["player_id"], []).append(r)
        players = sorted(by_player)
        rng.shuffle(players)
        n_hold = max(1, int(math.ceil(F.HOLDOUT_PLAYER_FRAC * len(players))))
        hp, tp = players[:n_hold], players[n_hold:]
        per_level = a.train_per_level or F.TRAIN_PER_LEVEL
        tr = round_robin(by_player, tp, per_level, rng)
        ho = round_robin(by_player, hp, a.holdout_per_level or F.HOLDOUT_PER_LEVEL, rng)
        rec_lv[lv] = {"players": len(players), "holdout_players": len(hp), "train_games": len(tr),
                      "holdout_games": len(ho), "fit_games": len(rows)}
        train += [{"rank": lv, "game_id": r["game_id"], "player_id": r["player_id"], "color": r["color"]} for r in tr]
        hold += [{"rank": lv, "game_id": r["game_id"], "player_id": r["player_id"], "color": r["color"]} for r in ho]
    rng.shuffle(train)
    rng.shuffle(hold)
    tp_all = {(g["rank"], g["player_id"]) for g in train}
    hp_all = {(g["rank"], g["player_id"]) for g in hold}
    checks = {"train_holdout_players_disjoint": not (tp_all & hp_all),
              "no_inner_player": not ({p for _r, p in tp_all | hp_all} & inner_players),
              "no_duplicate_game": len({(g["rank"], g["game_id"]) for g in train + hold}) == len(train) + len(hold)}
    if not all(checks.values()):
        raise F.FTError(f"set checks failed: {checks}")
    sets = {"format": F.SETS_FORMAT, "created_utc": F.utc_now(), "seed": F.SETS_SEED,
            "pos_per_side": F.CONFIG["pos_per_side"], "train": train, "holdout": hold}
    F.write_json_atomic(paths.sets, sets)
    rec = {"format": "c10-prereg12 ft sets record v1", "created_utc": sets["created_utc"],
           "sets_json": F.rel(paths.sets) if paths.root.startswith(F.REPO) else paths.sets,
           "sha256": F.sha_raw(paths.sets), "seconds": round(time.time() - t0, 1),
           "source": "tools/c10data load_rows part 'fit' (split v1; never inner, never validation)",
           "rule": "per level: fit players shuffled (random.Random(SETS_SEED)); first ceil(5 %) are holdout players; "
                   "round robin over players, one random not-yet-taken game per turn, until the quota; games whose "
                   "move sequence equals an inner game's are excluded",
           "train_games": len(train), "holdout_games": len(hold), "train_players": len(tp_all),
           "holdout_players": len(hp_all), "duplicates_with_inner_excluded": dup_excluded, "levels": rec_lv,
           "checks": checks}
    if paths.is_real():
        F.write_json_atomic(os.path.join(F.HERE, "sets_record.json"), rec)
    else:
        F.write_json_atomic(os.path.join(paths.root, "sets_record.json"), rec)
    print(json.dumps({k: rec[k] for k in ("sha256", "train_games", "holdout_games", "train_players",
                                          "holdout_players", "duplicates_with_inner_excluded", "checks")}))
    return 0


def load_sets(paths):
    if not os.path.exists(paths.sets):
        raise F.FTError(f"{paths.sets} missing: run `ft_pilot.py sets` first")
    with open(paths.sets, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("format") != F.SETS_FORMAT:
        raise F.FTError(f"{paths.sets}: format {s.get('format')!r}")
    return s


def set_games(paths, name):
    """-> list of row dicts with sgf_content, color, ref, y (y = level index; -1 for inner sets, never read)."""
    if name in ("bank", "mono"):
        import p1_extract as E
        rows = E.set_games(E.load_sets(F.P1_SETS), name)
        return [{**r, "y": -1} for r in rows]
    from c10data import load_rows
    s = load_sets(paths)
    fit = load_rows(F.TRAIN_CSV_DIR, F.SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    by = {(r["rank"], r["game_id"]): r for r in fit}
    lv = {x: i for i, x in enumerate(F.LEVELS)}
    out = []
    for j, g in enumerate(s[name]):
        r = by[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != g["color"]:
            raise F.FTError(f"{name} game {j}: player or colour differs from sets.json")
        out.append({**r, "ref": j, "y": lv[g["rank"]]})
    return out


# ================================================================ cache (CPU)
def chunk_path(root, name, ci):
    return os.path.join(root, name, f"chunk_{ci:05d}.npz")


def cache_chunk_ok(path, name, refs):
    try:
        with np.load(path, allow_pickle=False) as z:
            return str(z["format"]) == F.CACHE_FORMAT and str(z["set"]) == name and np.array_equal(z["ref"], refs)
    except Exception:
        return False


def turn_seed(name, ref):
    return [F.SETS_SEED, 1 if name == "train" else 2, int(ref)]


def cmd_cache(a):
    paths = Paths(a.out_root)
    if a.max_games and paths.is_real():
        raise F.FTError("--max-games is for smoke runs only: give --out-root another folder")
    games = set_games(paths, a.set)
    if a.max_games:
        games = games[:a.max_games]
    csize = a.chunk or F.CACHE_CHUNK[a.set]
    refs = np.array([g["ref"] for g in games], dtype=np.int64)
    chunks = [(ci, s, min(len(games), s + csize)) for ci, s in enumerate(range(0, len(games), csize))]
    todo = [c for c in chunks if not cache_chunk_ok(chunk_path(paths.cache, a.set, c[0]), a.set, refs[c[1]:c[2]])]
    L(paths, "cache start", set=a.set, games=len(games), chunks=len(chunks), todo=len(todo), workers=a.workers)
    if not todo:
        return 0
    per_side = F.CONFIG["pos_per_side"]
    sampled = a.set in ("train", "holdout")
    os.makedirs(os.path.join(paths.cache, a.set), exist_ok=True)
    t_run = time.time()
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers > 0 else None
    try:
        for k, (ci, s, e) in enumerate(todo):
            if F.stop_requested(paths.stop):
                L(paths, "cache stopping (STOP file)", set=a.set, chunks_left=len(todo) - k)
                break
            t0 = time.time()
            part = games[s:e]

            def turns_of(_gi, g):
                if not sampled:
                    return None
                return F.select_turns(g["sgf_content"], g["color"], per_side, turn_seed(a.set, g["ref"]))

            res = F.build_cache_chunk(pool, part, turns_of)
            path = chunk_path(paths.cache, a.set, ci)
            tmp = path + ".tmp.npz"
            np.savez(tmp, format=np.array(F.CACHE_FORMAT), set=np.array(a.set), ref=refs[s:e],
                     y=np.array([g["y"] for g in part], dtype=np.int8),
                     colour=np.array([0 if g["color"] == "B" else 1 for g in part], dtype=np.int8),
                     sampled=np.array(sampled), pos_per_side=np.array(per_side if sampled else -1),
                     meta_sha256=np.array(F.P.neutral_table_sha()), created_utc=np.array(F.utc_now()),
                     code_sha256=np.array(json.dumps(F.code_shas(), sort_keys=True)), **res)
            os.replace(tmp, path)
            el = time.time() - t_run
            L(paths, "cache chunk done", set=a.set, chunk=ci, games=e - s, failed=int(res["failed"].sum()),
              positions=int(len(res["move"])), seconds=round(time.time() - t0, 1),
              progress=f"{k + 1}/{len(todo)}", eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return 0


def cache_files(paths, name):
    d = os.path.join(paths.cache, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("chunk_") and f.endswith(".npz")
                  and ".tmp" not in f)


def expected_chunks(paths, name):
    if name in ("bank", "mono"):
        import p1_extract as E
        n = len(E.load_sets(F.P1_SETS)[name]["rows"])
    else:
        n = len(load_sets(paths)[name])
    return int(math.ceil(n / F.CACHE_CHUNK[name]))


# ================================================================ GPU helpers
def gpu_guard(device):
    """Before any CUDA call: refuse when another compute process is on the GPU."""
    if not device.startswith("cuda"):
        return None
    rows = F.other_gpu_compute()
    if rows:
        return rows
    torch = F.torch_mod()
    if not torch.cuda.is_available():
        raise F.FTError("CUDA requested but torch.cuda.is_available() is False")
    torch.backends.cudnn.benchmark = True
    return None


def precision_of(a):
    if a.precision:
        return a.precision
    return F.CONFIG["precision_cuda"] if a.device.startswith("cuda") else "fp32"


class ChunkData:
    """One cache chunk in memory with per-game position ranges."""

    def __init__(self, path):
        z = F.load_cache_chunk(path)
        self.z = z
        self.G = len(z["ref"])
        gidx = z["gidx"]
        self.start = np.searchsorted(gidx, np.arange(self.G), side="left")
        self.end = np.searchsorted(gidx, np.arange(self.G), side="right")
        self.usable = (~z["failed"]) & (z["counts"][:, 0] > 0)

    def batch(self, games):
        idx = np.concatenate([np.arange(self.start[g], self.end[g]) for g in games])
        local = np.empty(len(idx), dtype=np.int64)
        o = 0
        for j, g in enumerate(games):
            n = self.end[g] - self.start[g]
            local[o:o + n] = j
            o += n
        z = self.z
        return {"bits": z["bits"][idx], "glob": z["glob"][idx], "move": z["move"][idx].astype(np.int64),
                "pla": z["pla"][idx].astype(np.int64), "seg": local * 2 + z["side"][idx].astype(np.int64),
                "y": z["y"][games].astype(np.int64), "colour": z["colour"][games].astype(np.int64),
                "n_games": len(games)}


def to_dev(b, device):
    torch = F.torch_mod()
    return {"bits": torch.as_tensor(b["bits"], device=device), "glob": torch.as_tensor(b["glob"], device=device),
            "move": torch.as_tensor(b["move"], device=device), "pla": torch.as_tensor(b["pla"], device=device),
            "seg": torch.as_tensor(b["seg"], device=device), "y": torch.as_tensor(b["y"], device=device),
            "colour": torch.as_tensor(b["colour"], device=device), "n_games": b["n_games"]}


def forward_batch(net, d, device, precision):
    with F.autocast_ctx(device, precision):
        trunk = net.trunk(F.unpack_bits_torch(d["bits"]), d["glob"], d["pla"])
    h = net.head.pos_hidden(trunk.float(), d["move"])
    return net.head.game_logits(h, d["seg"], d["n_games"], d["colour"])


SCORE_OFF1 = math.exp(-1.0)


def holdout_eval(net, paths, device, precision, batch_games=32, max_chunks=None):
    torch = F.torch_mod()
    net.eval()
    files = cache_files(paths, "holdout")[:max_chunks] if max_chunks else cache_files(paths, "holdout")
    ys, lps = [], []
    with torch.no_grad():
        for p in files:
            cd = ChunkData(p)
            games = np.nonzero(cd.usable)[0]
            for s in range(0, len(games), batch_games):
                d = to_dev(cd.batch(games[s:s + batch_games]), device)
                lg = forward_batch(net, d, device, precision)
                lps.append(torch.log_softmax(lg.float(), 1).cpu().numpy())
                ys.append(d["y"].cpu().numpy())
    net.train()
    if not ys:
        return {"games": 0}
    y = np.concatenate(ys)
    lp = np.concatenate(lps)
    pred = lp.argmax(1)
    sc = np.where(pred == y, 1.0, np.where(np.abs(pred - y) == 1, SCORE_OFF1, 0.0))
    per_lv = [float(sc[y == k].mean()) for k in range(F.N_LEVELS) if (y == k).any()]
    return {"games": int(len(y)), "logloss": float(-lp[np.arange(len(y)), y].mean()),
            "accuracy": float((pred == y).mean()), "score_argmax": float(sc.mean()),
            "score_argmax_rank_balanced": float(np.mean(per_lv))}


def train_index(paths):
    files = cache_files(paths, "train")
    usable = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            usable.append(int(((~z["failed"]) & (z["counts"][:, 0] > 0)).sum()))
    return files, usable


def schedule(step, total, cfg):
    """(head multiplier, trunk multiplier) of the learning rates."""
    floor = cfg["cosine_floor"]

    def cos(t0):
        if total <= t0:
            return 1.0
        x = min(1.0, max(0.0, (step - t0) / (total - t0)))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * x))

    head = min(1.0, (step + 1) / 200.0) * cos(0)
    t0 = cfg["head_only_steps"]
    if step < t0:
        trunk = 0.0
    else:
        trunk = min(1.0, (step - t0 + 1) / max(1, cfg["trunk_warmup_steps"])) * cos(t0)
    return head, trunk


def make_optimizer(net, cfg):
    torch = F.torch_mod()
    head_p = [p for p in net.head.parameters()]
    hid = {id(p) for p in head_p}
    trunk_p = [p for _n, p in net.trainable_named() if id(p) not in hid]
    groups = [{"params": head_p, "lr": cfg["lr_head"], "weight_decay": cfg["wd_head"], "name": "head"}]
    if trunk_p:
        groups.append({"params": trunk_p, "lr": cfg["lr_trunk"], "weight_decay": cfg["wd_trunk"], "name": "trunk"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def save_ckpt(path, net, opt, scaler, state):
    torch = F.torch_mod()
    tmp = path + ".tmp"
    torch.save({"trainable": F.save_trainable(net), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None, "state": state}, tmp)
    os.replace(tmp, path)


def epoch_plan(files, usable, epoch, seed, bg):
    """[(file index, batch index within the chunk)] of one epoch, deterministic."""
    order = np.random.default_rng([seed, epoch]).permutation(len(files))
    plan = []
    for fi in order:
        nb = int(math.ceil(usable[fi] / bg))
        plan += [(int(fi), b) for b in range(nb)]
    return plan


def batch_games_of(cd, seed, epoch, fi, b, bg):
    games = np.nonzero(cd.usable)[0]
    perm = np.random.default_rng([seed, epoch, fi]).permutation(len(games))
    return games[perm[b * bg:(b + 1) * bg]]


def cmd_bench(a):
    paths = Paths(a.out_root)
    busy = gpu_guard(a.device)
    if busy:
        L(paths, "GPU busy, bench not started", processes=busy)
        return GPU_BUSY
    torch = F.torch_mod()
    prec = precision_of(a)
    cfg = dict(F.CONFIG)
    net, info = F.build_net(a.device, tiny=a.tiny)
    files = cache_files(paths, "train")
    if not files:
        raise F.FTError("no train cache chunk")
    cd = ChunkData(files[0])
    rec = {"format": "c10-prereg12 ft bench v1", "created_utc": F.utc_now(), "device": a.device, "precision": prec,
           "model": info, "torch": torch.__version__}
    # inference rate (trunk + head hidden), padded batches of a.batch positions
    games = np.nonzero(cd.usable)[0]
    b = cd.batch(games[:max(1, min(len(games), a.batch // 32 + 1))])
    P_ = len(b["move"])
    rows = np.resize(np.arange(P_), a.batch)
    d = to_dev({**b, "bits": b["bits"][rows], "glob": b["glob"][rows], "move": b["move"][rows],
                "pla": b["pla"][rows]}, a.device)
    net.eval()
    reps = a.reps
    with torch.no_grad():
        for i in range(reps + 1):
            if i == 1:
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.time()
            with F.autocast_ctx(a.device, prec):
                tr = net.trunk(F.unpack_bits_torch(d["bits"]), d["glob"], d["pla"])
            net.head.pos_hidden(tr.float(), d["move"])
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
    rec["infer_positions_per_s"] = a.batch * reps / (time.time() - t0)
    # training step rate; halve batch_games on CUDA OOM
    net.train()
    bg = a.batch_games or cfg["batch_games"]
    while True:
        try:
            opt = make_optimizer(net, cfg)
            for g in opt.param_groups:
                g["lr"] = 0.0          # timing only: weights unchanged
            scaler = torch.amp.GradScaler("cuda") if prec == "fp16" else None
            if a.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            npos = 0
            for i in range(reps + 1):
                if i == 1:
                    if a.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.time()
                    npos = 0
                bt = to_dev(cd.batch(games[(i * bg) % max(1, len(games) - bg):][:bg]), a.device)
                loss = torch.nn.functional.cross_entropy(forward_batch(net, bt, a.device, prec).float(), bt["y"])
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                npos += int(bt["move"].shape[0])
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            secs = time.time() - t0
            rec.update({"batch_games": bg, "train_positions_per_s": npos / secs, "train_steps_per_s": reps / secs,
                        "peak_memory_gib": (torch.cuda.max_memory_allocated() / 2 ** 30)
                        if a.device.startswith("cuda") else None})
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bg <= 2:
                raise
            bg //= 2
            rec.setdefault("oom_at_batch_games", []).append(bg * 2)
    os.makedirs(paths.train, exist_ok=True)
    F.write_json_atomic(os.path.join(paths.train, "bench.json"), rec)
    L(paths, "bench done", **{k: rec[k] for k in ("infer_positions_per_s", "train_positions_per_s", "batch_games")})
    return 0


def cmd_train(a):
    paths = Paths(a.out_root)
    done_path = os.path.join(paths.train, "train_done.json")
    if os.path.exists(done_path):
        L(paths, "train already done", file=done_path)
        return 0
    if F.stop_requested(paths.stop):
        L(paths, "train not starting (STOP file)")
        return 0
    busy = gpu_guard(a.device)
    if busy:
        L(paths, "GPU busy, train not started", processes=busy)
        return GPU_BUSY
    torch = F.torch_mod()
    prec = precision_of(a)
    cfg = dict(F.CONFIG)
    if a.max_hours is not None:
        cfg["max_train_gpu_hours"] = a.max_hours
    if a.eval_every:
        cfg["eval_every"] = a.eval_every
    if a.ckpt_every:
        cfg["ckpt_every"] = a.ckpt_every
    if a.head_only_steps is not None:
        cfg["head_only_steps"] = a.head_only_steps
        cfg["trunk_warmup_steps"] = max(1, a.head_only_steps)
    bench_path = os.path.join(paths.train, "bench.json")
    bench = None
    if os.path.exists(bench_path):
        with open(bench_path, encoding="utf-8") as f:
            bench = json.load(f)
        cfg["batch_games"] = int(bench.get("batch_games", cfg["batch_games"]))
    files, usable = train_index(paths)
    want = expected_chunks(paths, "train") if not a.allow_partial_cache else len(files)
    if len(files) != want:
        raise F.FTError(f"train cache has {len(files)} chunks, {want} expected (run cache --set train)")
    if not cache_files(paths, "holdout"):
        raise F.FTError("holdout cache missing")
    net, info = F.build_net(a.device, tiny=a.tiny)
    opt = make_optimizer(net, cfg)
    scaler = torch.amp.GradScaler("cuda") if prec == "fp16" else None
    bg = cfg["batch_games"]
    steps_epoch = sum(int(math.ceil(u / bg)) for u in usable)
    last = os.path.join(paths.train, "ckpt_last.pt")
    best = os.path.join(paths.train, "best.pt")
    os.makedirs(paths.train, exist_ok=True)
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        st = ck["state"]
        if st["config"] != cfg or st["tiny"] != bool(a.tiny):
            raise F.FTError("ckpt_last.pt was made with another configuration")
        F.load_trainable(net, ck["trainable"])
        opt.load_state_dict(ck["opt"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        L(paths, "train resume", step=st["step"], train_seconds=round(st["train_seconds"], 1))
    else:
        total = steps_epoch * cfg["epochs"]
        cap = None
        if bench and bench.get("train_steps_per_s"):
            cap = int(bench["train_steps_per_s"] * cfg["max_train_gpu_hours"] * 3600 * 0.9)
            total = min(total, cap)
        if a.max_steps:
            total = min(total, a.max_steps)
        st = {"step": 0, "train_seconds": 0.0, "planned_steps": int(total), "steps_epoch": steps_epoch,
              "cap_steps_from_bench": cap, "config": cfg, "tiny": bool(a.tiny), "history": [], "best": None,
              "model": {k: v for k, v in info.items() if k != "checkpoint"}, "precision": prec,
              "created_utc": F.utc_now(), "train_games_usable": int(sum(usable))}
        L(paths, "train start", planned_steps=st["planned_steps"], steps_epoch=steps_epoch, batch_games=bg,
          trainable_params=info["trainable_params"])
    net.train()
    seed = cfg["seed"]
    cache = {}
    t_seg = time.time()
    hard_cap = cfg["max_train_gpu_hours"] * 3600 * 1.1
    stopped = None
    run_losses = []

    def checkpoint():
        st["train_seconds"] += time.time() - t_seg
        save_ckpt(last, net, opt, scaler, st)
        return time.time()

    def do_eval():
        ev = holdout_eval(net, paths, a.device, prec)
        ev.update({"step": st["step"], "train_seconds": st["train_seconds"],
                   "train_loss_recent": float(np.mean(run_losses[-200:])) if run_losses else None})
        st["history"].append(ev)
        if ev.get("games") and (st["best"] is None or ev["logloss"] < st["best"]["logloss"]):
            st["best"] = {"step": st["step"], "logloss": ev["logloss"]}
            torch.save({"trainable": F.save_trainable(net), "state": {"step": st["step"], "config": cfg,
                                                                       "tiny": bool(a.tiny), "holdout": ev}},
                       best + ".tmp")
            os.replace(best + ".tmp", best)
        L(paths, "holdout eval", **ev)

    while st["step"] < st["planned_steps"]:
        if F.stop_requested(paths.stop):
            stopped = "STOP file"
            break
        if st["train_seconds"] + (time.time() - t_seg) > hard_cap:
            stopped = "hard time cap"
            break
        step = st["step"]
        epoch, within = divmod(step, steps_epoch)
        plan_key = ("plan", epoch)
        if plan_key not in cache:
            cache = {plan_key: epoch_plan(files, usable, epoch, seed, bg)}
        fi, b = cache[plan_key][within]
        if ("chunk", fi) not in cache:
            for k in [k for k in cache if k[0] == "chunk"]:
                del cache[k]
            cache[("chunk", fi)] = ChunkData(files[fi])
        cd = cache[("chunk", fi)]
        d = to_dev(cd.batch(batch_games_of(cd, seed, epoch, fi, b, bg)), a.device)
        mh, mt = schedule(step, st["planned_steps"], cfg)
        for g in opt.param_groups:
            g["lr"] = (cfg["lr_head"] * mh) if g["name"] == "head" else (cfg["lr_trunk"] * mt)
        loss = torch.nn.functional.cross_entropy(forward_batch(net, d, a.device, prec).float(), d["y"])
        opt.zero_grad(set_to_none=True)
        params = [p for _n, p in net.trainable_named()]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise F.FTError(f"non-finite loss at step {step}")
        run_losses.append(lv)
        st["step"] += 1
        if st["step"] % 100 == 0:
            L(paths, "train progress", step=st["step"], planned=st["planned_steps"],
              loss_recent=round(float(np.mean(run_losses[-100:])), 4), lr_head_mult=round(mh, 4),
              lr_trunk_mult=round(mt, 4), seconds=round(st["train_seconds"] + time.time() - t_seg, 1))
        if st["step"] % cfg["eval_every"] == 0:
            t_seg = checkpoint()
            do_eval()
            t_seg = checkpoint()
        elif st["step"] % cfg["ckpt_every"] == 0:
            t_seg = checkpoint()
    t_seg = checkpoint()
    if stopped:
        L(paths, "train stopped (rerun the same command to continue)", reason=stopped, step=st["step"])
        if stopped != "hard time cap":
            return 0
    if not st["history"] or st["history"][-1]["step"] != st["step"]:
        do_eval()
        t_seg = checkpoint()
    F.write_json_atomic(done_path, {"format": "c10-prereg12 ft train done v1", "created_utc": F.utc_now(),
                                    "steps": st["step"], "planned_steps": st["planned_steps"],
                                    "train_seconds": st["train_seconds"], "best": st["best"],
                                    "history": st["history"], "config": cfg, "model": st["model"],
                                    "precision": prec, "ended": stopped or "planned steps",
                                    "best_pt_sha256": F.sha_raw(best), "code_sha256": F.code_shas()})
    L(paths, "train done", steps=st["step"], best=st["best"])
    return 0


def cmd_extract(a):
    paths = Paths(a.out_root)
    if F.stop_requested(paths.stop):
        L(paths, "extract not starting (STOP file)", set=a.set)
        return 0
    best = os.path.join(paths.train, "best.pt")
    if not os.path.exists(os.path.join(paths.train, "train_done.json")):
        raise F.FTError("training not finished (train_done.json missing)")
    files = cache_files(paths, a.set)
    want = expected_chunks(paths, a.set) if not a.allow_partial_cache else len(files)
    if len(files) != want:
        raise F.FTError(f"{a.set} cache has {len(files)} chunks, {want} expected")
    best_sha = F.sha_raw(best)
    out_dir = os.path.join(paths.ft, a.set)
    todo = []
    for p in files:
        out = os.path.join(out_dir, os.path.basename(p))
        try:
            with np.load(out, allow_pickle=False) as z:
                if str(z["format"]) == F.FT_FORMAT and str(z["best_sha256"]) == best_sha:
                    continue
        except Exception:
            pass
        todo.append((p, out))
    L(paths, "extract start", set=a.set, chunks=len(files), todo=len(todo))
    if not todo:
        return 0
    busy = gpu_guard(a.device)
    if busy:
        L(paths, "GPU busy, extract not started", processes=busy)
        return GPU_BUSY
    torch = F.torch_mod()
    prec = precision_of(a)
    net, _info = F.build_net(a.device, tiny=a.tiny)
    ck = torch.load(best, map_location="cpu", weights_only=False)
    if ck["state"]["config"]["hidden"] != F.CONFIG["hidden"] or bool(ck["state"]["tiny"]) != bool(a.tiny):
        raise F.FTError("best.pt does not match this configuration")
    F.load_trainable(net, ck["trainable"])
    net.eval()
    import copy
    head64 = copy.deepcopy(net.head).cpu().double().eval()     # game level in float64 on the CPU
    os.makedirs(out_dir, exist_ok=True)
    Hd = F.CONFIG["hidden"]
    t_run = time.time()
    for k, (p, out) in enumerate(todo):
        if F.stop_requested(paths.stop):
            L(paths, "extract stopping (STOP file)", set=a.set, chunks_left=len(todo) - k)
            return 0
        t0 = time.time()
        z = F.load_cache_chunk(p)
        G = len(z["ref"])
        Pn = len(z["move"])
        sums = np.zeros((G * 2, Hd), dtype=np.float64)
        seg_all = z["gidx"].astype(np.int64) * 2 + z["side"].astype(np.int64)
        with torch.no_grad():
            for s in range(0, Pn, a.batch):
                e = min(Pn, s + a.batch)
                rows = np.arange(s, e)
                if a.device.startswith("cuda") and len(rows) < a.batch:
                    rows = np.concatenate([rows, np.full(a.batch - len(rows), e - 1)])
                bits = torch.as_tensor(z["bits"][rows], device=a.device)
                glob = torch.as_tensor(z["glob"][rows], device=a.device)
                pla = torch.as_tensor(z["pla"][rows].astype(np.int64), device=a.device)
                mv = torch.as_tensor(z["move"][rows].astype(np.int64), device=a.device)
                with F.autocast_ctx(a.device, prec):
                    tr = net.trunk(F.unpack_bits_torch(bits), glob, pla)
                h = net.head.pos_hidden(tr.float(), mv)[:e - s].double().cpu().numpy()
                np.add.at(sums, seg_all[s:e], h)
        counts = np.bincount(seg_all, minlength=G * 2).reshape(G, 2)
        if not np.array_equal(counts, z["counts"]):
            raise F.FTError(f"{p}: position counts differ from the cache record")
        with torch.no_grad():
            lg = head64.game_logits_from_sums(torch.as_tensor(sums.reshape(G, 2, Hd)), torch.as_tensor(counts),
                                              torch.as_tensor(z["colour"].astype(np.int64)))
            lp = torch.log_softmax(lg, 1).numpy()
        if not np.isfinite(lp).all():
            raise F.FTError(f"{p}: non-finite lp")
        pooled = (sums.reshape(G, 2, Hd) / np.maximum(counts, 1)[:, :, None]).astype(np.float32)
        tmp = out + ".tmp.npz"
        np.savez(tmp, format=np.array(F.FT_FORMAT), set=np.array(a.set), ref=z["ref"], lp=lp, pooled=pooled,
                 counts=counts.astype(np.int32), failed=z["failed"], n_moves=z["n_moves"],
                 best_sha256=np.array(best_sha), precision=np.array(prec), device=np.array(a.device),
                 tiny=np.array(bool(a.tiny)), created_utc=np.array(F.utc_now()), seconds=np.array(time.time() - t0),
                 code_sha256=np.array(json.dumps(F.code_shas(), sort_keys=True)))
        os.replace(tmp, out)
        el = time.time() - t_run
        L(paths, "extract chunk done", set=a.set, chunk=os.path.basename(p), games=G, positions=Pn,
          seconds=round(time.time() - t0, 1), positions_per_s=round(Pn / max(1e-9, time.time() - t0), 1),
          progress=f"{k + 1}/{len(todo)}", eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    L(paths, "extract done", set=a.set)
    return 0


def cmd_status(a):
    paths = Paths(a.out_root)
    out = {"root": paths.root, "stop_file": os.path.exists(paths.stop), "sets": os.path.exists(paths.sets)}
    for s in ("train", "holdout", "bank", "mono"):
        try:
            want = expected_chunks(paths, s)
        except Exception:
            want = None
        out[f"cache_{s}"] = f"{len(cache_files(paths, s))}/{want}"
    for s in ("bank", "mono"):
        d = os.path.join(paths.ft, s)
        out[f"ft_{s}"] = len([f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f]) \
            if os.path.isdir(d) else 0
    for f in ("bench.json", "ckpt_last.pt", "best.pt", "train_done.json"):
        out[f"train/{f}"] = os.path.exists(os.path.join(paths.train, f))
    print(json.dumps(out, indent=1))
    return 0


def cmd_gpu_free(_a):
    rows = F.other_gpu_compute()
    if rows:
        print(json.dumps({"gpu_free": False, "processes": rows}))
        return GPU_BUSY
    print(json.dumps({"gpu_free": True}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gpu=False):
        p.add_argument("--out-root", default=None, help="smoke root instead of data/prereg12")
        if gpu:
            p.add_argument("--device", default="cuda")
            p.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"])
            p.add_argument("--tiny", action="store_true", help="CPU smoke stand-in model (never reported)")
            p.add_argument("--allow-partial-cache", action="store_true", help="smoke only")

    p = sub.add_parser("sets")
    common(p)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-per-level", type=int, default=None, help="smoke only")
    p.add_argument("--holdout-per-level", type=int, default=None, help="smoke only")
    p = sub.add_parser("cache")
    common(p)
    p.add_argument("--set", required=True, choices=["train", "holdout", "bank", "mono"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--chunk", type=int, default=None)
    p = sub.add_parser("bench")
    common(p, gpu=True)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--batch-games", type=int, default=None)
    p.add_argument("--reps", type=int, default=20)
    p = sub.add_parser("train")
    common(p, gpu=True)
    p.add_argument("--max-steps", type=int, default=None, help="smoke only")
    p.add_argument("--max-hours", type=float, default=None, help="smoke only")
    p.add_argument("--eval-every", type=int, default=None, help="smoke only")
    p.add_argument("--ckpt-every", type=int, default=None, help="smoke only")
    p.add_argument("--head-only-steps", type=int, default=None, help="smoke only")
    p = sub.add_parser("extract")
    common(p, gpu=True)
    p.add_argument("--set", required=True, choices=["bank", "mono"])
    p.add_argument("--batch", type=int, default=1024)
    p = sub.add_parser("status")
    common(p)
    sub.add_parser("gpu-free")
    a = ap.parse_args(argv)
    smoke_flags = [getattr(a, k, None) for k in ("max_steps", "max_hours", "eval_every", "ckpt_every",
                                                 "head_only_steps", "train_per_level", "holdout_per_level")]
    if (any(x is not None for x in smoke_flags) or getattr(a, "tiny", False)
            or getattr(a, "allow_partial_cache", False)) and Paths(a.out_root).is_real():
        print("smoke options need --out-root (not data/prereg12)", file=sys.stderr)
        return 2
    fn = {"sets": cmd_sets, "cache": cmd_cache, "bench": cmd_bench, "train": cmd_train, "extract": cmd_extract,
          "status": cmd_status, "gpu-free": cmd_gpu_free}[a.cmd]
    try:
        return fn(a)
    except F.FTError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
