"""prereg08 P1, step 1 (GPU, resumable): neutral-metadata HumanSL probe features of probe / bank / mono games.

    Git Bash, repository root; GPU python .venv-gpu/Scripts/python.exe, CPU python .venv/Scripts/python.exe
    S=results/prereg08/p1
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe $S/p1_extract.py sets    -> data/prereg08/p1/sets.json
    PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe $S/p1_extract.py run --set probe   (then bank, mono)
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe $S/p1_extract.py status
    smoke: ... run --set probe --max-games 400 --out-dir data/prereg08/p1_smoke   (refused for data/prereg08/p1)
           ... check-smoke --out-dir data/prereg08/p1_smoke --games 6               (CPU fp32 recompute vs the chunk)

Sets (sets.json; p1_plan.json pins its SHA-256):
    probe  PROBE_PER_LEVEL games per level from fit players only (never inner, never validation; tools/c10data
           load_rows part 'fit', local_dir None): players of the level in a seeded order, round robin, each turn one
           random not-yet-taken game of that player (random.Random(PROBE_SEED)), until the level has its quota. So the
           probe spreads over every fit player of the level (~5 games each). Used only to fit the P1 mapping.
    bank   the PREREG_07 pilot's bank rows (11,813 inner games; every inner fit / eval pseudo-question game), taken from
           data/prereg07/pilot/plan.json (SHA-256 pinned below), same order.
    mono   the pilot's mono rows (61,356 inner games of the single-colour pseudo-questions not in the bank), same order.
Output: data/prereg08/p1/<set>/chunk_<i>.npz (git-ignored), CHUNK games each, atomic (tmp + os.replace); `run` skips a
chunk whose file exists and matches (format, set, refs, sides, neutral-metadata SHA). Per game: means [G, 2, 770] float32
(side 0 target, 1 opponent; layout in p1_lib.F_SLICES), counts [G, 2], n_moves, failed (replay errors recorded, not
fatal); per position: game index, side, turn, log p and entropy.

GPU etiquette (as pilot_run.py): `run --device cuda` refuses to start while nvidia-smi lists another compute process
(Type 'C'); --ignore-other-gpu-jobs overrides. To yield the GPU (e.g. to C4) create the empty file
data/prereg08/p1/STOP: the run ends after the chunk in progress; delete STOP and rerun the same command to continue.
--stop-after-min M: no new chunk after M minutes. Importing p1_lib installs hs_lib's data/local guard.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p1_lib as P  # noqa: E402  (pin check, data/local guard)

H = P.H
REPO = P.REPO
OUT_DIR = os.path.join(REPO, "data", "prereg08", "p1")
SETS = os.path.join(OUT_DIR, "sets.json")
PILOT_PLAN = os.path.join(REPO, "data", "prereg07", "pilot", "plan.json")
PILOT_PLAN_SHA = "8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946"
INNER_KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SETS_FORMAT = "c10-prereg08 p1 sets v1"
PROBE_PER_LEVEL = 3000
PROBE_SEED = 20261006
CHUNK = 200
SIDES = ("target", "opponent")
SET_NAMES = ("probe", "bank", "mono")
CHUNK_ROOT = OUT_DIR


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def log(msg, **kv):
    rec = {"time_utc": H.utc_now(), "msg": msg, **kv}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(CHUNK_ROOT, exist_ok=True)
    with open(os.path.join(CHUNK_ROOT, "run_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ================================================================ sets
def build_probe(rows, per_level=PROBE_PER_LEVEL, seed=PROBE_SEED):
    """rows: fit rows (dicts rank, game_id, player_id, color). -> list of probe game dicts (deterministic)."""
    by_player = {}
    for r in rows:
        by_player.setdefault((r["rank"], r["player_id"]), []).append(r)
    rng = random.Random(seed)
    out = []
    for lv in H.LEVELS:
        players = sorted(p for (rk, p) in by_player if rk == lv)
        rng.shuffle(players)
        pools = {p: sorted(by_player[(lv, p)], key=lambda x: x["game_id"]) for p in players}
        taken = 0
        while taken < per_level:
            moved = False
            for p in players:
                if taken >= per_level:
                    break
                pool = pools[p]
                if not pool:
                    continue
                r = pool.pop(rng.randrange(len(pool)))
                out.append({"rank": lv, "game_id": r["game_id"], "player_id": p, "color": r["color"]})
                taken += 1
                moved = True
            if not moved:
                raise P.P1Error(f"level {lv}: only {taken} fit games, {per_level} wanted")
    return out


def cmd_sets(a):
    from c10data import load_rows
    t0 = time.time()
    if P.sha_raw(PILOT_PLAN) != PILOT_PLAN_SHA:
        raise P.P1Error(f"{PILOT_PLAN}: SHA-256 differs from the pinned pilot plan")
    with open(PILOT_PLAN, encoding="utf-8") as f:
        pplan = json.load(f)
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    probe = build_probe(rows)
    inner_players = {r["player_id"] for r in load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=None,
                                                        check_counts=True)}
    if {g["player_id"] for g in probe} & inner_players:
        raise P.P1Error("probe set contains an inner player")
    if len({(g["rank"], g["game_id"]) for g in probe}) != len(probe):
        raise P.P1Error("probe set has a duplicate game")
    per_player = {}
    for g in probe:
        per_player[(g["rank"], g["player_id"])] = per_player.get((g["rank"], g["player_id"]), 0) + 1
    sets = {"format": SETS_FORMAT, "created_utc": H.utc_now(),
            "probe": {"source": "fit", "seed": PROBE_SEED, "per_level": PROBE_PER_LEVEL, "games": probe},
            "bank": {"source": "inner", "rows": pplan["sets"]["bank"]["rows"],
                     "from": "data/prereg07/pilot/plan.json sets.bank.rows"},
            "mono": {"source": "inner", "rows": pplan["sets"]["mono"]["rows"],
                     "from": "data/prereg07/pilot/plan.json sets.mono.rows"},
            "inputs_sha256": {"pilot_plan.json": PILOT_PLAN_SHA, "inner_keys.npz": P.sha_raw(INNER_KEYS),
                              "players_split.csv": P.sha_raw(SPLIT_CSV)},
            "counts": {"probe": len(probe), "probe_players": len(per_player),
                       "probe_games_per_player_max": max(per_player.values()),
                       "bank": len(pplan["sets"]["bank"]["rows"]), "mono": len(pplan["sets"]["mono"]["rows"])}}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.abspath(a.out) if a.out else SETS
    if os.path.exists(out):
        raise P.P1Error(f"{out} exists (p1_plan.json pins its SHA-256, which includes created_utc); not overwritten. "
                        "Use --out <other path> to rebuild and compare the game lists")
    write_json_atomic(out, sets)
    print(json.dumps({"written": out, "sha256": P.sha_raw(out), "counts": sets["counts"],
                      "seconds": round(time.time() - t0, 1)}, indent=1))
    return 0


def load_sets(path=SETS):
    if not os.path.exists(path):
        raise P.P1Error(f"{path} missing: run `p1_extract.py sets` first")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("format") != SETS_FORMAT:
        raise P.P1Error(f"{path}: format {s.get('format')!r}")
    return s


def set_games(sets, name):
    """-> list of row dicts (rank, game_id, player_id, color, sgf_content, ref); ref = inner_keys row (bank / mono)
    or the probe index."""
    from c10data import load_rows
    s = sets[name]
    if s["source"] == "inner":
        with np.load(INNER_KEYS, allow_pickle=False) as z:
            keys = {k: z[k] for k in ("rank", "game_id", "player_id", "color")}
        rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=None, check_counts=True)
        sgf = {(r["rank"], r["game_id"]): r for r in rows}
        out = []
        for i in s["rows"]:
            r = sgf[(str(keys["rank"][i]), str(keys["game_id"][i]))]
            if r["player_id"] != str(keys["player_id"][i]) or r["color"] != str(keys["color"][i]):
                raise P.P1Error(f"inner row {i}: player or colour differs from inner_keys.npz")
            out.append({**r, "ref": int(i)})
        return out
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    sgf = {(r["rank"], r["game_id"]): r for r in rows}
    out = []
    for j, g in enumerate(s["games"]):
        r = sgf[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != g["color"]:
            raise P.P1Error(f"probe game {j}: player or colour differs from sets.json")
        out.append({**r, "ref": j})
    return out


# ================================================================ run
def _featurize(args):
    i, sgf, colour = args
    try:
        return i, P.featurize_game(sgf, colour, sides=SIDES), None
    except H.HSError as e:
        return i, None, str(e)


def chunk_path(name, ci, root=None):
    return os.path.join(root or CHUNK_ROOT, name, f"chunk_{ci:05d}.npz")


def chunk_ok(path, name, refs):
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["format"]) == P.CHUNK_FORMAT and str(z["set"]) == name
                    and list(z["sides"]) == list(SIDES) and str(z["meta_sha256"]) == P.neutral_table_sha()
                    and np.array_equal(z["ref"], refs))
    except Exception:
        return False


def run_chunk(pool, sc, games, group_positions):
    G = len(games)
    means = np.zeros((G, 2, P.F_DIM), dtype=np.float32)
    counts = np.zeros((G, 2), dtype=np.int32)
    n_moves = np.zeros(G, dtype=np.int16)
    failed = np.zeros(G, dtype=bool)
    errors = []
    pos_game, pos_side, pos_turn, pos_lp, pos_ent = [], [], [], [], []
    buf = []
    t_gpu = [0.0]

    def flush():
        if not buf:
            return
        cat = {k: np.concatenate([f[k] for _, f in buf]) for k in ("bin", "glob", "legal", "move", "pla")}
        t = time.time()
        pf = sc.position_features(cat)
        t_gpu[0] += time.time() - t
        o = 0
        for gi, f in buf:
            T = len(f["move"])
            part = pf[o:o + T]
            o += T
            m, c = P.game_means(part, f["side"])
            means[gi], counts[gi] = m.astype(np.float32), c
            pos_game.append(np.full(T, gi, dtype=np.int32))
            pos_side.append(f["side"])
            pos_turn.append(f["turn"])
            pos_lp.append(part[:, 768])
            pos_ent.append(part[:, 769])
        buf.clear()

    args = [(gi, g["sgf_content"], g["color"]) for gi, g in enumerate(games)]
    pending = 0
    for gi, f, err in pool.imap(_featurize, args, chunksize=2):
        if err is not None:
            failed[gi] = True
            errors.append({"game": gi, "error": err})
            continue
        n_moves[gi] = f["n_moves"]
        buf.append((gi, f))
        pending += len(f["move"])
        if pending >= group_positions:
            flush()
            pending = 0
    flush()
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    return {"means": means, "counts": counts, "n_moves": n_moves, "failed": failed, "errors": json.dumps(errors),
            "pos_game": cat(pos_game, np.int32), "pos_side": cat(pos_side, np.int8),
            "pos_turn": cat(pos_turn, np.int16), "pos_logp": cat(pos_lp, np.float32),
            "pos_entropy": cat(pos_ent, np.float32)}, t_gpu[0]


def stop_requested(t_run, a):
    if os.path.exists(os.path.join(CHUNK_ROOT, "STOP")):
        return "STOP file"
    if a.stop_after_min and time.time() - t_run > a.stop_after_min * 60:
        return f"--stop-after-min {a.stop_after_min}"
    return None


def cmd_run(a):
    global CHUNK_ROOT
    import torch
    CHUNK_ROOT = os.path.abspath(a.out_dir)
    if a.max_games and os.path.normcase(CHUNK_ROOT) == os.path.normcase(os.path.abspath(OUT_DIR)):
        raise P.P1Error("--max-games is for smoke runs only; give --out-dir another folder "
                        "(e.g. data/prereg08/p1_smoke)")
    sets = load_sets(a.sets)
    games = set_games(sets, a.set)
    if a.max_games:
        games = games[:a.max_games]
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    chunks = [(ci, s, min(len(games), s + CHUNK)) for ci, s in enumerate(range(0, len(games), CHUNK))]
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    todo = [(ci, s, e) for ci, s, e in chunks if not chunk_ok(chunk_path(a.set, ci), a.set, refs_all[s:e])]
    log("run start", set=a.set, games=len(games), chunks=len(chunks), todo=len(todo), precision=a.precision,
        batch=a.batch, workers=a.workers, channels_last=a.channels_last, device=a.device)
    if not todo:
        return 0
    stopped = stop_requested(time.time(), a)
    if stopped:
        log("not starting", reason=stopped, set=a.set)
        return 0
    if a.device.startswith("cuda"):
        if not a.ignore_other_gpu_jobs:
            procs, _ = H.other_compute_processes()        # before any CUDA call of this process
            if procs:
                raise P.P1Error(f"nvidia-smi lists another compute process on the GPU, not starting: {procs}")
        if not torch.cuda.is_available():
            raise P.P1Error("CUDA requested but torch.cuda.is_available() is False")
        torch.backends.cudnn.benchmark = True
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=a.full_hash)
    model = P.load_model(device=a.device)
    sc = P.ProbeScorer(model, device=a.device, precision=a.precision, batch=a.batch, channels_last=a.channels_last)
    group_positions = a.batch * a.group_batches
    os.makedirs(os.path.join(CHUNK_ROOT, a.set), exist_ok=True)
    shas = P.code_shas()
    t_run = time.time()
    done_pos = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (ci, s, e) in enumerate(todo):
            stopped = stop_requested(t_run, a)
            if stopped:
                log("stopping before the next chunk (rerun the same command to continue)", reason=stopped,
                    set=a.set, chunks_done=k, chunks_left=len(todo) - k)
                break
            t0 = time.time()
            res, t_gpu = run_chunk(pool, sc, games[s:e], group_positions)
            secs = time.time() - t0
            npos = int(len(res["pos_side"]))
            done_pos += npos
            path = chunk_path(a.set, ci)
            tmp = path + ".tmp.npz"
            np.savez(tmp, format=np.array(P.CHUNK_FORMAT), set=np.array(a.set), sides=np.array(SIDES),
                     ref=refs_all[s:e], precision=np.array(a.precision), batch=np.array(a.batch),
                     channels_last=np.array(a.channels_last), device=np.array(a.device),
                     torch=np.array(torch.__version__), ckpt_bytes=np.array(ck["bytes"]),
                     ckpt_sha256=np.array(ck.get("sha256", "not hashed in this run")),
                     katago_commit=np.array(H.KATAGO_COMMIT), meta_sha256=np.array(P.neutral_table_sha()),
                     code_sha256=np.array(json.dumps(shas, sort_keys=True)),
                     created_utc=np.array(H.utc_now()), seconds=np.array(secs), **res)
            os.replace(tmp, path)
            el = time.time() - t_run
            log("chunk done", set=a.set, chunk=ci, games=e - s, failed=int(res["failed"].sum()), positions=npos,
                seconds=round(secs, 1), positions_per_s=round(npos / secs, 1),
                gpu_busy_frac=round(t_gpu / secs, 3), progress=f"{k + 1}/{len(todo)}",
                eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    sc.close()
    log("run end", set=a.set, chunks=len(todo), positions=done_pos, seconds=round(time.time() - t_run, 1),
        stopped=stopped)
    return 0


def cmd_status(a):
    sets = load_sets(a.sets)
    out = {}
    for name in SET_NAMES:
        n = sets["counts"][name]
        nch = (n + CHUNK - 1) // CHUNK
        d = os.path.join(os.path.abspath(a.out_dir), name)
        have = sorted(x for x in os.listdir(d) if x.startswith("chunk_") and x.endswith(".npz")
                      and ".tmp" not in x) if os.path.isdir(d) else []
        out[name] = {"games": n, "chunks": nch, "chunks_present": len(have)}
    out["STOP_file"] = os.path.exists(os.path.join(os.path.abspath(a.out_dir), "STOP"))
    print(json.dumps(out, indent=1))
    return 0


def cmd_check_smoke(a):
    """Recompute the first --games games of chunk 0 of a smoke run on the CPU in fp32 and compare."""
    root = os.path.abspath(a.out_dir)
    path = chunk_path(a.set, 0, root)
    sets = load_sets(a.sets)
    games = set_games(sets, a.set)
    with np.load(path, allow_pickle=False) as z:
        ref, means, counts, prec = z["ref"], z["means"], z["counts"], str(z["precision"])
    model = P.load_model(device="cpu")
    sc = P.ProbeScorer(model, device="cpu", precision="fp32", batch=256)
    rows = []
    for gi in range(min(a.games, len(ref))):
        g = games[gi]
        if g["ref"] != ref[gi]:
            raise P.P1Error("smoke chunk refs differ from the set")
        f = P.featurize_game(g["sgf_content"], g["color"])
        pf = sc.position_features(f)
        m, c = P.game_means(pf, f["side"])
        if not np.array_equal(c, counts[gi]):
            raise P.P1Error(f"game {gi}: counts differ")
        d = np.abs(m - means[gi].astype(np.float64))
        rows.append({"game": gi, "positions": int(c.sum()),
                     **{k: float(d[:, s:e].max()) for k, (s, e) in P.F_SLICES.items()},
                     "rel_trunk": float(d[:, :768].max() / max(1e-12, np.abs(m[:, :768]).max()))})
    rep = {"what": "P1 smoke: GPU chunk (per-game means) vs CPU fp32 recompute", "chunk": path,
           "chunk_precision": prec, "games": rows,
           "max": {k: max(r[k] for r in rows) for k in list(P.F_SLICES) + ["rel_trunk"]}}
    print(json.dumps(rep, indent=1))
    if a.report:
        write_json_atomic(a.report, rep)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--out", default=None, help="default data/prereg08/p1/sets.json")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=SET_NAMES)
    r.add_argument("--sets", default=SETS)
    r.add_argument("--device", default="cuda")
    r.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    r.add_argument("--batch", type=int, default=1024, help="positions per forward pass")
    r.add_argument("--group-batches", type=int, default=4, help="forward passes per featurised game group")
    r.add_argument("--workers", type=int, default=20, help="featurisation worker processes")
    r.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--max-chunks", type=int, default=0)
    r.add_argument("--max-games", type=int, default=0, help="smoke runs only: first N games of the set")
    r.add_argument("--stop-after-min", type=float, default=0)
    r.add_argument("--ignore-other-gpu-jobs", action="store_true")
    r.add_argument("--full-hash", action="store_true")
    r.add_argument("--out-dir", default=OUT_DIR)
    st = sub.add_parser("status")
    st.add_argument("--sets", default=SETS)
    st.add_argument("--out-dir", default=OUT_DIR)
    cs = sub.add_parser("check-smoke")
    cs.add_argument("--set", default="probe", choices=SET_NAMES)
    cs.add_argument("--sets", default=SETS)
    cs.add_argument("--out-dir", required=True)
    cs.add_argument("--games", type=int, default=6)
    cs.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    return {"sets": cmd_sets, "run": cmd_run, "status": cmd_status, "check-smoke": cmd_check_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
