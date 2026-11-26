"""PREREG_07 pilot, step 1 (GPU, resumable): KataGo HumanSL log-likelihoods of inner and fit games.

    GPU python: .venv-gpu/Scripts/python.exe (CUDA torch); run from the repository root in Git Bash.
    PY=.venv-gpu/Scripts/python.exe; S=results/prereg07/explore/pilot_humansl
    PYTHONDONTWRITEBYTECODE=1 $PY $S/pilot_run.py plan                     -> data/prereg07/pilot/plan.json
    PYTHONDONTWRITEBYTECODE=1 $PY $S/pilot_run.py run --set fitmap         (mapping-training fit games)
    PYTHONDONTWRITEBYTECODE=1 $PY $S/pilot_run.py run --set bank           (inner bank: every fit / eval question)
    PYTHONDONTWRITEBYTECODE=1 $PY $S/pilot_run.py run --set mono           (the other inner games of the
                                                                             single-colour pseudo-questions)
    PYTHONDONTWRITEBYTECODE=1 $PY $S/pilot_run.py status

Sets (plan): every game is a training-player game read through tools/c10data.load_rows (part 'inner' or 'fit',
local_dir=None); hs_lib's guard makes any access under data/local raise.
    bank    the 11,813 inner bank games of results/prereg02/rank/pseudo_inner.json (all 636 bank players; every
            prereg02 inner fit and eval pseudo-question is drawn from them), as rows of inner_keys.npz.
    mono    the inner games used by the PREREG_03 single-colour pseudo-questions that are not in the bank (row lists
            from results/prereg05/explore/u1/u1_oof.npz mono_games; 72,908 games in all, 61,356 outside the bank),
            grouped by player, players in a seeded order cycling through the levels (MONO_ORDER_SEED): stopping the
            run early still leaves complete players, so pilot_fuse can report the covered single-colour questions.
    fitmap  FITMAP_PER_LEVEL fit-player games per level, one game from each of FITMAP_PER_LEVEL distinct fit players
            (random.Random(FITMAP_SEED)); used only to learn the per-game HS -> 10-level mapping (pilot_fuse.py).
Profiles: PILOT_PROFILES below (C++ preset rows: KGS source, date 2020-03-01, same rank for both sides).
Output: data/prereg07/pilot/<set>/chunk_<i>.npz (git-ignored), one per CHUNK games, written atomically (tmp file +
os.replace); `run` skips every chunk whose file exists and matches the plan (profiles, sides, rows). A crash or Ctrl+C
loses at most the chunk in progress; rerun the same command to continue. Each chunk holds per game: sums [G, 2, N]
(side 0 target, 1 opponent), counts [G, 2], n_moves, failed (replay errors, recorded not fatal), and per position
(target / opponent side): game index, turn, side, logp [P, N] float32.

Defaults (2026-10-03 measurements, README.md): fp16 autocast, batch 1024 (73 positions x 14 profiles), channels_last
on, 8 featurisation workers, 16 forward passes per featurised group. Every forward pass has the same shape (hs_lib
pads the last one), so cuDNN autotunes once per process (~8-11 s) instead of once per new shape.
GPU etiquette: `run` refuses to start while nvidia-smi lists another compute process (--ignore-other-gpu-jobs to
override). To yield the GPU (e.g. to C4), create the empty file <out-dir>/STOP: the run ends after the chunk in
progress; delete STOP and rerun the same command to continue. --stop-after-min M ends the run before the first chunk
that would start after M minutes (for a fixed night window).
Smoke runs: --max-games N --out-dir data/prereg07/pilot_smoke (refused for the real pilot folder).
"""

import argparse
import hashlib
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

import hs_lib as H  # noqa: E402  (installs the data/local guard)

OUT_DIR = os.path.join(H.REPO, "data", "prereg07", "pilot")
PLAN = os.path.join(OUT_DIR, "plan.json")
CHUNK_ROOT = OUT_DIR          # run --out-dir moves the chunk files and run_log.jsonl (benchmarks; never the pilot)
INNER_KEYS = os.path.join(H.REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
PSEUDO = os.path.join(H.REPO, "results", "prereg02", "rank", "pseudo_inner.json")
U1_OOF = os.path.join(H.REPO, "results", "prereg05", "explore", "u1", "u1_oof.npz")
SPLIT_CSV = os.path.join(H.REPO, "results", "split_v1", "players_split.csv")
TRAIN_DIR = os.path.join(H.REPO, "data", "training")
FORMAT = "c10-prereg07 pilot chunk v1"
CHUNK = 200
FITMAP_PER_LEVEL = 300
FITMAP_SEED = 20261003
MONO_ORDER_SEED = 20261004
# Pilot grid (14 profiles, 20k..9d): chosen from the parity curves (hs_parity.json c_curves) and the GPU grid scan
# (hs_gpu_bench.json grid_scan); see README.md. C++ preset names (KGS, 2020-03-01, same rank both sides).
PILOT_PROFILES = ["rank_20k", "rank_15k", "rank_12k", "rank_10k", "rank_8k", "rank_6k", "rank_4k", "rank_2k",
                  "rank_1d", "rank_2d", "rank_3d", "rank_5d", "rank_7d", "rank_9d"]


def sha256_file(path):
    return H.sha256_file(path)


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


# ================================================================ plan
def cmd_plan(a):
    from c10data import LEVELS, load_rows
    t0 = time.time()
    with np.load(INNER_KEYS, allow_pickle=False) as z:
        keys = {k: z[k] for k in ("rank", "game_id", "player_id", "color")}
    n_inner = len(keys["rank"])
    row_of = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    with open(PSEUDO, encoding="utf-8") as f:
        pseudo = json.load(f)
    bank = sorted({row_of[(p["rank"], gid)] for p in pseudo["players"] for gid in p["bank"]})
    with np.load(U1_OOF, allow_pickle=False) as z:
        mg = z["mono_games"]
        eg = z["eval_games"]
    mono_all = set(int(x) for x in mg[mg >= 0])
    if not set(int(x) for x in eg[eg >= 0]) <= set(bank):
        raise H.HSError("u1 eval games are not all bank games")
    # mono rows grouped by player, players in a seeded order that cycles through the 10 levels, so that a partial
    # run covers every single-colour question of a level-balanced set of players (pilot_fuse reports the coverage)
    rest = sorted(mono_all - set(bank))
    by_pl = {}
    for i in rest:
        by_pl.setdefault((str(keys["rank"][i]), str(keys["player_id"][i])), []).append(i)
    prng = random.Random(MONO_ORDER_SEED)
    queues = []
    for lv in LEVELS:
        pl = sorted(p for p in by_pl if p[0] == lv)
        prng.shuffle(pl)
        queues.append(pl)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    mono = [i for p in order for i in by_pl[p]]
    # fitmap: one game per distinct fit player, FITMAP_PER_LEVEL players per level
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    by_player = {}
    for r in rows:
        by_player.setdefault((r["rank"], r["player_id"]), []).append(r)
    rng = random.Random(FITMAP_SEED)
    fitmap = []
    for lv in LEVELS:
        players = sorted(p for (rk, p) in by_player if rk == lv)
        for p in rng.sample(players, FITMAP_PER_LEVEL):
            r = rng.choice(sorted(by_player[(lv, p)], key=lambda x: x["game_id"]))
            fitmap.append({"rank": lv, "game_id": r["game_id"], "player_id": p, "color": r["color"]})
    plan = {"format": "c10-prereg07 pilot plan v1", "created_utc": H.utc_now(),
            "profiles": PILOT_PROFILES, "chunk": CHUNK, "katago_commit": H.KATAGO_COMMIT,
            "checkpoint": {"path": os.path.relpath(H.CKPT_PATH, H.REPO).replace("\\", "/"), "bytes": H.CKPT_BYTES,
                           "sha256": H.CKPT_SHA256},
            "rules": H.RULES, "inputs_sha256": {"inner_keys.npz": sha256_file(INNER_KEYS),
                                                "pseudo_inner.json": sha256_file(PSEUDO),
                                                "u1_oof.npz": sha256_file(U1_OOF),
                                                "players_split.csv": sha256_file(SPLIT_CSV)},
            "inner_games": n_inner,
            "sets": {"bank": {"source": "inner", "rows": bank},
                     "mono": {"source": "inner", "rows": mono, "order_seed": MONO_ORDER_SEED,
                              "order": "grouped by player; players shuffled within level, levels interleaved"},
                     "fitmap": {"source": "fit", "games": fitmap, "seed": FITMAP_SEED,
                                "per_level": FITMAP_PER_LEVEL}},
            "counts": {"bank": len(bank), "mono": len(mono), "mono_all": len(mono_all),
                       "bank_or_mono": len(set(bank) | mono_all), "fitmap": len(fitmap)},
            "seconds": None}
    plan["seconds"] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    write_json_atomic(PLAN, plan)
    log("plan written", counts=plan["counts"], path=PLAN)
    return 0


def load_plan():
    if not os.path.exists(PLAN):
        raise H.HSError(f"{PLAN} missing: run `pilot_run.py plan` first")
    with open(PLAN, encoding="utf-8") as f:
        return json.load(f)


def set_games(plan, name):
    """-> list of dicts (rank, game_id, player_id, color, sgf_content, ref) in the set's order; ref = inner_keys row
    (inner sets) or the fitmap index."""
    from c10data import load_rows
    s = plan["sets"][name]
    if s["source"] == "inner":
        with np.load(INNER_KEYS, allow_pickle=False) as z:
            keys = {k: z[k] for k in ("rank", "game_id", "player_id", "color")}
        rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=None, check_counts=True)
        sgf = {(r["rank"], r["game_id"]): r for r in rows}
        out = []
        for i in s["rows"]:
            r = sgf[(str(keys["rank"][i]), str(keys["game_id"][i]))]
            if r["player_id"] != str(keys["player_id"][i]) or r["color"] != str(keys["color"][i]):
                raise H.HSError(f"inner row {i}: player or colour differs from inner_keys.npz")
            out.append({**r, "ref": int(i)})
        return out
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    sgf = {(r["rank"], r["game_id"]): r for r in rows}
    out = []
    for j, g in enumerate(s["games"]):
        r = sgf[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != g["color"]:
            raise H.HSError(f"fitmap game {j}: player or colour differs from the plan")
        out.append({**r, "ref": j})
    return out


# ================================================================ run
def _featurize(args):
    i, sgf, colour, sides = args
    try:
        return i, H.featurize_game(sgf, colour, sides=sides), None
    except H.HSError as e:
        return i, None, str(e)


def chunk_path(name, ci):
    return os.path.join(CHUNK_ROOT, name, f"chunk_{ci:05d}.npz")


def chunk_ok(path, refs, profiles, sides):
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["format"]) == FORMAT and list(z["profiles"]) == list(profiles)
                    and list(z["sides"]) == list(sides) and np.array_equal(z["ref"], refs))
    except Exception:
        return False


def run_chunk(pool, sc, games, sides, group_positions):
    """Featurise (worker processes) and score (GPU) one chunk -> dict of arrays."""
    G, N = len(games), sc.N
    sums = np.zeros((G, 2, N), dtype=np.float64)
    counts = np.zeros((G, 2), dtype=np.int32)
    n_moves = np.zeros(G, dtype=np.int16)
    failed = np.zeros(G, dtype=bool)
    errors = []
    pos_game, pos_turn, pos_side, pos_logp = [], [], [], []
    buf = []

    def flush():
        if not buf:
            return
        cat = {k: np.concatenate([f[k] for _, f in buf]) for k in ("bin", "glob", "legal", "move", "pla")}
        lp = sc.logp_moves(cat)
        o = 0
        for gi, f in buf:
            T = len(f["move"])
            part = lp[o:o + T]
            o += T
            s, c = H.game_sums(part, f["side"])
            sums[gi], counts[gi] = s, c
            pos_game.append(np.full(T, gi, dtype=np.int32))
            pos_turn.append(f["turn"])
            pos_side.append(f["side"])
            pos_logp.append(part)
        buf.clear()

    args = [(gi, g["sgf_content"], g["color"], tuple(sides)) for gi, g in enumerate(games)]
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
    cat = lambda xs, dt, shape: np.concatenate(xs) if xs else np.zeros(shape, dt)  # noqa: E731
    return {"sums": sums, "counts": counts, "n_moves": n_moves, "failed": failed, "errors": json.dumps(errors),
            "pos_game": cat(pos_game, np.int32, (0,)), "pos_turn": cat(pos_turn, np.int16, (0,)),
            "pos_side": cat(pos_side, np.int8, (0,)), "pos_logp": cat(pos_logp, np.float32, (0, N))}


def stop_requested(t_run, a):
    """True when STOP exists in the chunk folder (yield the GPU, e.g. to C4) or the --stop-after-min budget is used."""
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
        raise H.HSError("--max-games is for smoke runs only; give --out-dir another folder (e.g. data/prereg07/pilot_smoke)")
    plan = load_plan()
    profiles = H.profiles_from_names(plan["profiles"])
    sides = ("target", "opponent") if a.sides == "both" else ("target",)
    games = set_games(plan, a.set)
    if a.max_games:
        games = games[:a.max_games]
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    chunks = [(ci, s, min(len(games), s + plan["chunk"])) for ci, s in enumerate(range(0, len(games), plan["chunk"]))]
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    todo = [(ci, s, e) for ci, s, e in chunks
            if not chunk_ok(chunk_path(a.set, ci), refs_all[s:e], plan["profiles"], sides)]
    log("run start", set=a.set, games=len(games), chunks=len(chunks), todo=len(todo), sides=list(sides),
        precision=a.precision, batch=a.batch, workers=a.workers, channels_last=a.channels_last)
    if not todo:
        return 0
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise H.HSError("CUDA requested but torch.cuda.is_available() is False")
    if a.device.startswith("cuda") and not a.ignore_other_gpu_jobs:
        procs, _ = H.other_compute_processes()
        if procs:
            raise H.HSError(f"nvidia-smi lists another compute process on the GPU, not starting: {procs}")
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=a.full_hash)
    torch.backends.cudnn.benchmark = True
    model, _ = H.load_model(H.CKPT_PATH, device=a.device, check=False)
    sc = H.Scorer(model, profiles, device=a.device, precision=a.precision, batch=a.batch,
                  channels_last=a.channels_last)
    group_positions = max(1, a.batch // len(profiles)) * a.group_batches
    os.makedirs(os.path.join(CHUNK_ROOT, a.set), exist_ok=True)
    hs_sha = sha256_file(os.path.join(HERE, "hs_lib.py"))
    t_run = time.time()
    done_evals = 0
    stopped = None
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (ci, s, e) in enumerate(todo):
            stopped = stop_requested(t_run, a)
            if stopped:
                log("stopping before the next chunk (rerun the same command to continue)", reason=stopped,
                    set=a.set, chunks_done=k, chunks_left=len(todo) - k)
                break
            t0 = time.time()
            res = run_chunk(pool, sc, games[s:e], sides, group_positions)
            secs = time.time() - t0
            evals = int(len(res["pos_side"]) * len(profiles))
            done_evals += evals
            path = chunk_path(a.set, ci)
            tmp = path + ".tmp.npz"
            np.savez(tmp, format=np.array(FORMAT), profiles=np.array(plan["profiles"]), sides=np.array(sides),
                     ref=refs_all[s:e], set=np.array(a.set), precision=np.array(a.precision),
                     batch=np.array(a.batch), channels_last=np.array(a.channels_last),
                     torch=np.array(torch.__version__), ckpt_bytes=np.array(ck["bytes"]),
                     ckpt_sha256=np.array(ck.get("sha256", "not hashed in this run")),
                     katago_commit=np.array(H.KATAGO_COMMIT), hs_lib_sha256=np.array(hs_sha),
                     created_utc=np.array(H.utc_now()), seconds=np.array(secs), **res)
            os.replace(tmp, path)
            el = time.time() - t_run
            log("chunk done", set=a.set, chunk=ci, games=e - s, failed=int(res["failed"].sum()), evals=evals,
                seconds=round(secs, 1), evals_per_s=round(evals / secs, 1), progress=f"{k + 1}/{len(todo)}",
                eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    log("run end", set=a.set, chunks=len(todo), evals=done_evals, seconds=round(time.time() - t_run, 1),
        stopped=stopped)
    return 0


def cmd_status(a):
    plan = load_plan()
    out = {}
    for name in plan["sets"]:
        n = plan["counts"][name]
        nch = (n + plan["chunk"] - 1) // plan["chunk"]
        d = os.path.join(os.path.abspath(a.out_dir), name)
        have = sorted(x for x in os.listdir(d) if x.startswith("chunk_") and x.endswith(".npz")
                      and ".tmp" not in x) if os.path.isdir(d) else []
        out[name] = {"games": n, "chunks": nch, "chunks_present": len(have)}
    print(json.dumps(out, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=["bank", "mono", "fitmap"])
    r.add_argument("--sides", default="target", choices=["target", "both"])
    r.add_argument("--device", default="cuda")
    r.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    r.add_argument("--batch", type=int, default=1024, help="network evaluations per forward pass")
    r.add_argument("--group-batches", type=int, default=16, help="forward passes per featurised game group")
    r.add_argument("--workers", type=int, default=8, help="featurisation worker processes")
    r.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                   help="channels_last memory format (default on: 1,805 vs 1,451 evals/s, hs_gpu_bench.json)")
    r.add_argument("--max-chunks", type=int, default=0, help="stop after this many chunks (0 = all)")
    r.add_argument("--max-games", type=int, default=0, help="smoke runs only: first N games of the set")
    r.add_argument("--stop-after-min", type=float, default=0,
                   help="do not start a new chunk after this many minutes (0 = no limit); rerun to continue")
    r.add_argument("--ignore-other-gpu-jobs", action="store_true",
                   help="start even when nvidia-smi lists another compute process (default: refuse)")
    r.add_argument("--full-hash", action="store_true", help="re-hash the checkpoint (SHA-256) before running")
    r.add_argument("--out-dir", default=OUT_DIR, help="chunk folder (default data/prereg07/pilot)")
    st = sub.add_parser("status")
    st.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    return {"plan": cmd_plan, "run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
