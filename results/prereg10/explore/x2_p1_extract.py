"""prereg10 explore (competition 1), step x2: KataGo HumanSL neutral-probe features (PREREG_08 P1 layout) for a pilot
set of fit players' BLACK games, to test whether P1 features identify players.

    Git Bash, repository root:
    X=results/prereg10/explore
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe     $X/x2_p1_extract.py sets  [--players-per-level 20 --games 40]
    PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe $X/x2_p1_extract.py run
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe     $X/x2_p1_extract.py status

Only fit players (tools/c10data.load_rows part 'fit', local_dir None); never data/local (hs_lib's guard is installed on
import of p1_lib). Players: fit players with >= 120 black games (the official-shape simulation population of PREREG_06),
per level the first N by sha256('c10-prereg10|p1pilot|<pid>'); per player G black games, the player's black games
(store file order) shuffled with random.Random('c10-prereg10|p1pilot|games|<pid>').

Reuses results/prereg08/p1/p1_lib.py + p1_extract.run_chunk unchanged (imported, nothing written there; bytecode off).
Output: data/prereg10/tmp/p1pilot/sets.json, chunk_<i>.npz (CHUNK games each; means [G,2,770] side 0 = target (black),
side 1 = opponent). Resumable: existing matching chunks are skipped. Yield the GPU: create data/prereg10/tmp/p1pilot/STOP.
`run` refuses to start while nvidia-smi lists another compute process.
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
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1_DIR = os.path.join(REPO, "results", "prereg08", "p1")
TOOLS = os.path.join(REPO, "tools")
for p in (P1_DIR, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402

import p1_lib as P  # noqa: E402  (pin check + data/local guard)
import p1_extract as PX  # noqa: E402

H = P.H
OUT = os.path.join(REPO, "data", "prereg10", "tmp", "p1pilot")
SETS = os.path.join(OUT, "sets.json")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
FEATS = os.path.join(REPO, "data", "prereg06", "feats")
SALT = "c10-prereg10|p1pilot"
CHUNK = 200


def cmd_sets(a):
    with np.load(os.path.join(FEATS, "meta.npz"), allow_pickle=False) as z:
        rank, gid, pid, color = z["rank"], z["game_id"], z["player_id"], z["color"]
    byb = {}
    for i in range(len(gid)):
        if color[i] == "B":
            byb.setdefault(str(pid[i]), []).append(i)
    lvl = {}
    for i in range(len(gid)):
        lvl.setdefault(str(pid[i]), str(rank[i]))
    games = []
    players = []
    for lv in H.LEVELS:
        el = sorted((p for p, rows in byb.items() if lvl[p] == lv and len(rows) >= 120),
                    key=lambda p: hashlib.sha256(("%s|%s" % (SALT, p)).encode()).hexdigest())
        for p in el[:a.players_per_level]:
            rows = list(byb[p])
            random.Random("%s|games|%s" % (SALT, p)).shuffle(rows)
            players.append(p)
            for r in rows[:a.games]:
                games.append({"rank": str(rank[r]), "game_id": str(gid[r]), "player_id": p, "color": "B",
                              "store_row": int(r)})
    sets = {"format": "c10-prereg10 p1pilot sets v1", "created_utc": H.utc_now(), "salt": SALT,
            "players_per_level": a.players_per_level, "games_per_player": a.games, "players": players,
            "games": games}
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(SETS):
        raise SystemExit("%s exists; not overwritten" % SETS)
    PX.write_json_atomic(SETS, sets)
    print(json.dumps({"players": len(players), "games": len(games), "sha256": P.sha_raw(SETS)}))
    return 0


def load_games():
    from c10data import load_rows
    with open(SETS, encoding="utf-8") as f:
        s = json.load(f)
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    sgf = {(r["rank"], r["game_id"]): r for r in rows}
    out = []
    for j, g in enumerate(s["games"]):
        r = sgf[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != "B":
            raise P.P1Error("pilot game %d: player or colour differs" % j)
        out.append({**r, "ref": j})
    return out


def cmd_run(a):
    import torch
    PX.CHUNK_ROOT = OUT          # run_log.jsonl goes to our folder
    games = load_games()
    if a.max_games:
        games = games[:a.max_games]
    refs = np.array([g["ref"] for g in games], np.int64)
    chunks = [(ci, s, min(len(games), s + CHUNK)) for ci, s in enumerate(range(0, len(games), CHUNK))]
    path = lambda ci: os.path.join(OUT, "chunk_%05d.npz" % ci)  # noqa: E731

    def ok(ci, s, e):
        try:
            with np.load(path(ci), allow_pickle=False) as z:
                return str(z["format"]) == P.CHUNK_FORMAT and np.array_equal(z["ref"], refs[s:e]) \
                    and str(z["meta_sha256"]) == P.neutral_table_sha()
        except Exception:
            return False
    todo = [c for c in chunks if not ok(*c)]
    PX.log("p10 pilot run start", games=len(games), chunks=len(chunks), todo=len(todo))
    if not todo:
        return 0
    procs, _ = H.other_compute_processes()
    if procs and not a.ignore_other_gpu_jobs:
        raise P.P1Error("another compute process on the GPU: %s" % procs)
    torch.backends.cudnn.benchmark = True
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=False)
    model = P.load_model(device="cuda")
    sc = P.ProbeScorer(model, device="cuda", precision="fp16", batch=1024, channels_last=True)
    t_run = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (ci, s, e) in enumerate(todo):
            if os.path.exists(os.path.join(OUT, "STOP")):
                PX.log("STOP file: stopping before the next chunk", chunks_left=len(todo) - k)
                break
            t0 = time.time()
            res, t_gpu = PX.run_chunk(pool, sc, games[s:e], 4096)
            secs = time.time() - t0
            tmp = path(ci) + ".tmp.npz"
            np.savez(tmp, format=np.array(P.CHUNK_FORMAT), set=np.array("p10pilot"), sides=np.array(PX.SIDES),
                     ref=refs[s:e], precision=np.array("fp16"), meta_sha256=np.array(P.neutral_table_sha()),
                     ckpt_bytes=np.array(ck["bytes"]), code_sha256=np.array(json.dumps(P.code_shas(), sort_keys=True)),
                     created_utc=np.array(H.utc_now()), seconds=np.array(secs), **res)
            os.replace(tmp, path(ci))
            el = time.time() - t_run
            npos = int(len(res["pos_side"]))
            PX.log("chunk done", chunk=ci, games=e - s, failed=int(res["failed"].sum()), positions=npos,
                   seconds=round(secs, 1), positions_per_s=round(npos / secs, 1), gpu_busy_frac=round(t_gpu / secs, 3),
                   progress="%d/%d" % (k + 1, len(todo)), eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
    sc.close()
    PX.log("p10 pilot run end", seconds=round(time.time() - t_run, 1))
    return 0


def cmd_status(_a):
    with open(SETS, encoding="utf-8") as f:
        n = len(json.load(f)["games"])
    have = [x for x in os.listdir(OUT) if x.startswith("chunk_") and x.endswith(".npz") and ".tmp" not in x]
    print(json.dumps({"games": n, "chunks": (n + CHUNK - 1) // CHUNK, "present": len(have),
                      "STOP": os.path.exists(os.path.join(OUT, "STOP"))}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--players-per-level", type=int, default=20)
    s.add_argument("--games", type=int, default=40)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=20)
    r.add_argument("--max-games", type=int, default=0)
    r.add_argument("--ignore-other-gpu-jobs", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()
    return {"sets": cmd_sets, "run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
