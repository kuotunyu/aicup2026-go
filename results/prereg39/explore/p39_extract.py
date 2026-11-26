"""prereg39 explore: KataGo HumanSL neutral-probe (P1) extraction of the listed games (PLAN.md 1, 2, 8).

    GPUPY results/prereg39/explore/p39_extract.py run [--sets L,D,pub2,priv2,E,E2] [--workers 8]
    PY    results/prereg39/explore/p39_extract.py status
    smoke (CPU, fake scorer, writes only under data/prereg39/tmp/<name>):
    GPUPY results/prereg39/explore/p39_extract.py run --smoke <name> --sets L,pub2 --max-games 20 --workers 2

Only the games of data/prereg39/lists/<set>.json without a reusable PREREG_10 mock chunk (mock_ref < 0) are
extracted, in list order, 200 per chunk -> data/prereg39/p1/<set>/chunk_%05d.npz (ref = index in the list). The
chunk layout is PREREG_10's (results/prereg08/p1/p1_lib.py and p1_extract.run_chunk imported unchanged; side 0 =
the listed player's colour). Finished chunks are skipped; a chunk that exists but does not match is never
overwritten. Yield the GPU: create data/prereg39/p1/STOP (stops before the next chunk); delete it and run the same
line again to resume. Refuses to start while nvidia-smi lists another compute process.
Only fit players (tools/c10data.load_rows part 'fit', local_dir None) and the official new-pool files; the
validation answers folder is never read (the hs_lib guard is installed when p1_lib is imported). Official games:
the root properties that the training games do not have (HD) are removed with tools/root_props.classify_game
(PREREG_22), so KataGo's parser sees the training form.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1_DIR = os.path.join(REPO, "results", "prereg08", "p1")
for p in (P1_DIR, os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg36", "explore"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg39")
LISTS = os.path.join(DATA, "lists")
OUT = os.path.join(DATA, "p1")
TMP = os.path.join(DATA, "tmp")
RECORD = os.path.join(HERE, "lists_record.json")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
ORDER = ("L", "D", "pub2", "priv2", "E", "E2")
OFFICIAL = {"pub2": "public", "priv2": "private"}
CHUNK = 200
ROOT_STATUS = {}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_list(name):
    """-> (games of the list, SHA-256 of the list file); the SHA must be the committed lists_record.json's."""
    p = os.path.join(LISTS, name + ".json")
    with open(RECORD, encoding="utf-8") as f:
        want = json.load(f)["lists_sha256"][name]
    sha = sha_file(p)
    if sha != want:
        raise SystemExit("%s differs from lists_record.json" % p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)["games"], sha


def fresh_games(name, games, fit_rows):
    """The games to extract: [{'sgf_content', 'color', 'ref'}] (ref = index in the list), mock_ref < 0 only."""
    out = []
    if name in OFFICIAL:
        import p39_lists as PL
        items = PL.official_items(OFFICIAL[name])
        if len(items) != len(games):
            raise SystemExit("%s: the pool files changed" % name)
        import root_props as RP          # PREREG_22: root properties the training games do not have (HD ...)
        n = {}
        for j, (g, (sgf, c)) in enumerate(zip(games, items)):
            if g["item"] != j or g["color"] != c:
                raise SystemExit("%s item %d differs from the list" % (name, j))
            status, new, _detail = RP.classify_game(sgf)
            n[status] = n.get(status, 0) + 1
            out.append({"sgf_content": new if status == "root_props" else sgf, "color": c, "ref": j})
        ROOT_STATUS[name] = n
        return out
    for j, g in enumerate(games):
        if g.get("mock_ref", -1) >= 0:
            continue
        r = fit_rows[(g["rank"], g["game_id"], g["player_id"])]
        if r["color"] != g["color"]:
            raise SystemExit("%s game %d: colour differs" % (name, j))
        out.append({"sgf_content": r["sgf_content"], "color": g["color"], "ref": j})
    return out


def chunk_plan(n):
    return [(ci, s, min(n, s + CHUNK)) for ci, s in enumerate(range(0, n, CHUNK))]


def cmd_status(_a):
    rep = {}
    for name in ORDER:
        games, _sha = read_list(name)
        n = len(games) if name in OFFICIAL else sum(1 for g in games if g.get("mock_ref", -1) < 0)
        d = os.path.join(OUT, name)
        have = [x for x in os.listdir(d) if x.startswith("chunk_") and x.endswith(".npz") and ".tmp" not in x] \
            if os.path.isdir(d) else []
        rep[name] = {"games_to_extract": n, "chunks": (n + CHUNK - 1) // CHUNK, "present": len(have)}
    rep["STOP"] = os.path.exists(os.path.join(OUT, "STOP"))
    print(json.dumps(rep, indent=1))
    return 0


def cmd_run(a):
    sets = [s for s in a.sets.split(",") if s]
    if any(s not in ORDER for s in sets) or sets != [s for s in ORDER if s in sets]:
        raise SystemExit("--sets must be a subset of %s in that order" % ",".join(ORDER))
    if a.smoke:
        if not a.max_games:
            raise SystemExit("--smoke needs --max-games")
        root = os.path.join(TMP, a.smoke)
    else:
        if a.max_games or a.fake_scorer:
            raise SystemExit("--max-games / --fake-scorer are for --smoke only")
        root = OUT
    stop = os.path.join(root, "STOP")
    fake = bool(a.smoke) and a.fake_scorer
    fit_rows = None
    if any(s not in OFFICIAL for s in sets):
        from c10data import load_rows
        rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
        fit_rows = {(r["rank"], r["game_id"], r["player_id"]): r for r in rows}
        del rows
    import p1_lib as P            # pin check + the answers-folder guard
    import p1_extract as PX
    H = P.H
    os.makedirs(root, exist_ok=True)
    PX.CHUNK_ROOT = root
    sc = None
    t_all = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name in sets:
            games_l, lsha = read_list(name)
            games = fresh_games(name, games_l, fit_rows)
            if a.max_games:
                games = games[:a.max_games]
            refs = np.array([g["ref"] for g in games], np.int64)
            d = os.path.join(root, name)
            os.makedirs(d, exist_ok=True)
            set_name = "p39" + name
            path = lambda ci: os.path.join(d, "chunk_%05d.npz" % ci)  # noqa: E731

            def ok(ci, s, e):
                try:
                    with np.load(path(ci), allow_pickle=False) as z:
                        return str(z["format"]) == P.CHUNK_FORMAT and np.array_equal(z["ref"], refs[s:e]) \
                            and str(z["meta_sha256"]) == P.neutral_table_sha() and str(z["set"]) == set_name \
                            and str(z["list_sha256"]) == lsha
                except Exception:
                    return False
            todo = []
            for ci, s, e in chunk_plan(len(games)):
                if not ok(ci, s, e):
                    if os.path.exists(path(ci)):
                        raise SystemExit("%s chunk %d exists but does not match (never overwritten)" % (name, ci))
                    todo.append((ci, s, e))
            PX.log("p39 extract start", set=name, games=len(games), todo=len(todo), smoke=bool(a.smoke),
                   root_status=ROOT_STATUS.get(name))
            if not todo:
                continue
            if sc is None:
                if fake:
                    import p10_p1
                    sc, ck, prec = p10_p1.FakeProbeScorer(), {"bytes": 0}, "fake-fp32"
                else:
                    procs, _ = H.other_compute_processes()
                    if procs:
                        raise SystemExit("another compute process on the GPU: %s" % procs)
                    import torch
                    torch.backends.cudnn.benchmark = True
                    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=False)
                    model = P.load_model(device="cuda")
                    sc, prec = P.ProbeScorer(model, device="cuda", precision="fp16", batch=1024,
                                             channels_last=True), "fp16"
            t_run, stopped = time.time(), False
            for k, (ci, s, e) in enumerate(todo):
                if os.path.exists(stop):
                    PX.log("STOP file: stopping before the next chunk", set=name, chunks_left=len(todo) - k)
                    stopped = True
                    break
                t0 = time.time()
                res, t_gpu = PX.run_chunk(pool, sc, games[s:e], 4096)
                secs = time.time() - t0
                tmp = path(ci) + ".tmp.npz"
                np.savez(tmp, format=np.array(P.CHUNK_FORMAT), set=np.array(set_name), sides=np.array(PX.SIDES),
                         ref=refs[s:e], precision=np.array(prec), meta_sha256=np.array(P.neutral_table_sha()),
                         ckpt_bytes=np.array(ck["bytes"]),
                         code_sha256=np.array(json.dumps(P.code_shas(), sort_keys=True)),
                         list_sha256=np.array(lsha), created_utc=np.array(H.utc_now()), seconds=np.array(secs),
                         **res)
                os.replace(tmp, path(ci))
                el = time.time() - t_run
                npos = int(len(res["pos_side"]))
                PX.log("chunk done", set=name, chunk=ci, games=e - s, failed=int(res["failed"].sum()),
                       positions=npos, seconds=round(secs, 1), positions_per_s=round(npos / max(secs, 1e-9), 1),
                       gpu_busy_frac=round(t_gpu / max(secs, 1e-9), 3), progress="%d/%d" % (k + 1, len(todo)),
                       eta_min=round(el / (k + 1) * (len(todo) - k - 1) / 60, 1))
            if stopped:
                break
    if sc is not None:
        sc.close()
    PX.log("p39 extract end", seconds=round(time.time() - t_all, 1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sets", default=",".join(ORDER))
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--smoke", default=None, help="name of a folder under data/prereg39/tmp")
    r.add_argument("--max-games", type=int, default=None)
    r.add_argument("--fake-scorer", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
