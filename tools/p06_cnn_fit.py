"""PREREG_06 step S4 (competition 1): per-game CNN embeddings of the fit players' black games on the local RTX 4090.

spec.md section 4 / PREREG_06 section 2.5. Run with the GPU environment, from the repository root (Git Bash):

    PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe tools/p06_cnn_fit.py [--smoke N] [--workers 6]

What it computes (the same code path as the prereg02 / test02 player notebooks):
  * games: every black game of the fit players with >= 105 black games (counted before any alternation check), in
    file order exactly as results/prereg06/explore/c1_headroom/c1h_lib.build_cache (1,082 players, 232,165 games);
    read through tools/c10data.load_rows(part="fit", local_dir=None) (player / game counts checked against EXPECTED);
  * sgf_windows: third_party/aicup2026_tutorial/utils.py SGFParsePlayerIdentification().extract_features(sgf, 'b',
    mode='inference') -> uint8; parse failure or no window -> emb_sum 0, n_windows 0 (as the notebooks);
  * embed: GoPlayerResNet() defaults, prereg01 best_model.pth (bytes read once, SHA-256 checked, then loaded from the
    same bytes), fp32, no autocast, cudnn.benchmark off, TF32 at the torch defaults (recorded), GPU_BATCH 4096, one
    forward per game; emb_sum = e.sum(0) (float32), n_windows = len(e); CHUNK 256 games per parallel parse.
Output: data/prereg06/cnn_fit/shard_NNNN.npz (8,192 games each: row, rank, game_id, player_id, emb_sum, n_windows;
SHA-256 in progress.json; a rerun continues after the last complete shard), then black_ge105.npz and the report
results/prereg06/design/cnn_fit_report.json. The first 256 games are embedded a second time at the end and the
largest absolute difference is recorded (expected 0).

Discipline (spec.md section 0): nothing under data/local is opened, listed or stat-ed (an import-time guard makes
that raise); file names containing the answers marker are refused; tools/score.py is never used; nvidia-smi is
checked before the start and the run refuses when another compute process ("C" type) is on the GPU.
Real mode requires this file to be committed and clean, unless C10_P06_PRECOMMIT=1 (the report then says
precommit: true and records the file's SHA-256, so the commit that adds the same bytes can be checked later).
--smoke N embeds the first N games only and writes to the session scratch (never into the repository).
"""
import sys
sys.dont_write_bytecode = True

import argparse
import builtins
import datetime
import hashlib
import io
import json
import os
import platform
import subprocess
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
TUT = os.path.join(REPO, "third_party", "aicup2026_tutorial")
WEIGHTS = os.path.join(REPO, "data", "prereg01_weights", "player", "best_model.pth")
WEIGHTS_SHA256 = "356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b"
OUT_DIR = os.path.join(REPO, "data", "prereg06", "cnn_fit")
REPORT = os.path.join(REPO, "results", "prereg06", "design", "cnn_fit_report.json")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
GPU_BATCH = 4096
CHUNK = 256
EMBED_DIM = 256
SHARD = 8192
MIN_BLACK = 105
EXPECT_PLAYERS, EXPECT_GAMES = 1082, 232165
RECHECK = 256
# third_party/aicup2026_tutorial originals (= colab/build_player_nb.py SOURCE_SHA256 at generation time)
TUTORIAL_FILES = ("utils.py", "network.py")

# ---------------------------------------------------------------- data/local guard (prefix test, not substring)
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))
_ANS = "ans" + "wers"


def forbidden(p):
    if isinstance(p, int):
        return False
    try:
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    if s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep):
        return True
    try:
        r = os.path.normcase(os.path.realpath(s))
    except (OSError, ValueError):
        return False
    return r == _FORBIDDEN or r.startswith(_FORBIDDEN + os.sep)


class Refused(RuntimeError):
    pass


def install_guard():
    if getattr(builtins, "_c10_p06_cnn_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def chk(p, what):
        if forbidden(p):
            raise Refused("p06_cnn_fit: refusing to %s %s (data/local)" % (what, p))
        if not isinstance(p, int) and _ANS in os.path.basename(str(os.fspath(p))).lower():
            raise Refused("p06_cnn_fit: refusing to %s %s (answers file)" % (what, p))

    def g_open(file, *a, **k):
        chk(file, "open")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        chk(path, "list")
        return real_listdir(path)

    def g_scandir(path="."):
        chk(path, "scan")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        chk(path, "stat")
        return real_stat(path, *a, **k)

    builtins.open, io.open, os.listdir, os.scandir, os.stat = g_open, g_open, g_listdir, g_scandir, g_stat
    builtins._c10_p06_cnn_guard = True


# ---------------------------------------------------------------- small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True).stdout.strip()


def tool_state():
    rel = "tools/p06_cnn_fit.py"
    tracked = bool(git("ls-files", rel))
    clean = tracked and subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    return {"path": rel, "sha256_lf": sha256_lf(os.path.join(REPO, rel)), "committed_and_clean": bool(clean)}


def nvidia_snapshot():
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    procs = []
    for line in full.splitlines():
        parts = line.strip("| ").split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[4] in ("C", "G", "C+G"):
            procs.append({"pid": parts[3], "type": parts[4], "name": parts[5]})
    return full, procs


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- worker side (same body as the notebooks)
_parser = None


def _init():
    global _parser
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if TUT not in sys.path:
        sys.path.insert(0, TUT)
    from utils import SGFParsePlayerIdentification
    _parser = SGFParsePlayerIdentification()


def sgf_windows(args):
    """(sgf, colour) -> uint8 (N, 17, 19, 19) or None (the notebooks' sgf_windows)."""
    sgf, color = args
    try:
        feats = _parser.extract_features(sgf, str(color).lower(), mode="inference")
        return np.asarray(feats, dtype=np.uint8) if len(feats) else None
    except Exception:
        return None


# ---------------------------------------------------------------- the games
def fit_black_games():
    """[(rank, game_id, player_id, sgf)] of the fit players with >= 105 black games, black games, file order."""
    if os.path.join(REPO, "tools") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "tools"))
    from c10data import load_rows
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None)
    n_black = Counter(r["player_id"] for r in rows if r["color"] == "B")
    keep = {p for p, n in n_black.items() if n >= MIN_BLACK}
    games = [(r["rank"], r["game_id"], r["player_id"], r["sgf_content"]) for r in rows
             if r["player_id"] in keep and r["color"] == "B"]
    return games, len(rows), len(keep)


def main():
    install_guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="embed only the first N games, output to --scratch")
    ap.add_argument("--scratch", default=None, help="output folder for --smoke (outside the repository)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t_start = time.time()
    smoke = a.smoke > 0
    if a.workers > 6:
        raise SystemExit("at most 6 CPU workers on this machine")
    ts = tool_state()
    precommit = False
    if not smoke and not ts["committed_and_clean"]:
        if os.environ.get("C10_P06_PRECOMMIT") == "1":
            precommit = True
        else:
            print("refused: tools/p06_cnn_fit.py is not committed and clean (real mode)", file=sys.stderr)
            return 2
    if smoke:
        if not a.scratch:
            raise SystemExit("--smoke needs --scratch (outside the repository)")
        out_dir = os.path.abspath(a.scratch)
        if os.path.normcase(out_dir).startswith(os.path.normcase(REPO) + os.sep):
            raise SystemExit("--smoke output must be outside the repository")
        report_path = os.path.join(out_dir, "cnn_fit_report.json")
    else:
        out_dir, report_path = OUT_DIR, REPORT
    if not smoke and os.path.exists(REPORT):
        print("refused: %s exists (no overwrite)" % REPORT, file=sys.stderr)
        return 2

    # weights first: refuse before any GPU work when the SHA is wrong
    with open(WEIGHTS, "rb") as f:
        wbytes = f.read()
    wsha = hashlib.sha256(wbytes).hexdigest()
    if wsha != WEIGHTS_SHA256:
        print("refused: best_model.pth SHA-256 %s is not %s" % (wsha, WEIGHTS_SHA256), file=sys.stderr)
        return 2
    tut_sha = {n: sha256_file(os.path.join(TUT, n)) for n in TUTORIAL_FILES}
    smi_text, smi_procs = nvidia_snapshot()
    others = [p for p in smi_procs if p["type"] == "C" or "python" in p["name"].lower()]
    if others and not smoke:
        print("refused: another compute process is on the GPU: %s" % others, file=sys.stderr)
        return 3

    t0 = time.time()
    games, n_fit_rows, n_players = fit_black_games()
    t_load = time.time() - t0
    if (n_players, len(games)) != (EXPECT_PLAYERS, EXPECT_GAMES):
        raise SystemExit("unexpected fit black-game set: %d players, %d games" % (n_players, len(games)))
    if smoke:
        games = games[:a.smoke]

    import torch
    if TUT not in sys.path:
        sys.path.insert(0, TUT)
    from network import GoPlayerResNet
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    net = GoPlayerResNet()
    net.load_state_dict(torch.load(io.BytesIO(wbytes), map_location="cpu"))
    net.eval().to(dev)
    del wbytes

    def embed(arr):
        outs = []
        with torch.no_grad():
            for i in range(0, len(arr), GPU_BATCH):
                x = torch.from_numpy(arr[i:i + GPU_BATCH]).float().to(dev)
                outs.append(net(x).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, EMBED_DIM), np.float32)

    os.makedirs(out_dir, exist_ok=True)
    prog_path = os.path.join(out_dir, "progress.json")
    progress = {"weights_sha256": wsha, "n_games": len(games), "shards": {}}
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("weights_sha256") == wsha and old.get("n_games") == len(games):
            progress = old
    n_shards = (len(games) + SHARD - 1) // SHARD
    t_embed = 0.0
    resumed = []
    with Pool(a.workers, initializer=_init) as pool:
        for s in range(n_shards):
            path = os.path.join(out_dir, "shard_%04d.npz" % s)
            rec = progress["shards"].get(str(s))
            if rec and os.path.exists(path) and sha256_file(path) == rec["sha256"]:
                resumed.append(s)
                continue
            lo, hi = s * SHARD, min((s + 1) * SHARD, len(games))
            part = games[lo:hi]
            jobs = [(g[3], "b") for g in part]
            emb_sum = np.zeros((len(part), EMBED_DIM), np.float32)
            n_win = np.zeros(len(part), np.int32)
            ts0 = time.time()
            for start in range(0, len(jobs), CHUNK):
                for g, arr in zip(range(start, start + CHUNK), pool.map(sgf_windows, jobs[start:start + CHUNK])):
                    if arr is None:
                        continue
                    e = embed(arr)
                    assert e.dtype == np.float32 and e.shape == (len(arr), EMBED_DIM), (e.dtype, e.shape)
                    emb_sum[g] = e.sum(0)
                    n_win[g] = len(e)
            t_embed += time.time() - ts0
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez(f, row=np.arange(lo, hi, dtype=np.int64), rank=np.array([g[0] for g in part]),
                         game_id=np.array([g[1] for g in part]), player_id=np.array([g[2] for g in part]),
                         emb_sum=emb_sum, n_windows=n_win)
            os.replace(tmp, path)
            progress["shards"][str(s)] = {"rows": [lo, hi], "sha256": sha256_file(path)}
            write_json(prog_path, progress)
            print("shard %d/%d  games %d-%d  %.0f s  (total %.0f s)" % (s + 1, n_shards, lo, hi,
                                                                      time.time() - ts0, time.time() - t_start),
                  flush=True)
        # merge
        parts = []
        for s in range(n_shards):
            with np.load(os.path.join(out_dir, "shard_%04d.npz" % s)) as z:
                parts.append({k: z[k] for k in z.files})
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        assert len(merged["row"]) == len(games) and np.array_equal(merged["row"], np.arange(len(games)))
        assert list(merged["game_id"]) == [g[1] for g in games]
        final = os.path.join(out_dir, "black_ge105.npz")
        with open(final + ".tmp", "wb") as f:
            np.savez(f, weights_sha256=np.array(wsha), **merged)
        os.replace(final + ".tmp", final)
        # recheck: the first 256 games once more
        nre = min(RECHECK, len(games))
        got = pool.map(sgf_windows, [(g[3], "b") for g in games[:nre]])
        maxdiff, nw_same = 0.0, True
        for g, arr in enumerate(got):
            es = embed(arr).sum(0) if arr is not None else np.zeros(EMBED_DIM, np.float32)
            nw = 0 if arr is None else len(arr)
            nw_same &= nw == int(merged["n_windows"][g])
            maxdiff = max(maxdiff, float(np.max(np.abs(es.astype(np.float64) - merged["emb_sum"][g].astype(np.float64)))))
    nw = merged["n_windows"]
    report = {
        "what": "PREREG_06 S4: per-game CNN embeddings of the fit players' black games (>= 105 black games), 4090",
        "mode": "smoke" if smoke else "real",
        "precommit": precommit,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_head": git("rev-parse", "HEAD"),
        "tool": ts,
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "workers": a.workers,
                "GPU_BATCH": GPU_BATCH, "CHUNK": CHUNK, "SHARD": SHARD},
        "inputs": {"weights": {"path": os.path.relpath(WEIGHTS, REPO).replace(os.sep, "/"), "sha256": wsha},
                   "tutorial_sha256": tut_sha,
                   "split_csv_sha256": sha256_file(SPLIT_CSV),
                   "training_csv_sha256": {lv: sha256_file(os.path.join(TRAIN_DIR, "train_%s.csv" % lv))
                                           for lv in ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")}},
        "data": {"fit_rows": n_fit_rows, "players_ge105_black": n_players, "games": int(len(nw)),
                 "games_without_windows": int((nw == 0).sum()), "windows_total": int(nw.sum()),
                 "windows_per_game_mean": round(float(nw.mean()), 3)},
        "output": {"path": os.path.relpath(final, REPO).replace(os.sep, "/") if not smoke else final,
                   "sha256": sha256_file(final), "bytes": os.path.getsize(final),
                   "shards": {k: v["sha256"] for k, v in sorted(progress["shards"].items(), key=lambda kv: int(kv[0]))},
                   "resumed_shards": resumed},
        "recheck_first_games": {"games": nre, "max_abs_diff_emb_sum": maxdiff, "n_windows_same": bool(nw_same)},
        "nvidia_smi_before": smi_text,
        "seconds": {"load_fit": round(t_load, 1), "embed": round(t_embed, 1), "total": round(time.time() - t_start, 1)},
    }
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("mode", "precommit", "data", "recheck_first_games", "seconds")}, indent=1))
    print("written", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
