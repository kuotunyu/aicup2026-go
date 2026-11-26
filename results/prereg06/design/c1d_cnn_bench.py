"""PREREG_06 design helper (competition 1): runtime of the fit-player CNN-embedding job on this machine (RTX 4090).

Times the two halves of the per-game embedding exactly as the prereg02 / test02 player notebooks compute it
(third_party/aicup2026_tutorial/utils.py SGFParsePlayerIdentification.extract_features in inference mode = one
(17, 19, 19) window per own move once 8 board states exist; GoPlayerResNet() defaults; fp32, no autocast; one
forward per game, GPU_BATCH 4096; emb_sum = e.sum(0), n_windows = len(e)):
  1. CPU window extraction with a multiprocessing pool (chunks of 256 games, as the notebooks),
  2. GPU forward per game (and, for comparison only, several games per forward).
The network is randomly initialised (seed 0): speed does not depend on the weights, and the prereg01 player weights
are not on this machine yet. Games: a fixed sample of the fit players' black games from the c1_headroom scratch cache
(results/prereg06/explore/c1_headroom/c1h_lib.build_cache: fit players with >= 105 black games), rebuilt as SGF
text from the cached move strings. No data/local, no tools/score.py; nothing is written in the repository.

    PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe results/prereg06/design/c1d_cnn_bench.py [--n 768]
        [--workers 6] [--out PATH]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
import pickle
import platform
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TUT = os.path.join(REPO, "third_party", "aicup2026_tutorial")
SCRATCH = os.environ.get(
    "C10_C1H_SCRATCH",
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\c1_headroom")
CHUNK = 256
GPU_BATCH = 4096
EMBED_DIM = 256

_parser = None


def _init():
    global _parser
    sys.dont_write_bytecode = True
    if TUT not in sys.path:
        sys.path.insert(0, TUT)
    from utils import SGFParsePlayerIdentification
    _parser = SGFParsePlayerIdentification()


def sgf_windows(args):
    """Same body as the notebooks' sgf_windows: (sgf, colour) -> uint8 (N, 17, 19, 19) or None."""
    sgf, color = args
    try:
        feats = _parser.extract_features(sgf, str(color).lower(), mode="inference")
        return np.asarray(feats, dtype=np.uint8) if len(feats) else None
    except Exception:
        return None


def to_sgf(pts):
    n = len(pts) // 2
    return "(;GM[1]SZ[19]" + "".join(";%s[%s]" % ("B" if i % 2 == 0 else "W", pts[2 * i:2 * i + 2])
                                      for i in range(n)) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(SCRATCH, "c1d_cnn_bench.json"))
    a = ap.parse_args()
    with open(os.path.join(SCRATCH, "cache_black.pkl"), "rb") as f:
        c = pickle.load(f)
    nb = c["n_black"]
    by = c["by_player"]
    n_ge105 = sum(len(v) for p, v in by.items())
    p120 = [p for p in by if nb[p] >= 120]
    n_ge120 = sum(len(by[p]) for p in p120)
    first120 = sum(min(len(by[p]), 120) for p in p120)
    order = sorted(range(len(c["gpts"])), key=lambda i: hashlib.sha256(("bench|%d" % i).encode()).hexdigest())
    rows = order[:a.n]
    jobs = [(to_sgf(c["gpts"][i]), "B") for i in rows]
    mean_len = float(np.mean([c["glen"][i] for i in rows]))
    all_len = float(np.mean(c["glen"]))

    import torch
    if TUT not in sys.path:
        sys.path.insert(0, TUT)
    from network import GoPlayerResNet
    dev = torch.device("cuda")
    torch.manual_seed(0)
    net = GoPlayerResNet().eval().to(dev)

    def embed(arr):
        outs = []
        with torch.no_grad():
            for i in range(0, len(arr), GPU_BATCH):
                x = torch.from_numpy(arr[i:i + GPU_BATCH]).float().to(dev)
                outs.append(net(x).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, EMBED_DIM), np.float32)

    # warm-up (cuDNN autotuning, CUDA context) on a few windows
    with Pool(a.workers, initializer=_init) as pool:
        warm = pool.map(sgf_windows, jobs[:8])
        for arr in warm:
            if arr is not None:
                embed(arr)
        torch.cuda.synchronize()
        t_ext, t_gpu, n_win, n_none = 0.0, 0.0, 0, 0
        arrs = []
        t_all = time.perf_counter()
        for s in range(0, len(jobs), CHUNK):
            t0 = time.perf_counter()
            got = pool.map(sgf_windows, jobs[s:s + CHUNK])
            t_ext += time.perf_counter() - t0
            t0 = time.perf_counter()
            for arr in got:
                if arr is None:
                    n_none += 1
                    continue
                e = embed(arr)
                _ = e.sum(0)
                n_win += len(e)
            torch.cuda.synchronize()
            t_gpu += time.perf_counter() - t0
            arrs += [x for x in got if x is not None][:64]
        t_total = time.perf_counter() - t_all
    # comparison only: several games per forward (not the notebooks' way; float results may differ slightly)
    cat = np.concatenate(arrs[:64])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(cat), GPU_BATCH):
        with torch.no_grad():
            net(torch.from_numpy(cat[i:i + GPU_BATCH]).float().to(dev)).cpu()
    torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0
    per_game = t_total / len(jobs)
    out = {
        "what": __doc__.split("\n")[0],
        "machine": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "cudnn": torch.backends.cudnn.version(), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "python": platform.python_version(),
                    "cpu_count": os.cpu_count(), "workers": a.workers},
        "sample": {"games": len(jobs), "mean_moves": round(mean_len, 1), "cache_mean_moves": round(all_len, 1),
                   "windows": n_win, "windows_per_game": round(n_win / max(len(jobs) - n_none, 1), 2),
                   "no_window_games": n_none},
        "seconds": {"extract_cpu": round(t_ext, 2), "embed_gpu_per_game": round(t_gpu, 2), "total": round(t_total, 2),
                    "per_game_ms": round(per_game * 1e3, 2),
                    "batched_forward_windows": int(len(cat)), "batched_forward_s": round(t_batched, 3),
                    "per_game_forward_s_same_windows_est": round(t_gpu * len(cat) / max(n_win, 1), 3)},
        "job_sizes": {"cached_black_games_players_ge105": n_ge105, "players_ge120": len(p120),
                      "black_games_players_ge120": n_ge120, "first120_black_games_players_ge120": first120},
        "estimate_hours_notebook_way": {
            "all_cached_ge105": round(n_ge105 * per_game / 3600, 2),
            "players_ge120_all_black": round(n_ge120 * per_game / 3600, 2),
            "players_ge120_first120": round(first120 * per_game / 3600, 2)},
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
