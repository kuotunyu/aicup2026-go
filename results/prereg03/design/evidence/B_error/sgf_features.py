"""Task B part 2a: cheap per-game features from the SGF text (CPU), for every training game.

Training rows come only from tools/c10data.load_rows(part="train") (leak-checked; split v1
training players = fit + inner). Validation players' games are never loaded. Writes
sgf_features.npz next to this file (scratch, not the repo).

The anonymised SGFs hold only SZ/B/W, strictly alternating from Black, no passes, no result,
so "pass/resign markers" do not exist; game length and end-of-game shape are the only
"ended early" proxies. Captures are approximated without a board engine: a move on a point
that was occupied before means the earlier stone there was captured (lower bound).
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
OUT = os.path.dirname(os.path.abspath(__file__))
PT = re.compile(r";[BW]\[([a-s]{2})\]")

NAMES = [
    "n_moves", "target_moved_last",
    "t_line1", "t_line2", "t_line3", "t_line4", "t_line5plus",
    "t_early_line12", "t_early_line34", "t_early_center",
    "o_line1", "o_line2",
    "t_mean_dist_prev", "t_tenuki", "t_tenuki_mid", "t_contact", "t_diag_contact", "t_mean_dist_own",
    "t_own_adjacent",
    "o_tenuki", "o_contact",
    "replay_rate", "t_captured_rate", "o_captured_rate", "first_replay_frac", "ko_points",
    "unique_points_frac", "last20_mean_line", "last20_t_line12",
    "t_first_44", "t_first_34", "t_first_33", "t_first_other",
    "t_first10_corner", "t_endgame_mean_line", "t_early_line1_count",
    "is_black",
]


def one(sgf, colour):
    pts = PT.findall(sgf)
    a = np.frombuffer("".join(pts).encode("ascii"), dtype=np.uint8).reshape(-1, 2).astype(np.int16) - 97
    x, y = a[:, 0], a[:, 1]
    N = len(x)
    par = 0 if colour == "B" else 1
    idx = np.arange(N)
    tm = (idx % 2) == par
    om = ~tm
    line = np.minimum(np.minimum(x, 18 - x), np.minimum(y, 18 - y)) + 1
    dprev = np.full(N, -1)
    dprev[1:] = np.maximum(np.abs(x[1:] - x[:-1]), np.abs(y[1:] - y[:-1]))
    mprev = np.full(N, -1)
    mprev[1:] = np.abs(x[1:] - x[:-1]) + np.abs(y[1:] - y[:-1])
    down = np.full(N, -1)
    down[2:] = np.maximum(np.abs(x[2:] - x[:-2]), np.abs(y[2:] - y[:-2]))
    mown = np.full(N, -1)
    mown[2:] = np.abs(x[2:] - x[:-2]) + np.abs(y[2:] - y[:-2])
    tv = tm & (idx >= 1)
    ov = om & (idx >= 1)
    tl = line[tm]
    early = tm & (idx < 50)
    mid = tv & (idx >= 50) & (idx < 150)
    # replays: stable sort by point keeps time order; a repeat means the previous stone there was captured
    p = x * 19 + y
    order = np.argsort(p, kind="stable")
    sp = p[order]
    rep = np.nonzero(sp[1:] == sp[:-1])[0]
    prev_occ = order[rep]          # the captured stone's move index
    replay_moves = order[rep + 1]
    cap_t = int(np.sum((prev_occ % 2) == par))
    cap_o = len(prev_occ) - cap_t
    cnt = np.bincount(p, minlength=361)
    first_rep = (replay_moves.min() / N) if len(replay_moves) else 1.0
    last = idx >= N - 20
    tfirst = np.nonzero(tm)[0][0]
    fx, fy = int(x[tfirst]), int(y[tfirst])
    l1, l2 = sorted((min(fx, 18 - fx) + 1, min(fy, 18 - fy) + 1))
    f44, f34, f33 = (l1, l2) == (4, 4), (l1, l2) == (3, 4), (l1, l2) == (3, 3)
    t10 = np.nonzero(tm)[0][:10]
    cx = np.minimum(x[t10], 18 - x[t10]) + 1
    cy = np.minimum(y[t10], 18 - y[t10]) + 1
    end = tm & (idx >= 150)
    nt = max(1, tm.sum())
    no = max(1, om.sum())
    return [
        N, float(((N - 1) % 2) == par),
        np.mean(tl == 1), np.mean(tl == 2), np.mean(tl == 3), np.mean(tl == 4), np.mean(tl >= 5),
        np.mean(line[early] <= 2), np.mean((line[early] >= 3) & (line[early] <= 4)), np.mean(line[early] >= 5),
        np.mean(line[om] == 1), np.mean(line[om] == 2),
        dprev[tv].mean(), np.mean(dprev[tv] >= 5), np.mean(dprev[mid] >= 5) if mid.any() else 0.0,
        np.mean(mprev[tv] == 1), np.mean((dprev[tv] == 1) & (mprev[tv] == 2)),
        down[tm & (idx >= 2)].mean(), np.mean(mown[tm & (idx >= 2)] == 1),
        np.mean(dprev[ov] >= 5), np.mean(mprev[ov] == 1),
        len(rep) / N, cap_t / nt, cap_o / no, first_rep, int(np.sum(cnt >= 3)),
        np.sum(cnt > 0) / 361.0, line[last].mean(), np.mean(line[last & tm] <= 2),
        float(f44), float(f34), float(f33), float(not (f44 or f34 or f33)),
        np.mean((cx <= 5) & (cy <= 5)), line[end].mean() if end.any() else line[tm].mean(),
        int(np.sum(tl[: 50] == 1)),
        float(par == 0),
    ]


def work(chunk):
    return np.asarray([one(s, c) for s, c in chunk], dtype=np.float32)


def main():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from c10data import LEVELS, inner_players, load_rows, read_split
    t0 = time.time()
    split_csv = os.path.join(REPO, "results", "split_v1", "players_split.csv")
    rows = load_rows(os.path.join(REPO, "data", "training"), split_csv, part="train",
                     local_dir=os.path.join(REPO, "data", "local"))
    inner = inner_players(read_split(split_csv))
    print(f"loaded {len(rows)} rows {time.time() - t0:.0f}s", flush=True)
    pairs = [(r["sgf_content"], r["color"]) for r in rows]
    chunks = [pairs[i:i + 4000] for i in range(0, len(pairs), 4000)]
    with Pool(16) as pool:
        parts = pool.map(work, chunks)
    X = np.concatenate(parts)
    assert X.shape == (len(rows), len(NAMES))
    np.savez_compressed(os.path.join(OUT, "sgf_features.npz"), X=X, names=np.array(NAMES),
                        rank=np.array([r["rank"] for r in rows]), level=np.array([LEVELS.index(r["rank"]) for r in rows]),
                        game_id=np.array([r["game_id"] for r in rows]), player_id=np.array([r["player_id"] for r in rows]),
                        is_inner=np.array([r["player_id"] in inner for r in rows]))
    print(f"features {X.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
