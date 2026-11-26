"""Task B (p04 proxy): read fit / inner through c10data.load_rows and store the moves compactly.

Read-only on the repo. Training data only via tools/c10data.load_rows (part fit / inner, leak-checked
against the local question/candidate files; no answer file is read). Output: moves_<part>.npz in this folder.

move code (uint16) = colour_bit * 512 + point, colour_bit 0 = B, 1 = W; point = y * 19 + x with
x = column letter index, y = row letter index of the SGF point (passes do not match the regex and are skipped).
"""
import os
import sys
import time
from array import array

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402

from c10data import LEVELS, load_rows  # noqa: E402
from p02_moves import MOVE_RE  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
LET = {c: i for i, c in enumerate("abcdefghijklmnopqrs")}


def encode(sgf, buf):
    n = 0
    for m in MOVE_RE.finditer(sgf):
        c, p = m.group(1), m.group(2)
        buf.append((512 if c == "W" else 0) + LET[p[1]] * 19 + LET[p[0]])
        n += 1
    return n


def run(part):
    t0 = time.time()
    rows = load_rows(os.path.join(REPO, "data", "training"),
                     os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                     part=part, local_dir=os.path.join(REPO, "data", "local"))
    print(f"{part}: {len(rows)} rows loaded ({time.time() - t0:.0f}s)", flush=True)
    buf = array("H")
    lens = array("i")
    rank, gid, pid, col = [], [], [], []
    for r in rows:
        lens.append(encode(r["sgf_content"], buf))
        rank.append(r["rank"])
        gid.append(r["game_id"])
        pid.append(r["player_id"])
        col.append(r["color"])
    n = len(rows)
    del rows
    lens = np.frombuffer(lens, dtype=np.int32).astype(np.int64)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    np.savez(os.path.join(OUT, f"moves_{part}.npz"), moves=np.frombuffer(buf, dtype=np.uint16), off=off,
             rank=np.array(rank), game_id=np.array(gid), player_id=np.array(pid), color=np.array(col),
             label=np.array([LEVELS.index(x) for x in rank], dtype=np.int8))
    print(f"{part}: {n} games, {len(buf)} moves, mean {len(buf) / n:.1f}, min {lens.min()} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for part in sys.argv[1:]:
        run(part)
