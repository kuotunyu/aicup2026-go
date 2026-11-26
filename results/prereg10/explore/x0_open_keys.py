"""prereg10 explore x0: opening keys of every row of the PREREG_06 fit feature store (data/prereg06/feats/meta.npz).

For each store row: points string of the first 4 / 6 / 8 / 10 plies (whole game, both colours), canonicalised under the
8 board symmetries (p02_moves.canon on that prefix length), plus the raw (non-canonical) first-6 string.
Used by x1_scenarios.py to build "novel opening" question games (question games whose opening prefix never occurs among
the owner's candidate games). Only fit rows (c10data.load_rows part 'fit', local_dir None); the p06 data/local guard
is installed. Output: data/prereg10/tmp/open_keys.npz (git-ignored).
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402

import p02_moves as pm  # noqa: E402
import p06_style as st  # noqa: E402
from c10data import load_rows  # noqa: E402

st.install_guard()
OUT = os.path.join(REPO, "data", "prereg10", "tmp", "open_keys.npz")
NS = (4, 6, 8, 10)


def main():
    t0 = time.time()
    with np.load(os.path.join(REPO, "data", "prereg06", "feats", "meta.npz"), allow_pickle=False) as z:
        rank, gid = z["rank"], z["game_id"]
    rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1",
                     "players_split.csv"), part="fit", local_dir=None, check_counts=True)
    sgf = {(r["rank"], r["game_id"]): r["sgf_content"] for r in rows}
    del rows
    keys = {n: [] for n in NS}
    raw6 = []
    for i in range(len(gid)):
        mv = pm.parse_moves(sgf[(str(rank[i]), str(gid[i]))], 12)
        raw6.append(pm.points_string(mv[:6]))
        for n in NS:
            keys[n].append(pm.points_string(pm.canon(mv[:n], n)))
    np.savez(OUT, raw6=np.array(raw6), **{"c%d" % n: np.array(keys[n]) for n in NS})
    print("rows", len(gid), "seconds", round(time.time() - t0, 1), "->", OUT)


if __name__ == "__main__":
    main()
