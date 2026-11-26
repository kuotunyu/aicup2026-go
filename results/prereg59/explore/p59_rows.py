"""prereg59: S0 (prereg43) PLAYER-side game vectors of every training game of the split v1 validation players
(rows of the prereg49 move cache listed in data/prereg59/val_rows.npy, written by `p59_eval.py rows`).

    GPUPY results/prereg59/explore/p59_rows.py embed --workers 2
        -> data/prereg59/emb_own_val/V2_step_320000_raw_tta1/rows_val.npz   (emb, count, store_row = cache rows)
    GPUPY results/prereg59/explore/p59_rows.py embed --limit 8 --workers 0 --device cpu
        -> data/prereg59/sanity/emb/V2_step_320000_raw_tta1/rows_val_limit8.npz   (run check only: the first 8 rows)

= prereg58's p58_rows.py embed (imported, not edited: p43_run.py embed = p42_embed.py, all sample moves, mean of the
per-move vectors) with the checkpoint, raw weights, tta 1, the cache and the side (own = the colour is NOT flipped)
fixed here. Skips when the output exists; an interrupted run resumes from p42_embed's part files.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "results", "prereg58", "explore"))
import p59_lib as K  # noqa: E402
import p58_rows as R58  # noqa: E402  (module level, so that the loader's worker processes can import it too)

CKPT = os.path.join("data", "prereg43", "S0", "ckpt", "step_320000.pt")


def cmd_embed(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run check only: the first N rows, to data/prereg59/sanity")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    rows_path, root, name = K.VAL_ROWS, os.path.join(K.DATA, "emb_own_val"), "val"
    if a.limit:
        rows = np.load(K.VAL_ROWS)[:a.limit]
        os.makedirs(os.path.join(K.DATA, "sanity"), exist_ok=True)
        rows_path = os.path.join(K.DATA, "sanity", "val_rows_limit%d.npy" % a.limit)
        np.save(rows_path, rows)
        root, name = os.path.join(K.DATA, "sanity", "emb"), "val_limit%d" % a.limit
    os.chdir(REPO)
    return R58.cmd_embed(["--cache-dir", os.path.relpath(K.CACHE49, REPO), "--rows", os.path.relpath(rows_path, REPO),
                          "--side", "own", "--name", name, "--emb-root", os.path.relpath(root, REPO),
                          "--ckpt", CKPT, "--weights", "raw", "--tta", "1", "--workers", str(a.workers),
                          "--device", a.device])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "embed":
        raise SystemExit("usage: p59_rows.py embed [--limit N] [--workers W] [--device cuda|cpu]")
    return cmd_embed(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
