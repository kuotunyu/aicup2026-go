"""prereg08 P1: synthetic chunk folders for the mechanics test of p1_fuse.py (no GPU, no network).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg08/p1/p1_synth.py --kind null    --out data/prereg08/p1_synth_null
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg08/p1/p1_synth.py --kind planted --out data/prereg08/p1_synth_planted
    then  p1_fuse.py --synthetic --p1-dir <out> --out <out>/fuse_report.json [--maxiter 200]

Writes <out>/sets.json (probe: the first --probe-per-level games per level of the real sets.json; bank: the real
bank rows; mono: none unless --mono) and chunk files in p1_extract's format with random features:
    null     every feature N(0, 1), independent of the level  -> P1 must add ~nothing over Q02H
    planted  as null, plus the side's level index (inner levels from inner_keys.npz: training players, never
             validation) times --signal added to the target-side log p and entropy slots -> P1 must help
"""

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p1_extract as E  # noqa: E402
import p1_lib as P  # noqa: E402

H = P.H


def write_chunks(root, name, refs, levels, kind, signal, rng):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    for ci, s in enumerate(range(0, len(refs), E.CHUNK)):
        r = refs[s:s + E.CHUNK]
        lv = levels[s:s + E.CHUNK].astype(np.float64)
        G = len(r)
        means = rng.standard_normal((G, 2, P.F_DIM))
        if kind == "planted":
            means[:, 0, 768] += signal * (lv - 4.5) + 0.0
            means[:, 0, 769] -= signal * (lv - 4.5)
        np.savez(E.chunk_path(name, ci, root), format=np.array(P.CHUNK_FORMAT), set=np.array(name),
                 sides=np.array(E.SIDES), ref=np.asarray(r, dtype=np.int64), precision=np.array("synthetic"),
                 device=np.array("none"), ckpt_sha256=np.array("synthetic"),
                 meta_sha256=np.array(P.neutral_table_sha()), means=means.astype(np.float32),
                 counts=np.full((G, 2), 100, dtype=np.int32), n_moves=np.full(G, 200, dtype=np.int16),
                 failed=np.zeros(G, dtype=bool))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["null", "planted"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-per-level", type=int, default=300)
    ap.add_argument("--signal", type=float, default=0.15)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if os.path.normcase(out) == os.path.normcase(os.path.abspath(E.OUT_DIR)):
        raise P.P1Error("never write synthetic chunks into data/prereg08/p1")
    real = E.load_sets()
    probe = []
    for lv in H.LEVELS:
        probe += [g for g in real["probe"]["games"] if g["rank"] == lv][:a.probe_per_level]
    sets = {"format": E.SETS_FORMAT, "synthetic": a.kind, "probe": {"source": "fit", "games": probe},
            "bank": real["bank"], "mono": real["mono"] if a.mono else {"source": "inner", "rows": []},
            "counts": {"probe": len(probe), "bank": len(real["bank"]["rows"]),
                       "mono": len(real["mono"]["rows"]) if a.mono else 0}}
    os.makedirs(out, exist_ok=True)
    E.write_json_atomic(os.path.join(out, "sets.json"), sets)
    rng = np.random.default_rng(a.seed)
    lv_of = {lv: i for i, lv in enumerate(H.LEVELS)}
    write_chunks(out, "probe", np.arange(len(probe)), np.array([lv_of[g["rank"]] for g in probe]), a.kind, a.signal,
                 rng)
    with np.load(E.INNER_KEYS, allow_pickle=False) as z:
        inner_lv = np.array([lv_of[str(x)] for x in z["rank"]])
    for name in ("bank",) + (("mono",) if a.mono else ()):
        rows = np.asarray(real[name]["rows"], dtype=np.int64)
        write_chunks(out, name, rows, inner_lv[rows], a.kind, a.signal, rng)
    print(json.dumps({"out": out, "counts": sets["counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
