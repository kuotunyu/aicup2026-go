"""prereg43: concatenation ensemble of per-game vectors (PLAN.md 2).

    PY results/prereg43/explore/p43_ens.py --in A.npz --in B.npz [--in C.npz] --label S01 --out <dir>/store_D.npz

Every input: the same games in the same order (store_row, or the pool's item order) and the same players hash.
Each model's game vector is L2-normalised, the vectors are concatenated and divided by sqrt(m); count = minimum.
ckpt_sha256 = sha256 of the sorted input checkpoint hashes joined by ','; embed_cfg records the members.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.inp) < 2:
        raise SystemExit("need at least two inputs")
    zs = [dict(np.load(p, allow_pickle=False)) for p in a.inp]
    ref = zs[0]
    for p, z in zip(a.inp[1:], zs[1:]):
        for k in ("store_row", "players_sha256_sorted_ids"):
            if (k in ref) != (k in z) or (k in ref and not np.array_equal(ref[k], z[k])):
                raise SystemExit("%s: %s differs from %s" % (p, k, a.inp[0]))
        if z["emb"].shape[0] != ref["emb"].shape[0]:
            raise SystemExit("%s: number of games differs" % p)
    parts = []
    for z in zs:
        e = z["emb"].astype(np.float64)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        parts.append(np.where(n > 0, e / np.where(n > 0, n, 1), 0.0))
    emb = (np.concatenate(parts, axis=1) / np.sqrt(len(zs))).astype(np.float32)
    count = np.min(np.stack([z["count"] for z in zs]), axis=0).astype(np.int32)
    shas = [str(z["ckpt_sha256"]) for z in zs]
    members = [{"file": os.path.abspath(p).replace("\\", "/"), "ckpt": str(z["ckpt"]), "ckpt_sha256": s,
                "embed_cfg": json.loads(str(z["embed_cfg"])) if "embed_cfg" in z else None}
               for p, z, s in zip(a.inp, zs, shas)]
    cfg = {"ensemble": "concat_l2_over_sqrt_m", "members": [{k: m[k] for k in ("ckpt", "ckpt_sha256", "embed_cfg")}
                                                           for m in members]}
    out = {"emb": emb, "count": count, "ckpt_sha256": np.array(hashlib.sha256(",".join(sorted(shas)).encode()).hexdigest()),
           "ckpt": np.array(a.label), "embed_cfg": np.array(json.dumps(cfg, sort_keys=True))}
    for k in ("store_row", "players_sha256_sorted_ids"):
        if k in ref:
            out[k] = ref[k]
    if "ckpt_step" in ref:
        out["ckpt_step"] = np.array(-1, np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(json.dumps({"out": a.out, "label": a.label, "games": int(len(emb)), "dim": int(emb.shape[1]),
                      "members": [m["ckpt"] for m in members]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
