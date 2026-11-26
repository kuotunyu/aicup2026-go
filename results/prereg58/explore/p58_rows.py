"""prereg58: S0 (prereg43) per-game vectors of an explicit list of rows of a prereg36-format move cache, through
p43_run.py embed (= p42_embed.py: all sample moves, mean of the per-move vectors, raw weights, tta 1, fp16 on cuda).

    GPUPY results/prereg58/explore/p58_rows.py embed --cache-dir <cache> --rows <rows.npy> --side opp|own --name NAME
          --ckpt data/prereg43/S0/ckpt/step_320000.pt --weights raw --tta 1 --workers 2 --emb-root <root>
        -> <root>/V2_step_320000_raw_tta1/rows_NAME.npz  (emb, count, store_row = the cache rows, + rows_cfg)
       side opp: the colour is flipped, so the vector is of the OPPONENT's moves (as p58_opp.py); own: the player's.
    GPUPY results/prereg58/explore/p58_rows.py head
        -> data/prereg58/s0_head.npz  (S0's class weights [3,685 x 256], class ids = sorted T36, class ranks; CPU)

p42_embed only knows "--source store --players D|E"; this wrapper replaces p36_embed.store_games by the explicit row
list and renames the output afterwards (p42_embed writes store_D.npz). Nothing of prereg36 / 42 / 43 is edited.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "results", "prereg43", "explore"))
import p58_lib as K  # noqa: E402
import p43_run as R  # noqa: E402  (puts results/prereg42/explore on sys.path)
import p42_lib as V  # noqa: E402

L = V.L          # p36_lib


def cmd_embed(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rows", required=True, help=".npy int64 rows of the cache")
    ap.add_argument("--side", required=True, choices=["opp", "own"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--emb-root", required=True)
    a, rest = ap.parse_known_args(argv)
    for bad in ("--source", "--players", "--src", "--pool", "--limit"):
        if bad in rest:
            raise SystemExit("%s is set by this wrapper" % bad)
    import p36_embed
    cache = L.Cache(K.guard(a.cache_dir))
    rows = np.load(K.guard(a.rows)).astype(np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= cache.n:
        raise SystemExit("--rows: not distinct rows of the cache")
    is_white = np.asarray(cache.color)[rows] == "W"
    if a.side == "opp":
        is_white = ~is_white

    def store_games(_players, _cache):
        return rows, os.path.join(cache.dir, "codes.npy"), cache.off[rows], cache.off[rows + 1], is_white

    p36_embed.store_games = store_games
    if os.path.isdir(a.emb_root):
        for d in sorted(os.listdir(a.emb_root)):
            if os.path.exists(os.path.join(a.emb_root, d, "rows_%s.npz" % a.name)):
                print("exists: %s" % os.path.join(a.emb_root, d, "rows_%s.npz" % a.name))
                return 0
    rc = R.main(["embed", "--source", "store", "--players", "D", "--emb-root", a.emb_root] + rest)
    if rc:
        return rc
    tag_dirs = [os.path.join(a.emb_root, d) for d in sorted(os.listdir(a.emb_root))
                if os.path.exists(os.path.join(a.emb_root, d, "store_D.npz"))]
    if len(tag_dirs) != 1:
        raise SystemExit("expected one store_D.npz under %s, found %d" % (a.emb_root, len(tag_dirs)))
    tag_dir = tag_dirs[0]
    final = os.path.join(tag_dir, "rows_%s.npz" % a.name)
    with np.load(os.path.join(tag_dir, "store_D.npz"), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "players_sha256_sorted_ids"}
    if not np.array_equal(d["store_row"], rows):
        raise SystemExit("store_row differs from --rows")
    cfg = {"cache_dir": os.path.relpath(os.path.abspath(a.cache_dir), REPO).replace("\\", "/"),
           "cache_codes_sha256": cache.meta.get("codes_sha256"), "side": a.side, "rows": int(len(rows)),
           "rows_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()}
    np.savez(final + ".tmp.npz", rows_cfg=np.array(json.dumps(cfg, sort_keys=True)), **d)
    os.replace(final + ".tmp.npz", final)
    os.remove(os.path.join(tag_dir, "store_D.npz"))
    print(json.dumps({"out": final, **cfg}), flush=True)
    return 0


def cmd_head(_argv):
    import torch
    torch.set_num_threads(2)
    ck = torch.load(K.S0_CKPT, map_location="cpu", weights_only=False)
    W = ck["head"]["weight"].float().numpy()
    cache = L.Cache()
    T = L.splits(cache)[2]
    if len(T) != len(W):
        raise SystemExit("head rows %d != T36 players %d" % (len(W), len(T)))
    rank_of = {}
    for p, r in zip(np.asarray(cache.pid).tolist(), np.asarray(cache.rank).tolist()):
        if rank_of.setdefault(p, str(r)) != str(r):
            raise SystemExit("player %s with two ranks" % p)
    cr = K.rank_index([rank_of[p] for p in T])
    os.makedirs(K.DATA, exist_ok=True)
    np.savez(K.HEAD_NPZ + ".tmp.npz", weight=W.astype(np.float32), class_id=np.array(T), class_rank=cr,
             ckpt_sha256=np.array(L.sha256_file(K.S0_CKPT)), ckpt_step=np.array(int(ck["step"])),
             class_ids_sha256=np.array(L.sha256_ids(T)), scale=np.array(float(L.SCALE)),
             config=np.array(json.dumps(ck.get("config", {}), sort_keys=True, default=str)))
    os.replace(K.HEAD_NPZ + ".tmp.npz", K.HEAD_NPZ)
    print(json.dumps({"out": K.HEAD_NPZ, "classes": int(len(W)), "step": int(ck["step"]),
                      "classes_per_rank": np.bincount(cr, minlength=K.L).tolist(),
                      "variant": ck.get("config", {}).get("variant")}), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("embed", "head"):
        raise SystemExit("usage: p58_rows.py embed ... | head")
    return {"embed": cmd_embed, "head": cmd_head}[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
