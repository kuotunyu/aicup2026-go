"""prereg44 explore: per-game logits of CNN44 (PLAN.md 3.4). Every target move of a game, identity symmetry.

    GPUPY results/prereg44/explore/p44_infer.py candidates --ckpt-dir <run>/ckpt --parts hold inner --workers 8
          -> <logits-root>/s<step>_<raw|ema>/<part>.npz for the 8 candidates of PLAN.md 3.4 (finished files skipped)
    GPUPY results/prereg44/explore/p44_infer.py one --ckpt <file> --weights ema --parts hold --out-root <dir>
    PY    results/prereg44/explore/p44_infer.py one --ckpt <smoke ckpt> --weights raw --parts hold --device cpu \
          --limit 64 --out-root data/prereg44/smoke/logits          # CPU smoke

The output npz: logits float32 (n, 10), positions, bad, rank / game_id / player_id / color (the cache's order),
cache_codes_sha256, ckpt_sha256, step, weights. Computing logits is not scoring: nothing here reads a label.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p44_lib as K  # noqa: E402

FORMAT = "c10-prereg44 logits v1"


def log(*x):
    print(*x, flush=True)


def load_net(ckpt, weights, dev):
    import torch
    ck = torch.load(K.guard(ckpt), map_location="cpu", weights_only=False)
    net = K.build()
    if weights == "raw":
        net.load_state_dict(ck["net"])
    elif weights == "ema":
        net.load_state_dict(ck["ema"])
    else:
        raise SystemExit(weights)
    return net.to(dev).eval(), int(ck["step"])


class Sub:
    """The first `limit` games of a cache (smoke)."""

    def __init__(self, c, limit):
        self.dir, self.off, self.n = c.dir, c.off, min(limit, c.n)
        self.color, self.meta = c.color[:self.n], c.meta
        self.rank, self.gid, self.pid = c.rank[:self.n], c.gid[:self.n], c.pid[:self.n]


def run_one(ckpt, weights, parts, out_root, dev, workers, limit=None, cache_root=K.CACHE):
    import torch
    net, step = load_net(ckpt, weights, dev)
    tag = "s%06d_%s" % (step, weights)
    ck_sha = K.sha256_file(ckpt)
    done = []
    for part in parts:
        out = os.path.join(out_root, tag, "%s.npz" % part)
        if os.path.exists(out):
            log("exists, skipped: %s" % out)
            done.append(out)
            continue
        c = K.open_cache(os.path.join(cache_root, part))
        if limit:
            c = Sub(c, limit)
        t0 = time.time()
        lg, pos, bad = K.infer_logits(net, c, dev, workers=workers, log=log)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, format=np.array(FORMAT), logits=lg, positions=pos, bad=bad,
                 rank=np.asarray(c.rank).astype(str), game_id=np.asarray(c.gid).astype(str),
                 player_id=np.asarray(c.pid).astype(str), color=np.asarray(c.color).astype(str),
                 cache_codes_sha256=np.array(c.meta["codes_sha256"]), cache_label=np.array(c.meta["label"]),
                 ckpt_sha256=np.array(ck_sha), step=np.array(step), weights=np.array(weights),
                 limit=np.array(-1 if not limit else int(limit)),
                 device=np.array(torch.cuda.get_device_name(0) if str(dev).startswith("cuda") else "cpu"),
                 seconds=np.array(round(time.time() - t0, 1)))
        os.replace(tmp, out)
        log(json.dumps({"out": out, "games": int(len(lg)), "bad": int(bad.sum()), "positions": int(pos.sum()),
                        "seconds": round(time.time() - t0, 1)}))
        done.append(out)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("one", "candidates"):
        p = sub.add_parser(name)
        p.add_argument("--parts", nargs="+", default=["hold", "inner"], choices=["hold", "inner"])
        p.add_argument("--out-root", default=K.LOGITS)
        p.add_argument("--cache-root", default=K.CACHE)
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--limit", type=int, default=None, help="smoke: only the first N games of every cache")
        if name == "one":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--weights", required=True, choices=list(K.CANDIDATE_WEIGHTS))
        else:
            p.add_argument("--ckpt-dir", required=True)
    a = ap.parse_args(argv)
    if a.device == "cpu" and not a.limit:
        raise SystemExit("--device cpu is for smoke runs (--limit) only")
    import torch
    if a.device == "cpu":
        torch.set_num_threads(max(1, min(8, a.threads)))
    dev = torch.device(a.device)
    if a.cmd == "one":
        run_one(a.ckpt, a.weights, a.parts, a.out_root, dev, a.workers, a.limit, a.cache_root)
        return 0
    for st in K.CANDIDATE_STEPS:
        ck = os.path.join(a.ckpt_dir, "step_%06d.pt" % st)
        if not os.path.exists(ck):
            raise SystemExit("missing candidate checkpoint %s" % ck)
        for w in K.CANDIDATE_WEIGHTS:
            run_one(ck, w, a.parts, a.out_root, dev, a.workers, a.limit, a.cache_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
