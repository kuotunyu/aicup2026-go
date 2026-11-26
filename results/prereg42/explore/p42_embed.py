"""prereg42 explore: per-game vectors of a checkpoint with the embed-time options of PLAN.md 2 (raw / EMA weights,
TTA-8, V3's pooling mode). Also embeds prereg36 / prereg37 checkpoints (the base system's CNN, spec p36).

    GPUPY results/prereg42/explore/p42_embed.py --ckpt data/prereg42/V2/ckpt/step_40000.pt --weights ema --players D
    GPUPY results/prereg42/explore/p42_embed.py --ckpt data/prereg36/ckpt/step_40000.pt --tta 8 --players D
    GPUPY results/prereg42/explore/p42_embed.py --ckpt <c*> [same options] --players E          # after tune.json
    GPUPY results/prereg42/explore/p42_embed.py --ckpt <c*> [same options] --source pool --src <tests dir> --pool public

One game = the pooled vector of ALL the target's sample moves (no subsampling); with --tta 8 the mean of the 8 board
symmetries' game vectors. Mean pooling (V1 / V2 / V4 / p36) = mean of the per-move vectors, as p36_embed.py. V3:
'full' = the transformer pooling of all moves at once; 'chunk24' = the moves dealt round-robin into ~24-move chunks,
each chunk pooled and L2-normalised, the chunk vectors averaged. Games without a sample get count 0 and zeros.

Output data/prereg42/emb/<tag>/store_<D|E>.npz or pool_<public|private>.npz in p36_embed.py's layout (emb, count,
store_row, ckpt_sha256, ckpt = <tag>, ckpt_step) plus embed_cfg (JSON). tag = <run>_<checkpoint stem>_<weights>_tta<n>
[_<mode>], run = the variant (V1..V4) or 'base' + the first 8 hex of the checkpoint's SHA-256 for a prereg36-format
checkpoint. Resumable (parts in <output>.parts/). --compare <npz>: record the largest difference to another file of
the same games (the seam 'p42_embed --tta 1' == p36_embed on a prereg36 checkpoint).
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
import p42_lib as V  # noqa: E402

L = V.L
CHUNK = 16
PART_ITEMS = 128
MAX_POS = 4096


class GameSet:
    """Item i = games [i * CHUNK, (i + 1) * CHUNK): x uint8, local game index and move number per position."""

    def __init__(self, codes, start, end, is_white):
        self.codes, self.start, self.end, self.is_white = codes, start, end, is_white
        self.n = len(is_white)

    def __len__(self):
        return (self.n + CHUNK - 1) // CHUNK

    def __getitem__(self, i):
        import torch
        if isinstance(self.codes, str):
            self.codes = np.load(self.codes, mmap_mode="r")
        xs, seg, ts = [], [], []
        for j, g in enumerate(range(i * CHUNK, min(self.n, (i + 1) * CHUNK))):
            try:
                x, t = L.planes_for_codes(np.asarray(self.codes[self.start[g]:self.end[g]]),
                                          "w" if self.is_white[g] else "b")
            except ValueError:
                continue
            if len(x):
                xs.append(x)
                seg.append(np.full(len(x), j, np.int64))
                ts.append(np.asarray(t, np.int64))
        if not xs:
            z = torch.zeros(0, dtype=torch.int64)
            return i, torch.zeros((0, L.N_PLANES, 19, 19), dtype=torch.uint8), z, z
        return i, torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(seg)), \
            torch.from_numpy(np.concatenate(ts))


def load_net(ckpt_path, weights, dev):
    """(EmbedNet in eval mode, spec, run name, step) of a prereg42 or a prereg36-format checkpoint."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    if "variant" in cfg:
        spec = V.spec_of(cfg["variant"])
        if cfg["spec"] != spec:
            raise SystemExit("the checkpoint's spec differs from p42_lib.VARIANTS[%s]" % cfg["variant"])
        net, _h = V.build(spec)
        if weights == "ema":
            net.load_state_dict(ck["ema"])
        else:
            net.load_state_dict(ck["net"])
        run = cfg["variant"]
    else:
        if weights != "raw":
            raise SystemExit("a prereg36-format checkpoint has no EMA weights")
        spec = V.spec_of("p36")
        net, _h = V.build(spec)
        net.trunk.load_state_dict(ck["net"])
        run = "base" + L.sha256_file(ckpt_path)[:8]
    return net.to(dev).eval(), spec, run, int(ck["step"])


def game_vectors(net, spec, x, seg, t, n_games, tta, mode, dev, cuda):
    """(n_games, 256) float32 vectors of one loader item and the number of positions per game."""
    import torch
    import torch.nn.functional as F
    syms = range(8) if tta == 8 else range(1)
    acc = torch.zeros(n_games, L.EMB_DIM, dtype=torch.float32, device=dev)
    seg_d, t_d = seg.to(dev), t.to(dev)
    cnt = torch.bincount(seg_d, minlength=n_games)
    if len(seg) == 0:
        return acc, cnt
    if mode == "chunk24":
        cid, owner, nch = V.chunk_segments(seg.numpy(), n_games)
        cid, owner = torch.from_numpy(cid).to(dev), torch.from_numpy(owner).to(dev)
    for s in syms:
        es = []
        for b in range(0, len(seg), MAX_POS):
            xb = V.sym_t(x[b:b + MAX_POS].to(dev), s).float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                es.append(net.trunk(xb).float())
        e = torch.cat(es)
        if not V.is_tx(spec):
            v = L.segment_mean(e, seg_d, n_games)
        elif mode == "chunk24":
            vc = F.normalize(net.pool(e, cid, nch, t_d).float(), dim=1)
            v = L.segment_mean(vc, owner, n_games)
        else:
            v = net.pool(e, seg_d, n_games, t_d).float()
        acc += v
    acc /= len(syms)
    acc[cnt == 0] = 0.0
    return acc, cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="raw", choices=["raw", "ema"])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 8])
    ap.add_argument("--mode", default=None, choices=["full", "chunk24"], help="V3 / V3b only (required there)")
    ap.add_argument("--source", default="store", choices=["store", "pool"])
    ap.add_argument("--players", choices=["D", "E"])
    ap.add_argument("--src")
    ap.add_argument("--pool", choices=["public", "private"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--emb-root", default=V.EMB)
    ap.add_argument("--limit", type=int, default=None, help="smoke / seam checks only: the first N games")
    ap.add_argument("--compare", default=None, help="an npz of the same games: record the largest difference")
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from torch.utils.data import DataLoader
    cuda = a.device == "cuda"
    dev = torch.device(a.device)
    if not cuda:
        torch.set_num_threads(2)
    net, spec, run, step = load_net(a.ckpt, a.weights, dev)
    if V.is_tx(spec) != (a.mode is not None):
        raise SystemExit("--mode full|chunk24 is required for V3 / V3b and only for them")
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    tag = "%s_%s_%s_tta%d" % (run, stem, a.weights, a.tta) + ("_%s" % a.mode if a.mode else "")
    out_dir = os.path.join(a.emb_root, tag)
    extra = {}
    if a.source == "store":
        if not a.players:
            raise SystemExit("--source store needs --players D or E")
        import p36_embed
        cache = L.Cache()
        E, D, _T, _per = L.splits(cache)
        who = {"D": D, "E": E}[a.players]
        rows, codes, start, end, is_white = p36_embed.store_games(who, cache)
        name = "store_%s" % a.players
        extra = {"store_row": rows, "players_sha256_sorted_ids": L.sha256_ids(who)}
    else:
        if not (a.src and a.pool):
            raise SystemExit("--source pool needs --src and --pool")
        import p36_embed
        codes, start, end, is_white = p36_embed.pool_games(a.src, a.pool)
        name = "pool_%s" % a.pool
    if a.limit:
        is_white, start, end = is_white[:a.limit], start[:a.limit], end[:a.limit]
        extra = {k: (v[:a.limit] if k == "store_row" else v) for k, v in extra.items()}
        name += "_limit%d" % a.limit
    n = len(is_white)
    out = os.path.join(out_dir, name + ".npz")
    if os.path.exists(out):
        print("exists: %s" % out)
        return 0
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    ck_sha = L.sha256_file(a.ckpt)
    embed_cfg = {"ckpt_sha256": ck_sha, "weights": a.weights, "tta": a.tta, "mode": a.mode, "spec": spec,
                 "games": int(n), "moves": int((np.asarray(end) - np.asarray(start)).sum()), "chunk": CHUNK,
                 "part_items": PART_ITEMS}
    meta_path = os.path.join(parts_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) != embed_cfg:
                raise SystemExit("%s belongs to another checkpoint / setting / game list; remove it first" % parts_dir)
    else:
        L.write_json(meta_path, embed_cfg)
    ds = GameSet(codes, start, end, is_white)
    n_items = len(ds)
    n_parts = (n_items + PART_ITEMS - 1) // PART_ITEMS

    def part_path(p):
        return os.path.join(parts_dir, "part_%05d.npz" % p)

    todo = [p for p in range(n_parts) if not os.path.exists(part_path(p))]
    items = [i for p in todo for i in range(p * PART_ITEMS, min(n_items, (p + 1) * PART_ITEMS))]
    loader = DataLoader(ds, batch_size=None, sampler=items, num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None)
    buf_e, buf_c, n_pos = {}, {}, 0
    with torch.no_grad():
        for i, x, seg, t in loader:
            i = int(i)
            g = min(n, (i + 1) * CHUNK) - i * CHUNK
            v, c = game_vectors(net, spec, x, seg, t, g, a.tta, a.mode, dev, cuda)
            buf_e[i] = v.cpu().numpy()
            buf_c[i] = c.cpu().numpy().astype(np.int32)
            n_pos += len(seg)
            p = i // PART_ITEMS
            if i == min(n_items, (p + 1) * PART_ITEMS) - 1:
                rr = range(p * PART_ITEMS, i + 1)
                np.savez(part_path(p) + ".tmp.npz", emb=np.concatenate([buf_e.pop(j) for j in rr]),
                         count=np.concatenate([buf_c.pop(j) for j in rr]))
                os.replace(part_path(p) + ".tmp.npz", part_path(p))
                print("part %d/%d  %.0f s" % (p + 1, n_parts, time.time() - t0), flush=True)
    emb, cnt = [], []
    for p in range(n_parts):
        with np.load(part_path(p)) as z:
            emb.append(z["emb"])
            cnt.append(z["count"])
    emb = np.concatenate(emb).astype(np.float32) if emb else np.zeros((0, L.EMB_DIM), np.float32)
    cnt = np.concatenate(cnt) if cnt else np.zeros(0, np.int32)
    assert len(emb) == n == len(cnt)
    rep = {"out": out, "tag": tag, "games": int(n), "games_without_sample": int((cnt == 0).sum()),
           "positions_this_run": int(n_pos), "seconds": round(time.time() - t0, 1)}
    if a.compare:
        with np.load(a.compare, allow_pickle=False) as z:
            ref, rc = z["emb"][:n], z["count"][:n]
        rep["compare"] = {"file": a.compare, "count_equal": bool((rc == cnt).all()),
                          "max_abs_diff": float(np.abs(ref - emb).max()) if n else 0.0}
        embed_cfg["compare"] = rep["compare"]
    np.savez(out + ".tmp.npz", emb=emb, count=cnt, ckpt_sha256=np.array(ck_sha), ckpt=np.array(tag),
             ckpt_step=np.array(step), embed_cfg=np.array(json.dumps(embed_cfg, sort_keys=True)), **extra)
    os.replace(out + ".tmp.npz", out)
    for p in range(n_parts):
        os.remove(part_path(p))
    os.remove(meta_path)
    os.rmdir(parts_dir)
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
