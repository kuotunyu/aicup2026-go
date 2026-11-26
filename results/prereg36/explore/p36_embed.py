"""prereg36 explore: per-game embeddings of a checkpoint (PLAN.md 2, "推論向量").

    GPUPY results/prereg36/explore/p36_embed.py --ckpt data/prereg36/ckpt/step_04000.pt --source store --players D
    GPUPY results/prereg36/explore/p36_embed.py --ckpt <chosen> --source store --players E
    GPUPY results/prereg36/explore/p36_embed.py --ckpt <chosen> --source pool --src <official tests dir> --pool public

One game = the mean of the per-move embeddings (not normalised) over ALL the target's sample moves, no
augmentation; float32. Output data/prereg36/emb/<checkpoint stem>/store_<D|E>.npz (rows = feature-store rows of
those players: emb, count, store_row) or pool_<public|private>.npz (rows = p36_lib.pool_items order). Games with
no sample (fewer than 8 non-pass moves before the target's last move, or an illegal move) have count 0 and a zero
vector. Resumable: finished parts are kept in <output>.parts/ and skipped on the next run.
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
import p36_lib as L  # noqa: E402

CHUNK = 16          # games per loader item
PART_ITEMS = 128    # items per part file (2,048 games)
MAX_POS = 4096      # positions per forward


class GameSet:
    """Item i = the planes of games [i * CHUNK, (i + 1) * CHUNK): x uint8, game index (local) per position."""

    def __init__(self, codes, start, end, is_white):
        """codes: uint16 array, or the path of the cache's codes.npy (memory-mapped in every worker)."""
        self.codes, self.start, self.end, self.is_white = codes, start, end, is_white
        self.n = len(is_white)

    def __len__(self):
        return (self.n + CHUNK - 1) // CHUNK

    def __getitem__(self, i):
        import torch
        if isinstance(self.codes, str):
            self.codes = np.load(self.codes, mmap_mode="r")
        xs, seg = [], []
        for j, g in enumerate(range(i * CHUNK, min(self.n, (i + 1) * CHUNK))):
            try:
                x = L.planes_for_codes(np.asarray(self.codes[self.start[g]:self.end[g]]),
                                      "w" if self.is_white[g] else "b")[0]
            except ValueError:
                continue
            if len(x):
                xs.append(x)
                seg.append(np.full(len(x), j, np.int64))
        if not xs:
            return i, torch.zeros((0, L.N_PLANES, 19, 19), dtype=torch.uint8), torch.zeros(0, dtype=torch.int64)
        return i, torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(seg))


def store_games(players, cache):
    """Feature-store rows of the players and their games as (rows, codes path, start, end, is_white)."""
    with np.load(os.path.join(L.FEATS, "meta.npz"), allow_pickle=False) as z:
        pid, gid, rk, col = z["player_id"], z["game_id"], z["rank"], z["color"]
    rows = np.flatnonzero(np.isin(pid, np.asarray(sorted(players))))
    key = cache.key_index()
    idx = np.array([key[(str(rk[r]), str(gid[r]), str(pid[r]))] for r in rows.tolist()], np.int64)
    assert (cache.color[idx] == col[rows]).all() and (cache.pid[idx] == pid[rows]).all()
    return rows, os.path.join(cache.dir, "codes.npy"), cache.off[idx], cache.off[idx + 1], col[rows] == "W"


def pool_games(src, pool):
    import p35_apply as pa
    cands, qs = pa.read_pool(src, pool)
    items = L.pool_items(cands, qs)[0]
    parts = [L.moves_of(s) for s, _c in items]
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    return np.concatenate(parts), off[:-1], off[1:], np.array([c.upper() == "W" for _s, c in items])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, choices=["store", "pool"])
    ap.add_argument("--players", choices=["D", "E"])
    ap.add_argument("--src")
    ap.add_argument("--pool", choices=["public", "private"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None, help="default data/prereg36/emb/<checkpoint stem>")
    ap.add_argument("--limit", type=int, default=None, help="tests only: the first N games")
    a = ap.parse_args(argv)
    t0 = time.time()
    stem = os.path.splitext(os.path.basename(a.ckpt))[0]
    out_dir = a.out_dir or os.path.join(L.EMB, stem)
    extra = {}
    if a.source == "store":
        if not a.players:
            raise SystemExit("--source store needs --players D or E")
        cache = L.Cache()
        E, D, _T, _per = L.splits(cache)
        rows, codes, start, end, is_white = store_games({"D": D, "E": E}[a.players], cache)
        name = "store_%s" % a.players
        extra = {"store_row": rows, "players_sha256_sorted_ids": L.sha256_ids({"D": D, "E": E}[a.players])}
    else:
        if not (a.src and a.pool):
            raise SystemExit("--source pool needs --src and --pool")
        codes, start, end, is_white = pool_games(a.src, a.pool)
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
    meta_path = os.path.join(parts_dir, "meta.json")
    meta = {"ckpt_sha256": ck_sha, "games": int(n), "moves": int((end - start).sum()), "chunk": CHUNK, "part_items": PART_ITEMS}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) != meta:
                raise SystemExit("%s belongs to another checkpoint or game list; remove it first" % parts_dir)
    else:
        L.write_json(meta_path, meta)

    import torch
    from torch.utils.data import DataLoader
    cuda = a.device == "cuda"
    dev = torch.device(a.device)
    if not cuda:
        torch.set_num_threads(2)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = L.build_model()[0]
    net.load_state_dict(ck["net"])
    net = net.to(dev).eval()
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
        for i, x, seg in loader:
            i = int(i)
            g = min(n, (i + 1) * CHUNK) - i * CHUNK
            s = torch.zeros(g, L.EMB_DIM, dtype=torch.float32, device=dev)
            c = torch.zeros(g, dtype=torch.float32, device=dev)
            for b in range(0, len(seg), MAX_POS):
                xb = x[b:b + MAX_POS].to(dev).float()
                sb = seg[b:b + MAX_POS].to(dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    e = net(xb)
                s.index_add_(0, sb, e.float())
                c.index_add_(0, sb, torch.ones(len(sb), dtype=torch.float32, device=dev))
            buf_e[i] = (s / c.clamp(min=1)[:, None]).cpu().numpy()
            buf_c[i] = c.cpu().numpy().astype(np.int32)
            n_pos += len(seg)
            p = i // PART_ITEMS
            if i == min(n_items, (p + 1) * PART_ITEMS) - 1:
                rng_ = range(p * PART_ITEMS, i + 1)
                np.savez(part_path(p) + ".tmp.npz", emb=np.concatenate([buf_e.pop(j) for j in rng_]),
                         count=np.concatenate([buf_c.pop(j) for j in rng_]))
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
    np.savez(out + ".tmp.npz", emb=emb, count=cnt, ckpt_sha256=np.array(ck_sha), ckpt=np.array(stem),
             ckpt_step=np.array(int(ck["step"])), **extra)
    os.replace(out + ".tmp.npz", out)
    for p in range(n_parts):
        os.remove(part_path(p))
    os.remove(meta_path)
    os.rmdir(parts_dir)
    print(json.dumps({"out": out, "games": int(n), "games_without_sample": int((cnt == 0).sum()),
                      "positions_this_run": int(n_pos), "seconds": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
