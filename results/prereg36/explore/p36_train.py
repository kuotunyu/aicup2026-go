"""prereg36 explore: train the per-move metric CNN (PLAN.md 2).

    bash results/prereg36/explore/run_train.sh                 # 40,000 steps, checkpoints every 4,000
    bash results/prereg36/explore/run_train.sh --resume        # continue from the last checkpoint
    GPUPY results/prereg36/explore/p36_train.py --smoke --steps 200 --workers 8     # timing only, scratch dir

One sample = one game of one T player: up to 24 random target moves (17 planes each, the position the player is
about to play in), one random board symmetry for the whole segment, per-move embeddings averaged and
L2-normalised; cosine softmax (scale 30, no margin) over the T players. Batch = 96 games: player uniform, then one
of his games uniform. AdamW lr 2e-3 wd 1e-4, OneCycle pct 0.05, fp16 autocast, channels_last, seed 0.

Batch number s is drawn from numpy default_rng([seed, s]) inside the worker, so the data stream does not depend on
the number of workers and --resume continues the same stream.

Stop cleanly: create data/prereg36/STOP (checked every 100 steps: saves a checkpoint and exits with code 4).
Refuses to start on the GPU when nvidia-smi lists another python compute process.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p36_lib as L  # noqa: E402

TOTAL_STEPS = 40000
CKPT_EVERY = 4000
LR = 2e-3
WD = 1e-4
PCT = 0.05
SEED = 0
STOP_EVERY = 100


class TrainSet:
    """Map-style dataset: item s = batch s (x uint8 (P, 17, 19, 19), segment index (P,), player labels (B,))."""

    def __init__(self, cache_dir, off, is_white, t_games, t_off, seed, batch_games, moves):
        self.cache_dir, self.off, self.is_white = cache_dir, off, is_white
        self.t_games, self.t_off = t_games, t_off
        self.seed, self.batch_games, self.moves = seed, batch_games, moves
        self.codes = None

    def __len__(self):
        return 1 << 62

    def batch(self, s):
        if self.codes is None:
            self.codes = np.load(os.path.join(self.cache_dir, "codes.npy"), mmap_mode="r")
        rng = np.random.default_rng([self.seed, int(s)])
        n_t = len(self.t_off) - 1
        xs, segs, ys = [], [], []
        while len(ys) < self.batch_games:
            p = int(rng.integers(n_t))
            g = int(self.t_games[self.t_off[p] + int(rng.integers(self.t_off[p + 1] - self.t_off[p]))])
            codes = np.asarray(self.codes[self.off[g]:self.off[g + 1]])
            try:
                x, _t = L.planes_for_codes(codes, "w" if self.is_white[g] else "b", self.moves, rng)
            except ValueError:
                continue
            if len(x) == 0:
                continue
            xs.append(L.sym(x, int(rng.integers(8))))
            segs.append(np.full(len(x), len(ys), np.int64))
            ys.append(p)
        return np.ascontiguousarray(np.concatenate(xs)), np.concatenate(segs), np.asarray(ys, np.int64)

    def __getitem__(self, s):
        import torch
        x, seg, y = self.batch(s)
        return torch.from_numpy(x), torch.from_numpy(seg), torch.from_numpy(y)


def make_trainset(cache, batch_games=L.BATCH_GAMES, moves=L.SEG_MOVES, seed=SEED):
    """The T players (sorted ids = class indices) and their games; asserts that no E or D game is in it."""
    E, D, T, _per = L.splits(cache)
    cls = {p: i for i, p in enumerate(T)}
    lab = np.array([cls.get(p, -1) for p in cache.pid.tolist()], np.int64)
    games = np.flatnonzero(lab >= 0)
    order = games[np.argsort(lab[games], kind="stable")]
    t_off = np.concatenate([[0], np.cumsum(np.bincount(lab[games], minlength=len(T)))]).astype(np.int64)
    held = set(E) | set(D)
    assert not any(p in held for p in np.unique(cache.pid[order]).tolist())
    assert (np.diff(t_off) >= L.T_MIN_GAMES).all()
    ds = TrainSet(cache.dir, cache.off, cache.color == "W", order, t_off, seed, batch_games, moves)
    return ds, T, {"E": L.sha256_ids(E), "D": L.sha256_ids(D), "T": L.sha256_ids(T)}


def other_python_on_gpu():
    """Names of python compute processes nvidia-smi lists (other than this one); None if nvidia-smi fails."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        pid, _, name = line.partition(",")
        base = os.path.basename(name.strip().replace("\\", "/")).lower()
        if base.startswith("python") and pid.strip() != str(os.getpid()):
            out.append(line.strip())
    return out


def last_checkpoint(d):
    files = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return files[-1] if files else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="stop after this step number (schedule unchanged)")
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS, help="length of the OneCycle schedule")
    ap.add_argument("--smoke", action="store_true", help="timing run: writes only under --out (default "
                                                         "data/prereg36/smoke), 30 steps unless --steps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-games", type=int, default=L.BATCH_GAMES, help="smoke only")
    ap.add_argument("--moves", type=int, default=L.SEG_MOVES, help="smoke only")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY, help="smoke only")
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args(argv)
    if not a.smoke and (a.batch_games != L.BATCH_GAMES or a.moves != L.SEG_MOVES or a.ckpt_every != CKPT_EVERY
                        or a.out or a.device != "cuda"):
        raise SystemExit("--batch-games / --moves / --ckpt-every / --out / --device cpu are for --smoke only")
    if a.smoke:
        out = a.out or os.path.join(L.DATA, "smoke")
        ckpt_dir, log_path, rec_path = os.path.join(out, "ckpt"), os.path.join(out, "train_log.jsonl"), \
            os.path.join(out, "train_record.json")
        stop_path = os.path.join(out, "STOP")
        if a.steps is None:
            a.steps = 30
    else:
        ckpt_dir, log_path, rec_path = L.CKPT, os.path.join(L.DATA, "train_log.jsonl"), \
            os.path.join(HERE, "train_record.json")
        stop_path = os.path.join(L.DATA, "STOP")
    end = a.total_steps if a.steps is None else min(a.steps, a.total_steps)
    if os.path.exists(stop_path):
        raise SystemExit("STOP file present: %s" % stop_path)
    if a.device == "cuda":
        others = other_python_on_gpu()
        if others is None:
            raise SystemExit("nvidia-smi could not be queried; not starting")
        if others:
            raise SystemExit("another python compute process is on the GPU; not starting:\n  " + "\n  ".join(others))

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    cuda = a.device == "cuda"
    dev = torch.device(a.device)
    torch.manual_seed(SEED)
    if not cuda:
        torch.set_num_threads(2)
    cache = L.Cache()
    ds, T, ids_sha = make_trainset(cache, a.batch_games, a.moves)
    config = {"total_steps": a.total_steps, "batch_games": a.batch_games, "segment_moves": a.moves, "lr": LR,
              "weight_decay": WD, "onecycle_pct": PCT, "seed": SEED, "scale": L.SCALE, "blocks": L.BLOCKS,
              "channels": L.CHANNELS, "emb_dim": L.EMB_DIM, "fp16": cuda, "channels_last": False,
              "n_classes": len(T), "T_sha256_sorted_ids": ids_sha["T"], "cache_codes_sha256":
                  cache.meta["codes_sha256"]}
    net, head = L.build_model(len(T))
    net = net.to(dev)
    head = head.to(dev)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=a.total_steps, pct_start=PCT)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    start, seconds, positions = 0, 0.0, 0
    os.makedirs(ckpt_dir, exist_ok=True)
    last = last_checkpoint(ckpt_dir)
    if a.resume:
        if last is None:
            raise SystemExit("--resume: no checkpoint in %s" % ckpt_dir)
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["config"] != config:
            raise SystemExit("--resume: the checkpoint's config differs:\n%s\n%s" % (ck["config"], config))
        net.load_state_dict(ck["net"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, seconds, positions = ck["step"], ck["seconds"], ck["positions"]
        print("resumed from %s (step %d)" % (last, start), flush=True)
    elif last is not None:
        raise SystemExit("checkpoints already exist in %s; use --resume" % ckpt_dir)

    def save(step):
        path = os.path.join(ckpt_dir, "step_%05d.pt" % step)
        torch.save({"net": net.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "config": config,
                    "seconds": seconds + time.time() - t0, "positions": positions, "git_head": L.git_head()},
                   path + ".tmp")
        os.replace(path + ".tmp", path)
        print("saved %s" % path, flush=True)

    loader = DataLoader(ds, batch_size=None, sampler=range(start, end), num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None, persistent_workers=False)
    net.train()
    head.train()
    t0 = time.time()
    tw, pw, lw, nw = time.time(), 0, 0.0, 0
    step, stopped = start, False
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for s, (x, seg, y) in zip(range(start, end), loader):
            x = x.to(dev, non_blocking=True).float()
            seg = seg.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                e = net(x)
            emb = F.normalize(L.segment_mean(e, seg, len(y)), dim=1)
            loss = F.cross_entropy(head(emb), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step = s + 1
            positions += len(seg)
            pw += len(seg)
            lw += float(loss.detach())
            nw += 1
            if step % a.log_every == 0 or step == end:
                dt = time.time() - tw
                log.write(json.dumps({"step": step, "loss": round(lw / nw, 5), "positions_per_s": round(pw / dt, 1),
                                      "steps_per_s": round(nw / dt, 3), "lr": sched.get_last_lr()[0],
                                      "elapsed_s": round(seconds + time.time() - t0, 1)}) + "\n")
                log.flush()
                tw, pw, lw, nw = time.time(), 0, 0.0, 0
            if not np.isfinite(float(loss.detach())):
                raise SystemExit("loss is not finite at step %d" % step)
            if step % a.ckpt_every == 0:
                save(step)
            if step % STOP_EVERY == 0 and os.path.exists(stop_path) and step < end:
                if step % a.ckpt_every:
                    save(step)
                stopped = True
                break
    seconds += time.time() - t0
    if stopped:
        print("STOP file: saved at step %d, exiting (resume with --resume after removing %s)" % (step, stop_path),
              flush=True)
        return 4
    if step == end and step % a.ckpt_every and (a.smoke or step == a.total_steps):
        save(step)
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    rec = {"format": "c10-prereg36 train v1", "config": config, "finished": step == a.total_steps, "steps": step,
           "smoke": bool(a.smoke), "device": a.device, "workers": a.workers, "git_head": L.git_head(),
           "torch": torch.__version__, "counts": {"T_players": len(T), "T_games": int(len(ds.t_games)),
                                                  "positions_seen": int(positions), "games_seen":
                                                      int(step * a.batch_games)},
           "split_sha256_sorted_ids": ids_sha, "seconds": round(seconds, 1),
           "checkpoints": {os.path.basename(p): L.sha256_file(p) for p in cks},
           "regular_checkpoints": [os.path.basename(p) for p in cks
                                   if int(os.path.basename(p)[5:10]) % a.ckpt_every == 0]}
    L.write_json(rec_path, rec)
    print(json.dumps({k: rec[k] for k in ("finished", "steps", "seconds", "counts")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
