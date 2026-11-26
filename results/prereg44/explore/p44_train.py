"""prereg44 explore: train CNN44 (PLAN.md 3.3).

    GPUPY results/prereg44/explore/p44_train.py --bench 300                # timing only (no files)
    GPUPY results/prereg44/explore/p44_train.py --auto-resume --workers 8   # 200,000 steps, checkpoints every 20,000
    PY    results/prereg44/explore/p44_train.py --smoke --device cpu --workers 0   # CPU smoke (tiny batches, 3 steps)

Reads only data/prereg44/cache/train (A players; asserted). AdamW lr 2e-3 / wd 1e-4, OneCycle pct 0.05, fp16 autocast
on cuda, seed 0 for the weights; batch s drawn from default_rng([44, s]) inside the worker. EMA (0.999) of the whole
RankNet in every checkpoint. Outputs: <out>/ckpt/step_NNNNNN.pt, <out>/train_log.jsonl, the record --record.
Stop cleanly: create <out>/STOP (checked every 100 steps; saves and exits 4); also exits 4 when free RAM < --min-free-ram-mb.
Refuses to start on the GPU when nvidia-smi lists another python compute process.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p44_lib as K  # noqa: E402

STOP_EVERY = 100
BENCH_WARMUP = 30
FORMAT = "c10-prereg44 train v1"


def free_ram_mb():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1048576.0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def other_python_on_gpu():
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


def code_sha():
    h = {}
    for f in ("p44_lib.py", "p44_train.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h[f] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    with open(os.path.join(K.P36, "p36_lib.py"), "rb") as fh:
        h["p36_lib.py"] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return h


def git_head():
    try:
        return subprocess.run(["git", "-C", K.REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def last_checkpoint(d):
    files = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return files[-1] if files else None


def loss_of(b, net, cuda):
    import torch
    import torch.nn.functional as F
    x, seg, y = b["x"].float(), b["seg"], b["y"]
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        e = net.trunk(x)
    return F.cross_entropy(net.pool_head(e, seg, len(y)), y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", action="store_true", help="resume if a checkpoint exists, else start (Colab)")
    ap.add_argument("--steps", type=int, default=None, help="stop after this step number (schedule unchanged)")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="timing only: N steps after a warm-up, no files")
    ap.add_argument("--smoke", action="store_true", help="tiny batches, 3 steps, outputs under data/prereg44/smoke")
    ap.add_argument("--cache", default=os.path.join(K.CACHE, "train"))
    ap.add_argument("--out", default=None, help="default data/prereg44/run (smoke: data/prereg44/smoke/run)")
    ap.add_argument("--record", default=None, help="default results/prereg44/explore/train_record.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (cpu only)")
    ap.add_argument("--min-free-ram-mb", type=float, default=4096)
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args(argv)
    if a.device == "cpu" and not (a.smoke or a.bench):
        raise SystemExit("--device cpu is for --smoke / --bench only")
    total, ckpt_every = K.TOTAL_STEPS, K.CKPT_EVERY
    small = None
    if a.smoke:
        small = (4, 6)
        out = a.out or os.path.join(K.DATA, "smoke", "run")
        rec_path = a.record or os.path.join(out, "train_record.json")
        if a.steps is None:
            a.steps = 3
        ckpt_every = 2
    else:
        out = a.out or K.RUN
        rec_path = a.record or os.path.join(HERE, "train_record.json")
    ckpt_dir, log_path, stop_path = (os.path.join(out, "ckpt"), os.path.join(out, "train_log.jsonl"),
                                     os.path.join(out, "STOP"))
    end = total if a.steps is None else min(a.steps, total)
    if os.path.exists(stop_path) and not a.bench:
        raise SystemExit("STOP file present: %s" % stop_path)
    if a.device == "cuda":
        others = other_python_on_gpu()
        if others is None:
            raise SystemExit("nvidia-smi could not be queried; not starting")
        if others:
            raise SystemExit("another python compute process is on the GPU; not starting:\n  " + "\n  ".join(others))

    import torch
    from torch.utils.data import DataLoader

    cuda = a.device == "cuda"
    dev = torch.device(a.device)
    torch.manual_seed(0)
    if not cuda:
        torch.set_num_threads(max(1, min(8, a.threads)))
    cache = K.open_cache(a.cache)
    if cache.meta.get("label") != "train":
        raise SystemExit("--cache is not the prereg44 train cache")
    idx = K.level_index(cache)
    kw = {} if small is None else {"batch_games": small[0], "moves": small[1]}
    ds = K.RankBatches(cache.dir, cache.off, cache.color == "W", idx, K.SEED, **kw)
    config = {"plan": "PLAN.md 3", "total_steps": total, "ckpt_every": ckpt_every, "lr": K.LR,
              "weight_decay": K.WD, "onecycle_pct": K.PCT, "weight_seed": 0, "sampler_seed": K.SEED,
              "batch_games": ds.batch_games, "moves": ds.moves, "min_player_games": K.MIN_PLAYER_GAMES,
              "trunk": {"name": "GPSENet (prereg42 V2)", "blocks": K.BLOCKS, "channels": K.CHANNELS,
                        "emb_dim": K.EMB_DIM, "se_ratio": K.SE_RATIO, "gpool_blocks": list(K.GPOOL_BLOCKS),
                        "gpool_ch": K.GPOOL_CH, "source": K.P42_SOURCE},
              "head": "Linear(256, 10) on the mean of the per-move vectors; cross-entropy",
              "fp16": cuda, "ema_decay": K.EMA_DECAY, "cache_codes_sha256": cache.meta["codes_sha256"],
              "sampled_players_per_level": [int(len(b)) for b in idx["by_level"]], "small": small}
    net = K.build().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=K.LR, weight_decay=K.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=total, pct_start=K.PCT)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = K.Ema(net)
    n_params = int(sum(p.numel() for p in net.parameters()))

    if a.bench:
        loader = DataLoader(ds, batch_size=None, sampler=range(0, BENCH_WARMUP + a.bench), num_workers=a.workers,
                            pin_memory=cuda, prefetch_factor=4 if a.workers else None)
        net.train()
        t1, pos = None, 0
        for s, b in enumerate(loader):
            if s == BENCH_WARMUP:
                if cuda:
                    torch.cuda.synchronize()
                t1, pos = time.time(), 0
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = loss_of(b, net, cuda)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            pos += len(b["seg"])
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t1
        sps = a.bench / dt
        res = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers,
               "steps_timed": a.bench, "steps_per_s": round(sps, 3), "positions_per_s": round(pos / dt, 1),
               "projected_hours_%d_steps" % total: round(total / sps / 3600.0, 2), "parameters": n_params,
               "torch": torch.__version__, "small": small}
        print(json.dumps(res), flush=True)
        return 0

    start, seconds, positions = 0, 0.0, 0
    os.makedirs(ckpt_dir, exist_ok=True)
    last = last_checkpoint(ckpt_dir)
    if a.auto_resume and last is not None:
        a.resume = True
    if a.resume:
        if last is None:
            raise SystemExit("--resume: no checkpoint in %s" % ckpt_dir)
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["config"] != config:
            raise SystemExit("--resume: the checkpoint's config differs:\n%s\n%s" % (ck["config"], config))
        net.load_state_dict(ck["net"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, seconds, positions = ck["step"], ck["seconds"], ck["positions"]
        print("resumed from %s (step %d)" % (last, start), flush=True)
    elif last is not None:
        raise SystemExit("checkpoints already exist in %s; use --resume" % ckpt_dir)
    t0 = time.time()

    def save(step):
        path = os.path.join(ckpt_dir, "step_%06d.pt" % step)
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "config": config,
                    "seconds": seconds + time.time() - t0, "positions": positions, "git_head": git_head(),
                    "code_sha256": code_sha()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print("saved %s" % path, flush=True)

    loader = DataLoader(ds, batch_size=None, sampler=range(start, end), num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None, persistent_workers=False)
    net.train()
    tw, pw, lw, nw = time.time(), 0, 0.0, 0
    step, stopped, why = start, False, None
    with open(log_path, "a", encoding="utf-8", newline="\n") as logf:
        for s, b in zip(range(start, end), loader):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = loss_of(b, net, cuda)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(net)
            step = s + 1
            positions += len(b["seg"])
            pw += len(b["seg"])
            lv = float(loss.detach())
            if not np.isfinite(lv):
                raise SystemExit("loss is not finite at step %d" % step)
            lw += lv
            nw += 1
            if step % a.log_every == 0 or step == end:
                dt = time.time() - tw
                logf.write(json.dumps({"step": step, "loss": round(lw / nw, 5), "positions_per_s": round(pw / dt, 1),
                                       "steps_per_s": round(nw / dt, 3), "lr": sched.get_last_lr()[0],
                                       "elapsed_s": round(seconds + time.time() - t0, 1)}) + "\n")
                logf.flush()
                tw, pw, lw, nw = time.time(), 0, 0.0, 0
            if step % ckpt_every == 0:
                save(step)
            if step % STOP_EVERY == 0 and step < end:
                fr = free_ram_mb() if a.min_free_ram_mb > 0 else None
                if os.path.exists(stop_path):
                    why = "STOP file"
                elif fr is not None and fr < a.min_free_ram_mb:
                    why = "free RAM %.0f MB < %.0f MB" % (fr, a.min_free_ram_mb)
                if why:
                    if step % ckpt_every:
                        save(step)
                    stopped = True
                    break
    seconds += time.time() - t0
    if stopped:
        print("%s: saved at step %d, exiting (resume with --auto-resume; remove %s if present)" % (why, step, stop_path),
              flush=True)
        return 4
    if step == end and step % ckpt_every and (a.smoke or step == total):
        save(step)
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    rec = {"format": FORMAT, "config": config, "finished": step == total, "steps": step, "smoke": bool(a.smoke),
           "device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers, "git_head": git_head(),
           "code_sha256": code_sha(), "torch": torch.__version__, "parameters": n_params,
           "positions_seen": int(positions), "seconds": round(seconds, 1),
           "checkpoints": {os.path.basename(p): K.sha256_file(p) for p in cks}}
    K.write_json(rec_path, rec)
    print(json.dumps({k: rec[k] for k in ("finished", "steps", "seconds", "positions_seen")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
