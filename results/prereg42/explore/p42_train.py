"""prereg42 explore: train one recipe variant (PLAN.md 2, 3).

    GPUPY results/prereg42/explore/p42_train.py --variant V2 --bench           # timing only (300 steps, no files)
    bash  results/prereg42/explore/run_train.sh --variant V2 --workers 4        # 40,000 steps, checkpoints every 8,000
    bash  results/prereg42/explore/run_train.sh --variant V2 --workers 4 --resume
    PY    results/prereg42/explore/p42_train.py --variant V1 --smoke --device cpu --workers 0    # CPU smoke, minutes

Everything not named in PLAN.md 2 is prereg36's: T players, the move cache data/prereg36/cache, OneCycle lr 2e-3 /
wd 1e-4 / pct 0.05, seed 0, fp16 autocast, NCHW, 40,000 steps. Batch s is drawn from numpy default_rng([0, s]) inside
the worker (independent of the number of workers; --resume continues the same stream). EMA (decay 0.999) of the
embedding network is kept in every checkpoint.

Outputs: data/prereg42/<variant>/ckpt/step_NNNNN.pt, train_log.jsonl, STOP; the record
results/prereg42/explore/train_record_<variant>.json (--out / --record move them, e.g. to Drive on Colab).
Stop cleanly: create <out>/STOP (checked every 100 steps; saves and exits with code 4). The run also saves and
exits 4 when the free RAM falls below --min-free-ram-mb. Refuses to start on the GPU when nvidia-smi lists another
python compute process.
"""
import argparse
import glob
import hashlib
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
STOP_EVERY = 100
BENCH_WARMUP = 30
BUDGET_HOURS = 4.5


def free_ram_mb():
    """Available physical memory in MB (Windows GlobalMemoryStatusEx / Linux MemAvailable); None if unknown."""
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


def code_sha():
    h = {}
    for f in ("p42_lib.py", "p42_train.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h[f] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    with open(os.path.join(V.P36, "p36_lib.py"), "rb") as fh:
        h["p36_lib.py"] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return h


def other_python_on_gpu():
    import p36_train
    return p36_train.other_python_on_gpu()


def last_checkpoint(d):
    files = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return files[-1] if files else None


def loss_of(batch, net, head, spec, cuda, step):
    """One forward pass and the loss (PLAN.md 2). batch: dict of tensors on the device."""
    import torch
    import torch.nn.functional as F
    x = batch["x"].float()
    seg, y, t = batch["seg"], batch["y"], batch["t"]
    if spec["head"] == "subcenter":
        head.m = V.margin_at(step)
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        e = net.trunk(x)
        if V.is_tx(spec):
            n_seg = len(batch["gset"]) if spec["sampler"] == "episode" else len(y)
            g = net.pool(e, seg, n_seg, t)
    if not V.is_tx(spec):
        n_seg = len(batch["gset"]) if spec["sampler"] == "episode" else len(y)
        g = net.pool(e, seg, n_seg, t)
    g = g.float()
    if spec["sampler"] != "episode":
        emb = F.normalize(g, dim=1)
        return F.cross_entropy(V.head_logits(head, spec, emb, y), y)
    gn = F.normalize(g, dim=1)
    sets = F.normalize(L.segment_mean(gn, batch["gset"], 2 * len(y)), dim=1)
    a, b = sets[0::2], sets[1::2]
    cls = 0.5 * (F.cross_entropy(V.head_logits(head, spec, a, y), y) +
                 F.cross_entropy(V.head_logits(head, spec, b, y), y))
    sim = V.PROTO_SCALE * (a @ b.T)
    tgt = torch.arange(len(y), device=sim.device)
    proto = 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt))
    return cls + proto


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(V.VARIANTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", action="store_true", help="resume if a checkpoint exists, else start (Colab)")
    ap.add_argument("--steps", type=int, default=None, help="stop after this step number (schedule unchanged)")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="timing only: N steps after a warm-up, no files")
    ap.add_argument("--smoke", action="store_true", help="tiny batches, 3 steps, outputs under data/prereg42/smoke")
    ap.add_argument("--out", default=None, help="default data/prereg42/<variant> (smoke: data/prereg42/smoke/<variant>)")
    ap.add_argument("--record", default=None, help="default results/prereg42/explore/train_record_<variant>.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-free-ram-mb", type=float, default=4096)
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args(argv)
    spec = V.spec_of(a.variant)
    if a.device == "cpu" and not (a.smoke or a.bench):
        raise SystemExit("--device cpu is for --smoke / --bench only")
    small = None
    total, ckpt_every = V.TOTAL_STEPS, V.CKPT_EVERY
    if a.smoke:
        small = (4, 12) if spec["sampler"] == "episode" else (4, 6)
        out = a.out or os.path.join(V.DATA, "smoke", a.variant)
        rec_path = a.record or os.path.join(out, "train_record.json")
        if a.steps is None:
            a.steps = 3
        ckpt_every = 2
    else:
        out = a.out or os.path.join(V.DATA, a.variant)
        rec_path = a.record or os.path.join(HERE, "train_record_%s.json" % a.variant)
    ckpt_dir, log_path, stop_path = os.path.join(out, "ckpt"), os.path.join(out, "train_log.jsonl"), \
        os.path.join(out, "STOP")
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
    torch.manual_seed(V.SEED)
    if not cuda:
        torch.set_num_threads(2)
    cache = L.Cache()
    ds, T, ids_sha = V.make_batches(spec, cache, V.SEED, small)
    config = {"variant": a.variant, "spec": spec, "total_steps": total, "lr": V.LR, "weight_decay": V.WD,
              "onecycle_pct": V.PCT, "seed": V.SEED, "scale": L.SCALE, "blocks": L.BLOCKS, "channels": L.CHANNELS,
              "emb_dim": L.EMB_DIM, "fp16": cuda, "channels_last": False, "ema_decay": V.EMA_DECAY,
              "n_classes": len(T), "T_sha256_sorted_ids": ids_sha["T"], "cache_codes_sha256": cache.meta["codes_sha256"],
              "small": small}
    if spec["sampler"] == "episode":
        config["episode"] = {"players": ds.players, "set_moves": ds.set_moves, "ks": list(V.EP_KS),
                             "proto_scale": V.PROTO_SCALE}
    else:
        config["game"] = {"batch_games": ds.batch_games, "moves": ds.moves}
    if spec["trunk"] == "gpse":
        config["gpse"] = {"se_ratio": V.SE_RATIO, "gpool_blocks": list(V.GPOOL_BLOCKS), "gpool_ch": V.GPOOL_CH}
    if spec["pool"] == "txres":
        config["txres"] = {"layers": V.TX_LAYERS, "heads": V.TX_HEADS, "ff": V.TX_FF, "dropout": V.TX_DROPOUT,
                           "pos_bucket": V.POS_BUCKET, "pos_buckets": V.POS_BUCKETS, "pos_init_std": V.TXRES_POS_STD,
                           "pool_lr_mult": V.POOL_LR_MULT["txres"], "readout": "mean + Linear0(LN(masked mean))",
                           "cls": False, "token_layernorm": True}
    if spec["pool"] == "tx":
        config["tx"] = {"layers": V.TX_LAYERS, "heads": V.TX_HEADS, "ff": V.TX_FF, "dropout": V.TX_DROPOUT,
                        "pos_bucket": V.POS_BUCKET, "pos_buckets": V.POS_BUCKETS}
    if spec["head"] == "subcenter":
        config["subcenter"] = {"k": V.SUBCENTERS, "margin": V.MARGIN, "ramp": V.MARGIN_RAMP}
    net, head = V.build(spec, len(T))
    net, head = net.to(dev), head.to(dev)
    groups = V.param_groups(net, head, spec)
    if len(groups) == 1:            # V1 / V2 / V3 / V4: exactly the original optimiser and schedule
        opt = torch.optim.AdamW(groups[0]["params"], lr=V.LR, weight_decay=V.WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=V.LR, total_steps=total, pct_start=V.PCT)
    else:                           # V3b: pooling lr x0.1, same OneCycle shape (PLAN.md revision 3)
        opt = torch.optim.AdamW(groups, lr=V.LR, weight_decay=V.WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total,
                                                    pct_start=V.PCT)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = V.Ema(net)
    n_params = int(sum(p.numel() for p in net.parameters()))

    if a.bench:
        loader = DataLoader(ds, batch_size=None, sampler=range(0, BENCH_WARMUP + a.bench), num_workers=a.workers,
                            pin_memory=cuda, prefetch_factor=4 if a.workers else None)
        net.train()
        head.train()
        t1, pos = None, 0
        for s, b in enumerate(loader):
            if s == BENCH_WARMUP:
                if cuda:
                    torch.cuda.synchronize()
                t1, pos = time.time(), 0
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = loss_of(b, net, head, spec, cuda, s)
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
        hours = total / sps / 3600.0
        res = {"variant": a.variant, "device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers,
               "steps_timed": a.bench, "steps_per_s": round(sps, 3), "positions_per_s": round(pos / dt, 1),
               "projected_hours_%d_steps" % total: round(hours, 2), "budget_hours": BUDGET_HOURS,
               "ok": bool(hours <= BUDGET_HOURS), "embed_net_parameters": n_params, "torch": torch.__version__}
        print(json.dumps(res), flush=True)
        return 0 if res["ok"] or not cuda else 5

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
        head.load_state_dict(ck["head"])
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
        path = os.path.join(ckpt_dir, "step_%05d.pt" % step)
        torch.save({"net": net.state_dict(), "head": head.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "config": config, "seconds": seconds + time.time() - t0, "positions": positions,
                    "git_head": L.git_head(), "code_sha256": code_sha()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print("saved %s" % path, flush=True)

    loader = DataLoader(ds, batch_size=None, sampler=range(start, end), num_workers=a.workers, pin_memory=cuda,
                        prefetch_factor=4 if a.workers else None, persistent_workers=False)
    net.train()
    head.train()
    tw, pw, lw, nw = time.time(), 0, 0.0, 0
    step, stopped, why = start, False, None
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for s, b in zip(range(start, end), loader):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = loss_of(b, net, head, spec, cuda, s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(net)
            step = s + 1
            positions += len(b["seg"])
            pw += len(b["seg"])
            lw += float(loss.detach())
            nw += 1
            if not np.isfinite(float(loss.detach())):
                raise SystemExit("loss is not finite at step %d" % step)
            if step % a.log_every == 0 or step == end:
                dt = time.time() - tw
                log.write(json.dumps({"step": step, "loss": round(lw / nw, 5), "positions_per_s": round(pw / dt, 1),
                                      "steps_per_s": round(nw / dt, 3), "lr": sched.get_last_lr()[0],
                                      "elapsed_s": round(seconds + time.time() - t0, 1)}) + "\n")
                log.flush()
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
        print("%s: saved at step %d, exiting (resume with --resume; remove %s if present)" % (why, step, stop_path),
              flush=True)
        return 4
    if step == end and step % ckpt_every and (a.smoke or step == total):
        save(step)
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    rec = {"format": "c10-prereg42 train v1", "config": config, "finished": step == total, "steps": step,
           "smoke": bool(a.smoke), "device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers,
           "git_head": L.git_head(), "code_sha256": code_sha(), "torch": torch.__version__,
           "embed_net_parameters": n_params,
           "counts": {"T_players": len(T), "positions_seen": int(positions)},
           "split_sha256_sorted_ids": ids_sha, "seconds": round(seconds, 1),
           "checkpoints": {os.path.basename(p): L.sha256_file(p) for p in cks}}
    os.makedirs(os.path.dirname(os.path.abspath(rec_path)), exist_ok=True)
    L.write_json(rec_path, rec)
    print(json.dumps({k: rec[k] for k in ("finished", "steps", "seconds", "counts")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
