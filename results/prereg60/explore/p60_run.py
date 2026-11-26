"""prereg60 (PLAN.md Method): fine-tune c49 with the 18-plane input and samples at every move; per-game vectors.

    GPUPY results/prereg60/explore/p60_run.py train [--workers 4] [--resume | --auto-resume] [--steps N]
    GPUPY results/prereg60/explore/p60_run.py train --smoke --device cpu --workers 0 [--steps N<=3] [--auto-resume]
    GPUPY results/prereg60/explore/p60_run.py embed --ckpt data/prereg60/S0/ckpt/step_030000.pt --source store --players E
    GPUPY results/prereg60/explore/p60_run.py embed --ckpt ... --source pool --src data/test_v2_download/x/tests --pool public|private
    GPUPY results/prereg60/explore/p60_run.py embed --ckpt data/prereg49/S0/ckpt/step_320000.pt --source store --players E \
          --limit 200 --no-early --device cpu --workers 0 --compare <c49's store_E.npz>          # seam (a)

Nothing of prereg36 / 42 / 43 / 49 / 50 is edited: the trunk and head (p42_lib.build, V2), the loss
(p42_train.loss_of), the EMA (p42_lib.Ema), c49's class set and cache (p49_run.t_index49, p49_lib.Cache49), the
per-game vectors (p42_embed.main / game_vectors) are imported; the planes, the samplers and the widening are
p60_lib's.

train (PLAN.md Method):
  * start = c49's raw weights (data/prereg49/S0/ckpt/step_320000.pt, sha256 checked), first convolution widened with
    a zero channel; c49's head (5,771 classes); a new EMA (0.999) and a new optimiser.
  * batch s (s = 0 .. 29,999) = batch (320,000 + s) of c49's sampler stream (seed 0, T49, 96 games x <= 24 moves, one
    board symmetry per game) with p60_lib's planes: the stream continues where c49's training ended (NOTES.md 1).
  * AdamW (weight decay 1e-4), lr 5e-4 x p60_lib.lr_factor (500 warm-up steps, cosine to 0 at step 30,000), fp16
    autocast on the GPU, seed 0.
  * checkpoints data/prereg60/S0/ckpt/step_NNNNNN.pt: 10,000 / 20,000 / 30,000 are kept; every 1,000 steps a resume
    checkpoint replaces the previous one. --resume / --auto-resume pick the highest step number.
  * data/prereg60/STOP or <out>/STOP (checked every 100 steps) or free RAM < --min-free-ram-mb -> save, exit code 4.
  * refuses to start on the GPU when nvidia-smi lists another python compute process.
  * at step 30,000: results/prereg60/explore/train_record.json (checkpoint hashes; p60_eval judge needs it).
embed: p42_embed.main with p60_lib.GameSet60 and an 18-plane loader; raw weights, tta 1; output
  data/prereg60/emb/p60_step_030000_raw_tta1/<store_E | pool_public | pool_private>.npz (p42_embed's layout).
  --no-early (old sample set) / --zero-action (plane 17 all zero) are debugging switches for the seam: they need
  --limit; every --limit run writes under data/prereg60/seam (CPU) or data/prereg60/seam_gpu (GPU). The c49
  checkpoint itself (widened) is accepted only with --limit.

No score of any kind is computed here. Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p60_lib as P  # noqa: E402

V, L, EM, REPO = P.V, P.L, P.EM, P.REPO
DATA = os.path.join(REPO, "data", "prereg60")
OUT = os.path.join(DATA, "S0")
EMB = os.path.join(DATA, "emb")
SEAM = os.path.join(DATA, "seam")
SEAM_GPU = os.path.join(DATA, "seam_gpu")
STOP_FILE = os.path.join(DATA, "STOP")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
C49_CKPT = os.path.join(REPO, "data", "prereg49", "S0", "ckpt", "step_320000.pt")
C49_SHA = "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"
VARIANT = "V2"
SEED = 0
TOTAL_STEPS = 30000
LR = 5e-4
WARMUP = 500
KEEP = (10000, 20000, 30000)
RESUME_EVERY = 1000
STOP_EVERY = 100
SMOKE_STEPS = 3
SMOKE_SMALL = (4, 6)                      # games per batch, moves per game
CODE_FILES = (os.path.join(HERE, "p60_lib.py"), os.path.join(HERE, "p60_run.py"),
              os.path.join(V.HERE, "p42_lib.py"), os.path.join(V.HERE, "p42_train.py"),
              os.path.join(V.HERE, "p42_embed.py"), os.path.join(V.P36, "p36_lib.py"))


def code_sha():
    out = {}
    for f in CODE_FILES:
        with open(f, "rb") as fh:
            out[os.path.basename(f)] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def ckpt_file(out, step):
    return os.path.join(out, "ckpt", "step_%06d.pt" % int(step))


def emb_tag(step=TOTAL_STEPS):
    return "p60_step_%06d_raw_tta1" % int(step)


def _rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), REPO).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def stop_requested(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ================================================================ train
def cmd_train(a):
    import p42_train as T42
    import p49_lib as P49
    import p49_run as R49
    cuda = a.device == "cuda"
    if not cuda and not a.smoke:
        raise SystemExit("--device cpu is for --smoke only")
    if a.smoke:
        a.steps = SMOKE_STEPS if a.steps is None else a.steps
        if a.steps > SMOKE_STEPS:
            raise SystemExit("--smoke runs at most %d steps" % SMOKE_STEPS)
        out = os.path.join(DATA, "smoke", "S0")
        rec_path = os.path.join(out, "train_record.json")
        keep, resume_every = (), 1
    else:
        out, rec_path, keep, resume_every = OUT, TRAIN_RECORD, KEEP, RESUME_EVERY
    end = TOTAL_STEPS if a.steps is None else min(int(a.steps), TOTAL_STEPS)
    ckpt_dir, log_path = os.path.join(out, "ckpt"), os.path.join(out, "train_log.jsonl")
    stops = (os.path.join(out, "STOP"), STOP_FILE)
    hit = stop_requested(stops)
    if hit:
        print("STOP file present: %s" % hit, flush=True)
        return 4
    if cuda:
        others = T42.other_python_on_gpu()
        if others is None:
            raise SystemExit("nvidia-smi could not be queried; not starting")
        if others:
            raise SystemExit("another python compute process is on the GPU; not starting:\n  " + "\n  ".join(others))

    import torch
    from torch.utils.data import DataLoader

    dev = torch.device(a.device)
    torch.manual_seed(SEED)
    if not cuda:
        torch.set_num_threads(2)
    base_sha = L.sha256_file(C49_CKPT)
    if base_sha != C49_SHA:
        raise SystemExit("%s is not the c49 checkpoint" % _rel(C49_CKPT))
    ck0 = torch.load(C49_CKPT, map_location="cpu", weights_only=False)
    bcfg = ck0["config"]
    spec = V.spec_of(VARIANT)
    want = {"variant": VARIANT, "spec": spec, "blocks": L.BLOCKS, "channels": L.CHANNELS, "emb_dim": L.EMB_DIM,
            "scale": L.SCALE, "weight_decay": V.WD, "ema_decay": V.EMA_DECAY, "seed": SEED,
            "game": {"batch_games": L.BATCH_GAMES, "moves": L.SEG_MOVES}}
    bad = {k: (bcfg.get(k), v) for k, v in want.items() if bcfg.get(k) != v}
    if bad or list(bcfg["gpse"]["gpool_blocks"]) != list(V.GPOOL_BLOCKS):
        raise SystemExit("the c49 checkpoint's recipe is not the one this code builds: %s" % bad)
    base_step = int(ck0["step"])

    cache = P49.Cache49()
    T, order, t_off, ids = R49.t_index49(cache)
    if len(T) != bcfg["n_classes"] or ids["T"] != bcfg["T_sha256_sorted_ids"] or \
            cache.meta["codes_sha256"] != bcfg["cache_codes_sha256"]:
        raise SystemExit("the sampler's classes / cache are not c49's")
    games, moves = SMOKE_SMALL if a.smoke else (L.BATCH_GAMES, L.SEG_MOVES)
    ds = P.GameBatches60(cache.dir, cache.off, cache.color == "W", order, t_off, SEED, batch_games=games,
                         moves=moves, early=True, action=True)
    config = {"variant": VARIANT, "spec": spec,
              "p60": {"base_ckpt_sha256": base_sha, "base_step": base_step, "n_planes": P.N_PLANES,
                      "early_moves": True, "action_plane": True, "total_steps": TOTAL_STEPS, "lr": LR,
                      "warmup": WARMUP, "schedule": "linear warm-up, cosine to 0 at total_steps",
                      "weight_decay": V.WD, "ema_decay": V.EMA_DECAY, "seed": SEED, "scale": L.SCALE, "fp16": cuda,
                      "n_classes": len(T), "T_sha256_sorted_ids": ids["T"],
                      "cache_codes_sha256": cache.meta["codes_sha256"], "batch_games": games, "moves": moves,
                      "first_batch": base_step, "start_weights": "raw", "smoke": bool(a.smoke)}}
    net, head = V.build(spec, len(T))
    net.load_state_dict(ck0["net"])
    head.load_state_dict(ck0["head"])
    del ck0
    P.widen(net)
    net, head = net.to(dev), head.to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=LR, weight_decay=V.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: P.lr_factor(s, WARMUP, TOTAL_STEPS))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = V.Ema(net)

    start, seconds, positions = 0, 0.0, 0
    os.makedirs(ckpt_dir, exist_ok=True)
    last = P.last_ckpt(ckpt_dir)
    if a.auto_resume and last is not None:
        a.resume = True
    if a.resume:
        if last is None:
            raise SystemExit("--resume: no checkpoint in %s" % ckpt_dir)
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["config"] != config:
            raise SystemExit("--resume: the checkpoint's config differs:\n%s\n%s" % (ck["config"], config))
        if ck["step"] > end:
            raise SystemExit("--resume: %s is past step %d" % (last, end))
        net.load_state_dict(ck["net"])
        head.load_state_dict(ck["head"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, seconds, positions = int(ck["step"]), ck["seconds"], ck["positions"]
        del ck
        print("resumed from %s (step %d)" % (last, start), flush=True)
    elif last is not None:
        raise SystemExit("checkpoints already exist in %s; use --resume" % ckpt_dir)
    t0 = time.time()

    def save(step):
        path = ckpt_file(out, step)
        torch.save({"net": net.state_dict(), "head": head.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "config": config, "seconds": seconds + time.time() - t0, "positions": positions,
                    "git_head": L.git_head(), "code_sha256": code_sha()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        P.prune(ckpt_dir, keep, step)
        print("saved %s" % path, flush=True)

    loader = DataLoader(ds, batch_size=None, sampler=range(base_step + start, base_step + end),
                        num_workers=a.workers, pin_memory=cuda, prefetch_factor=4 if a.workers else None,
                        persistent_workers=False)
    net.train()
    head.train()
    tw, pw, lw, nw = time.time(), 0, 0.0, 0
    step, why = start, None
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for s, b in zip(range(start, end), loader):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = T42.loss_of(b, net, head, spec, cuda, s)
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
                log.write(json.dumps({"step": step, "loss": round(lw / nw, 5), "positions_per_s": round(pw / dt, 1),
                                      "steps_per_s": round(nw / dt, 3), "lr": opt.param_groups[0]["lr"],
                                      "elapsed_s": round(seconds + time.time() - t0, 1)}) + "\n")
                log.flush()
                tw, pw, lw, nw = time.time(), 0, 0.0, 0
            saved = step in keep or step % resume_every == 0 or step == end
            if saved:
                save(step)
            if step % STOP_EVERY == 0 and step < end:
                fr = T42.free_ram_mb() if a.min_free_ram_mb > 0 else None
                hit = stop_requested(stops)
                if hit:
                    why = "STOP file %s" % hit
                elif fr is not None and fr < a.min_free_ram_mb:
                    why = "free RAM %.0f MB < %.0f MB" % (fr, a.min_free_ram_mb)
                if why:
                    if not saved:
                        save(step)
                    break
    seconds += time.time() - t0
    if why:
        print("%s: saved at step %d, exiting with code 4 (resume with --auto-resume)" % (why, step), flush=True)
        return 4
    if step == TOTAL_STEPS or a.smoke:
        cks = {os.path.basename(ckpt_file(out, n)): L.sha256_file(ckpt_file(out, n))
               for n in sorted(set(keep) | {step}) if os.path.exists(ckpt_file(out, n))}
        rec = {"format": "c10-prereg60 train v1", "config": config, "finished": step == TOTAL_STEPS, "steps": step,
               "smoke": bool(a.smoke), "base_ckpt": _rel(C49_CKPT), "base_ckpt_sha256": base_sha,
               "model": "raw weights of %s" % os.path.basename(ckpt_file(out, TOTAL_STEPS)),
               "device": torch.cuda.get_device_name(0) if cuda else "cpu", "workers": a.workers,
               "git_head": L.git_head(), "code_sha256": code_sha(), "torch": torch.__version__,
               "counts": {"T_players": len(T), "positions_seen": int(positions)},
               "seconds": round(seconds, 1), "checkpoints": cks}
        L.write_json(rec_path, rec)
        print(json.dumps({k: rec[k] for k in ("finished", "steps", "seconds", "counts")}), flush=True)
    else:
        print(json.dumps({"finished": False, "steps": step, "seconds": round(seconds, 1)}), flush=True)
    return 0


# ================================================================ embed
def load_net60(early, action):
    """A replacement of p42_embed.load_net for 18-plane checkpoints (prereg60's, or c49 widened for the seam)."""

    def load_net(ckpt_path, weights, dev):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        spec = V.spec_of(VARIANT)
        if cfg.get("variant") != VARIANT or cfg.get("spec") != spec:
            raise SystemExit("%s is not a V2 checkpoint" % ckpt_path)
        if weights != "raw":
            raise SystemExit("PLAN.md: the raw weights are the model")
        if "p60" in cfg:
            net, _h = P.build_wide(spec)
            net.load_state_dict(ck["net"])
            run = "p60"
        else:
            if L.sha256_file(ckpt_path) != C49_SHA:
                raise SystemExit("%s is neither a prereg60 checkpoint nor c49" % ckpt_path)
            net, _h = V.build(spec)
            net.load_state_dict(ck["net"])
            P.widen(net)
            run = "c49w"
        run += ("" if early else "_noearly") + ("" if action else "_zeroact")
        return net.to(dev).eval(), spec, run, int(ck["step"])

    return load_net


def cmd_embed(rest):
    rest = list(rest)
    early, action = "--no-early" not in rest, "--zero-action" not in rest
    rest = [x for x in rest if x not in ("--no-early", "--zero-action")]
    for f in ("--weights", "--tta", "--mode", "--emb-root"):
        if f in rest:
            raise SystemExit("%s is fixed by prereg60 (raw, tta 1, data/prereg60/emb)" % f)
    if "--ckpt" not in rest:
        raise SystemExit("--ckpt is required")
    ckpt = rest[rest.index("--ckpt") + 1]
    debug = not (early and action)
    is_c49 = os.path.abspath(ckpt) == os.path.abspath(C49_CKPT)
    limited = "--limit" in rest
    if (debug or is_c49) and not limited:
        raise SystemExit("--no-early / --zero-action / the c49 checkpoint are for the seam: --limit N is required")
    cpu = "--device" in rest and rest[rest.index("--device") + 1] == "cpu"
    if cpu and not limited:
        raise SystemExit("--device cpu is for --limit N (seam / smoke) only")
    root = (SEAM if cpu else SEAM_GPU) if limited else EMB
    EM.load_net = load_net60(early, action)
    EM.GameSet = lambda codes, start, end, is_white: P.GameSet60(codes, start, end, is_white, early, action)
    return EM.main(rest + ["--weights", "raw", "--tta", "1", "--emb-root", root])


# ================================================================ command line
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "embed"):
        raise SystemExit("usage: p60_run.py train ... | embed ...")
    if argv[0] == "embed":
        return cmd_embed(argv[1:])
    ap = argparse.ArgumentParser(prog="p60_run.py train")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", action="store_true", help="resume if a checkpoint exists, else start")
    ap.add_argument("--steps", type=int, default=None, help="stop after this step number (schedule unchanged)")
    ap.add_argument("--smoke", action="store_true",
                    help="<= %d steps, %d games x %d moves per batch, outputs under data/prereg60/smoke; CPU allowed"
                         % (SMOKE_STEPS, SMOKE_SMALL[0], SMOKE_SMALL[1]))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-free-ram-mb", type=float, default=4096)
    ap.add_argument("--log-every", type=int, default=100)
    return cmd_train(ap.parse_args(argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
