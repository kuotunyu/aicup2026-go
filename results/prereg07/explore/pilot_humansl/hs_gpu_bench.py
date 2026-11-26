"""PREREG_07 pilot: RTX 4090 throughput, precision agreement, a rank-grid scan, and CPU featurisation scaling.

    GPU part (at most ~8 minutes in total, refuses to start when nvidia-smi lists another compute process):
    PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_gpu_bench.py gpu
    CPU part (featurisation with N worker processes, no GPU):
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_gpu_bench.py cpu

Fit-player fixture games only (data/prereg07/fixtures/fit_sample.json). Writes hs_gpu_bench.json / hs_cpu_feat.json
next to this file (numbers for README.md).
gpu: 1 evals/s for fp16 and bf16 autocast (and fp32 for reference) at batches 256 / 512 / 1024 / 2048, TIME_PER_CELL s
       each after warm-up (cudnn.benchmark on); fp16 also with channels_last at 1024.
     2 agreement on the same positions x PILOT profiles: log p(played move) fp16 / bf16 vs fp32 on the GPU, and GPU fp32
       vs CPU fp32 on a subset.
     3 grid scan: GRID_GAMES fixture games per level x all 29 KataGo rank profiles (KGS preset) and the same with source
       FOX: per-level mean log-likelihood per move and where it peaks (design input for the pilot grid).
cpu: positions/s of featurize_game with a spawn Pool of N processes (N in WORKERS) on the same games.
"""

import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (data/local guard)
import hs_parity as PAR  # noqa: E402
from pilot_run import PILOT_PROFILES  # noqa: E402

TIME_PER_CELL = 6.0
BUDGET_S = 480.0          # sections that would start after this many seconds are skipped (task cap: 10 min of GPU)
BATCHES = [256, 512, 1024, 2048]
GRID_GAMES = 3
WORKERS = [1, 2, 4, 8, 12, 16, 20, 23]


def other_compute_processes():
    """Rows of nvidia-smi's process table whose Type is exactly 'C' (pure compute), i.e. another job on the GPU."""
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    rows = []
    in_proc = False
    for line in out.splitlines():
        if "Processes:" in line:
            in_proc = True
            continue
        if in_proc:
            m = re.match(r"\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C|G|C\+G)\s+(.*?)\s+(\S+)\s*\|", line)
            if m and m.group(2) == "C":
                rows.append({"pid": int(m.group(1)), "name": m.group(3).strip()})
    return rows, out


def featurize_many(games, sides=("target",)):
    return [H.featurize_game(g["sgf_content"], g["color"], sides=sides) for g in games]


def cat(feats):
    return {k: np.concatenate([f[k] for f in feats]) for k in ("bin", "glob", "legal", "move", "pla", "side")}


def bench_cell(torch, model, profiles, feats, precision, batch, channels_last=False):
    sc = H.Scorer(model, profiles, device="cuda", precision=precision, batch=batch, channels_last=channels_last)
    per = max(1, batch // sc.N)
    n_pos = len(feats["move"])
    # inputs resident on the GPU so that the cell measures the network (+ masking / gather), not featurisation
    sub = lambda s, e: {k: v[s:e] for k, v in feats.items()}  # noqa: E731
    tw = time.time()
    sc.logp_moves(sub(0, per * 2))                          # warm-up (cudnn autotune for this shape)
    torch.cuda.synchronize()
    warm = time.time() - tw
    t0, evals, s = time.time(), 0, 0
    while time.time() - t0 < TIME_PER_CELL:
        e = s + per
        if e > n_pos:
            s, e = 0, per
        sc.logp_moves(sub(s, e))
        evals += per * sc.N
        s = e
    torch.cuda.synchronize()
    secs = time.time() - t0
    if channels_last:
        model.to(memory_format=torch.contiguous_format)
    return {"precision": precision, "batch_evals": per * sc.N, "positions_per_batch": per,
            "channels_last": channels_last, "evals": evals, "seconds": secs, "evals_per_s": evals / secs,
            "warmup_seconds_incl_cudnn_autotune": warm,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}


def cmd_gpu():
    import torch
    procs, smi = other_compute_processes()
    if procs:
        raise SystemExit(f"another compute process is on the GPU, not starting: {procs}")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False")
    t_all = time.time()
    torch.backends.cudnn.benchmark = True
    fx = PAR.load_fixture()
    rep = {"what": "PREREG_07 pilot: RTX 4090 HumanSL benchmark (fit-player fixture games)", "created_utc": H.utc_now(),
           "environment": H.environment(), "nvidia_smi_compute_processes_before": procs,
           "checkpoint": H.check_checkpoint(H.CKPT_PATH, full_hash=False)}
    out_path = os.path.join(HERE, "hs_gpu_bench.json")
    rep["budget_seconds"] = BUDGET_S
    rep["skipped"] = []

    def save():
        rep["gpu_seconds_total"] = time.time() - t_all
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)

    def time_left(section):
        if time.time() - t_all > BUDGET_S:
            rep["skipped"].append(section)
            print(f"budget: skipping {section}", flush=True)
            save()
            return False
        return True

    model, _ = H.load_model(H.CKPT_PATH, device="cuda", check=False)
    profiles = H.profiles_from_names(PILOT_PROFILES)
    games = [g for g in fx["games"]][:20]
    t = time.time()
    feats = cat(featurize_many(games))
    rep["bench_positions"] = int(len(feats["move"]))
    rep["featurise_seconds_one_process"] = time.time() - t
    # ---- 1 throughput
    cells = []
    rep["throughput"] = cells
    plan_cells = [(p, b, False) for p in ("fp16", "bf16") for b in BATCHES] + [("fp16", 1024, True), ("fp32", 1024, False)]
    for prec, b, cl in plan_cells:
        if not time_left(f"throughput {prec} {b}{' channels_last' if cl else ''}"):
            continue
        cells.append(bench_cell(torch, model, profiles, feats, prec, b, channels_last=cl))
        print(json.dumps(cells[-1]), flush=True)
        save()
    rep["throughput_note"] = (f"fp32 cell uses PyTorch defaults (cuDNN TF32 convolutions on); inputs featurised beforehand; each cell runs Scorer.logp_moves (H2D copy of uint8 "
                              f"planes, expand over {len(profiles)} profiles, forward, mask, log-softmax, gather) for "
                              f"{TIME_PER_CELL} s after one warm-up call; cudnn.benchmark on")
    # ---- 2 agreement
    if time_left("agreement"):
        n_agree = min(len(feats["move"]), 600)
        sub = {k: v[:n_agree] for k, v in feats.items()}
        lp = {}
        torch.backends.cudnn.allow_tf32 = False          # the fp32 reference is true fp32 (cuDNN TF32 off)
        torch.backends.cuda.matmul.allow_tf32 = False
        lp["fp32"] = H.Scorer(model, profiles, device="cuda", precision="fp32", batch=1024).logp_moves(sub)
        torch.backends.cudnn.allow_tf32 = True           # PyTorch default (TF32 convolutions)
        lp["fp32_tf32"] = H.Scorer(model, profiles, device="cuda", precision="fp32", batch=1024).logp_moves(sub)
        for prec in ("fp16", "bf16"):
            lp[prec] = H.Scorer(model, profiles, device="cuda", precision=prec, batch=1024).logp_moves(sub)
        agree = {"positions": n_agree, "profiles": len(profiles)}
        for prec in ("fp16", "bf16", "fp32_tf32"):
            d = np.abs(lp[prec].astype(np.float64) - lp["fp32"])
            agree[f"{prec}_vs_fp32_logp_max_abs"] = float(d.max())
            agree[f"{prec}_vs_fp32_logp_mean_abs"] = float(d.mean())
            agree[f"{prec}_vs_fp32_logp_p99_abs"] = float(np.percentile(d, 99))
            agree[f"{prec}_vs_fp32_mean_logp_over_positions_diff"] = float(np.abs(
                lp[prec].astype(np.float64).mean(0) - lp["fp32"].mean(0)).max())
            same_argmax = (np.argmax(lp[prec], 1) == np.argmax(lp["fp32"], 1)).mean()
            agree[f"{prec}_vs_fp32_same_best_profile_per_position"] = float(same_argmax)
        model_cpu, _ = H.load_model(H.CKPT_PATH, device="cpu", check=False)
        n_cpu = 48
        cpu_sub = {k: v[:n_cpu] for k, v in feats.items()}
        two = H.profiles_from_names([PILOT_PROFILES[0], PILOT_PROFILES[-1]])
        lp_cpu = H.Scorer(model_cpu, two, device="cpu", precision="fp32", batch=64).logp_moves(cpu_sub)
        torch.backends.cudnn.allow_tf32 = False
        lp_gpu = H.Scorer(model, two, device="cuda", precision="fp32", batch=64).logp_moves(cpu_sub)
        torch.backends.cudnn.allow_tf32 = True
        agree["gpu_fp32_vs_cpu_fp32_logp_max_abs"] = float(np.abs(lp_cpu.astype(np.float64) - lp_gpu).max())
        agree["gpu_fp32_vs_cpu_fp32_positions"] = n_cpu
        agree["gpu_fp32_vs_cpu_fp32_profiles"] = [p.name for p in two]
        agree["fp32_reference"] = "GPU fp32 with cudnn.allow_tf32 = False and matmul TF32 off"
        rep["agreement"] = agree
        print(json.dumps(agree), flush=True)
        del model_cpu
        save()
    # ---- 3 grid scan (fp16): all grid games in one padded call per source, split per game afterwards
    scan = {}
    rep["grid_scan"] = scan
    grid_games = []
    for lv in H.LEVELS:
        grid_games += [g for g in fx["games"] if g["rank"] == lv][:GRID_GAMES]
    gfeats = featurize_many(grid_games) if time_left("grid featurisation") else []
    lens = [len(f["move"]) for f in gfeats]
    gcat = cat(gfeats) if gfeats else None
    for source in ("KGS", "FOX"):
        if gcat is None or not time_left(f"grid scan {source}"):
            continue
        profs = [H.Profile(r, source=source) for r in H.RANKS]
        sc = H.Scorer(model, profs, device="cuda", precision="fp16", batch=2048)
        t = time.time()
        lp_all = sc.logp_moves(gcat).astype(np.float64)
        secs = time.time() - t
        per_game, o = [], 0
        for g, n in zip(grid_games, lens):
            lpg = lp_all[o:o + n]
            o += n
            per_game.append({"level": g["rank"], "mean_logp": lpg.mean(0).tolist(),
                             "sum_logp": lpg.sum(0).tolist(), "moves": int(n)})
        lvl = {}
        for lv in H.LEVELS:
            gs = [x for x in per_game if x["level"] == lv]
            tot = np.sum([x["sum_logp"] for x in gs], axis=0)
            cnt = sum(x["moves"] for x in gs)
            mean = tot / cnt
            lvl[lv] = {"peak": H.RANKS[int(np.argmax(mean))], "mean_logp_per_move": mean.tolist(),
                       "per_game_peaks": [H.RANKS[int(np.argmax(x["mean_logp"]))] for x in gs]}
        evals = sum(x["moves"] for x in per_game) * len(profs)
        scan[source] = {"ranks": H.RANKS, "games_per_level": GRID_GAMES, "per_level": lvl, "evals": evals,
                        "seconds_incl_autotune": secs, "evals_per_s_incl_autotune": evals / secs}
        print(source, {lv: v["peak"] for lv, v in lvl.items()}, f"{evals / secs:.0f} evals/s", flush=True)
        save()
    procs_after, _ = other_compute_processes()
    rep["nvidia_smi_compute_processes_after"] = procs_after
    save()
    print("->", out_path, f"{rep['gpu_seconds_total']:.0f}s")


def _feat(args):
    sgf, colour = args
    f = H.featurize_game(sgf, colour)
    return len(f["move"])


def cmd_cpu():
    fx = PAR.load_fixture()
    games = fx["games"]
    args = [(g["sgf_content"], g["color"]) for g in games]
    rep = {"what": "featurize_game throughput with N spawn worker processes (fit fixture games, target side)",
           "created_utc": H.utc_now(), "logical_cpus": os.cpu_count(), "games": len(games), "rows": []}
    for n in WORKERS:
        use = args if n > 1 else args[:24]
        with mp.get_context("spawn").Pool(n) as pool:
            pool.map(_feat, use[:n], chunksize=1)          # start-up / imports outside the timing
            t = time.time()
            T = sum(pool.imap_unordered(_feat, use, chunksize=1))
            secs = time.time() - t
        rep["rows"].append({"workers": n, "games": len(use), "positions": T, "seconds": secs,
                            "positions_per_s": T / secs})
        print(json.dumps(rep["rows"][-1]), flush=True)
    with open(os.path.join(HERE, "hs_cpu_feat.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    {"gpu": cmd_gpu, "cpu": cmd_cpu}[sys.argv[1]]()
