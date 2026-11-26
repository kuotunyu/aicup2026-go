"""prereg06 explore / katago: build a RANDOM-WEIGHT stand-in for b18c384nbt-humanv0 on CPU and time it.

No weights are downloaded. The real checkpoint stores its own config; here we use the closest public config
"b18c384nbt-mish-meta" with version 14 (lightvector, KataGo issue #1009: HumanSL was trained as version 14 with
mish activations). What this measures:
  - parameter count and multiply-accumulates (MACs) per position for the trunk + heads (hooks on Conv2d/Linear),
  - that KataGo's python SGFMetadata builds a 192-wide metadata row for every rank profile we would use,
    and that different profiles give different rows,
  - CPU forward throughput (positions x profiles per second) with at most 8 threads,
  - the masked log-softmax we would use to read p(actual move | profile) from policy channel 0.
Random weights mean the policy values are meaningless; only shapes, cost and plumbing are checked.

Usage (repo root, Git Bash):
  .venv/Scripts/python.exe results/prereg06/explore/katago/kg_model_bench.py --katago-python <clone>/python
Writes results/prereg06/explore/katago/kg_model_bench.json (only file written).
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kg_guard import install_guard  # noqa: E402

install_guard()

import numpy as np  # noqa: E402
import torch  # noqa: E402

# inverse rank used by KataGo: 9d=1 ... 1d=9, 1k=10 ... 20k=29 (cpp/neuralnet/sgfmetadata.cpp getProfile)
PROFILES = ["20k", "18k", "16k", "14k", "13k", "12k", "11k", "10k", "9k", "8k", "7k", "6k", "5k", "4k", "3k", "2k",
            "1k", "1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d"]


def inverse_rank(s):
    n = int(s[:-1])
    return 10 - n if s.endswith("d") else 9 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago-python", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "kg_model_bench.json"))
    args = ap.parse_args()
    torch.set_num_threads(min(args.threads, 8))

    sys.path.insert(0, os.path.abspath(args.katago_python))
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model
    from katago.game.sgfmetadata import SGFMetadata

    config = copy.deepcopy(modelconfigs.config_of_name["b18c384nbt-mish-meta"])
    config["version"] = 14
    torch.manual_seed(0)
    model = Model(config, 19)
    model.initialize()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # ---- MAC count with hooks (per single position, metadata encoder included)
    macs = {"conv": 0, "linear": 0}

    def conv_hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        macs["conv"] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * k

    def lin_hook(mod, inp, out):
        macs["linear"] += out.numel() // out.shape[0] * mod.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    # ---- metadata rows for every profile, built exactly like C++ makeBasicRankProfile (KGS, byo-yomi 1200+5x30, 2020-03-01)
    def kgs_profile(r, source=SGFMetadata.SOURCE_KGS):
        inv = inverse_rank(r)
        return SGFMetadata(inverseBRank=inv, inverseWRank=inv, bIsHuman=True, wIsHuman=True,
                           gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True,
                           mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
                           gameDate=datetime.date(2020, 3, 1), source=source)

    rows = np.stack([kgs_profile(r).get_metadata_row(nextPlayer="b", boardArea=361) for r in PROFILES])
    fox_rows = np.stack([kgs_profile(r, SGFMetadata.SOURCE_FOX).get_metadata_row(nextPlayer="b", boardArea=361)
                         for r in PROFILES])
    meta_checks = {
        "row_width": int(rows.shape[1]),
        "all_profiles_distinct": bool(len({r.tobytes() for r in rows}) == len(PROFILES)),
        "rank_ones_for_5k": int(rows[PROFILES.index("5k"), 6:40].sum()),
        "rank_ones_for_9d": int(rows[PROFILES.index("9d"), 6:40].sum()),
        "kgs_vs_fox_rows_differ_only_in_source_onehot": bool(
            np.array_equal(np.nonzero((rows != fox_rows).any(axis=0))[0], np.array([151 + 2, 151 + 3]))),
    }

    # ---- one dummy position (empty board, black to move, Japanese-rule globals) replicated over profiles
    bin_in = torch.zeros(1, 22, 19, 19)
    bin_in[:, 0] = 1.0                      # on-board mask
    glob_in = torch.zeros(1, 19)
    glob_in[:, 5] = -6.5 / 20.0             # self komi for black, territory scoring at move 0
    glob_in[:, 9] = 1.0                     # SCORING_TERRITORY
    glob_in[:, 10] = 1.0                    # TAX_SEKI
    with torch.no_grad():
        model(bin_in, glob_in, input_meta=torch.tensor(rows[:1]))
    for h in hooks:
        h.remove()
    macs_total = macs["conv"] + macs["linear"]

    # ---- CPU throughput: batch = positions x profiles rows
    b = args.batch
    reps = (b + len(PROFILES) - 1) // len(PROFILES)
    meta_b = torch.tensor(np.tile(rows, (reps, 1))[:b])
    with torch.no_grad():
        model(bin_in.expand(b, -1, -1, -1).contiguous(), glob_in.expand(b, -1).contiguous(), input_meta=meta_b)  # warm-up
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = model(bin_in.expand(b, -1, -1, -1).contiguous(), glob_in.expand(b, -1).contiguous(), input_meta=meta_b)
        dt = time.perf_counter() - t0
        policy_logits = model.postprocess_output(out)[0][0]      # (N, num_policy_outputs, 362)
        logits0 = policy_logits[:, 0, :]
        legal = torch.ones(362, dtype=torch.bool)                  # empty board: every point + pass legal
        logp = torch.log_softmax(logits0.masked_fill(~legal, float("-inf")), dim=-1)
        sample_move = 3 * 19 + 15                                  # the point "pd" (x=15, y=3)
        lp = logp[:len(PROFILES), sample_move]

    res = {
        "what": "random-weight stand-in for b18c384nbt-humanv0 (config b18c384nbt-mish-meta, version 14); CPU only",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "params": int(n_params),
        "params_fp32_megabytes": n_params * 4 / 1e6,
        "macs_per_eval": int(macs_total),
        "gflops_per_eval": 2 * macs_total / 1e9,
        "macs_conv": int(macs["conv"]),
        "macs_linear": int(macs["linear"]),
        "policy_output_shape": list(policy_logits.shape),
        "profiles": PROFILES,
        "metadata_checks": meta_checks,
        "cpu_batch": b,
        "cpu_iters": args.iters,
        "cpu_seconds": dt,
        "cpu_evals_per_second": b * args.iters / dt,
        "logp_pd_first_profiles_randomweights": [float(x) for x in lp[:5]],
        "note": "shared desktop CPU with other jobs running; fp32; random weights so outputs are meaningless",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
