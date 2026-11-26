"""prereg08 exploration: tiny CPU-only forward/backward timing of the KataGo HumanSL net (b18c384nbt-humanv0).

Purpose: get the RATIO of training-step cost to inference cost for the fine-tune options, so the measured 4090
inference speed (hs_gpu_bench.json, 1,805 evals/s fp16 channels_last) can be turned into a training-throughput
ESTIMATE. The 4090 is busy (PREREG_07 mono run), so this never touches CUDA.

Light CPU: torch threads fixed to 2 (the mono run feeds the GPU from CPU workers). Positions are real fit-player
positions from data/prereg07/fixtures/fit_sample.json (fit players only; never data/local).

Modes (batch B positions, one metadata row per position = one Yugen level profile):
  infer      : torch.inference_mode forward (what the pilot does)
  A_meta     : only metadata_encoder params trainable (grad flows back through the frozen trunk activations)
  B_head     : new 10-level conditioning = a fresh 10x384 embedding added at the trunk input (same place as the
               metadata encoder output); trunk frozen -> same backward path as A_meta
  C_full     : all params trainable (full fine-tune)
  D_probe    : trunk frozen and run under no_grad; only a small strength head on pooled trunk output trains
  E_full_se  : C_full + strength head (multi-task)
Output: ft_cpu_timing.json next to this file.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results", "prereg07", "explore", "pilot_humansl"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hs_lib as H  # noqa: E402  (installs the data/local guard)

torch.set_num_threads(2)
assert not torch.cuda.is_available() or True  # CUDA is never used below (device fixed to cpu)
DEV = "cpu"
B = int(os.environ.get("B", "16"))
REPS = int(os.environ.get("REPS", "3"))


def main():
    fx = json.load(open(os.path.join(REPO, "data", "prereg07", "fixtures", "fit_sample.json"), encoding="utf-8"))
    g = fx["games"][0]
    f = H.featurize_game(g["sgf_content"], g["color"])
    idx = np.linspace(0, len(f["move"]) - 1, B).astype(int)
    binp = torch.tensor(f["bin"][idx]).float()
    glob = torch.tensor(f["glob"][idx]).float()
    mv = torch.tensor(f["move"][idx].astype(np.int64))
    legal = torch.tensor(f["legal"][idx])
    prof = H.profiles_from_names(["rank_5k", "rank_1d"])
    meta_tab = torch.tensor(H.metadata_table(prof))
    meta = meta_tab[torch.arange(B) % 2, torch.tensor(f["pla"][idx].astype(np.int64))]

    t0 = time.time()
    model, state = H.load_model(device=DEV)
    load_s = time.time() - t0
    model.train(False)   # keep norm layers in eval behaviour (KataGo uses fixup-like norms; no batch stats)
    n_all = sum(p.numel() for p in model.parameters())
    n_meta = sum(p.numel() for p in model.metadata_encoder.parameters())
    n_policy = sum(p.numel() for n, p in model.named_parameters() if n.startswith("policy_head"))
    c_trunk = model.c_trunk
    level_emb = torch.nn.Parameter(torch.zeros(10, c_trunk))
    head = torch.nn.Sequential(torch.nn.Linear(2 * c_trunk, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
    lvl = torch.arange(B) % 10

    # hook: trunk output (input of the policy head) for the probe / strength head
    feats = {}

    def hook(_m, inp, _out):
        feats["trunk"] = inp[0]
    model.policy_head.register_forward_hook(hook)

    def policy_loss(out):
        logits = out[0][0][:, 0, :].float().masked_fill(~legal, float("-inf"))
        return torch.nn.functional.cross_entropy(logits, mv)

    def strength_loss():
        t = feats["trunk"]
        pooled = torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1)
        return torch.nn.functional.cross_entropy(head(pooled.float()), lvl)

    def set_trainable(which):
        for n, p in model.named_parameters():
            p.requires_grad_(which == "all" or (which == "meta" and n.startswith("metadata_encoder")))

    def step(mode):
        if mode == "infer":
            with torch.inference_mode():
                model(binp, glob, input_meta=meta)
            return
        if mode == "A_meta":
            set_trainable("meta")
            loss = policy_loss(model(binp, glob, input_meta=meta))
        elif mode == "B_head":
            set_trainable("none")
            # fresh level embedding added to the metadata path: emulate by feeding meta + grad-carrying offset
            x_meta_offset = level_emb[lvl]
            orig = model.metadata_encoder.forward
            model.metadata_encoder.forward = lambda im, eo: orig(im, eo) + x_meta_offset
            try:
                loss = policy_loss(model(binp, glob, input_meta=meta))
            finally:
                model.metadata_encoder.forward = orig
        elif mode == "C_full":
            set_trainable("all")
            loss = policy_loss(model(binp, glob, input_meta=meta))
        elif mode == "D_probe":
            set_trainable("none")
            with torch.no_grad():
                model(binp, glob, input_meta=meta)
            feats["trunk"] = feats["trunk"].detach()
            loss = strength_loss()
        elif mode == "E_full_se":
            set_trainable("all")
            loss = policy_loss(model(binp, glob, input_meta=meta)) + strength_loss()
        loss.backward()
        model.zero_grad(set_to_none=True)
        level_emb.grad = None
        head.zero_grad(set_to_none=True)

    res = {}
    for mode in ["infer", "A_meta", "B_head", "C_full", "D_probe", "E_full_se"]:
        step(mode)   # warm-up
        ts = []
        for _ in range(REPS):
            t = time.time()
            step(mode)
            ts.append(time.time() - t)
        res[mode] = {"median_s": float(np.median(ts)), "all_s": ts}
        print(mode, res[mode], flush=True)
    base = res["infer"]["median_s"]
    for m in res:
        res[m]["ratio_to_infer"] = res[m]["median_s"] / base
    out = {
        "what": "prereg08 explore: CPU-only fwd/bwd timing ratios of b18c384nbt-humanv0 (no CUDA used)",
        "created_utc": H.utc_now(), "environment": H.environment(), "torch_threads": torch.get_num_threads(),
        "batch_positions": B, "reps": REPS, "checkpoint_load_s": load_s, "checkpoint_load_how": state["_hs_load"],
        "params": {"all": n_all, "metadata_encoder": n_meta, "policy_head": n_policy, "c_trunk": c_trunk,
                   "new_level_embedding": 10 * c_trunk,
                   "strength_head": sum(p.numel() for p in head.parameters())},
        "config_keys": {k: state["config"][k] for k in ("version", "trunk_num_channels", "num_blocks")
                        if k in state["config"]} if isinstance(state.get("config"), dict) else None,
        "timing": res,
        "source_game": {"player_id": g["player_id"], "game_id": g["game_id"], "rank": g["rank"]},
    }
    json.dump(out, open(os.path.join(HERE, "ft_cpu_timing.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
