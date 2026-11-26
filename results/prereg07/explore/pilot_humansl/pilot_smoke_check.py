"""Check the GPU smoke chunks of pilot_run.py against an independent CPU fp32 computation (fit / inner players only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/pilot_smoke_check.py
    -> results/prereg07/explore/pilot_humansl/pilot_smoke_check.json

Inputs: data/prereg07/pilot_smoke/{fitmap,bank}/chunk_00000.npz, written by
    pilot_run.py run --set fitmap|bank --max-games 40 --out-dir data/prereg07/pilot_smoke   (GPU, fp16, channels_last)
Checks (tolerances fixed before this script was first run):
  1 format seam: pilot_fuse.load_set / curves read the chunks; refs equal the plan's first 40 rows; no failed game;
    counts[:, 0] equal the number of target moves of a fresh CPU featurisation; sums equal the per-position logp sums.
  2 values: for CHECK_GAMES games per set and CHECK_PROFILES, every position's log p(played move) from the GPU chunk
    (fp16 autocast, channels_last, padded batches) within 0.05 (absolute) of hs_lib's CPU fp32 path (no padding), and
    the per-game mean log p per move within 0.002. (hs_gpu_bench.json measured fp16 vs fp32 on the GPU: max 0.0084,
    per-profile means over 600 positions 0.0002; the bounds leave room for a different game sample.)
"""

import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (data/local guard)
import pilot_fuse as PF  # noqa: E402
import pilot_run as PR  # noqa: E402

SMOKE = os.path.join(H.REPO, "data", "prereg07", "pilot_smoke")
OUT = os.path.join(HERE, "pilot_smoke_check.json")
CHECK_GAMES = 2
CHECK_PROFILES = ["rank_20k", "rank_1d", "rank_9d"]
TOL_POS = 0.05
TOL_MEAN = 0.002


def main():
    t0 = time.time()
    plan = PR.load_plan()
    N = len(plan["profiles"])
    pidx = [plan["profiles"].index(p) for p in CHECK_PROFILES]
    model, _ = H.load_model(H.CKPT_PATH, device="cpu")
    sc = H.Scorer(model, H.profiles_from_names(CHECK_PROFILES), device="cpu", precision="fp32", batch=96, pad=False)
    rep = {"what": "pilot_run GPU smoke chunks vs CPU fp32 (hs_lib)", "created_utc": H.utc_now(),
           "smoke_dir": os.path.relpath(SMOKE, H.REPO).replace("\\", "/"), "profiles_checked": CHECK_PROFILES,
           "tolerances": {"position_logp_abs": TOL_POS, "game_mean_logp_abs": TOL_MEAN}, "sets": {}}
    ok_all = True
    for name in ("fitmap", "bank"):
        st = PF.load_set(name, N, SMOKE)
        x, ok = PF.curves(st, 0)
        with np.load(os.path.join(SMOKE, name, "chunk_00000.npz"), allow_pickle=False) as z:
            ch = {k: z[k] for k in z.files}
        games = PR.set_games(plan, name)[:len(st["ref"])]
        refs_ok = bool(np.array_equal(st["ref"], [g["ref"] for g in games]))
        n_tgt = [len(H.featurize_game(g["sgf_content"], g["color"])["move"]) for g in games]
        counts_ok = bool(np.array_equal(st["counts"][:, 0], n_tgt)) and bool((st["counts"][:, 1] == 0).all())
        sums_from_pos = np.zeros((len(games), N))
        np.add.at(sums_from_pos, ch["pos_game"], ch["pos_logp"].astype(np.float64))
        sums_ok = float(np.abs(sums_from_pos - st["sums"][:, 0, :]).max())
        rec = {"games": int(len(st["ref"])), "failed": int(st["failed"].sum()), "refs_equal_plan": refs_ok,
               "counts_equal_featurisation": counts_ok, "max_abs_sums_vs_position_sum": sums_ok,
               "curves_finite": bool(np.isfinite(x[ok]).all()), "curves_ok_games": int(ok.sum()),
               "chunk_meta": {k: str(ch[k]) for k in ("precision", "batch", "channels_last", "torch", "ckpt_bytes",
                                                     "hs_lib_sha256", "seconds", "created_utc") if k in ch},
               "mean_curve_over_games": x[ok].mean(0).tolist(), "values": []}
        worst_pos, worst_mean = 0.0, 0.0
        for gi in range(CHECK_GAMES):
            g = games[gi]
            f = H.featurize_game(g["sgf_content"], g["color"])
            lp_cpu = sc.logp_moves(f).astype(np.float64)                      # [T, 3]
            lp_gpu = ch["pos_logp"][ch["pos_game"] == gi][:, pidx].astype(np.float64)
            turn_ok = bool(np.array_equal(ch["pos_turn"][ch["pos_game"] == gi], f["turn"]))
            dpos = float(np.abs(lp_cpu - lp_gpu).max())
            dmean = float(np.abs(lp_cpu.mean(0) - lp_gpu.mean(0)).max())
            worst_pos, worst_mean = max(worst_pos, dpos), max(worst_mean, dmean)
            rec["values"].append({"game_id": g["game_id"], "level": g["rank"], "colour": g["color"],
                                  "positions": int(len(lp_cpu)), "turns_equal": turn_ok,
                                  "max_abs_position_logp": dpos, "max_abs_game_mean_logp": dmean,
                                  "cpu_mean_logp": lp_cpu.mean(0).tolist(), "gpu_mean_logp": lp_gpu.mean(0).tolist()})
            print(name, gi, g["rank"], len(lp_cpu), f"pos {dpos:.5f} mean {dmean:.6f}", flush=True)
            ok_all &= turn_ok
        rec["max_abs_position_logp"] = worst_pos
        rec["max_abs_game_mean_logp"] = worst_mean
        rec["passed"] = bool(refs_ok and counts_ok and sums_ok < 1e-3 and rec["failed"] == 0 and rec["curves_finite"]
                             and worst_pos <= TOL_POS and worst_mean <= TOL_MEAN)
        ok_all &= rec["passed"]
        rep["sets"][name] = rec
    rep["passed"] = bool(ok_all)
    rep["seconds"] = time.time() - t0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("values", "mean_curve_over_games")}
                          if isinstance(v, dict) and k == "sets" else v) for k, v in rep.items() if k != "sets"}))
    for name, v in rep["sets"].items():
        print(name, {k: v[k] for k in ("passed", "refs_equal_plan", "counts_equal_featurisation",
                                       "max_abs_sums_vs_position_sum", "max_abs_position_logp",
                                       "max_abs_game_mean_logp")})
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
