"""Synthetic chunk folders for a mechanics smoke test of pilot_fuse.py (NOT HumanSL output; never a result).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_synth.py null    data/prereg07/smoke_synth/null
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_synth.py planted data/prereg07/smoke_synth/planted
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/pilot_fuse.py --pilot-dir data/prereg07/smoke_synth/null    --out data/prereg07/smoke_synth/null_report.json
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/pilot_fuse.py --pilot-dir data/prereg07/smoke_synth/planted --out data/prereg07/smoke_synth/planted_report.json

mode null   : per-game curves are pure noise -> Q02+HS must stay ~ Q02 (fusion weight ~ 0, gate fails).
mode planted: curves peak at a profile tied to the game's true level, plus noise -> Q02+HS must beat Q02 clearly.
The inner labels (results/prereg02/rank/dumps/inner_keys.npz 'label', training players) are used ONLY to plant the
synthetic signal. Same generator as the first run of 2026-10-02 (scratchpad make_synth.py, default_rng(7)), moved here
so it is kept; the chunk files carry only the keys pilot_fuse.load_set reads.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (data/local guard)

REPO = H.REPO
LEVELS = H.LEVELS


def main():
    mode, out = sys.argv[1], os.path.abspath(sys.argv[2])
    if mode not in ("null", "planted"):
        raise SystemExit("mode is null or planted")
    with open(os.path.join(REPO, "data", "prereg07", "pilot", "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    plan["synthetic"] = mode
    N = len(plan["profiles"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f)
    with np.load(os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz"), allow_pickle=False) as k:
        lab = k["label"].astype(int)
    rng = np.random.default_rng(7)

    def curve(levels):
        G = len(levels)
        x = -3.0 + 0.3 * rng.standard_normal((G, 1)) + 0.05 * rng.standard_normal((G, N))
        if mode == "planted":
            centre = 1.0 + levels * (N - 3) / 9.0
            j = np.arange(N)[None, :]
            x = x - 0.004 * (j - centre[:, None]) ** 2 + 0.02 * rng.standard_normal((G, N))
        return x

    for name in ("fitmap", "bank", "mono"):
        s = plan["sets"][name]
        if name == "fitmap":
            refs = np.arange(len(s["games"]))
            lv = np.array([LEVELS.index(g["rank"]) for g in s["games"]])
        else:
            refs = np.array(s["rows"])
            lv = lab[refs]
        cnt = np.full(len(refs), 110, dtype=np.int32)
        x = curve(lv)
        os.makedirs(os.path.join(out, name), exist_ok=True)
        for ci, st in enumerate(range(0, len(refs), 200)):
            e = min(len(refs), st + 200)
            sums = np.zeros((e - st, 2, N))
            sums[:, 0, :] = x[st:e] * cnt[st:e, None]
            counts = np.zeros((e - st, 2), dtype=np.int32)
            counts[:, 0] = cnt[st:e]
            np.savez(os.path.join(out, name, f"chunk_{ci:05d}.npz"), ref=refs[st:e], sums=sums, counts=counts,
                     failed=np.zeros(e - st, bool), precision=np.array("synthetic"), ckpt_sha256=np.array("synthetic"),
                     hs_lib_sha256=np.array("synthetic"), sides=np.array(["target"]))
    print("ok", out)


if __name__ == "__main__":
    main()
