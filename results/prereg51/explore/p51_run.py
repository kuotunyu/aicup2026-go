"""prereg51 (PLAN.md 2): two arms of prereg49's c49 recipe (V2, seed 0, T49 players), each changing one thing.

    L : 480,000 steps (c49: 320,000), trunk B10C128 as c49               -> 4090
    B : 320,000 steps, larger trunk (B15C128, or B12C128 per PLAN.md 3)   -> Colab G4

    GPUPY results/prereg51/explore/p51_run.py bench --arm B --size B15C128 [--steps-timed 300]
    GPUPY results/prereg51/explore/p51_run.py train --arm L|B [--size NAME for B] [--auto-resume ...] [--out DIR --record F]
    GPUPY results/prereg51/explore/p51_run.py embed --arm L|B [--size NAME for B] [p42_embed.py arguments]

Nothing of prereg36 / 42 / 43 / 49 / 50 is edited. Like p50_run: sets p42_lib's constants, the trunk size
(global-pooling blocks = index % 3 == 2), checkpoints every 16,000 steps (L) / 5,000 steps (B, Colab), resume picks the highest step number;
players = T49 (p49_lib, prereg49 adopted).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "results", r, "explore") for r in ("prereg42", "prereg43", "prereg49", "prereg50")):
    if p not in sys.path:
        sys.path.insert(0, p)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p42_lib as V  # noqa: E402
import p43_run as R43  # noqa: E402
import p50_run as R50  # noqa: E402

L = V.L
SEED = 0
ARMS = {"L": {"steps": 480000, "sizes": ("B10C128",), "ckpt_every": 16000},
        "B": {"steps": 320000, "sizes": ("B15C128", "B12C128"), "ckpt_every": 5000}}   # B: <= ~30 min on Colab


def pop(rest, flag, default=None):
    if flag not in rest:
        return default
    i = rest.index(flag)
    v = rest[i + 1]
    del rest[i:i + 2]
    return v


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("bench", "train", "embed"):
        raise SystemExit("usage: p51_run.py bench | train | embed --arm L|B [--size NAME] ...")
    rest = argv[1:]
    for f in ("--seed", "--variant"):
        if f in rest:
            raise SystemExit("%s is fixed by prereg51" % f)
    arm = pop(rest, "--arm")
    if arm not in ARMS:
        raise SystemExit("--arm must be L or B")
    size = pop(rest, "--size", ARMS[arm]["sizes"][0])
    if size not in ARMS[arm]["sizes"]:
        raise SystemExit("arm %s allows sizes %s" % (arm, ARMS[arm]["sizes"]))
    R50.set_size(size)
    V.TOTAL_STEPS, V.CKPT_EVERY = ARMS[arm]["steps"], ARMS[arm]["ckpt_every"]
    if argv[0] == "embed":
        import p42_embed as Em
        return Em.main(rest)
    V.SEED = SEED
    V.DATA = os.path.join(REPO, "data", "prereg51")
    import p42_train as T
    if T.V is not V or T.L is not L:
        raise SystemExit("p42_train did not import the same modules")
    T.last_checkpoint = R50.last_checkpoint_by_step
    if argv[0] == "bench":
        n = pop(rest, "--steps-timed", "300")
        T.BUDGET_HOURS = 1e9
        print(json.dumps({"arm": arm, "size": size, "blocks": L.BLOCKS, "channels": L.CHANNELS,
                          "gpool_blocks": list(V.GPOOL_BLOCKS), "total_steps": V.TOTAL_STEPS}), flush=True)
        return T.main(["--variant", R43.VARIANT, "--bench", n] + rest)
    import p49_lib as P
    import p49_run as R49
    L.Cache = P.Cache49
    V.t_index = R49.t_index49
    if "--smoke" in rest:
        return T.main(["--variant", R43.VARIANT] + rest)
    out = pop(rest, "--out", os.path.join(REPO, "data", "prereg51", arm))
    rec = pop(rest, "--record", os.path.join(HERE, "train_record_%s.json" % arm))
    return T.main(["--variant", R43.VARIANT, "--out", out, "--record", rec] + rest)


if __name__ == "__main__":
    sys.exit(main())
