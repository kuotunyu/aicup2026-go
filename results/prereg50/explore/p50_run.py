"""prereg50 (PLAN.md 2): prereg43's c* recipe (V2, 320,000 steps, seed 0) with a larger trunk, on the players of the
base system (T49 when prereg49 is adopted, else prereg36's T).

    GPUPY results/prereg50/explore/p50_run.py bench --size B15C160 [--steps-timed 300]   # timing only, no files
    GPUPY results/prereg50/explore/p50_run.py train --size <chosen> [--resume | --auto-resume | --smoke ...]
    GPUPY results/prereg50/explore/p50_run.py embed --size <chosen> [p42_embed.py arguments]

Nothing of prereg36 / prereg42 / prereg43 / prereg49 is edited. This wrapper sets p42_lib's constants as p43_run.py
does, sets the trunk size (blocks, channels; global-pooling blocks = every block with index % 3 == 2, which is
(2, 5, 8) for 10 blocks), checkpoints every 16,000 steps, and replaces p42_train.last_checkpoint by a version that
picks the highest step number (the original sorts file names, which puts step_96000 after step_288000).
Outputs: data/prereg50/S0/, results/prereg50/explore/train_record.json.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P42 = os.path.join(REPO, "results", "prereg42", "explore")
P43 = os.path.join(REPO, "results", "prereg43", "explore")
P49 = os.path.join(REPO, "results", "prereg49", "explore")
for p in (P42, P43, P49, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p42_lib as V  # noqa: E402
import p43_run as R43  # noqa: E402

L = V.L
SEED = 0
CKPT_EVERY = 16000
# PLAN.md 3: candidates, largest first (blocks, channels)
SIZES = {"B20C160": (20, 160), "B15C160": (15, 160), "B15C128": (15, 128), "B12C128": (12, 128),
         "B10C128": (10, 128)}
OUT = os.path.join(REPO, "data", "prereg50", "S0")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
BASE_FILE = os.path.join(HERE, "base.json")   # {"players": "T49" | "T36"} committed by PLAN.md 4


def last_checkpoint_by_step(d):
    files = glob.glob(os.path.join(d, "step_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))


def set_size(name):
    if name not in SIZES:
        raise SystemExit("--size must be one of %s" % sorted(SIZES))
    b, c = SIZES[name]
    L.BLOCKS, L.CHANNELS = b, c
    V.GPOOL_BLOCKS = tuple(i for i in range(b) if i % 3 == 2)


def pop_size(rest):
    if "--size" not in rest:
        raise SystemExit("--size is required")
    i = rest.index("--size")
    name = rest[i + 1]
    del rest[i:i + 2]
    set_size(name)
    return name


def use_players():
    """T49 (prereg49 adopted) or T36; read from base.json, which PLAN.md 4 commits from prereg49's eval.json."""
    with open(BASE_FILE, encoding="utf-8") as f:
        base = json.load(f)["players"]
    if base == "T49":
        import p49_lib as P
        import p49_run as R49
        L.Cache = P.Cache49
        V.t_index = R49.t_index49
    elif base != "T36":
        raise SystemExit("base.json players must be T49 or T36")
    return base


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("bench", "train", "embed"):
        raise SystemExit("usage: p50_run.py bench | train | embed --size NAME ...")
    rest = argv[1:]
    for f in ("--seed", "--variant", "--out", "--record"):
        if f in rest:
            raise SystemExit("%s is fixed by prereg50" % f)
    size = pop_size(rest)
    V.TOTAL_STEPS, V.CKPT_EVERY = R43.TOTAL_STEPS, CKPT_EVERY
    if argv[0] == "embed":
        import p42_embed as Em
        return Em.main(rest)
    V.SEED = SEED
    V.DATA = os.path.join(REPO, "data", "prereg50")
    import p42_train as T
    if T.V is not V or T.L is not L:
        raise SystemExit("p42_train did not import the same modules")
    T.last_checkpoint = last_checkpoint_by_step
    if argv[0] == "bench":
        n = 300
        if "--steps-timed" in rest:
            i = rest.index("--steps-timed")
            n = int(rest[i + 1])
            del rest[i:i + 2]
        T.BUDGET_HOURS = 1e9
        print(json.dumps({"size": size, "blocks": L.BLOCKS, "channels": L.CHANNELS,
                          "gpool_blocks": list(V.GPOOL_BLOCKS)}), flush=True)
        return T.main(["--variant", R43.VARIANT, "--bench", str(n)] + rest)
    use_players()
    if "--smoke" in rest:
        return T.main(["--variant", R43.VARIANT] + rest)
    return T.main(["--variant", R43.VARIANT, "--out", OUT, "--record", TRAIN_RECORD] + rest)


if __name__ == "__main__":
    sys.exit(main())
