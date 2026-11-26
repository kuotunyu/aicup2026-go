"""prereg43: the prereg42 V2 code with a 320,000-step schedule and a chosen seed (PLAN.md 1).

    GPUPY results/prereg43/explore/p43_run.py train --seed S [p42_train.py arguments; --variant is fixed to V2]
    GPUPY results/prereg43/explore/p43_run.py embed [p42_embed.py arguments]

Nothing of results/prereg42/explore is edited: this wrapper only sets three constants of p42_lib before handing over.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P42 = os.path.join(REPO, "results", "prereg42", "explore")
sys.path.insert(0, P42)
import p42_lib as V  # noqa: E402

TOTAL_STEPS = 320000
CKPT_EVERY = 32000
SEEDS = (0, 1)
VARIANT = "V2"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "embed"):
        raise SystemExit("usage: p43_run.py train --seed S ... | embed ...")
    V.TOTAL_STEPS, V.CKPT_EVERY = TOTAL_STEPS, CKPT_EVERY
    if argv[0] == "train":
        rest = argv[1:]
        if "--seed" not in rest:
            raise SystemExit("--seed is required")
        i = rest.index("--seed")
        seed = int(rest[i + 1])
        if seed not in SEEDS:
            raise SystemExit("seed must be one of %s" % (SEEDS,))
        del rest[i:i + 2]
        if "--variant" in rest:
            raise SystemExit("--variant is fixed to %s" % VARIANT)
        V.SEED = seed
        import p42_train as T
        if T.V is not V:
            raise SystemExit("p42_train did not import the same p42_lib module")
        return T.main(["--variant", VARIANT] + rest)
    import p42_embed as E
    return E.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
