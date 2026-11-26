"""prereg37 explore: the prereg36 metric CNN code with a 4x longer schedule (PLAN.md 2).

    GPUPY results/prereg37/explore/p37_run.py train [p36_train.py arguments]
    GPUPY results/prereg37/explore/p37_run.py embed [p36_embed.py arguments]

Nothing of results/prereg36/explore is edited: this wrapper only redirects the outputs to data/prereg37 (checkpoints,
embeddings, logs, STOP) and results/prereg37/explore (train_record.json) and sets the two schedule constants. The
move cache (data/prereg36/cache) and the splits are those of prereg36.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
sys.path.insert(0, P36)
import p36_lib as L  # noqa: E402

TOTAL_STEPS = 160000
CKPT_EVERY = 16000
DATA = os.path.join(REPO, "data", "prereg37")


def patch():
    L.DATA = DATA
    L.CKPT = os.path.join(DATA, "ckpt")
    L.EMB = os.path.join(DATA, "emb")
    os.makedirs(DATA, exist_ok=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "embed"):
        raise SystemExit("usage: p37_run.py train|embed ...")
    patch()
    if argv[0] == "train":
        import p36_train as T
        T.TOTAL_STEPS, T.CKPT_EVERY, T.HERE = TOTAL_STEPS, CKPT_EVERY, HERE
        return T.main(argv[1:])
    import p36_embed as E
    return E.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
