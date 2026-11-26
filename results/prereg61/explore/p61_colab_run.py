"""prereg61 revision 1: finish prereg61's training on Colab. p62_run with one extra arm P61 = seed 0, 200,000 steps,
keep (50k, 100k, 150k, 200k) = prereg61's own schedule (lr / warm-up are prereg61's, set by p61_run.configure).

    PY results/prereg61/explore/p61_colab_run.py train --auto-resume --out DIR --record FILE [--workers N]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results", "prereg62", "explore"))
import p62_run  # noqa: E402

p62_run.ARMS["P61"] = {"seed": 0, "steps": 200000, "keep": (50000, 100000, 150000, 200000)}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "train":
        raise SystemExit("usage: p61_colab_run.py train ...")
    return p62_run.main(["train", "--arm", "P61"] + argv[1:])


if __name__ == "__main__":
    sys.exit(main())
