"""prereg68: p62_run with two more arms (prereg61 recipe, 400,000 steps): X2 = seed 1, X3 = seed 2.

    PY results/prereg68/explore/p68_colab_run.py train --arm X2|X3 --auto-resume --out DIR --record FILE [--workers N]
    PY results/prereg68/explore/p68_colab_run.py embed --arm X2|X3 --ckpt ... [p42_embed arguments]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results", "prereg62", "explore"))
import p62_run  # noqa: E402

KEEP = (100000, 200000, 300000, 400000)
p62_run.ARMS["X2"] = {"seed": 1, "steps": 400000, "keep": KEEP}
p62_run.ARMS["X3"] = {"seed": 2, "steps": 400000, "keep": KEEP}

if __name__ == "__main__":
    sys.exit(p62_run.main())
