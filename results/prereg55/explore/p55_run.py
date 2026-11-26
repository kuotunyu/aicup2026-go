"""prereg55 (PLAN.md): c49's recipe with seed 1 (V2, B10C128, 320,000 steps, T49 players) on Colab G4 -> an equally
strong second member for prereg54's score-level ensemble. Only the seed differs from c49.

    GPUPY results/prereg55/explore/p55_run.py train --arm S [--auto-resume ...] [--out DIR --record F]
    GPUPY results/prereg55/explore/p55_run.py embed --arm S [p42_embed.py arguments]

Wraps p51_run.py unchanged: adds arm S (320,000 steps, B10C128, checkpoints every 5,000 steps) and sets the seed to 1.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P51 = os.path.join(REPO, "results", "prereg51", "explore")
if P51 not in sys.path:
    sys.path.insert(0, P51)
import p51_run as R  # noqa: E402

SEED = 1
R.ARMS["S"] = {"steps": 320000, "sizes": ("B10C128",), "ckpt_every": 5000}
R.SEED = SEED


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--arm" in argv and argv[argv.index("--arm") + 1] != "S":
        raise SystemExit("prereg55 trains arm S only")
    if len(argv) > 1 and argv[0] == "train" and "--out" not in argv:
        argv += ["--out", os.path.join(REPO, "data", "prereg55", "S")]
    if len(argv) > 1 and argv[0] == "train" and "--record" not in argv:
        argv += ["--record", os.path.join(HERE, "train_record_S.json")]
    return R.main(argv)


if __name__ == "__main__":
    sys.exit(main())
