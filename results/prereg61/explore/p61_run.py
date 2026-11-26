"""prereg61 (PLAN.md Method): prereg60's code and recipe with only the schedule changed. A wrapper over p60_run.

    GPUPY results/prereg61/explore/p61_run.py train [--workers 4] [--resume | --auto-resume] [--steps N]
    GPUPY results/prereg61/explore/p61_run.py train --smoke --device cpu --workers 0 [--steps N<=3] [--auto-resume]
    GPUPY results/prereg61/explore/p61_run.py embed --ckpt data/prereg61/S0/ckpt/step_200000.pt --source store --players E
    GPUPY results/prereg61/explore/p61_run.py embed --ckpt ... --source pool --src data/test_v2_download/x/tests --pool public|private

Nothing of prereg60 is edited. p60_run is imported and its module constants are replaced before its functions run
(cmd_train / cmd_embed read them as module globals at call time - checked line by line, NOTES.md 1):

  DATA / OUT / EMB / SEAM / SEAM_GPU / STOP_FILE  -> data/prereg61/...
  TRAIN_RECORD                                    -> results/prereg61/explore/train_record.json
  TOTAL_STEPS 200,000   LR 1e-3   WARMUP 2,000    (cosine to 0 at step 200,000: p60_lib.lr_factor(s, 2000, 200000))
  KEEP (50000, 100000, 150000, 200000)            resume checkpoint every 1,000 steps, pruned, picked by step number
  CODE_FILES                                      + this file

Everything else is prereg60's: c49 raw start with the widened first convolution, c49's head, T49 sampler stream
continued at batch 320,000 (batch s of this run = batch 320,000 + s, s = 0 .. 199,999), 96 x 24, AdamW wd 1e-4,
fp16, seed 0, new EMA (kept, not used), the STOP / low-RAM pause (exit code 4), the refusal to start when another
python compute process is on the GPU. The checkpoints carry prereg60's config key ("p60") with this run's numbers;
the vectors' tag is p60_step_200000_raw_tta1 (run name of p60_run.load_net60 + the checkpoint's stem) under
data/prereg61/emb.

No score of any kind is computed here. Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P60D = os.path.join(REPO, "results", "prereg60", "explore")
for _p in (P60D,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p60_run as R  # noqa: E402

P = R.P
DATA = os.path.join(REPO, "data", "prereg61")
TOTAL_STEPS = 200000
LR = 1e-3
WARMUP = 2000
KEEP = (50000, 100000, 150000, 200000)
RESUME_EVERY = 1000
TRAIN_RECORD = os.path.join(HERE, "train_record.json")


def emb_tag(step=TOTAL_STEPS):
    return "p60_step_%06d_raw_tta1" % int(step)


def configure():
    """Point p60_run at prereg61's folder and schedule. Idempotent."""
    R.DATA = DATA
    R.OUT = os.path.join(DATA, "S0")
    R.EMB = os.path.join(DATA, "emb")
    R.SEAM = os.path.join(DATA, "seam")
    R.SEAM_GPU = os.path.join(DATA, "seam_gpu")
    R.STOP_FILE = os.path.join(DATA, "STOP")
    R.TRAIN_RECORD = TRAIN_RECORD
    R.TOTAL_STEPS = TOTAL_STEPS
    R.LR = LR
    R.WARMUP = WARMUP
    R.KEEP = KEEP
    R.RESUME_EVERY = RESUME_EVERY
    R.emb_tag = emb_tag
    me = os.path.abspath(__file__)
    if me not in R.CODE_FILES:
        R.CODE_FILES = tuple(R.CODE_FILES) + (me,)
    return R


def main(argv=None):
    configure()
    return R.main(argv)


if __name__ == "__main__":
    sys.exit(main())
