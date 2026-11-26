"""prereg62 (PLAN.md): prereg61's recipe, arms S1 (seed 1), S2 (seed 2), X (seed 0, 400,000 steps), for Colab G4.
A wrapper over p61_run (which wraps p60_run); nothing of prereg60 / prereg61 is edited.

    PY results/prereg62/explore/p62_run.py train --arm S1|S2|X [--workers N] [--auto-resume] [--out DIR] [--record F]
                                                 [--min-free-ram-mb 2048] [--log-every 1000]
    PY results/prereg62/explore/p62_run.py train --arm S1 --smoke --device cpu --workers 0 [--steps N<=3] [--auto-resume]
    PY results/prereg62/explore/p62_run.py embed --arm S1 --ckpt data/prereg62/S1/ckpt/step_200000.pt --source store --players E

An arm fixes the seed and the number of steps; nothing else differs (PLAN.md table):

    S1: seed 1, 200,000 steps, kept 50k/100k/150k/200k      S2: seed 2, the same
    X : seed 0, 400,000 steps, kept 100k/200k/300k/400k

What this wrapper sets on p60_run after p61_run.configure() (lr 1e-3, warm-up 2,000 stay prereg61's):

  DATA / OUT / STOP_FILE -> data/prereg62/<arm> (or --out: the Colab Drive folder), EMB / SEAM / SEAM_GPU under DATA,
  TRAIN_RECORD -> results/prereg62/explore/train_record_<arm>.json (or --record),
  SEED, TOTAL_STEPS, KEEP per arm; RESUME_EVERY 5,000 (Colab: <= ~20 min of work lost);
  CODE_FILES + this file;
  p60_lib.prune -> prune2: keeps the kept steps AND the two most recent checkpoints (the newest resumable one is
  never the only copy while Drive is syncing);
  before a training run: a newest checkpoint that torch cannot read is renamed <name>.bad, so that resume (highest
  step number, p60_lib.last_ckpt) takes the one before;
  seed != 0 only: p60_run compares its SEED with the seed stored in the c49 checkpoint (0) inside its "is this
  c49's recipe" check. torch.load is wrapped for that one file: the wrapper checks itself that the stored seed is
  0 and then hands p60_run a config whose seed is the arm's, so that single comparison passes and all the others
  (variant, spec, blocks, channels, emb_dim, scale, wd, EMA decay, batch shape; classes, cache hash and the
  checkpoint's sha256 further down) are p60_run's, untouched.

The seed controls (PLAN.md): the sampler stream numpy.random.default_rng([seed, 320000 + s]) of batch s;
torch.manual_seed (no random operation in the V2 forward pass or loss; all weights come from c49); the number
written in the config. Checkpoints carry prereg60's config key "p60"; the vectors' tag is
p60_step_<steps>_raw_tta1 under data/prereg62/<arm>/emb.

No score of any kind is computed here. Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P61D = os.path.join(REPO, "results", "prereg61", "explore")
if P61D not in sys.path:
    sys.path.insert(0, P61D)
import p61_run as W61  # noqa: E402

R, P = W61.R, W61.P
ARMS = {"S1": {"seed": 1, "steps": 200000, "keep": (50000, 100000, 150000, 200000)},
        "S2": {"seed": 2, "steps": 200000, "keep": (50000, 100000, 150000, 200000)},
        "X": {"seed": 0, "steps": 400000, "keep": (100000, 200000, 300000, 400000)}}
RESUME_EVERY = 5000
RECENT = 2
C49_SEED = 0


def pop(rest, flag, default=None):
    if flag not in rest:
        return default
    i = rest.index(flag)
    v = rest[i + 1]
    del rest[i:i + 2]
    return v


def prune2(d, keep_steps, latest):
    """p60_lib.prune, but the RECENT (2) highest step numbers present are kept as well as keep_steps and `latest`."""
    steps = sorted((P.step_of(p), p) for p in glob.glob(os.path.join(d, "step_*.pt")))
    recent = {s for s, _ in steps[-RECENT:]}
    gone = []
    for s, p in steps:
        if s not in keep_steps and s != latest and s not in recent:
            os.remove(p)
            gone.append(os.path.basename(p))
    return sorted(gone)


def set_aside_unreadable(ckpt_dir):
    """Rename a newest checkpoint that cannot be read to <name>.bad (repeatedly). Returns the renamed names."""
    import torch
    bad = []
    while os.path.isdir(ckpt_dir):
        last = P.last_ckpt(ckpt_dir)
        if last is None:
            break
        try:
            ck = torch.load(last, map_location="cpu", weights_only=False)
            ok = isinstance(ck, dict) and "net" in ck and "step" in ck
            del ck
        except Exception as e:  # noqa: BLE001  (truncated / half-synced file)
            print("cannot read %s: %s" % (last, str(e)[:200]), flush=True)
            ok = False
        if ok:
            break
        os.replace(last, last + ".bad")
        bad.append(os.path.basename(last))
        print("set aside %s.bad; resuming from the checkpoint before it" % os.path.basename(last), flush=True)
    return bad


def pass_base_seed(seed):
    """See the module docstring: only for seed != 0, only for the c49 checkpoint file."""
    if seed == C49_SEED:
        return
    import torch
    orig = torch.load
    if getattr(orig, "_p62", False):
        return
    base = os.path.abspath(R.C49_CKPT)

    def load(f, *a, **k):
        ck = orig(f, *a, **k)
        if isinstance(f, str) and os.path.abspath(f) == base:
            if ck["config"].get("seed") != C49_SEED:
                raise SystemExit("the c49 checkpoint's stored seed is not %d" % C49_SEED)
            ck["config"] = dict(ck["config"], seed=seed)
        return ck

    load._p62 = True
    torch.load = load


def configure(arm, out=None, record=None):
    a = ARMS[arm]
    W61.configure()
    data = os.path.join(REPO, "data", "prereg62", arm)
    R.DATA = data
    R.OUT = os.path.abspath(out) if out else data
    R.EMB = os.path.join(data, "emb")
    R.SEAM = os.path.join(data, "seam")
    R.SEAM_GPU = os.path.join(data, "seam_gpu")
    R.STOP_FILE = os.path.join(data, "STOP")
    R.TRAIN_RECORD = os.path.abspath(record) if record else os.path.join(HERE, "train_record_%s.json" % arm)
    R.SEED = a["seed"]
    R.TOTAL_STEPS = a["steps"]
    R.KEEP = a["keep"]
    R.RESUME_EVERY = RESUME_EVERY
    R.emb_tag = lambda step=a["steps"]: "p60_step_%06d_raw_tta1" % int(step)
    me = os.path.abspath(__file__)
    if me not in R.CODE_FILES:
        R.CODE_FILES = tuple(R.CODE_FILES) + (me,)
    P.prune = prune2
    return R


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "embed"):
        raise SystemExit("usage: p62_run.py train|embed --arm S1|S2|X ...")
    rest = argv[1:]
    arm = pop(rest, "--arm")
    if arm not in ARMS:
        raise SystemExit("--arm must be one of %s" % sorted(ARMS))
    out, record = pop(rest, "--out"), pop(rest, "--record")
    if argv[0] == "embed":
        if out or record:
            raise SystemExit("--out / --record are for train")
        configure(arm)
        return R.main(["embed"] + rest)
    configure(arm, out, record)
    print("prereg62 arm %s: seed %d, %d steps, lr %g, warm-up %d, keep %s, out %s" % (
        arm, R.SEED, R.TOTAL_STEPS, R.LR, R.WARMUP, list(R.KEEP), R.OUT), flush=True)
    run_out = os.path.join(R.DATA, "smoke", "S0") if "--smoke" in rest else R.OUT
    set_aside_unreadable(os.path.join(run_out, "ckpt"))
    pass_base_seed(R.SEED)
    return R.main(["train"] + rest)


if __name__ == "__main__":
    sys.exit(main())
