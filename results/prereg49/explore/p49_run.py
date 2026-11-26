"""prereg49 (PLAN.md 2): prereg43's c* recipe (V2, 320,000 steps, checkpoints every 32,000, seed 0) on the T49 players.

    GPUPY results/prereg49/explore/p49_run.py seam                 # CPU: new cache + T36 -> batches 0..2 == old cache's
    GPUPY results/prereg49/explore/p49_run.py train [--resume | --steps N | --smoke ...]
    GPUPY results/prereg49/explore/p49_run.py embed [p42_embed.py arguments]

Nothing of prereg36 / prereg42 / prereg43 is edited. This wrapper sets p42_lib's constants as p43_run.py does, points
p36_lib.Cache at data/prereg49/cache and replaces p42_lib.t_index by the T49 version (every cache player outside E with
>= 20 games). Outputs: data/prereg49/S0/ (ckpt, train_log.jsonl, STOP), results/prereg49/explore/train_record.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P42 = os.path.join(REPO, "results", "prereg42", "explore")
P43 = os.path.join(REPO, "results", "prereg43", "explore")
for p in (P42, P43, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p42_lib as V  # noqa: E402
import p43_run as R43  # noqa: E402
import p49_lib as P  # noqa: E402

L = V.L
SEED = 0
OUT = os.path.join(REPO, "data", "prereg49", "S0")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
_ORIG_T_INDEX = V.t_index
_E = {}


def e_ids():
    if "E" not in _E:
        _E["E"] = P.e_players()
    return _E["E"]


def t_index49(cache):
    """= p42_lib.t_index with T = T49 (sorted ids = class indices); asserts no E game and >= 20 games per class."""
    E = e_ids()
    T = P.t49(cache, E)
    cls = {p: i for i, p in enumerate(T)}
    lab = np.array([cls.get(p, -1) for p in cache.pid.tolist()], np.int64)
    games = np.flatnonzero(lab >= 0)
    order = games[np.argsort(lab[games], kind="stable")]
    t_off = np.concatenate([[0], np.cumsum(np.bincount(lab[games], minlength=len(T)))]).astype(np.int64)
    Es = set(E)
    assert not any(p in Es for p in np.unique(cache.pid[order]).tolist())
    assert (np.diff(t_off) >= L.T_MIN_GAMES).all()
    _E2, D, _T36, _per = L.splits(L.Cache())
    return T, order, t_off, {"E": L.sha256_ids(E), "D": L.sha256_ids(D), "T": L.sha256_ids(T)}


def seam():
    """New cache with prereg36's T (the original t_index on the new cache would also count the new players, so the
    T36 index is built from the old cache) -> the first three GameBatches equal the old cache's, byte for byte."""
    old, new = L.Cache(), P.Cache49()
    T36, order, t_off, _ids = _ORIG_T_INDEX(old)
    wh_old, wh_new = old.color == "W", new.color == "W"
    a = V.GameBatches(old.dir, old.off, wh_old, order, t_off, SEED)
    b = V.GameBatches(new.dir, new.off, wh_new, order, t_off, SEED)
    same = []
    for s in range(3):
        x, y = a.batch(s), b.batch(s)
        same.append(all(np.array_equal(x[k], y[k]) for k in x))
    T = P.t49(new, e_ids())
    rec = {"rule": "new cache + T36 index: GameBatches 0..2 identical to the old cache's", "batches_identical": same,
           "T49_players": len(T), "T36_players": len(T36)}
    print(json.dumps(rec))
    if not all(same):
        raise SystemExit("seam failed")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("seam", "train", "embed"):
        raise SystemExit("usage: p49_run.py seam | train ... | embed ...")
    if argv[0] == "seam":
        return seam()
    V.TOTAL_STEPS, V.CKPT_EVERY = R43.TOTAL_STEPS, R43.CKPT_EVERY
    if argv[0] == "embed":
        import p42_embed as Em
        return Em.main(argv[1:])
    rest = argv[1:]
    for f in ("--seed", "--variant", "--out", "--record"):
        if f in rest:
            raise SystemExit("%s is fixed by prereg49" % f)
    V.SEED = SEED
    V.DATA = os.path.join(REPO, "data", "prereg49")
    L.Cache = P.Cache49
    V.t_index = t_index49
    import p42_train as T
    if T.V is not V or T.L is not L:
        raise SystemExit("p42_train did not import the same modules")
    if "--smoke" in rest:
        return T.main(["--variant", R43.VARIANT] + rest)
    return T.main(["--variant", R43.VARIANT, "--out", OUT, "--record", TRAIN_RECORD] + rest)


if __name__ == "__main__":
    sys.exit(main())
