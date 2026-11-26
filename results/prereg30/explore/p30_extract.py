"""prereg30 (GPU, resumable): results/prereg08/p1/p1_extract.py, unchanged, with the v2 scorer (PLAN.md 2).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; GPU python .venv-gpu/Scripts/python.exe
    results/prereg30/explore/p30_extract.py run --set probe --sets data/prereg30/pilot/sets.json \
        --out-dir data/prereg30/pilot --workers 12
    CPU smoke: ... run --set probe --sets data/prereg30/pilot/sets.json --out-dir data/prereg30/smoke \
        --device cpu --precision fp32 --batch 64 --group-batches 1 --workers 2 --max-games 4

Every argument goes to p1_extract.main (same defaults as P1: fp16, batch 1024, group 4, channels-last, 200 games per
chunk, atomic chunk files, finished chunks skipped, STOP file in the out dir to yield the GPU, nvidia-smi check for other
compute processes). Three things are replaced, nothing in p1_extract.py / p1_lib.py is edited:
    * p1_lib.ProbeScorer -> p30_scorer.V2Scorer (same forward passes and P1 values; also lmax / lead / winp per position);
    * p1_extract.run_chunk -> run_chunk below (p1_extract.run_chunk line for line, plus the per-position pos_lmax,
      pos_lead, pos_winp arrays, the format mark p30_format and the prereg30 code SHA-256s in every chunk);
    * p1_extract.chunk_ok -> also requires the p30 format mark (a P1-only chunk is never taken as done).
A sets.json whose probe has source 'val' (prereg25's V) takes its games from prereg25's p25_extract.set_games (full run,
PLAN.md 5.1); any other source goes to p1_extract.set_games unchanged.
"""

import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p30_lib as V  # noqa: E402
import p30_scorer as S  # noqa: E402  (p1_lib pin check and data/local guard)
import p1_extract as E  # noqa: E402

P = E.P
_ORIGINAL_CHUNK_OK = E.chunk_ok


def chunk_ok(path, name, refs):
    if not _ORIGINAL_CHUNK_OK(path, name, refs):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return "p30_format" in z.files and str(z["p30_format"]) == V.V2_FORMAT
    except Exception:
        return False


def run_chunk(pool, sc, games, group_positions):
    """p1_extract.run_chunk plus pos_lmax / pos_lead / pos_winp (sc is a V2Scorer)."""
    G = len(games)
    means = np.zeros((G, 2, P.F_DIM), dtype=np.float32)
    counts = np.zeros((G, 2), dtype=np.int32)
    n_moves = np.zeros(G, dtype=np.int16)
    failed = np.zeros(G, dtype=bool)
    errors = []
    pos_game, pos_side, pos_turn, pos_lp, pos_ent = [], [], [], [], []
    pos_extra = []
    buf = []
    t_gpu = [0.0]

    def flush():
        if not buf:
            return
        cat = {k: np.concatenate([f[k] for _, f in buf]) for k in ("bin", "glob", "legal", "move", "pla")}
        t = time.time()
        pf = sc.position_features(cat)
        ex = sc.last_extra
        t_gpu[0] += time.time() - t
        if ex is None or len(ex) != len(pf):
            raise V.P30Error("scorer did not return the per-position extras")
        o = 0
        for gi, f in buf:
            T = len(f["move"])
            part = pf[o:o + T]
            pos_extra.append(ex[o:o + T])
            o += T
            m, c = P.game_means(part, f["side"])
            means[gi], counts[gi] = m.astype(np.float32), c
            pos_game.append(np.full(T, gi, dtype=np.int32))
            pos_side.append(f["side"])
            pos_turn.append(f["turn"])
            pos_lp.append(part[:, 768])
            pos_ent.append(part[:, 769])
        buf.clear()

    args = [(gi, g["sgf_content"], g["color"]) for gi, g in enumerate(games)]
    pending = 0
    for gi, f, err in pool.imap(E._featurize, args, chunksize=2):
        if err is not None:
            failed[gi] = True
            errors.append({"game": gi, "error": err})
            continue
        n_moves[gi] = f["n_moves"]
        buf.append((gi, f))
        pending += len(f["move"])
        if pending >= group_positions:
            flush()
            pending = 0
    flush()
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    ex = np.concatenate(pos_extra) if pos_extra else np.zeros((0, 3), np.float32)
    return {"means": means, "counts": counts, "n_moves": n_moves, "failed": failed, "errors": json.dumps(errors),
            "pos_game": cat(pos_game, np.int32), "pos_side": cat(pos_side, np.int8),
            "pos_turn": cat(pos_turn, np.int16), "pos_logp": cat(pos_lp, np.float32),
            "pos_entropy": cat(pos_ent, np.float32),
            "pos_lmax": np.ascontiguousarray(ex[:, 0]), "pos_lead": np.ascontiguousarray(ex[:, 1]),
            "pos_winp": np.ascontiguousarray(ex[:, 2]),
            "p30_format": np.array(V.V2_FORMAT), "p30_code_sha256": np.array(S.code_record())}, t_gpu[0]


def set_games(sets, name):
    s = sets[name]
    if s.get("source") == "val":
        sys.path.insert(0, os.path.join(REPO, "results", "prereg25", "explore"))
        import p25_extract  # noqa: E402  (its import re-wraps p1_extract.set_games; undone on the next line)
        E.set_games = set_games
        return p25_extract.set_games(sets, name)
    return _ORIGINAL_SET_GAMES(sets, name)


_ORIGINAL_SET_GAMES = E.set_games
P.ProbeScorer = S.V2Scorer          # p1_extract.cmd_run builds P.ProbeScorer(...)
E.run_chunk = run_chunk
E.chunk_ok = chunk_ok
E.set_games = set_games

if __name__ == "__main__":
    sys.exit(E.main())
