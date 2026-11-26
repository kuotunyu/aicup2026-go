"""prereg45: run prereg39's P1 extractor unchanged on the prereg45 lists (PLAN.md 1).

    PY results/prereg45/explore/p45_extract.py run --sets M1,M2 --workers 4
    PY results/prereg45/explore/p45_extract.py status

Only the module constants are pointed at prereg45 (lists, record, output, order); chunks are written to
data/prereg45/p1/<set>/ (chunk set name 'p39M1' / 'p39M2', as built by p39_extract). Resumable; STOP file
data/prereg45/p1/STOP stops before the next chunk.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P39 = os.path.join(REPO, "results", "prereg39", "explore")
if P39 not in sys.path:
    sys.path.insert(0, P39)
import p39_extract as X  # noqa: E402

X.DATA = os.path.join(REPO, "data", "prereg45")
X.LISTS = os.path.join(X.DATA, "lists")
X.OUT = os.path.join(X.DATA, "p1")
X.TMP = os.path.join(X.DATA, "tmp")
X.RECORD = os.path.join(HERE, "lists_record.json")
X.ORDER = ("M1", "M2")
X.OFFICIAL = {}

# Memory (war room, 11/15 00:18): keep only the rows of the prereg45 lists after loading, so the main process does not
# hold every fit game's SGF for the whole run. The kept rows are the same objects; nothing else changes.
import json  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "tools"))
import c10data  # noqa: E402

_load_rows = c10data.load_rows


def _filtered(args, kw):
    keys = set()
    for n in X.ORDER:
        with open(os.path.join(X.LISTS, n + ".json"), encoding="utf-8") as f:
            keys |= {(g["rank"], g["game_id"], g["player_id"]) for g in json.load(f)["games"]}
    rows = [r for r in _load_rows(*args, **kw) if (r["rank"], r["game_id"], r["player_id"]) in keys]
    if len(rows) != len(keys):
        raise SystemExit("%d list games, %d rows found" % (len(keys), len(rows)))
    return rows


def _load_rows_lists(*args, **kw):
    # 11/15 00:35: the full load runs in a short-lived child process, so the main process only ever holds the
    # list rows (the child's ~4 GB is returned to the system when it exits).
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as p:
        return p.apply(_filtered, (args, kw))


c10data.load_rows = _load_rows_lists

# 11/15 00:35 (war room): sharing the 4090 with another job is approved for this run; the "no other compute process"
# start check is skipped. P1 numbers do not depend on what else runs on the card.
if os.environ.get("C10_P45_SHARE_GPU") == "1":
    sys.path.insert(0, os.path.join(REPO, "results", "prereg08", "p1"))
    import p1_lib as _P  # noqa: E402
    _P.H.other_compute_processes = lambda: ([], "")

if __name__ == "__main__":
    sys.exit(X.main())
