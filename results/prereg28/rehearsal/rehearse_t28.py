"""PREREG_28 rehearsal on PREREG_27's synthetic v2 trees (tools/testset_v2.py fake): what tools/t28_n.py decides for
the size condition on each tree, next to what tools/t06_player.py alone decides.

    PY results/prereg28/rehearsal/rehearse_t28.py --tree same=<S1> --tree grow-player=<S2> [--out FILE]

For each tree (S/download/ as written by `testset_v2.py fake`): the competition-1 counts of each pool, counted the way
06 and test02 count them (questions = rows of the question file, candidates = distinct player_id of the candidate
file), go into pool reports whose other 06-ON inputs are all set to true; then on_conditions of t06_player.py and of
the t28 wrapper are compared. Variants derived from each tree: 'q_extra' (one more question than candidates) and
'c_extra' (one more candidate) must be refused by both. The full test-day chain (T1 -> test02 -> 06) cannot run on
these trees on CPU: T1.4 / T1.5 and the seam need data/local (never opened here) and test02 needs the Colab CNN; at
N != 400 T1.2 stops with FAIL_STRUCT until PREREG_27 rev 3 (PREREG_27 10.2). Nothing here opens data/local.
"""
import argparse
import copy
import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import t06_player as t6  # noqa: E402
import t28_n as t28  # noqa: E402

csv.field_size_limit(1 << 30)
NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
ORIG = t6.on_conditions


def counts(dl, pool):
    qf, cf = t6.TEST_FILES[pool]
    with open(os.path.join(dl, qf), encoding="utf-8", newline="") as f:
        nq = sum(1 for _ in csv.DictReader(f))
    with open(os.path.join(dl, cf), encoding="utf-8", newline="") as f:
        nc = len({r["player_id"] for r in csv.DictReader(f)})
    return nq, nc


def report(nq, nc):
    return {"used06": True, "selected": "V2", "switch_allowed_at_this_level": True, "n_questions": nq,
            "n_candidates": nc, "test02": {"n_questions": nq, "n_candidates": nc, "guard_level": 1},
            "detector": {"usable": True, "opens": True, "r_forced_cost": 1.0, "r_star": 2.2515761243137895}}


def t2():
    return {"c_copies": {"player_identification_test_public.csv": 0, "player_identification_test_private.csv": 0},
            "abd_failed": [], "units_present": ["a", "b", "c", "d"], "verdict": "PASS", "fake": True}


def decide(pub, priv):
    args = ({"secondary_not_negative": True}, t2(), "rehearsal", False, NOW)
    a = ORIG(copy.deepcopy(pub), copy.deepcopy(priv), *args)
    b = t28.wrap(ORIG)(copy.deepcopy(pub), copy.deepcopy(priv), *args)
    return {"t06_size_ok": a[t28.COND]["ok"], "t06_all_on": all(v["ok"] for v in a.values()),
            "t28_size_ok": b[t28.COND]["ok"], "t28_all_on": all(v["ok"] for v in b.values()),
            "t28_identical_to_t06": a == b, "t28_size_detail": b[t28.COND]["detail"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", required=True, help="name=path of a testset_v2 fake root")
    ap.add_argument("--out", default=os.path.join(HERE, "rehearsal_t28.json"))
    a = ap.parse_args(argv)
    out = {"what": "PREREG_28 rehearsal: t28 vs t06 size condition on testset_v2 fake trees", "trees": {}}
    for item in a.tree:
        name, root = item.split("=", 1)
        dl = os.path.join(root, "download")
        c = {p: counts(dl, p) for p in t6.TEST_POOLS}
        rec = {"counts": {p: {"questions": v[0], "candidates": v[1]} for p, v in c.items()}}
        pub, priv = report(*c["public"]), report(*c["private"])
        rec["as_built"] = decide(pub, priv)
        q = report(c["private"][0] + 1, c["private"][1])
        rec["q_extra"] = decide(pub, q)
        cc = report(c["private"][0], c["private"][1] + 1)
        rec["c_extra"] = decide(pub, cc)
        out["trees"][name] = rec
        print(name, rec["counts"], {k: (v["t06_all_on"], v["t28_all_on"], v["t28_identical_to_t06"])
                                    for k, v in rec.items() if k != "counts"})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
