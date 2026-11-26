"""Fake test07 HS sets for drills, from the committed validation HS (results/prereg07/design/spec.md 9).

    PY tools/t07_fake_tests.py from-val --root S --results-dir R7 [--sets seam,public,private] [--mode dry-run]
                                        [--rerun]
    PY tools/t07_fake_tests.py compare  --hs R7/hs/public_hs.npz --set public [--json OUT]

from-val  S holds the tree of `t02_fake_tests.py from-val` and `t04_fake_tests.py from-val` (S/FAKE_TREE.json,
          S/FAKE_TREE_T04.json): rank public = validation questions 1-400, private = 401-800, question ids re-prefixed
          val_ -> pub_ / priv_. Written to R7/hs/ (R7 outside the repository, or results/test07_rehearsal):
            seam_hs.npz     val_hs.npz rows 0-399 and val_seam_moves.npz (pos_game, pos_turn, pos_logp), byte for byte
            public_hs.npz   val_hs.npz rows of validation questions 1-400, question_id re-prefixed
            private_hs.npz  the rows of questions 401-800
            hs_<set>_report.json  fake_from_val true, mode as --mode, environment = val_report.json's, no chunks.
          tools/t07_rank.py accepts such records outside the real mode only ("the substitute entry point" of spec 9).
compare   A drill HS file of public / private computed on the GPU (t07_rank.py hs --max-games N on the from-val tree)
          against the validation HS of the same games (question id re-prefixed back, slot): failed / counts / n_moves
          equal, the largest per-game per-profile mean log p difference (spec 9: <= 0.002) and the lp_HS difference.
Rule: the games are validation games; only the HS seams and apply may use these files, never a content statistic.
No label file is opened; nothing is written inside the repository except under results/test07_rehearsal.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_test_bundle as tb  # noqa: E402

VAL_HS = os.path.join(REPO, "results", "prereg07", "hs", "val_hs.npz")
VAL_MOVES = os.path.join(REPO, "results", "prereg07", "hs", "val_seam_moves.npz")
VAL_REPORT = os.path.join(REPO, "results", "prereg07", "hs", "val_report.json")
HS_MAP = os.path.join(REPO, "results", "prereg07", "rank", "hs_map.npz")
REHEARSAL_R7 = os.path.join(REPO, "results", "test07_rehearsal")
PREFIX = {"public": "pub_", "private": "priv_"}
SLICES = {"public": (0, 400), "private": (400, 800)}
SETS = ("seam", "public", "private")
HS_FORMAT = "c10-test07 hs v1"
REPORT_FORMAT = "c10-test07 hs report v1"
RULE = "validation games: only the HS seams and apply; never run content statistics"


class FakeError07(SystemExit):
    def __init__(self, message):
        super().__init__(f"t07_fake_tests: {message}")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def question_bounds(qids):
    order, first = [], {}
    for i, q in enumerate(qids):
        if q not in first:
            first[q] = i
            order.append(q)
        elif order[-1] != q:
            raise FakeError07(f"val_hs: question {q} is not contiguous")
    return order, [first[q] for q in order] + [len(qids)]


def save(path, arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def from_val(root, r7, sets=SETS, mode="dry-run", rerun=False, log=print):
    root, r7 = os.path.abspath(root), os.path.abspath(r7)
    if tb.inside(root, REPO):
        raise FakeError07(f"--root {root} must be outside the repository")
    if tb.inside(r7, REPO) and not os.path.normcase(r7) == os.path.normcase(REHEARSAL_R7):
        raise FakeError07(f"--results-dir {r7} must be outside the repository or {REHEARSAL_R7}")
    for f in ("FAKE_TREE.json", "FAKE_TREE_T04.json"):
        if not os.path.exists(os.path.join(root, f)):
            raise FakeError07(f"{os.path.join(root, f)} missing: build the t02 and t04 from-val trees first")
    vh, vm, vr = load(VAL_HS), load(VAL_MOVES), tb.read_json(VAL_REPORT)
    qids = [str(x) for x in vh["question_id"]]
    order, bounds = question_bounds(qids)
    hs_dir = os.path.join(r7, "hs")
    os.makedirs(hs_dir, exist_ok=True)
    env = {k: vr["environment"].get(k) for k in ("python", "torch", "numpy", "cuda", "cudnn", "gpu", "driver")}
    now = tb.now_iso()
    out = {}
    for s in sets:
        npz, rep_p = os.path.join(hs_dir, f"{s}_hs.npz"), os.path.join(hs_dir, f"hs_{s}_report.json")
        if (os.path.exists(npz) or os.path.exists(rep_p)) and not rerun:
            raise FakeError07(f"{npz} exists; pass --rerun")
        if s == "seam":
            r0, r1 = 0, 400
            q = np.array(qids[r0:r1])
        else:
            lo, hi = SLICES[s]
            r0, r1 = bounds[lo], bounds[hi]
            q = np.array([PREFIX[s] + x[len("val_"):] for x in qids[r0:r1]])
            tq = tb.read_json(os.path.join(root, "FAKE_TREE.json"))
            if tq.get("format") != "c10-test02 fake tree v1":
                raise FakeError07("FAKE_TREE.json is not a t02 from-val tree")
        arrays = {"format": np.array(HS_FORMAT), "set": np.array(s), "question_id": q,
                  "slot": vh["slot"][r0:r1].astype(np.int64), "color": vh["color"][r0:r1],
                  "sums": vh["sums"][r0:r1], "counts": vh["counts"][r0:r1], "n_moves": vh["n_moves"][r0:r1],
                  "failed": vh["failed"][r0:r1], "profiles": vh["profiles"]}
        if s == "seam":
            arrays.update(pos_game=vm["pos_game"], pos_turn=vm["pos_turn"], pos_logp=vm["pos_logp"])
        save(npz, arrays)
        failed, counts, nm = vh["failed"][r0:r1], vh["counts"][r0:r1], vh["n_moves"][r0:r1]
        rep = {"format": REPORT_FORMAT, "what": "FAKE (t07_fake_tests from-val): validation HS rows, no GPU run",
               "set": s, "mode": mode, "created": now, "fake_scorer": False, "fake_from_val": True,
               "max_games_drill": False, "games": int(r1 - r0), "questions": len(dict.fromkeys(q.tolist())),
               "failed": int(failed.sum()), "failed_or_no_move": int((failed | (nm == 0)).sum()),
               "no_target_moves": int(((counts == 0) & ~failed).sum()), "chunks": [],
               "earliest_chunk_created_utc": now, "latest_chunk_created_utc": now, "environment": env,
               "source": {"val_hs.npz": tb.sha256_file(VAL_HS), "rows": [int(r0), int(r1)],
                          **({"val_seam_moves.npz": tb.sha256_file(VAL_MOVES)} if s == "seam" else {})},
               "output": {"file": f"{s}_hs.npz", "sha256": tb.sha256_file(npz)}, "rule": RULE}
        tb.write_json(rep_p, rep)
        out[s] = rep
        log(f"t07 from-val: {s} rows {r0}-{r1} -> {npz}")
    return out


def compare(hs_path, s):
    """A GPU drill HS file of public / private against the validation HS rows of the same games."""
    z, vh = load(hs_path), load(VAL_HS)
    idx = {(str(q), int(sl)): i for i, (q, sl) in enumerate(zip(vh["question_id"], vh["slot"]))}
    rows = []
    for q, sl in zip(z["question_id"], z["slot"]):
        q = str(q)
        if not q.startswith(PREFIX[s]):
            raise FakeError07(f"{q} does not carry the {PREFIX[s]} prefix")
        rows.append(idx[("val_" + q[len(PREFIX[s]):], int(sl))])
    rows = np.array(rows)
    eq = {k: bool(np.array_equal(np.asarray(z[k]), np.asarray(vh[k])[rows])) for k in ("failed", "counts", "n_moves")}
    eq["color"] = [str(x) for x in z["color"]] == [str(x) for x in vh["color"][rows]]
    ok = ~np.asarray(z["failed"], dtype=bool) & (np.asarray(z["counts"]) > 0)
    ma = np.asarray(z["sums"], dtype=np.float64)[ok] / np.asarray(z["counts"])[ok, None]
    mb = np.asarray(vh["sums"], dtype=np.float64)[rows][ok] / np.asarray(vh["counts"])[rows][ok, None]
    m = load(HS_MAP)

    def lp(sums, counts):
        x = sums / counts[:, None]
        Z = ((x - m["mu"]) / m["sd"]) @ m["W"] + m["b"]
        Z = Z - Z.max(1, keepdims=True)
        return Z - np.log(np.exp(Z).sum(1, keepdims=True))
    dl = float(np.max(np.abs(lp(np.asarray(z["sums"], dtype=np.float64)[ok], np.asarray(z["counts"])[ok])
                             - lp(np.asarray(vh["sums"], dtype=np.float64)[rows][ok],
                                  np.asarray(vh["counts"])[rows][ok])))) if ok.any() else None
    dm = float(np.max(np.abs(ma - mb))) if ok.any() else None
    return {"games": int(len(rows)), "equal": eq, "max_abs_profile_mean": dm, "max_abs_lp_hs": dl,
            "bitwise_equal_sums": bool(np.array_equal(np.asarray(z["sums"]), np.asarray(vh["sums"])[rows])),
            "passed": bool(all(eq.values()) and dm is not None and dm <= 0.002),
            "rule": "spec 9 item 2: failed / counts / n_moves equal, per game per profile mean log p <= 0.002"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="t07_fake_tests.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-val")
    p.add_argument("--root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--mode", default="dry-run", choices=("dry-run", "rehearsal"))
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--hs", required=True)
    p.add_argument("--set", required=True, choices=("public", "private"))
    p.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "from-val":
        sets = [x for x in a.sets.split(",") if x]
        if any(x not in SETS for x in sets):
            raise FakeError07(f"--sets {a.sets}")
        from_val(a.root, a.results_dir, sets, a.mode, a.rerun)
        return 0
    rec = compare(a.hs, a.set)
    print(json.dumps(rec, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
