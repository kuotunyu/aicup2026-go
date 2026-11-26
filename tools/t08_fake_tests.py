"""Fake test08 P1 sets for drills, from the committed validation P1 run (PREREG_08 rev 3 section 13; as
tools/t07_fake_tests.py for 07).

    PY tools/t08_fake_tests.py from-val --root S --results-dir R8 [--p1-data-dir H] [--sets seam,public,private]
                                        [--mode dry-run|rehearsal] [--rerun]
    PY tools/t08_fake_tests.py compare  --results-dir R8 [--p1-data-dir H] --set public [--json OUT]

from-val  S holds the tree of `t02_fake_tests.py from-val` and `t04_fake_tests.py from-val` (S/FAKE_TREE.json,
          S/FAKE_TREE_T04.json): rank public = validation questions 1-400, private = 401-800, question ids re-prefixed
          val_ -> pub_ / priv_ (the same games as t07_fake_tests.py). The features are the validation P1 chunks
          data/prereg08/val/chunk_*.npz that results/prereg08/feat/val_report.json lists (SHA-256 checked), copied row
          for row into test08-format chunks of 200 games in H/<set>/ (H default R8/p1_data; data/test08_rehearsal/p1
          for R8 = results/test08_rehearsal), plus R8/p1/<set>_p1.npz and p1_<set>_report.json (fake_from_val true,
          mode as --mode, environment = val_report.json's). seam = validation games 0-399; its means equal
          results/prereg08/feat/val_seam_feats.npz (means, counts, n_moves, failed, question id, slot and colour
          checked). tools/t08_rank.py accepts such records outside the real
          mode only.
compare   A drill P1 set of public / private computed on the GPU (t08_rank.py p1 --max-games N on the from-val tree)
          against the validation P1 chunks of the same games (question id re-prefixed back, slot): failed / counts /
          n_moves equal, the largest standardised feature difference (<= 0.05) and the lp_P1TO difference (<= 0.04, PREREG_08 rev 4).
          Records the validation rows and whether every drill chunk starts off the validation chunk grid (private
          starts at validation row 4,787, so its 200-game chunks straddle the validation chunks: a different GPU
          batch composition; public starts at row 0, the seam's chunks).
Rule: the games are validation games; only the P1 seams and apply may use these files, never a content statistic.
No label file is opened; nothing is written inside the repository except under results/test08_rehearsal and
data/test08_rehearsal.
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
import t08_rank as t8  # noqa: E402

VAL_REPORT = os.path.join(REPO, "results", "prereg08", "feat", "val_report.json")
VAL_P1 = os.path.join(REPO, "results", "prereg08", "feat", "val_p1.npz")
VAL_SEAM = os.path.join(REPO, "results", "prereg08", "feat", "val_seam_feats.npz")
P1_MAP = os.path.join(REPO, "results", "prereg08", "rank", "p1_map.npz")
PREFIX = {"public": "pub_", "private": "priv_"}
SLICES = {"public": (0, 400), "private": (400, 800)}
SETS = ("seam", "public", "private")
RULE = "validation games: only the P1 seams and apply; never run content statistics"


class FakeError08(SystemExit):
    def __init__(self, message):
        super().__init__(f"t08_fake_tests: {message}")


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
            raise FakeError08(f"val_p1: question {q} is not contiguous")
    return order, [first[q] for q in order] + [len(qids)]


def val_chunks():
    """The validation P1 chunks val_report.json lists (SHA-256 checked) -> (report, per-game arrays, chunk env)."""
    vr = tb.read_json(VAL_REPORT)
    parts, env = [], None
    for ch in vr["chunks"]:
        p = os.path.join(REPO, *ch["file"].split("/"))
        if not os.path.exists(p) or tb.sha256_file(p) != ch["sha256"]:
            raise FakeError08(f"{ch['file']} missing or changed since val_report.json")
        with np.load(p, allow_pickle=False) as z:
            parts.append({k: np.asarray(z[k]) for k in ("ref", "means", "counts", "n_moves", "failed")})
            if env is None:
                env = {k: str(z[k]) for k in ("precision", "device", "torch", "cudnn", "gpu_name", "driver",
                                              "ckpt_sha256", "katago_commit", "meta_sha256", "code_sha256")}
                env.update(batch=int(z["batch"]), channels_last=bool(z["channels_last"]), ckpt_bytes=int(z["ckpt_bytes"]))
    d = {k: np.concatenate([x[k] for x in parts]) for k in ("ref", "means", "counts", "n_moves", "failed")}
    if not np.array_equal(d["ref"], np.arange(len(d["ref"]))):
        raise FakeError08("validation chunk refs are not in order")
    return vr, d, env


def dirs(r8, h):
    r8 = os.path.abspath(r8)
    if tb.inside(r8, REPO) and os.path.normcase(r8) != os.path.normcase(t8.REHEARSAL_R8):
        raise FakeError08(f"--results-dir {r8} must be outside the repository or {t8.REHEARSAL_R8}")
    if h:
        h = os.path.abspath(h)
    else:
        h = t8.REHEARSAL_H if os.path.normcase(r8) == os.path.normcase(t8.REHEARSAL_R8) else os.path.join(r8, "p1_data")
    if tb.inside(h, REPO) and not tb.inside(h, os.path.dirname(t8.REHEARSAL_H)):
        raise FakeError08(f"--p1-data-dir {h} inside the repository must be under data/test08_rehearsal")
    return r8, h


def from_val(root, r8, h=None, sets=SETS, mode="dry-run", rerun=False, log=print):
    root = os.path.abspath(root)
    if tb.inside(root, REPO):
        raise FakeError08(f"--root {root} must be outside the repository")
    r8, h = dirs(r8, h)
    for f in ("FAKE_TREE.json", "FAKE_TREE_T04.json"):
        if not os.path.exists(os.path.join(root, f)):
            raise FakeError08(f"{os.path.join(root, f)} missing: build the t02 and t04 from-val trees first")
    tq = tb.read_json(os.path.join(root, "FAKE_TREE.json"))
    if tq.get("format") != "c10-test02 fake tree v1":
        raise FakeError08("FAKE_TREE.json is not a t02 from-val tree")
    vr, d, cenv = val_chunks()
    v1 = load(VAL_P1)
    qids = [str(x) for x in v1["question_id"]]
    if len(qids) != len(d["ref"]):
        raise FakeError08("val_p1.npz and the validation chunks differ in length")
    for k in ("counts", "n_moves", "failed"):
        if not np.array_equal(np.asarray(v1[k]).astype(np.int64), np.asarray(d[k]).astype(np.int64)):
            raise FakeError08(f"val_p1.npz {k} differs from the validation chunks")
    order, bounds = question_bounds(qids)
    env = {k: vr["environment"].get(k) for k in ("python", "torch", "numpy", "cuda", "cudnn", "gpu", "driver")}
    now = tb.now_iso()
    p1_dir = os.path.join(r8, "p1")
    os.makedirs(p1_dir, exist_ok=True)
    out = {}
    for s in sets:
        npz, rep_p = os.path.join(p1_dir, f"{s}_p1.npz"), os.path.join(p1_dir, f"p1_{s}_report.json")
        cdir = os.path.join(h, s)
        if (os.path.exists(npz) or os.path.exists(rep_p) or os.path.isdir(cdir)) and not rerun:
            raise FakeError08(f"{npz} or {cdir} exists; pass --rerun")
        if s == "seam":
            r0, r1 = 0, t8.SEAM_GAMES
            q = qids[r0:r1]
        else:
            lo, hi = SLICES[s]
            r0, r1 = bounds[lo], bounds[hi]
            q = [PREFIX[s] + x[len("val_"):] for x in qids[r0:r1]]
        if os.path.isdir(cdir):
            for f in os.listdir(cdir):
                if f.startswith("chunk_") and f.endswith(".npz"):
                    os.remove(os.path.join(cdir, f))
        os.makedirs(cdir, exist_ok=True)
        n = r1 - r0
        rows = []
        for ci, b in enumerate(range(0, n, t8.CHUNK)):
            e = min(n, b + t8.CHUNK)
            p = os.path.join(cdir, f"chunk_{ci:05d}.npz")
            tmp = p + ".tmp.npz"
            np.savez(tmp, format=np.array(t8.FORMATS["chunk"]), set=np.array(f"test08_{s}"),
                     sides=np.array(t8.r8.RUN["sides"]), ref=np.arange(b, e, dtype=np.int64),
                     precision=np.array(cenv["precision"]), batch=np.array(cenv["batch"]),
                     channels_last=np.array(cenv["channels_last"]), device=np.array(cenv["device"]),
                     torch=np.array(cenv["torch"]), cudnn=np.array(cenv["cudnn"]), gpu_name=np.array(cenv["gpu_name"]),
                     driver=np.array(cenv["driver"]), ckpt_bytes=np.array(cenv["ckpt_bytes"]),
                     ckpt_sha256=np.array(cenv["ckpt_sha256"]), katago_commit=np.array(cenv["katago_commit"]),
                     meta_sha256=np.array(cenv["meta_sha256"]), code_sha256=np.array(cenv["code_sha256"]),
                     keys_sha256=np.array("fake_from_val"), questions_sha256=np.array("fake_from_val"),
                     tool_sha256=np.array("t08_fake_tests from-val"), mode=np.array(mode), created_utc=np.array(now),
                     seconds=np.array(0.0), gpu_seconds=np.array(0.0), means=d["means"][r0 + b:r0 + e],
                     counts=d["counts"][r0 + b:r0 + e], n_moves=d["n_moves"][r0 + b:r0 + e],
                     failed=d["failed"][r0 + b:r0 + e], errors=np.array("[]"), pos_side=np.zeros(0, dtype=np.int8))
            os.replace(tmp, p)
        plan = [(ci, 0, 0) for ci in range(len(range(0, n, t8.CHUNK)))]
        rows, dd = t8.read_chunks(cdir, plan, with_means=True)
        arrays = {"format": np.array(t8.FORMATS["p1"]), "set": np.array(s), "question_id": np.array(q),
                  "slot": np.asarray(v1["slot"][r0:r1]).astype(np.int64), "color": np.asarray(v1["color"][r0:r1]),
                  "counts": dd["counts"], "n_moves": dd["n_moves"], "failed": dd["failed"], "scal": dd["scal"],
                  "scal_fields": np.array(["logp", "entropy"]), "sides": np.array(t8.r8.RUN["sides"])}
        if s == "seam":
            arrays["means"] = dd["means"]
            ref = load(VAL_SEAM)
            same = {"means": np.array_equal(ref["means"], dd["means"]),
                    "counts": np.array_equal(ref["counts"], dd["counts"]),
                    "n_moves": np.array_equal(np.asarray(ref["n_moves"]).astype(np.int64), dd["n_moves"]),
                    "failed": np.array_equal(np.asarray(ref["failed"], dtype=bool), dd["failed"]),
                    "question_id": [str(x) for x in ref["question_id"]] == q,
                    "slot": [int(x) for x in ref["slot"]] == [int(x) for x in arrays["slot"]],
                    "color": [str(x) for x in ref["color"]] == [str(x) for x in arrays["color"]]}
            if not all(same.values()):
                raise FakeError08("seam rows 0-399 of the validation chunks differ from val_seam_feats.npz: "
                                  f"{[k for k, v in same.items() if not v]}")
        tmp = npz + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, npz)
        failed, nm = dd["failed"], dd["n_moves"]
        rep = {"format": t8.FORMATS["p1_report"], "what": "FAKE (t08_fake_tests from-val): validation P1 rows, no "
                                                          "GPU run",
               "set": s, "mode": mode, "created": now, "fake_scorer": False, "fake_from_val": True,
               "max_games_drill": False, "games": int(n), "questions": len(dict.fromkeys(q)),
               "failed": int(failed.sum()), "failed_or_no_move": int((failed | (nm == 0)).sum()),
               "no_target_moves": int(((dd["counts"][:, 0] == 0) & ~failed).sum()), "chunks": rows,
               "chunk_dir": cdir, "earliest_chunk_created_utc": now, "latest_chunk_created_utc": now,
               "environment": env, "source": {"val_report.json": tb.sha256_file(VAL_REPORT), "rows": [int(r0), int(r1)]},
               "output": {"file": f"{s}_p1.npz", "sha256": tb.sha256_file(npz)}, "rule": RULE}
        tb.write_json(rep_p, rep)
        out[s] = rep
        log(f"t08 from-val: {s} rows {r0}-{r1} -> {npz} ({len(rows)} chunks in {cdir})")
    return out


def compare(r8, h, s):
    """A GPU drill P1 set of public / private against the validation P1 rows of the same games."""
    r8, h = dirs(r8, h)
    rep = tb.read_json(os.path.join(r8, "p1", f"p1_{s}_report.json"))
    z = load(os.path.join(r8, "p1", f"{s}_p1.npz"))
    _rows, dd = t8.read_chunks(os.path.join(h, s), [(ci, 0, 0) for ci in range(len(rep["chunks"]))], with_means=True)
    _vr, d, _env = val_chunks()
    v1 = load(VAL_P1)
    idx = {(str(q), int(sl)): i for i, (q, sl) in enumerate(zip(v1["question_id"], v1["slot"]))}
    rows = []
    for q, sl in zip(z["question_id"], z["slot"]):
        q = str(q)
        if not q.startswith(PREFIX[s]):
            raise FakeError08(f"{q} does not carry the {PREFIX[s]} prefix")
        rows.append(idx[("val_" + q[len(PREFIX[s]):], int(sl))])
    rows = np.array(rows)
    chunk = int(_vr["chunks"][0]["games"])
    starts = rows[::t8.CHUNK]                      # the first validation row of every drill chunk
    eq = {k: bool(np.array_equal(np.asarray(dd[k]).astype(np.int64), np.asarray(d[k])[rows].astype(np.int64)))
          for k in ("failed", "counts", "n_moves")}
    eq["color"] = [str(x) for x in z["color"]] == [str(x) for x in v1["color"][rows]]
    model = load(P1_MAP)
    ok = ~np.asarray(dd["failed"], dtype=bool) & (np.asarray(dd["counts"])[:, 0] > 0)
    if ok.any():
        xa = t8.std_x(dd["means"], dd["counts"], model["mu"], model["sd"])[ok]
        xb = t8.std_x(d["means"][rows], d["counts"][rows], model["mu"], model["sd"])[ok]
        df = float(np.max(np.abs(xa - xb)))
        dl = float(np.max(np.abs(t8.lp_np(xa, model) - t8.lp_np(xb, model))))
    else:
        df = dl = None
    return {"games": int(len(rows)), "equal": eq, "max_abs_std_feature": df, "max_abs_lp_p1": dl,
            "bitwise_equal_means": bool(np.array_equal(dd["means"], d["means"][rows])),
            "positions_per_s": rep.get("positions_per_s"), "positions": rep.get("positions"),
            "gpu_seconds": rep.get("gpu_seconds"),
            "validation_rows": [int(rows.min()), int(rows.max())] if len(rows) else None,
            "rows_contiguous": bool(len(rows) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows)))),
            "chunk_starts_val_rows": [int(x) for x in starts],
            "chunk_boundaries_differ_from_val": bool(len(rows) and all(int(x) % chunk != 0 for x in starts)),
            "passed": bool(all(eq.values()) and df is not None and df <= t8.TOL_FEAT and dl <= t8.TOL_LP),
            "rule": "failed / counts / n_moves equal; standardised features <= 0.05; lp_P1TO <= 0.04 (PREREG_08 5.2, rev 4)"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="t08_fake_tests.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-val")
    p.add_argument("--root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--p1-data-dir")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--mode", default="dry-run", choices=("dry-run", "rehearsal"))
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--p1-data-dir")
    p.add_argument("--set", required=True, choices=("public", "private"))
    p.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "from-val":
        sets = [x for x in a.sets.split(",") if x]
        if any(x not in SETS for x in sets):
            raise FakeError08(f"--sets {a.sets}")
        from_val(a.root, a.results_dir, a.p1_data_dir, sets, a.mode, a.rerun)
        return 0
    rec = compare(a.results_dir, a.p1_data_dir, a.set)
    print(json.dumps(rec, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
