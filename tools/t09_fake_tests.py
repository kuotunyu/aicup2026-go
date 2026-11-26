"""Fake test09 inputs for drills and the tol_E measurement (PREREG_09 5.2; rev 3 section 13; as
tools/t08_fake_tests.py for 08).

    PY tools/t09_fake_tests.py from-val      --root S --t08-results-dir R8 [--p1-data-dir H] [--mode dry-run|rehearsal]
                                             [--rerun]
    PY tools/t09_fake_tests.py add-positions --t08-results-dir R8 [--p1-data-dir H]
    PY tools/t09_fake_tests.py compare       --t08-results-dir R8 [--p1-data-dir H] --set seam|public|private
                                             [--json OUT]
    PY tools/t09_fake_tests.py measure-tol   --tree S [--json OUT]

from-val       tools/t08_fake_tests.py from-val (imported, unchanged: seam = validation games 0-399, public / private
               = the validation P1 rows of questions 1-400 / 401-800) followed by add-positions. Run it before
               tools/t08_rank.py apply --dry-run (08's report binds the P1 reports, which add-positions rewrites).
add-positions  08's from-val chunks carry no per-position arrays (t08_fake_tests writes pos_side of length 0), but 09's
               EX needs them (PREREG_09 1.3 item 1). This copies pos_game / pos_side / pos_turn / pos_logp /
               pos_entropy of the same games from the validation chunks data/prereg08/val/chunk_*.npz (SHA-256 against
               results/prereg08/feat/val_report.json; the positions of a game keep their order, pos_game is rebased to
               the drill chunk) into every drill chunk and rewrites the chunk rows of p1_<set>_report.json (the
               <set>_p1.npz files and every array t08 reads stay as they were, so 08's outputs equal its golden).
               Only fake_from_val records, only before 08's apply (R8/rank/t08_apply_report.json absent).
compare        A P1 set of 08's drill tree (GPU or from-val) against the validation P1 rows of the same games (seam:
               rows 0-399; public / private: question ids re-prefixed back, slot): failed / counts / n_moves equal, the
               largest standardised EX difference, standardised X difference and lp_P1TO_E difference (numpy float64
               forward with results/prereg09/rank/p1e_map.npz, before the x 4); whether every chunk starts off the
               validation chunk grid.
measure-tol    PREREG_09 5.2 / rev 4 (before any test data; the standardised EX difference is the judged 42
               features, the two frac_lowp features recorded only): on the 4090 rehearsal tree S of 08 (S/gpu/test08: the seam on
               the GPU, the validation chunking; S/gpu_pub/test08: private first 1,000 games on the GPU, off the
               validation chunk grid) -> m = the largest lp_P1TO_E difference of the off-grid games, tol_E =
               max(0.04, 2.5 x m); the standardised EX difference is recorded too.
Rule: the games are validation games; only the seams, the measurement and apply may use these files, never a content
statistic. No label file is opened; nothing is written inside the repository except the measurement JSON (--json),
results/test08_rehearsal and data/test08_rehearsal.
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
import t08_fake_tests as t8f  # noqa: E402
import t08_rank as t8  # noqa: E402
import t09_rank as t9  # noqa: E402
import p09_forward as FW  # noqa: E402

VAL_REPORT = t8f.VAL_REPORT
VAL_P1 = t8f.VAL_P1
P1E_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
PREFIX = t8f.PREFIX
SETS = t8f.SETS
POS_KEYS = t9.POS_KEYS
POS_DTYPES = {"pos_game": np.int32, "pos_side": np.int8, "pos_turn": np.int16, "pos_logp": np.float32,
              "pos_entropy": np.float32}       # as the validation chunks data/prereg08/val/chunk_*.npz
FLOOR_E = 0.04
FACTOR_E = 2.5
RULE = "validation games: only the P1 seams, the tol_E measurement and apply; never run content statistics"


class FakeError09(SystemExit):
    def __init__(self, message):
        super().__init__(f"t09_fake_tests: {message}")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def val_index():
    """val_report.json's chunks (SHA-256 checked when read) -> (report, first row of every chunk)."""
    vr = tb.read_json(VAL_REPORT)
    starts = np.cumsum([0] + [int(c["games"]) for c in vr["chunks"]])
    return vr, starts


def val_rows(rows):
    """The validation P1 data of the given (sorted or unsorted) rows: means, counts, n_moves, failed, ex32 and the
    per-row positions -> dict (row order as given)."""
    vr, starts = val_index()
    rows = np.asarray(rows, dtype=np.int64)
    need = sorted(set(int(np.searchsorted(starts, r, side="right") - 1) for r in rows))
    data, pos = {}, {}
    for ci in need:
        ch = vr["chunks"][ci]
        p = os.path.join(REPO, *ch["file"].split("/"))
        if not os.path.exists(p) or tb.sha256_file(p) != ch["sha256"]:
            raise FakeError09(f"{ch['file']} missing or changed since val_report.json")
        z = load(p)
        G = len(z["ref"])
        if not np.array_equal(np.asarray(z["ref"]), np.arange(starts[ci], starts[ci] + G)):
            raise FakeError09(f"{ch['file']}: refs are not the validation rows {starts[ci]}..{starts[ci] + G - 1}")
        ex = FW.ex_raw32({k: np.asarray(z[k]) for k in POS_KEYS}, G)
        pg = np.asarray(z["pos_game"])
        for g in range(G):
            r = int(starts[ci]) + g
            data[r] = (z["means"][g], z["counts"][g], z["n_moves"][g], z["failed"][g], ex[g])
            sel = np.flatnonzero(pg == g)
            pos[r] = {k: np.asarray(z[k])[sel] for k in POS_KEYS}
    out = {"means": np.stack([data[int(r)][0] for r in rows]), "counts": np.stack([data[int(r)][1] for r in rows]),
           "n_moves": np.array([data[int(r)][2] for r in rows]), "failed": np.array([data[int(r)][3] for r in rows]),
           "ex32": np.stack([data[int(r)][4] for r in rows]), "ref": np.arange(len(rows))}
    return out, pos


def dirs(r8, h):
    r8 = os.path.abspath(r8)
    return t8f.dirs(r8, h)


def add_positions(r8, h=None, sets=SETS, log=print):
    r8, h = dirs(r8, h)
    if os.path.exists(os.path.join(r8, "rank", "t08_apply_report.json")):
        raise FakeError09(f"{r8}/rank/t08_apply_report.json exists: add the positions before 08's apply (its report "
                          f"binds the P1 reports)")
    out = {}
    for s in sets:
        rep_p = os.path.join(r8, "p1", f"p1_{s}_report.json")
        rep = tb.read_json(rep_p)
        if not rep.get("fake_from_val"):
            raise FakeError09(f"{rep_p} is not a from-val record (only drills get copied positions)")
        r0, r1 = [int(x) for x in rep["source"]["rows"]]
        _d, pos = val_rows(np.arange(r0, r1))
        cdir = os.path.join(h, s)
        for ci, ch in enumerate(rep["chunks"]):
            p = os.path.join(cdir, ch["file"])
            if tb.sha256_file(p) != ch["sha256"]:
                raise FakeError09(f"{p} differs from its report")
            z = load(p)
            ref = np.asarray(z["ref"]).astype(np.int64)
            parts = {k: [] for k in POS_KEYS}
            for g, rr in enumerate(ref):
                q = pos[r0 + int(rr)]
                n = len(q["pos_side"])
                for k in POS_KEYS:
                    parts[k].append(np.full(n, g, dtype=q[k].dtype) if k == "pos_game" else q[k])
            for k in POS_KEYS:
                z[k] = np.concatenate(parts[k]).astype(POS_DTYPES[k]) if parts[k] else np.zeros(0, POS_DTYPES[k])
            tmp = p + ".tmp.npz"
            np.savez(tmp, **z)
            os.replace(tmp, p)
        plan = [(ci, 0, 0) for ci in range(len(rep["chunks"]))]
        rows, _dd = t8.read_chunks(cdir, plan, with_means=False)
        rep["chunks"] = rows
        rep["positions_copied_by"] = "t09_fake_tests add-positions (validation chunks, same games)"
        tb.write_json(rep_p, rep)
        out[s] = sum(r["positions"] for r in rows)
        log(f"t09 add-positions: {s} rows {r0}-{r1}: {out[s]} positions -> {cdir}")
    return out


def from_val(root, r8, h=None, mode="dry-run", rerun=False, log=print):
    t8f.from_val(root, r8, h, SETS, mode, rerun, log)
    return add_positions(r8, h, SETS, log)


def drill_rows(r8, s):
    """The validation rows of a drill set's games (seam: 0-399; public / private by question id and slot)."""
    z = load(os.path.join(r8, "p1", f"{s}_p1.npz"))
    if s == "seam":
        return np.arange(len(z["failed"])), z
    v1 = load(VAL_P1)
    idx = {(str(q), int(sl)): i for i, (q, sl) in enumerate(zip(v1["question_id"], v1["slot"]))}
    rows = []
    for q, sl in zip(z["question_id"], z["slot"]):
        q = str(q)
        if not q.startswith(PREFIX[s]):
            raise FakeError09(f"{q} does not carry the {PREFIX[s]} prefix")
        rows.append(idx[("val_" + q[len(PREFIX[s]):], int(sl))])
    return np.array(rows), z


def compare(r8, h, s):
    """A drill P1 set of 08's tree against the validation P1 rows of the same games, with 09's frozen map."""
    r8, h = dirs(r8, h)
    rep = tb.read_json(os.path.join(r8, "p1", f"p1_{s}_report.json"))
    rows, _z = drill_rows(r8, s)
    _r, a = t9.read_set_chunks(os.path.join(h, s), [c["file"] for c in rep["chunks"]], [c["sha256"] for c in
                                                                                        rep["chunks"]])
    if len(a["failed"]) != len(rows):
        raise FakeError09(f"{s}: {len(a['failed'])} chunk games, {len(rows)} rows")
    b, _pos = val_rows(rows)
    mp = FW.load_map(P1E_MAP)
    rec = t9.compare_sets(a, b, mp)
    _vr, starts = val_index()
    chunk = int(starts[1] - starts[0])
    first = rows[::t8.CHUNK]
    rec.update(set=s, report_mode=rep.get("mode"), fake_from_val=bool(rep.get("fake_from_val")),
               environment={k: (rep.get("environment") or {}).get(k) for k in ("torch", "cudnn", "gpu", "driver")},
               positions_per_s=rep.get("positions_per_s"), validation_rows=[int(rows.min()), int(rows.max())],
               rows_contiguous=bool(np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows)))),
               chunk_starts_val_rows=[int(x) for x in first],
               chunk_boundaries_differ_from_val=bool(all(int(x) % chunk != 0 for x in first)),
               map_sha256=tb.sha256_file(P1E_MAP), rule=RULE)
    return rec


def ex_breakdown(r8, s, top=8):
    """Record only: the standardised EX difference per feature of a drill set against the validation rows (which
    of the 44 features moves; the low-probability fractions are counts over a hard threshold)."""
    r8, h = dirs(r8, None)
    rep = tb.read_json(os.path.join(r8, "p1", f"p1_{s}_report.json"))
    rows, _z = drill_rows(r8, s)
    _r, a = t9.read_set_chunks(os.path.join(h, s), [c["file"] for c in rep["chunks"]])
    b, _pos = val_rows(rows)
    mp = FW.load_map(P1E_MAP)
    ok = FW.usable_games(a["failed"], a["counts"]) & FW.usable_games(b["failed"], b["counts"])
    d = np.abs(t9.std_ex(a["ex32"][ok], mp) - t9.std_ex(b["ex32"][ok], mp))
    names = FW.ex_names()
    col = d.max(0)
    rest = [i for i, n in enumerate(names) if "frac_lowp" not in n]
    return {"largest": [{"feature": names[i], "max_abs": float(col[i]),
                         "q99": float(np.quantile(d[:, i], 0.99)), "sd_ex": float(mp["sd_ex"][i])}
                        for i in np.argsort(-col)[:top]],
            "max_abs_without_frac_lowp": float(d[:, rest].max()),
            "games_with_a_frac_lowp_change": int((d[:, [i for i in range(len(names)) if i not in rest]] > 0)
                                                 .any(1).sum()),
            "per_game_max_quantiles": {q: float(np.quantile(d.max(1), float(q))) for q in ("0.5", "0.9", "0.99")}}


def measure_tol(tree):
    """PREREG_09 5.2: tol_E from the 4090 rehearsal tree of 08 (D:/c10_t08_reh): the GPU seam (validation chunking)
    and the private first 1,000 games (off the validation chunk grid)."""
    tree = os.path.abspath(tree)
    seam = compare(os.path.join(tree, "gpu", "test08"), None, "seam")
    priv = compare(os.path.join(tree, "gpu_pub", "test08"), None, "private")
    if not priv["chunk_boundaries_differ_from_val"] or priv["fake_from_val"] or seam["fake_from_val"]:
        raise FakeError09("the measurement needs GPU sets; private must start off the validation chunk grid")
    if not (seam["counts_equal"] and priv["counts_equal"]):
        raise FakeError09("failed / counts / n_moves differ: no measurement")
    m = float(priv["max_abs_lp_p1e"])
    tol = max(FLOOR_E, FACTOR_E * m)
    priv["std_ex_by_feature"] = ex_breakdown(os.path.join(tree, "gpu_pub", "test08"), "private")
    return {"what": "PREREG_09 5.2 tol_E (before any test data): the largest lp_P1TO_E difference (per game, before "
                    "the x 4, numpy float64 forward with the frozen p1e_map.npz) between P1 chunks computed on the "
                    "RTX 4090 in the 08 rehearsal and the validation chunks of the same games",
            "tree": tree, "map_sha256": tb.sha256_file(P1E_MAP), "val_report_sha256": tb.sha256_file(VAL_REPORT),
            "seam_gpu_same_chunking": seam, "private_gpu_off_grid": priv,
            "m": m, "rule": f"tol_E = max({FLOOR_E}, {FACTOR_E} x m)", "tol_E": tol,
            "std_ex_rule": "PREREG_09 rev 4: judged over the 42 EX features other than t_frac_lowp / o_frac_lowp "
                           "(those two recorded only)",
            "std_ex_max_off_grid_judged_42": priv["max_abs_std_ex"],
            "std_ex_max_off_grid_all_44": priv["max_abs_std_ex_all_44"],
            "frac_lowp_record_only_off_grid": priv["frac_lowp_record_only"], "std_ex_tolerance": t9.TOL_EX,
            "std_ex_within_tolerance": bool(priv["max_abs_std_ex"] <= t9.TOL_EX),
            "lp_p1e_within_tol_E": bool(m <= tol),
            "tool_sha256_lf": {n: tb.sha256_text_file(os.path.join(HERE, n)) for n in ("t09_fake_tests.py", "t09_rank.py",
                                                                                   "p09_forward.py")}}


def write_json(path, obj):
    path = os.path.abspath(path)
    allowed = [os.path.join(REPO, "results", "test09", "design"), os.path.join(REPO, "results", "test09_rehearsal")]
    if tb.inside(path, REPO) and not any(tb.inside(path, r) for r in allowed):
        raise FakeError09(f"--json {path}: inside the repository only under results/test09/design or "
                          f"results/test09_rehearsal")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t9.jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="t09_fake_tests.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-val")
    p.add_argument("--root", required=True)
    p.add_argument("--t08-results-dir", required=True)
    p.add_argument("--p1-data-dir")
    p.add_argument("--mode", default="dry-run", choices=("dry-run", "rehearsal"))
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("add-positions")
    p.add_argument("--t08-results-dir", required=True)
    p.add_argument("--p1-data-dir")
    p = sub.add_parser("compare")
    p.add_argument("--t08-results-dir", required=True)
    p.add_argument("--p1-data-dir")
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--json")
    p = sub.add_parser("measure-tol")
    p.add_argument("--tree", required=True)
    p.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "from-val":
        from_val(a.root, a.t08_results_dir, a.p1_data_dir, a.mode, a.rerun)
        return 0
    if a.cmd == "add-positions":
        add_positions(a.t08_results_dir, a.p1_data_dir)
        return 0
    rec = compare(a.t08_results_dir, a.p1_data_dir, a.set) if a.cmd == "compare" else measure_tol(a.tree)
    print(json.dumps(t9.jsonable(rec), indent=1, ensure_ascii=False))
    if a.json:
        write_json(a.json, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
