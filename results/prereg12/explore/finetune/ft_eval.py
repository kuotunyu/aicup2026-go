"""prereg12 exploration (competition 2): inner evaluation of the fine-tuning pilot (PLAN.md section 5). CPU only.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg12/explore/finetune
    $S/ft_eval.py prep-inner                -> data/prereg12/q09_inner/ (the P1 TO / EX arrays of the bank / mono
                                               games, built with tools/p09_map.read_chunk_rows from the 08 manifest's
                                               chunks, SHA-checked; = p09_map prep's inner part)
    $S/ft_eval.py eval                      -> $S/eval_report.json (the pilot gate; judged once)
    $S/ft_eval.py eval --fake-ft --out-root data/prereg12/smoke   (CPU smoke: lp_FT = noisy Q09 lp; never a result)

Machinery (imported, never edited): tools/p09_rank_fuse.py Inner (u1_lib inner inputs, HS baseline, Q02 / Q02H / Q08,
5-fold player OOF fusion = p04 generalised calibration, rule B, PlayerBootstrap) and p09_forward (Q09's frozen
p1e_map.npz). Seams before anything else: Q02H, Q08, Q08 single-colour = p09 REF (1e-12); Q09 inner and single-colour
= results/prereg09/rank/tune_report.json (1e-12).
Candidate (fixed in PLAN.md): Q09+FT = OP, O4, SWA, SWB, HS, P1TO (= 4 lp_P1TO_E), FT (= 4 lp_FT, masked mean over
the question's usable games). Gate vs Q09: G1 inner rank-balanced difference >= +0.010; G2 paired PlayerBootstrap 95 %
lower bound > 0; G4 single-colour (120,544 questions) lower bound > 0 and black and white differences each > 0.
Never data/local (u1_lib / hs_lib guards), never validation features, never tools/score.py.
"""

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402  (pins the p08 tools before importing them)
import p09_map as M  # noqa: E402
import p09_forward as FW  # noqa: E402

R = r.R
L = r.L
SCALE_FT = 4.0
GATE = 0.010
TOL = 1e-12
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
DATA = os.path.join(REPO, "data", "prereg12")
PREP_DIR = os.path.join(DATA, "q09_inner")
REPORT = os.path.join(HERE, "eval_report.json")
FT_FORMAT = "c10-prereg12 ft inner v1"
FORBIDDEN = ("data/local", "data/prereg08/val", "data/prereg10/val", "results/prereg08/feat/val_",
             "results/prereg09/feat/val_", "results/prereg10/feat/val_")


def guard_path(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    for f in FORBIDDEN:
        if ("/" + f.lower()) in q:
            raise SystemExit(f"refused path {p} (validation material)")
    return p


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


class InnerPrep:
    """Duck type of p09_rank_fuse.Prep for the inner part only (inner_ref, inner_src, inner_X, inner_ex,
    inner_rows)."""

    def __init__(self, d):
        self.dir = d
        self.inner_X = np.load(os.path.join(d, "inner_X.npy"), mmap_mode="r")
        self.inner_ex = np.load(os.path.join(d, "inner_ex.npy"), mmap_mode="r")
        with np.load(os.path.join(d, "inner_meta.npz"), allow_pickle=False) as z:
            self.inner_ref = np.asarray(z["ref"]).astype(np.int64)
            self.inner_src = np.asarray(z["src"]).astype(str)
        if len(self.inner_ref) != r.INNER_USABLE:
            raise SystemExit(f"inner rows {len(self.inner_ref)} != {r.INNER_USABLE}")

    def inner_rows(self, sets=("bank", "mono")):
        return np.nonzero(np.isin(self.inner_src, np.array(list(sets))))[0]


def cmd_prep_inner(a):
    d = guard_path(os.path.abspath(a.prep_dir))
    os.makedirs(d, exist_ok=True)
    man = R.read_json(r.P08_MANIFEST)
    if r.sha_lf(r.P08_MANIFEST) != r.P08_FROZEN_PINS["results/prereg08/feat/inner_manifest.json"]:
        raise SystemExit("08 inner_manifest.json differs from its pin")
    t0 = time.time()
    Xs, EXs, refs, src, shas = [], [], [], [], {}
    for s in ("bank", "mono"):
        rows = (man["sets"].get(s) or {}).get("chunks") or []
        shas[s] = []
        for x in rows:
            c = M.read_chunk_rows(guard_path(r.path_of(x["file"])), x["sha256"], ())
            Xs.append(c["X"])
            EXs.append(c["EX"])
            refs.append(c["ref"])
            src += [s] * len(c["ref"])
            shas[s].append(c["sha"])
        print(f"prep-inner {s}: {len(rows)} chunks, {time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(Xs)
    EX = np.concatenate(EXs)
    np.save(os.path.join(d, "inner_X.npy"), X)
    np.save(os.path.join(d, "inner_ex.npy"), EX)
    np.savez(os.path.join(d, "inner_meta.npz"), ref=np.concatenate(refs), src=np.array(src, dtype=str))
    rec = {"format": "c10-prereg12 q09 inner prep v1", "created": r.now_iso(), "rows": int(len(X)),
           "expected_rows": r.INNER_USABLE, "chunk_sha256_checked_against": "results/prereg08/feat/inner_manifest.json",
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(d, "prep_record.json"), rec)
    if len(X) != r.INNER_USABLE:
        raise SystemExit(f"{len(X)} inner rows, {r.INNER_USABLE} expected")
    print(json.dumps(rec))
    return 0


def load_ft(ft_root, n_inner):
    """FT chunks of bank and mono -> (lp_full [n_inner, 10], usable_full [n_inner], record)."""
    lp = np.zeros((n_inner, L))
    usable = np.zeros(n_inner, dtype=bool)
    seen = np.zeros(n_inner, dtype=bool)
    rec = {"chunks": {}, "best_sha256": set(), "tiny": set()}
    for s in ("bank", "mono"):
        dd = guard_path(os.path.join(ft_root, "ft", s))
        files = sorted(f for f in os.listdir(dd) if f.startswith("chunk_") and f.endswith(".npz") and ".tmp" not in f)
        rec["chunks"][s] = len(files)
        for f in files:
            with np.load(os.path.join(dd, f), allow_pickle=False) as z:
                if str(z["format"]) != FT_FORMAT:
                    raise SystemExit(f"{f}: not an FT chunk")
                ref = np.asarray(z["ref"]).astype(np.int64)
                if seen[ref].any():
                    raise SystemExit(f"{s}/{f}: an inner row twice")
                seen[ref] = True
                lp[ref] = np.asarray(z["lp"], dtype=np.float64)
                usable[ref] = (~np.asarray(z["failed"], dtype=bool)) & (np.asarray(z["counts"])[:, 0] > 0)
                rec["best_sha256"].add(str(z["best_sha256"]))
                rec["tiny"].add(bool(z["tiny"]))
    rec["best_sha256"] = sorted(rec["best_sha256"])
    rec["tiny"] = sorted(rec["tiny"])
    rec["rows"] = int(seen.sum())
    rec["usable"] = int(usable.sum())
    return lp, usable, rec


def tune_ref():
    with open(Q09_TUNE, encoding="utf-8") as f:
        t = json.load(f)
    c = t["combinations"]["Q09"]
    return {"Q09": c["oof"]["rank_balanced"], "Q09_mono": c["single_colour"]["rank_balanced"]}


def gates(cmp):
    by = cmp["mono_by_colour"]
    g1 = cmp["inner_diff"] >= GATE - TOL
    g2 = cmp["inner_ci95"][0] > 0
    g4 = (cmp["mono_ci95"][0] is not None and cmp["mono_ci95"][0] > 0
          and all(by.get(c) is not None and by[c] > 0 for c in ("B", "W")))
    return {"G1": {"threshold": GATE, "delta": cmp["inner_diff"], "passed": bool(g1)},
            "G2": {"ci95": cmp["inner_ci95"], "passed": bool(g2)},
            "G4": {"mono_diff": cmp["mono_diff"], "mono_ci95": cmp["mono_ci95"], "by_colour": by,
                   "passed": bool(g4)},
            "passed": bool(g1 and g2 and g4)}


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.fake_ft) or os.path.normcase(os.path.abspath(a.out_root)) != os.path.normcase(DATA)
    out_path = os.path.join(os.path.abspath(a.out_root), "eval_report.json") if smoke else REPORT
    if not smoke and os.path.exists(out_path) and not (a.rerun and a.reason):
        raise SystemExit(f"{out_path} exists: the gate is judged once (--rerun --reason TEXT only after a commit "
                         f"that names a concrete program error)")
    prep = InnerPrep(guard_path(os.path.abspath(a.prep_dir)))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        fc = json.load(f)
    if map_sha != fc["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=lambda *x: print(*x, flush=True))
    ref = tune_ref()
    lp_q09 = r.inner_map_lp(prep, FW.load_map(Q09_MAP))
    q09 = inner.evaluate(lp_q09)
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    seam_ok = all(abs(x - y) <= TOL for x, y in seam.values())
    print(f"seam {'ok' if seam_ok else 'FAILED'}: {seam}", flush=True)
    if not seam_ok:
        rep = {"format": "c10-prereg12 ft eval v1", "status": "stop: seam failed", "seam": seam, "smoke": smoke}
        write_json(out_path, rep)
        return 1
    n_inner = inner.n_inner
    if a.fake_ft:
        rng = np.random.default_rng(0)
        z = lp_q09 + rng.normal(0.0, 0.5, size=lp_q09.shape)
        lp_rows = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
        lp_full = inner.lp_full(lp_rows)
        usable = inner.covered.copy()
        ft_rec = {"fake": "lp_Q09 + N(0, 0.5) noise, renormalised (smoke only)"}
    else:
        lp_full, usable, ft_rec = load_ft(os.path.abspath(a.out_root), n_inner)
        if True in ft_rec["tiny"]:
            raise SystemExit("FT chunks were made with the tiny smoke model")
    mask_equal = bool(np.array_equal(usable, inner.covered))
    ft_rec["usable_equals_p1_usable"] = mask_equal
    ft_rec["usable_minus_p1"] = int((usable & ~inner.covered).sum())
    ft_rec["p1_minus_usable"] = int((inner.covered & ~usable).sum())
    if not mask_equal and not smoke:
        rep = {"format": "c10-prereg12 ft eval v1", "status": "stop: FT usable games differ from P1's", "ft": ft_rec}
        write_json(out_path, rep)
        return 1
    E_q09, _ = inner.E_p1(lp_q09)
    E_ft4, _ = R.masked_inner(inner.d, lp_full, usable, scale=SCALE_FT)
    E_ft1, _ = R.masked_inner(inner.d, lp_full, usable, scale=1.0)
    terms = list(r.Q08_TERMS) + ["FT"]
    cand = inner.run(terms, {"HS": inner.E_hs, "P1TO": E_q09, "FT": E_ft4})
    cmp_ = inner.compare(cand, q09)
    g = gates(cmp_)
    print(f"Q09+FT {cand['rb']!r} vs Q09 {q09['rb']!r}: {json.dumps(g)}", flush=True)
    rec_only = {}
    c1 = inner.run(terms, {"HS": inner.E_hs, "P1TO": E_q09, "FT": E_ft1})
    rec_only["Q09+FT_x1"] = inner.short(c1, q09)
    rep_ft = inner.run(list(r.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_ft4})
    rec_only["Q02H+FT_x4_instead_of_P1TO_E"] = inner.short(rep_ft, q09)
    rec_only["direction_ft_vs_q02"] = r.direction_ref(inner, E_ft4, inner.q02)
    rep = {"format": "c10-prereg12 ft eval v1", "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke": smoke, "fake_ft": bool(a.fake_ft),
           "what": "PLAN.md 5: inner gate of the HumanSL fine-tuning pilot (training players only; no validation "
                   "scoring; competition 2 has no validation score left)",
           "status": "gates passed" if g["passed"] else "gates not passed: stop", "seam": seam,
           "candidate": {"name": "Q09+FT", "terms": terms, "scale_FT": SCALE_FT, **inner.short(cand, q09)},
           "baseline": {"name": "Q09", **inner.short(q09)}, "gates": g, "record_only": rec_only, "ft": ft_rec,
           "q09_map_sha256": map_sha, "inner_rows": int(len(prep.inner_ref))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_json(out_path, rep)
    print(f"-> {out_path}: {rep['status']}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-inner")
    p.add_argument("--prep-dir", default=PREP_DIR)
    p = sub.add_parser("eval")
    p.add_argument("--prep-dir", default=PREP_DIR)
    p.add_argument("--out-root", default=DATA, help="FT root (data/prereg12) or a smoke root")
    p.add_argument("--fake-ft", action="store_true", help="smoke: noisy Q09 lp in place of FT")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"prep-inner": cmd_prep_inner, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
