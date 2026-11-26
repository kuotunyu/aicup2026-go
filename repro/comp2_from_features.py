"""Layer 1, competition 2 (rank, TEAM_11248): stored per-game features -> the final submission 69, CPU only.

    python repro/comp2_from_features.py [--root REPO] [--out DIR] [--tolerant]

Calls the frozen test-day tool tools/t69_rank.py in its rehearsal mode (apply --out DIR); nothing of the method is
re-implemented here. That tool recomputes Q25's seven terms from the stored per-game sources, checks seam A (Q25 =
results/test25 byte for byte), computes the opponent term OPP3 from the stored c49 vectors, checks seam T (merge off
= results/test64 byte for byte), applies M8 per pool and rule B, runs check_submission --official, and writes
DIR/rank/submission_rank.csv. This script then compares that file with the uploaded one by SHA-256.

One thing is changed for this process only: t64_rank.K6_64 (the registered competition deadline 2026-11-24 23:00,
after which the tool refuses to produce anything) is moved, because a reproduction necessarily runs after it.
No file is edited; data/local is never opened.

--tolerant (for a platform whose floating-point last bits differ from the original Windows run, e.g. Linux): the
byte-for-byte seams A and T are compared with a tolerance instead, in this process only, and the real differences
are written to DIR/tolerant_report.json (repro/_tolerant.py). PASS is still only the final CSV's SHA-256. Without
the flag nothing is different.
"""
import argparse
import datetime as dt
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

EXPECTED = "c7595f2eb43c185973c55c7b4b03875de41c5ce92327f7042fc3769dbbae36c8"   # results/test69/rank/submission_rank.csv
UPLOADED = "results/test69/rank/submission_rank.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="repository root (default: $C10_ROOT, else the parent of repro/)")
    ap.add_argument("--out", help="output folder outside the repository (default: a new temporary folder)")
    ap.add_argument("--list-inputs", metavar="JSON", help="also write the list of repository files this run read "
                                                          "(path, bytes, SHA-256)")
    ap.add_argument("--tolerant", action="store_true",
                    help="compare the byte-for-byte seams with a tolerance and record the differences in "
                         "<out>/tolerant_report.json (repro/_tolerant.py); PASS is still the final SHA-256 only")
    a = ap.parse_args()
    root = C.repo_root(a.root)
    out = C.out_dir(a.out, root, "comp2")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v):
            sys.exit(f"ERROR: unset {v} (the registered byte seams use the default BLAS threads)")
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "tools"))
    t0 = time.time()
    if a.list_inputs:
        C.trace_start(root)
    import t69_rank as t69                                    # imports t64 / t59 / t25 / ... (frozen, unedited)
    t69.t64.K6_64 = dt.datetime(9999, 1, 1, tzinfo=t69.t25.TW)  # the deadline guard only (see the module docstring)
    rep = None
    if a.tolerant:
        import _tolerant as T
        rep = T.Report("competition 2 (69): tools/t69_rank.py apply", out)
        T.install_comp2(t69, rep)
        print(f"TOLERANT mode: seams A and T compared with a tolerance; differences -> {rep.path}", flush=True)
    final = os.path.join(out, "rank", "submission_rank.csv")
    try:
        rc = t69.main(["apply", "--out", out, "--rerun", "--reason", "repro/comp2_from_features.py"])
    finally:
        if rep:
            rep.write()
    if rc != 0:
        print(f"FAIL: tools/t69_rank.py apply returned {rc}")
        return 1
    ok = C.verdict("competition 2 (69)", final, EXPECTED, os.path.join(root, UPLOADED))
    if rep:
        rep.write(C.final_record(final, EXPECTED, os.path.join(root, UPLOADED)))
        print(f"tolerant report: {rep.path}")
    print(f"seconds {time.time() - t0:.1f}")
    if a.list_inputs:
        C.trace_write(root, os.path.abspath(a.list_inputs))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
