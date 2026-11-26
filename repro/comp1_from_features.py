"""Layer 1, competition 1 (player, TEAM_11249): stored features -> the final submission v2-9, CPU only.

    python repro/comp1_from_features.py [--root REPO] [--tests-dir DIR] [--out DIR] [--pool public|private|both]
                                        [--tolerant]

Calls the registered program results/prereg67/explore/p67_eval.py (apply per pool, then assemble); nothing of the
method is re-implemented here. For each of the two new pools (pub2_ / priv2_) apply reads the official v2 test files,
rebuilds the hand-made terms (G) from the SGFs, loads the stored P1 features and the stored CNN vectors of the seven
members, refits the L-player LDA (checked against prereg43 tune.json), rebuilds v2-8's top5 as a seam (byte for byte
= results/prereg66/apply/<pool>/top5.csv), and writes DIR/<pool>/top5.csv. assemble puts the 800 original rows of
the stored v1 06-ON submission (results/test06/player/on/submission_player.csv) in front of the 800 new rows.
This script then compares DIR/submission_player.csv with the uploaded file by SHA-256.

The original 800 rows (pub_ / priv_, 06-ON) are NOT recomputed here: they are copied from the stored v1 output, as
in the registered assemble step. See repro/README.md, "competition 1, original questions".
No file of the repository is edited; data/local is never opened.

--tolerant (for a platform whose floating-point last bits differ from the original Windows run, e.g. Linux): the
L-LDA refit is compared with the stored Windows fit (repro/ref_lda_L.npz) within a tolerance instead of by hash,
and a difference in the rebuilt v2-8 top5 is recorded instead of stopping the run, in this process only; the real
differences are written to DIR/tolerant_report.json (repro/_tolerant.py). PASS is still only the final CSV's
SHA-256. Without the flag nothing is different.
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

EXPECTED = "36b222d866267abffda03d126d94db9ed5bea3406b2f8b47323622fd5329a026"   # results/prereg67/apply/submission_player.csv
UPLOADED = "results/prereg67/apply/submission_player.csv"
TOP5 = {"public": "results/prereg67/apply/public/top5.csv", "private": "results/prereg67/apply/private/top5.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="repository root (default: $C10_ROOT, else the parent of repro/)")
    ap.add_argument("--tests-dir", help="folder with the official v2 player test CSVs (default: $C10_TESTS_PLAYER, "
                                        "else <root>/data/test_v2_download/x/tests)")
    ap.add_argument("--out", help="output folder outside the repository (default: a new temporary folder)")
    ap.add_argument("--pool", default="both", choices=["public", "private", "both"],
                    help="both (default) also assembles and checks the final file")
    ap.add_argument("--list-inputs", metavar="JSON", help="also write the list of repository files this run read "
                                                          "(path, bytes, SHA-256)")
    ap.add_argument("--tolerant", action="store_true",
                    help="compare the exact seams with a tolerance and record the differences in "
                         "<out>/tolerant_report.json (repro/_tolerant.py); PASS is still the final SHA-256 only")
    a = ap.parse_args()
    root = C.repo_root(a.root)
    src = os.path.abspath(a.tests_dir or os.environ.get("C10_TESTS_PLAYER")
                          or os.path.join(root, "data", "test_v2_download", "x", "tests"))
    out = C.out_dir(a.out, root, "comp1")
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "results", "prereg67", "explore"))
    t0 = time.time()
    if a.list_inputs:
        C.trace_start(root)
    import p67_eval as P67                                    # imports p66 / p63 / p49 / p43 / p39 / p38 / ... (unedited)
    if os.name != "nt":                                     # Windows: p38_backend.polite() does this itself
        print(f"BLAS threads limited to 1 (as in the original Windows run): {C.blas_one_thread_posix()}", flush=True)
    rep = None
    if a.tolerant:
        import _tolerant as T
        rep = T.Report("competition 1 (v2-9): results/prereg67/explore/p67_eval.py apply / assemble", out)
        T.install_comp1(P67, rep)
        print(f"TOLERANT mode: the L-LDA refit and the v2-8 top5 seam compared with a tolerance / recorded; "
              f"differences -> {rep.path}", flush=True)
    ok = True
    pools = ["public", "private"] if a.pool == "both" else [a.pool]
    final = os.path.join(out, "submission_player.csv")
    try:
        for pool in pools:
            t1 = time.time()
            if rep:
                rep.context = pool
            rc = P67.main(["apply", "--src", src, "--pool", pool, "--out", out])
            if rc != 0:
                print(f"FAIL: p67_eval.py apply --pool {pool} returned {rc}")
                return 1
            got, ref = os.path.join(out, pool, "top5.csv"), os.path.join(root, TOP5[pool])
            same = C.same_lf(got, ref)                        # the stored copy may be CRLF in a Windows checkout
            ok &= same
            print(f"new questions, {pool}: top5.csv {'= ' if same else '!= '}{TOP5[pool]} "
                  f"({'PASS' if same else 'FAIL'}; {time.time() - t1:.1f} s)", flush=True)
            if rep:
                d = C.rows_diff(got, ref)
                rep.add(f"v2-9 top5 of the pool vs {TOP5[pool]} (the entry point's own comparison, not a seam "
                        f"of the tool)", "discrete", same, same, rows_different=d["rows_different"],
                        rows=[d["rows_produced"], d["rows_uploaded"]], first_differences=d["first_differences"])
        if rep:
            rep.context = ""
        if a.pool == "both":
            rc = P67.main(["assemble", "--src", src, "--out", out])
            if rc != 0:
                print(f"FAIL: p67_eval.py assemble returned {rc}")
                return 1
            ok &= C.verdict("competition 1 (v2-9)", final, EXPECTED, os.path.join(root, UPLOADED))
    finally:
        if rep:
            rep.write(C.final_record(final, EXPECTED, os.path.join(root, UPLOADED)) if a.pool == "both" else None)
            print(f"tolerant report: {rep.path}")
    print(f"seconds {time.time() - t0:.1f}")
    if a.list_inputs:
        C.trace_write(root, os.path.abspath(a.list_inputs))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
