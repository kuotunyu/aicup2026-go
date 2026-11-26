"""Tree H01: exactly the real test placement '(;SZ[19]HD[n];' with n in {0,1} (~20 % HD1). Usage: PY rehearse_h01.py WT F OUT"""
import hashlib
import json
import os
import sys

WT, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rehearse as R  # noqa: E402  (results/root_props/rehearse.py; argv-driven: same WT / F / OUT)

tc, tp, NAMES = R.tc, R.tp, R.NAMES


def add_hd01(src, dst):
    os.makedirs(dst, exist_ok=True)
    st = {}
    for comp in ("rank", "player"):
        for _pool, kind, name in tc.FILES[comp]:
            text = open(os.path.join(src, name), "rb").read().decode("utf-8")
            want = {g[3] for g in tc.file_games(tc.Table(os.path.join(src, name)), kind)}
            ins, k, hd1 = [], 0, 0
            for rec in tp.scan_cells(text):
                for v, a, _b, _q in rec:
                    if v in want and v.startswith("(;SZ[19]"):
                        k += 1
                        n = int(R.u01("h01", name, k) < 0.20)
                        hd1 += n
                        ins.append((a + 8, "HD[%d]" % n))
            parts, pos = [], 0
            for i, s in sorted(ins):
                parts += [text[pos:i], s]
                pos = i
            parts.append(text[pos:])
            open(os.path.join(dst, name), "wb").write("".join(parts).encode("utf-8"))
            st[name] = {"games": k, "HD1": hd1}
    return st


def main():
    clean = os.path.join(OUT, "dl_clean")
    dl = os.path.join(OUT, "dl_H01")
    res = {"injected": add_hd01(clean, dl)}
    import hs_lib as H
    A, Hh = R.games_of(clean), R.games_of(dl)
    kat = {}
    for n in NAMES:
        ok_a = ok_h = 0
        for (_c, sa), (_d, sh) in zip(A[n], Hh[n]):
            for s, which in ((sa, "a"), (sh, "h")):
                try:
                    H.parse_moves(s)
                    if which == "a":
                        ok_a += 1
                    else:
                        ok_h += 1
                except H.HSError:
                    pass
        kat[n] = {"games": len(A[n]), "clean_ok": ok_a, "H01_ok": ok_h}
    res["hs_lib_parse_raw_H01"] = kat
    res["sample"] = [g[1][:24] for g in Hh[NAMES[0]][:3]]
    root = os.path.join(OUT, "H01")
    R.run("t02_test_bundle.py", "ingest", "--src", dl, "--root", root)
    res["t1_raw"] = {c: (R.t1(root, c), R.t1j(root, c)["verdict"]) for c in ("rank", "player")}
    res["t1_raw_equals_clean_except_other_props"] = {
        c: R.t1cmp(R.t1j(root, c)) == R.t1cmp(R.t1j(os.path.join(OUT, "A"), c)) for c in ("rank", "player")}
    res["rp_prepare_exit"] = R.run("root_props.py", "prepare", "--root", root)
    pj = tc.read_json(os.path.join(root, "results", "test02", "checks", "root_props", "prepare.json"))
    res["files"] = {n: {k: pj["files"][n][k] for k in ("games", "games_edited", "removed_props", "hd_values",
                                                     "plain_form_before", "plain_form_after", "counts")} for n in NAMES}
    norm = os.path.join(root, "data", "test_rootprops", "normalised")
    res["normalised_equals_clean"] = {n: R.sha(os.path.join(norm, n)) == R.sha(os.path.join(clean, n)) for n in NAMES}
    R.run("t02_test_bundle.py", "ingest", "--src", norm, "--root", root, "--rerun", "--reason", R.R11)
    res["t1_copy"] = {c: (R.t1(root, c, R.R11), R.t1j(root, c)["verdict"]) for c in ("rank", "player")}
    res["verify_exit"] = R.run("root_props.py", "verify", "--root", root)
    R.run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", root)
    a_built = {n: R.sha(os.path.join(OUT, "A", "data", "test", n)) for n in NAMES}
    res["built_equals_A"] = {n: R.sha(os.path.join(root, "data", "test", n)) == a_built[n] for n in NAMES}
    with open(os.path.join(OUT, "rehearsal_h01.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
