"""PREREG_02 rev 11 scratch rehearsal (kept as results/root_props/rehearse.py). Usage: PY rehearse.py <worktree> <fake tree F> <out>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

WT, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PY = sys.executable
for d in ("tools", "tests", os.path.join("results", "prereg07", "explore", "pilot_humansl"),
          os.path.join("results", "prereg08", "p1")):
    sys.path.insert(0, os.path.join(WT, d))
import t02_checks as tc  # noqa: E402
import trailing_pass as tp  # noqa: E402
import root_props as rp  # noqa: E402
from test_trailing_pass import inject  # noqa: E402

NAMES = tp.ALL_NAMES
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "rehearsal.log"), "a", encoding="utf-8")
R9 = "PREREG_02 rev 9 rehearsal: trailing passes removed"
R11 = "PREREG_02 rev 11 rehearsal: root properties removed"
RES = ["B+R", "W+R", "B+12.5", "W+0.5", "W+T", "0"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def u01(*key):
    return int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:8], 16) / 2 ** 32


def run(*args, ok=(0,)):
    cmd = [PY, "-B", os.path.join(WT, "tools", args[0])] + list(args[1:])
    r = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    LOG.write("$ %s\n-> %d\n%s%s\n" % (" ".join(cmd), r.returncode, r.stdout[-6000:], r.stderr[-4000:]))
    LOG.flush()
    if r.returncode not in ok:
        raise SystemExit("FAILED %s (%d): %s %s" % (args, r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    return r.returncode


def t1(root, comp, rerun=None):
    a = ["t02_checks.py", "t1", "--comp", comp, "--dry-run", "--tests-dir", os.path.join(root, "data", "test", "raw"),
         "--out", os.path.join(root, "results", "test02", "checks")]
    if rerun:
        a += ["--rerun", "--reason", rerun]
    return run(*a, ok=(0, 2, 3))


def t1j(root, comp):
    return tc.read_json(os.path.join(root, "results", "test02", "checks", comp, "t1.json"))


def t1cmp(d):
    """the T1 content without provenance, file SHA-256s and the record-only other-property counts"""
    d = json.loads(json.dumps(d))
    d.pop("provenance", None)
    for k in list(d):
        if k in ("created", "files", "rerun_reason", "out_dir", "tests_dir"):
            d.pop(k)
    for f in ((d.get("subchecks") or {}).get("T1.4") or {}).get("files", {}).values():
        f.pop("other_properties_record_only", None)
        f.get("counts", {}).pop("regex_vs_parse_mismatch_record_only", None)
    return d


def t45(root):
    out = {}
    for comp in ("rank", "player"):
        run("t02_checks.py", "run", "--comp", comp, "--dry-run", "--root", root, "--only", "T4,T5")
        for c in ("t4", "t5"):
            d = tc.read_json(os.path.join(root, "results", "test02", "checks", comp, c + ".json"))
            d.pop("provenance", None)
            out["%s/%s" % (comp, c)] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return out


def add_root_props(src_dir, dst_dir, salt):
    """Every T1.4 game of the six files gets HD[n] right after SZ[19] (v1.1 placement); 15 % also CA[UTF-8]RE[..]
    after it, 15 % CA[UTF-8] before SZ[19] and RE[..] after HD. HD: 80 % 0, 16 % 1, 4 % 2-9."""
    os.makedirs(dst_dir, exist_ok=True)
    stats = {}
    for comp in ("rank", "player"):
        for _pool, kind, name in tc.FILES[comp]:
            text = open(os.path.join(src_dir, name), "rb").read().decode("utf-8")
            t = tc.Table(os.path.join(src_dir, name))
            want = {g[3] for g in tc.file_games(t, kind)}
            ins, k, hd, forms = [], 0, {}, {"hd": 0, "hd_ca_re": 0, "ca_first": 0}
            for rec in tp.scan_cells(text):
                for v, a, _b, _q in rec:
                    if v not in want or not v.startswith("(;SZ[19]"):
                        continue
                    k += 1
                    u = u01(salt, name, k, "hd")
                    n = 0 if u < 0.80 else 1 if u < 0.96 else 2 + int(u01(salt, name, k, "h") * 8)
                    hd[n] = hd.get(n, 0) + 1
                    re_ = RES[int(u01(salt, name, k, "re") * len(RES))]
                    w = u01(salt, name, k, "form")
                    if w < 0.70:
                        ins.append((a + 8, "HD[%d]" % n))
                        forms["hd"] += 1
                    elif w < 0.85:
                        ins.append((a + 8, "HD[%d]CA[UTF-8]RE[%s]" % (n, re_)))
                        forms["hd_ca_re"] += 1
                    else:
                        ins.append((a + 2, "CA[UTF-8]"))
                        ins.append((a + 8, "HD[%d]RE[%s]" % (n, re_)))
                        forms["ca_first"] += 1
            parts, pos = [], 0
            for i, s in sorted(ins):
                parts.append(text[pos:i])
                parts.append(s)
                pos = i
            parts.append(text[pos:])
            open(os.path.join(dst_dir, name), "wb").write("".join(parts).encode("utf-8"))
            stats[name] = {"games": k, "hd": dict(sorted(hd.items())), "forms": forms}
    return stats


def games_of(d):
    out = {}
    for comp in ("rank", "player"):
        for _pool, kind, name in tc.FILES[comp]:
            t = tc.Table(os.path.join(d, name))
            out[name] = [(g[2], g[3]) for g in tc.file_games(t, kind)]
    return out


def parser_compare(clean_dir, h_dir):
    """Direct outputs of the SGF consumers on the raw H games vs the clean games (same order)."""
    import numpy as np
    import p02_moves
    import p02_player_open as po
    import p02_rank_opening as pr
    import p04_opp
    import p06_style
    sys.path.insert(0, os.path.join(WT, "third_party", "aicup2026_tutorial"))
    import utils as tut
    import hs_lib as H
    A, Hh = games_of(clean_dir), games_of(h_dir)
    res = {}
    fns = {
        "p02_moves.parse_moves": lambda c, s: p02_moves.parse_moves(s),
        "p02_player_open.game_tokens(raw)": lambda c, s: po.game_tokens(c if c in "BW" else "B", po.moves_of(s), "raw"),
        "p02_rank_opening.game_string": lambda c, s: pr.game_string(s),
        "p04_opp.encode_one": lambda c, s: p04_opp.encode_one(s).tobytes(),
        "p06_style.parse_all": lambda c, s: p06_style.parse_all(s),
        "t02_checks.game_keys": lambda c, s: tc.game_keys([p for _c, p in tc.MOVE_RE.findall(s)]),
        "t02_checks.sgf_check.moves+reasons": lambda c, s: (lambda g: (g["moves"], g["reasons"], g["setup"], g["sz"],
                                                                        g["pass"]))(tc.sgf_check(s)),
    }
    for name in NAMES:
        a, h = A[name], Hh[name]
        assert len(a) == len(h)
        r = {}
        for fn, f in fns.items():
            diff = sum(f(ca, sa) != f(ch, sh) for (ca, sa), (ch, sh) in zip(a, h))
            r[fn] = {"games": len(a), "differ": diff}
        # sgfmill (official tutorial parsers used by the R20 / Q04 / 04 notebooks): every 50th game
        rk, pl = tut.SGFParseRankPrediction(), tut.SGFParsePlayerIdentification()
        nd_r = nd_p = ns = 0
        for i in range(0, len(a), 50):
            ns += 1
            fa, ea = rk.features_from_sgf_content(a[i][1])
            fh, eh = rk.features_from_sgf_content(h[i][1])
            nd_r += not (ea is None and eh is None and np.array_equal(fa, fh))
            col = "b" if (a[i][0] or "B").upper() == "B" else "w"
            pa = pl.extract_features(a[i][1], col)
            ph = pl.extract_features(h[i][1], col)
            nd_p += not (len(pa) == len(ph) and all(np.array_equal(x, y) for x, y in zip(pa, ph)))
        r["sgfmill rank features (every 50th)"] = {"games": ns, "differ": nd_r}
        r["sgfmill player features (every 50th)"] = {"games": ns, "differ": nd_p}
        # KataGo chain parser (07 HS, 08 / 19 P1, 10 B'): hs_lib.parse_moves on every game
        fa_ok = fh_ok = 0
        same = 0
        for (_ca, sa), (_ch, sh) in zip(a, h):
            try:
                ma = H.parse_moves(sa)
                fa_ok += 1
            except H.HSError:
                ma = None
            try:
                mh = H.parse_moves(sh)
                fh_ok += 1
            except H.HSError:
                mh = None
            same += ma == mh and ma is not None
        r["hs_lib.parse_moves (KataGo input)"] = {"games": len(a), "clean_ok": fa_ok, "H_ok": fh_ok,
                                                  "identical": same}
        res[name] = r
    return res


def main():
    src = os.path.join(F, "data", "test")
    dl = {k: os.path.join(OUT, "dl_" + k) for k in ("clean", "H", "P", "HP")}
    os.makedirs(dl["clean"], exist_ok=True)
    for n in NAMES:
        shutil.copyfile(os.path.join(src, n), os.path.join(dl["clean"], n))
    res = {"inputs": {"fake_tree": F, "FAKE_TREE_sha256": sha(os.path.join(F, "FAKE_TREE.json")),
                      "clean_sha256": {n: sha(os.path.join(dl["clean"], n)) for n in NAMES}}}
    res["injected_H"] = add_root_props(dl["clean"], dl["H"], "H")
    res["injected_P"] = inject(dl["clean"], dl["P"], 0.3, 0.0, ("rank", "player"), "pass")
    res["injected_HP"] = inject(dl["H"], dl["HP"], 0.3, 0.0, ("rank", "player"), "pass")
    res["H_sha256"] = {n: sha(os.path.join(dl["H"], n)) for n in NAMES}
    res["sample_H_cells"] = [g[1][:60] for g in games_of(dl["H"])[NAMES[0]][:6]]

    # direct parser comparison, raw H (no normalisation) vs clean
    res["parsers_raw_H_vs_clean"] = parser_compare(dl["clean"], dl["H"])

    roots = {k: os.path.join(OUT, k) for k in ("A", "H", "HP")}
    # A: clean
    A = roots["A"]
    run("t02_test_bundle.py", "ingest", "--src", dl["clean"], "--root", A)
    a = {"t1": {c: (t1(A, c), t1j(A, c)["verdict"]) for c in ("rank", "player")}}
    a["rp_prepare_exit"] = run("root_props.py", "prepare", "--root", A, ok=(2,))
    a["rootprops_dir_exists"] = os.path.exists(os.path.join(A, "data", "test_rootprops"))
    run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", A)
    a["built_sha256"] = {n: sha(os.path.join(A, "data", "test", n)) for n in NAMES}
    a["t4_t5"] = t45(A)
    res["A"] = a

    # H: every game with root properties
    Hr = roots["H"]
    run("t02_test_bundle.py", "ingest", "--src", dl["H"], "--root", Hr)
    h = {"t1_raw": {c: (t1(Hr, c), t1j(Hr, c)["verdict"]) for c in ("rank", "player")}}
    h["t1_raw_equals_A_except_other_props"] = {c: t1cmp(t1j(Hr, c)) == t1cmp(t1j(A, c)) for c in ("rank", "player")}
    h["t1_raw_other_props"] = {c: {n: v.get("other_properties_record_only") for n, v in
                                   t1j(Hr, c)["subchecks"]["T1.4"]["files"].items()} for c in ("rank", "player")}
    h["rp_prepare_exit"] = run("root_props.py", "prepare", "--root", Hr)
    pj = tc.read_json(os.path.join(Hr, "results", "test02", "checks", "root_props", "prepare.json"))
    h["decisions"] = {c: pj["comps"][c]["decision"] for c in pj["comps"]}
    h["files"] = {n: {k: pj["files"][n][k] for k in ("games", "games_edited", "counts", "removed_props", "hd_values",
                                                     "plain_form_before", "plain_form_after")} for n in NAMES}
    norm = os.path.join(Hr, "data", "test_rootprops", "normalised")
    h["normalised_equals_clean_bytes"] = {n: sha(os.path.join(norm, n)) == res["inputs"]["clean_sha256"][n] for n in NAMES}
    h["verify_before_ingest_exit"] = run("root_props.py", "verify", "--root", Hr, ok=(1,))
    run("t02_test_bundle.py", "ingest", "--src", norm, "--root", Hr, "--rerun", "--reason", R11)
    h["t1_copy"] = {c: (t1(Hr, c, R11), t1j(Hr, c)["verdict"]) for c in ("rank", "player")}
    h["verify_exit"] = run("root_props.py", "verify", "--root", Hr, "--rerun", "--reason", "after ingest and t1")
    h["prepare_again_exit_refused"] = run("root_props.py", "prepare", "--root", Hr, "--rerun", "--reason", "x", ok=(1,))
    run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", Hr)
    h["built_equals_A"] = {n: sha(os.path.join(Hr, "data", "test", n)) == a["built_sha256"][n] for n in NAMES}
    h["t4_t5_equal_A"] = {k: v == a["t4_t5"][k] for k, v in t45(Hr).items()}
    res["H"] = h

    # HP: root properties and 30 % trailing passes: rev 9 first, then rev 11
    P = roots["HP"]
    run("t02_test_bundle.py", "ingest", "--src", dl["HP"], "--root", P)
    hp = {"t1_raw": {c: (t1(P, c), t1j(P, c)["verdict"]) for c in ("rank", "player")}}
    hp["rp_before_rev9_exit_refused"] = run("root_props.py", "prepare", "--root", P, ok=(1,))
    hp["tp_prepare_exit"] = run("trailing_pass.py", "prepare", "--root", P)
    strip = os.path.join(P, "data", "test_pass", "stripped")
    hp["rev9_copy_equals_H_bytes"] = {n: sha(os.path.join(strip, n)) == res["H_sha256"][n] for n in NAMES}
    run("t02_test_bundle.py", "ingest", "--src", strip, "--root", P, "--rerun", "--reason", R9)
    hp["t1_rev9"] = {c: (t1(P, c, R9), t1j(P, c)["verdict"]) for c in ("rank", "player")}
    hp["rp_before_rev9_verify_exit_refused"] = run("root_props.py", "prepare", "--root", P, ok=(1,))
    hp["tp_verify_exit"] = run("trailing_pass.py", "verify", "--root", P)
    hp["rp_prepare_exit"] = run("root_props.py", "prepare", "--root", P)
    norm = os.path.join(P, "data", "test_rootprops", "normalised")
    hp["normalised_equals_clean_bytes"] = {n: sha(os.path.join(norm, n)) == res["inputs"]["clean_sha256"][n] for n in NAMES}
    run("t02_test_bundle.py", "ingest", "--src", norm, "--root", P, "--rerun", "--reason", R11)
    hp["t1_copy"] = {c: (t1(P, c, R11), t1j(P, c)["verdict"]) for c in ("rank", "player")}
    hp["verify_exit"] = run("root_props.py", "verify", "--root", P)
    run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", P)
    hp["built_equals_A"] = {n: sha(os.path.join(P, "data", "test", n)) == a["built_sha256"][n] for n in NAMES}
    # order: rev 11 first, then rev 9 (in process) gives the same bytes; rev 11 twice changes nothing
    tmp1, tmp2 = os.path.join(OUT, "order_rp_first"), os.path.join(OUT, "order_rp_twice")
    os.makedirs(tmp1, exist_ok=True)
    os.makedirs(tmp2, exist_ok=True)
    same_order, idem = {}, {}
    for comp in ("rank", "player"):
        for _pool, kind, n in tc.FILES[comp]:
            _r, nb = rp.analyse_file(os.path.join(dl["HP"], n), kind)
            open(os.path.join(tmp1, n), "wb").write(nb)
            _r2, nb2 = tp.analyse_file(os.path.join(tmp1, n), kind)
            same_order[n] = tp.sha256_bytes(nb2) == res["inputs"]["clean_sha256"][n]
            _r3, nb3 = rp.analyse_file(os.path.join(tmp1, n), kind)
            idem[n] = nb3 is None
    hp["rev11_then_rev9_equals_clean"] = same_order
    hp["rev11_idempotent"] = idem
    res["HP"] = hp
    with open(os.path.join(OUT, "rehearsal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("A", "H", "HP")}, indent=1, default=str)[:12000])
    print(json.dumps(res["parsers_raw_H_vs_clean"], indent=1))


if __name__ == "__main__":
    main()
