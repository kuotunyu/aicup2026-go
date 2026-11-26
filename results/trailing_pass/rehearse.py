"""PREREG_02 rev 9 scratch rehearsal (not part of the repo). Usage: PY rehearse.py <worktree> <fake tree F> <out dir>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

WT, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PY = sys.executable
sys.path.insert(0, os.path.join(WT, "tools"))
sys.path.insert(0, os.path.join(WT, "tests"))
import t02_checks as tc  # noqa: E402
import trailing_pass as tp  # noqa: E402
from test_trailing_pass import inject  # noqa: E402

NAMES = tp.ALL_NAMES
LOG = open(os.path.join(OUT, "rehearsal.log"), "a", encoding="utf-8")
REASON = "PREREG_02 rev 9 rehearsal: trailing passes removed"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(*args, ok=(0,)):
    cmd = [PY, "-B"] + [os.path.join(WT, "tools", args[0])] + list(args[1:])
    r = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    LOG.write("$ %s\n-> %d\n%s%s\n" % (" ".join(cmd), r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
    LOG.flush()
    if r.returncode not in ok:
        raise SystemExit("FAILED %s (%d): %s %s" % (args, r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return r.returncode


def t1(root, comp, rerun=False):
    a = ["t02_checks.py", "t1", "--comp", comp, "--dry-run", "--tests-dir", os.path.join(root, "data", "test", "raw"),
         "--out", os.path.join(root, "results", "test02", "checks")]
    if rerun:
        a += ["--rerun", "--reason", REASON]
    return run(*a, ok=(0, 2, 3))


def t1v(root, comp):
    return tc.read_json(os.path.join(root, "results", "test02", "checks", comp, "t1.json"))["verdict"]


def t45(root):
    out = {}
    for comp in ("rank", "player"):
        run("t02_checks.py", "run", "--comp", comp, "--dry-run", "--root", root, "--only", "T4,T5")
        for c in ("t4", "t5"):
            d = tc.read_json(os.path.join(root, "results", "test02", "checks", comp, c + ".json"))
            d.pop("provenance", None)
            out["%s/%s" % (comp, c)] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return out


def prep(root):
    return tc.read_json(os.path.join(root, "results", "test02", "checks", "trailing_pass", "prepare.json"))


def main():
    src = os.path.join(F, "data", "test")
    dl = {k: os.path.join(OUT, "dl_" + k) for k in ("clean", "pass", "control", "midonly")}
    os.makedirs(dl["clean"], exist_ok=True)
    for n in NAMES:
        shutil.copyfile(os.path.join(src, n), os.path.join(dl["clean"], n))
    stats = {"pass": inject(dl["clean"], dl["pass"], 0.3, 0.0, ("rank", "player"), "pass"),
             "control": inject(dl["clean"], dl["control"], 0.3, 0.03, ("rank", "player"), "control"),
             "midonly": inject(dl["clean"], dl["midonly"], 0.0, 0.03, ("rank",), "midonly")}
    res = {"inputs": {"fake_tree": F, "fake_tree_FAKE_TREE_sha256": sha(os.path.join(F, "FAKE_TREE.json")),
                      "clean_sha256": {n: sha(os.path.join(dl["clean"], n)) for n in NAMES}},
           "injected": stats}
    roots = {k: os.path.join(OUT, k) for k in ("A_clean", "B_pass", "C_control", "D_midonly")}
    # A: clean
    A = roots["A_clean"]
    run("t02_test_bundle.py", "ingest", "--src", dl["clean"], "--root", A)
    for c in ("rank", "player"):
        t1(A, c)
    a = {"t1": {c: t1v(A, c) for c in ("rank", "player")},
         "prepare_exit": run("trailing_pass.py", "prepare", "--root", A, ok=(2,))}
    run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", A)
    a["built_sha256"] = {n: sha(os.path.join(A, "data", "test", n)) for n in NAMES}
    a["t4_t5"] = t45(A)
    res["A_clean"] = a
    # B: 30 % trailing passes in all six files
    B = roots["B_pass"]
    run("t02_test_bundle.py", "ingest", "--src", dl["pass"], "--root", B)
    b = {"t1_raw": {c: (t1(B, c), t1v(B, c)) for c in ("rank", "player")}}
    b["t1_raw_T14"] = {c: {n: {k: v[k] for k in ("games", "violating_games", "share")} for n, v in
                           tc.read_json(os.path.join(B, "results/test02/checks", c, "t1.json"))["subchecks"]["T1.4"]["files"].items()}
                       for c in ("rank", "player")}
    b["prepare_exit"] = run("trailing_pass.py", "prepare", "--root", B)
    p = prep(B)
    b["decisions"] = {c: (p["comps"][c]["decision"], p["comps"][c].get("copy_t1_predicted", {}).get("verdict"))
                      for c in p["comps"]}
    b["files"] = {n: {k: p["files"][n][k] for k in ("games", "games_stripped", "pass_nodes_removed", "sha256_before",
                                                     "sha256_after", "bytes_before", "bytes_after")} for n in NAMES}
    strip = os.path.join(B, "data", "test_pass", "stripped")
    b["stripped_equals_clean_bytes"] = {n: sha(os.path.join(strip, n)) == res["inputs"]["clean_sha256"][n] for n in NAMES}
    b["verify_before_ingest_exit"] = run("trailing_pass.py", "verify", "--root", B, ok=(1,))
    run("t02_test_bundle.py", "ingest", "--src", strip, "--root", B, "--rerun", "--reason", REASON)
    b["t1_copy"] = {c: (t1(B, c, rerun=True), t1v(B, c)) for c in ("rank", "player")}
    b["verify_exit"] = run("trailing_pass.py", "verify", "--root", B, "--rerun", "--reason", "after ingest and t1")
    run("t02_test_bundle.py", "build", "--kind", "rehearsal", "--root", B)
    b["built_sha256"] = {n: sha(os.path.join(B, "data", "test", n)) for n in NAMES}
    b["built_equals_A"] = {n: b["built_sha256"][n] == a["built_sha256"][n] for n in NAMES}
    b["built_t1_verdicts"] = {c: tc.read_json(os.path.join(B, "results/test02/checks/built", c, "t1.json"))["verdict"]
                              for c in ("rank", "player")}
    b["t4_t5"] = t45(B)
    b["t4_t5_equal_A"] = {k: b["t4_t5"][k] == a["t4_t5"][k] for k in a["t4_t5"]}
    res["B_pass"] = b
    # C: rank with mid-game passes (3 %) and trailing (30 %); player trailing only
    C = roots["C_control"]
    run("t02_test_bundle.py", "ingest", "--src", dl["control"], "--root", C)
    cc = {"t1_raw": {c: (t1(C, c), t1v(C, c)) for c in ("rank", "player")}}
    cc["prepare_exit"] = run("trailing_pass.py", "prepare", "--root", C)
    p = prep(C)
    cc["decisions"] = {c: (p["comps"][c]["decision"], p["comps"][c]["reason"],
                           p["comps"][c].get("copy_t1_predicted", {}).get("verdict")) for c in p["comps"]}
    cc["rank_counts"] = {n: p["files"][n]["counts"] for n in NAMES if p["files"][n]["comp"] == "rank"}
    cc["rank_edited"] = {n: p["files"][n]["edited"] for n in NAMES if p["files"][n]["comp"] == "rank"}
    run("t02_test_bundle.py", "ingest", "--src", os.path.join(C, "data", "test_pass", "stripped"), "--root", C,
        "--rerun", "--reason", REASON)
    cc["t1_after"] = {c: (t1(C, c, rerun=True), t1v(C, c)) for c in ("rank", "player")}
    cc["verify_exit"] = run("trailing_pass.py", "verify", "--root", C)
    cc["player_files_equal_clean"] = {n: sha(os.path.join(C, "data", "test", "raw", n)) == res["inputs"]["clean_sha256"][n]
                                      for n in NAMES if "player" in n}
    res["C_control"] = cc
    # D: mid-game passes only, rank
    D = roots["D_midonly"]
    run("t02_test_bundle.py", "ingest", "--src", dl["midonly"], "--root", D)
    dd = {"t1_raw": {c: (t1(D, c), t1v(D, c)) for c in ("rank", "player")}}
    dd["prepare_exit"] = run("trailing_pass.py", "prepare", "--root", D, ok=(2,))
    p = prep(D)
    dd["decisions"] = {c: (p["comps"][c]["decision"], p["comps"][c]["reason"]) for c in p["comps"]}
    dd["test_pass_dir_exists"] = os.path.exists(os.path.join(D, "data", "test_pass"))
    res["D_midonly"] = dd
    with open(os.path.join(OUT, "rehearsal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("injected",)}, indent=1))
    print("A", a["t1"], a["prepare_exit"])
    print("B", b["t1_raw"], b["decisions"], all(b["stripped_equals_clean_bytes"].values()), b["t1_copy"],
          b["verify_exit"], all(b["built_equals_A"].values()), b["built_t1_verdicts"], b["t4_t5_equal_A"])
    print("C", cc["t1_raw"], cc["decisions"], cc["t1_after"], cc["verify_exit"], cc["player_files_equal_clean"])
    print("D", dd)


if __name__ == "__main__":
    main()
