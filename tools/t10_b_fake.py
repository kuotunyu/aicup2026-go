"""Fake PREREG_10 B trees for the 10 test-day dry runs (part B; PREREG_10 rev 4, section 13). Never touches the
repository's results/ or data/: every output goes under --root (outside the repository).

    PY tools/t10_b_fake.py fake-p10 --root S [--name p10]
        S/<name>/ shaped like results/prereg10 (+ val/ = data/prereg10/val): the real frozen_config.json and
        p1_lda.npz copied; SYNTHETIC validation P1 chunks of all 44,9xx validation games (tools/t10_b.synthetic_means:
        no KataGo, a deterministic summary of the SGF moves; set p10val) and val_seam_feats.npz / val_report.json;
        player_val_top5.csv and apply_report.json from tools/p10_apply.apply_core (06 recomputed with the 06 functions,
        byte-identical to results/prereg06 or it stops), score_allowed set true ('fake finalize'); a FAKE
        score_H10P.json (B - A +0.001 [-0.004, +0.006], A side = score_H6P's B side) and RESULTS.md with the adopted
        mark. Nothing is scored: tools/score.py is never run, no label file is opened.
    PY tools/t10_b_fake.py drill --p10 S/p10 --out S/p10_x --kind seam_perturbed|not_adopted|inconsistent
        a copy without val/ (the GPU seam and b_state do not read the chunks): seam_perturbed shifts the side-0 means of
        5 seam games by +1 (val_report.json's seam SHA updated: the GPU seam fails, B not used); not_adopted makes H10P fail (-0.001) with the mark
        不採用 (every B path exit 3, nothing written); inconsistent keeps 採用 with that failing score (B not used, recorded).
    PY tools/t10_b_fake.py fake-p13 --p10 S/p10 --out S/q10 [--kind adopted|not_adopted|inconsistent|audit_unbound]
        PREREG_10 rev 5 (section 16): B adopted through PREREG_13. S/q10 is a copy of the fake PREREG_10 tree (files
        hard-linked where possible; the files changed here are written fresh) where H10P never ran: no
        score_H10P.json, no H10P mark, apply_report.json score_allowed false (as the real one). Next to it S/q10_p13
        (the dry-run PREREG_13 folder of --prereg10-dir S/q10, tools/t10_b.p13_dir) shaped like results/prereg13: the
        two view top-5 files (stand-ins), view_report.json (gates passed, bound to S/q10's freeze / LDA / val_report /
        apply_report, score_allowed true after a fake finalize that records view_report_sha256_at_audit),
        prescore_audit.json bound to the view report as it was before the finalize, a FAKE score_H13P.json (B - A
        +0.001 [-0.004, +0.006], n_view 263) and RESULTS.md with the H13P adopted mark. Drill kinds: not_adopted
        (score below the n_view-scaled bound, mark 不採用: exit 3, nothing written), inconsistent (mark 採用 with that
        score), audit_unbound (the audit bound another view_report.json).
data/local is read only for the two validation game files, inside the p06 guard (as the 06 / 10 seam).
"""

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import t10_b as tb  # noqa: E402
import p10_apply as PA  # noqa: E402
import p10_fuse as F  # noqa: E402
import p02_player_fuse as pf  # noqa: E402

t6, st = tb.t6, tb.st
SYN_ENV = {"scorer": "synthetic", "torch": "none", "cudnn": "none", "gpu": "none"}


def out_ok(path):
    if st.inside_repo(path):
        raise SystemExit(f"{path} is inside the repository: fake trees live outside it")
    if ("ans" + "wer") in os.path.abspath(path).lower():
        raise SystemExit("refusing a path with the label-file marker")
    return path


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t6.jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")


def cmd_fake_p10(a):
    t0 = time.time()
    P = out_ok(os.path.abspath(os.path.join(a.root, a.name)))
    if os.path.exists(P):
        raise SystemExit(f"{P} exists")
    paths = tb.p10_paths(P)
    real = tb.p10_paths(tb.REAL_P10)
    os.makedirs(os.path.dirname(paths["frozen"]), exist_ok=True)
    for k in ("frozen", "lda"):
        shutil.copyfile(real[k], paths[k])
    fc = t6.read_json(paths["frozen"])
    with st.allow_local():
        cand_rows = pf.read_candidates(PA.VAL_CANDIDATES)
        q_rows = pf.read_questions(PA.VAL_QUESTIONS)
        in_sha = {"candidates_csv": t6.sha_raw(PA.VAL_CANDIDATES), "questions_csv": t6.sha_raw(PA.VAL_QUESTIONS)}
    keys, cols = PA.val_keys(cand_rows, q_rows)
    sgfs = [r[3] for r in cand_rows] + [x for _q, g in q_rows for _s, _c, x in g]
    games = [{"sgf_content": x, "color": c, "ref": j} for j, (x, c) in enumerate(zip(sgfs, cols))]
    ksha = PA.keys_sha(keys)
    os.makedirs(paths["val_dir"], exist_ok=True)
    files = []
    for ci, s in enumerate(range(0, len(games), tb.CHUNK)):
        e = min(len(games), s + tb.CHUNK)
        res, _t = tb.synthetic_chunk(games[s:e])
        p = os.path.join(paths["val_dir"], "chunk_%05d.npz" % ci)
        tb.save_chunk(p, tb.VAL_SET, np.arange(s, e), ksha, "synthetic", SYN_ENV, tb.SYN, res, 0.0, 0.0)
        files.append(p)
    X, usable, failed, have, cnt = F.load_side0(files, len(games), tb.VAL_SET)
    n = min(tb.SEAM_GAMES, len(games))
    os.makedirs(os.path.dirname(paths["val_seam"]), exist_ok=True)
    np.savez(paths["val_seam"], side0=X[:n], counts0=cnt[:n], failed=failed[:n], ref=np.arange(n),
             keys_sha256=np.array(ksha))
    vr = {"format": F.FORMATS["val_report"], "mode": "synthetic", "fake": True, "complete": bool(have.all()),
          "games": len(games), "chunks": [F.chunk_record(p) for p in files], "failed": int(failed.sum()),
          "no_target_moves": int(((cnt == 0) & ~failed).sum()), "usable": int(usable.sum()), "keys_sha256": ksha,
          "inputs_sha256": in_sha, "frozen_config_sha256": t6.sha_lf(paths["frozen"]), "set": tb.VAL_SET,
          "seam": {"file": "feat/val_seam_feats.npz", "sha256": t6.sha_raw(paths["val_seam"]), "games": n},
          "what": "FAKE (tools/t10_b_fake.py): synthetic P1 of the validation games, for the 10 dry runs only"}
    write_json(paths["val_report"], vr)
    model = F.load_lda(paths["lda"])
    rep = PA.apply_core(PA.VAL_CANDIDATES, PA.VAL_QUESTIONS, PA.P02_PLAYER, PA.P06_PLAYER, files, tb.VAL_SET, vr,
                        model, float(fc["w_star"]), os.path.dirname(paths["frozen"]))
    rep.update(w_star=float(fc["w_star"]), score_allowed=True, score_blocked_reasons=[], fake=True,
               finalized="fake (tools/t10_b_fake.py): no audit, no score")
    write_json(paths["apply"], rep)
    h6p = t6.read_json(os.path.join(tb.P06_PLAYER, "score_H6P.json"))
    write_json(paths["h10p"], {"fake": True, "A_mean": h6p["B_mean"], "B_mean": h6p["B_mean"] + 0.001,
                               "B_minus_A": {"n": 400, "strata": 1, "mean_diff": 0.001, "ci95": [-0.004, 0.006]},
                               "what": "FAKE score record for the 10 dry runs (tools/score.py was not run)"})
    with open(paths["results_md"], "w", encoding="utf-8", newline="\n") as f:
        f.write("# FAKE PREREG_10 results (tools/t10_b_fake.py, dry runs only)\n\n" + tb.B_ADOPTED_MARK + "\n")
    print(json.dumps({"p10": P, "games": len(games), "chunks": len(files), "pool_gate": rep["pool_gate"],
                      "changed_top1_vs_06": rep["changed_top1_vs_06"], "seconds": round(time.time() - t0, 1)}))
    return 0


def cmd_drill(a):
    src, dst = os.path.abspath(a.p10), out_ok(os.path.abspath(a.out))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("val"))
    P = tb.p10_paths(dst)
    if a.kind == "seam_perturbed":
        with np.load(P["val_seam"], allow_pickle=False) as z:
            arr = {k: np.array(z[k]) for k in z.files}
        ok = np.flatnonzero(~arr["failed"] & (arr["counts0"] > 0))[:5]
        arr["side0"][ok] += 1.0
        np.savez(P["val_seam"], **arr)
        vr = t6.read_json(P["val_report"])
        vr["seam"]["sha256"] = t6.sha_raw(P["val_seam"])
        write_json(P["val_report"], vr)
        print(json.dumps({"drill": a.kind, "games_perturbed": ok.tolist()}))
    elif a.kind in ("not_adopted", "inconsistent"):
        h = t6.read_json(P["h10p"])
        h["B_minus_A"] = dict(h["B_minus_A"], mean_diff=-0.001)
        write_json(P["h10p"], h)
        if a.kind == "not_adopted":
            with open(P["results_md"], "w", encoding="utf-8", newline="\n") as f:
                f.write("# FAKE PREREG_10 results: H10P not adopted (drill)\n\nH10P：B **不採用**\n")
        print(json.dumps({"drill": a.kind}))
    return 0


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def fresh_write(path, write):
    """Write path as a new file (never through a hard link shared with the source tree)."""
    if os.path.exists(path):
        os.remove(path)
    write(path)


def write_text(path, text):
    def w(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    fresh_write(path, w)


def cmd_fake_p13(a):
    """PREREG_10 rev 5: a fake tree where B is adopted through PREREG_13 (H10P never ran)."""
    src, dst = os.path.abspath(a.p10), out_ok(os.path.abspath(a.out))
    if st.inside_repo(src):
        raise SystemExit(f"--p10 {src} is inside the repository: fake-p13 copies a fake tree (tools/t10_b_fake.py "
                         f"fake-p10), never results/prereg10")
    if os.path.normcase(src) == os.path.normcase(dst):
        raise SystemExit("--out must differ from --p10 (the source fake tree is never changed)")
    d13 = out_ok(dst + "_p13")
    for d in (dst, d13):
        if os.path.exists(d):
            shutil.rmtree(d)
    shutil.copytree(src, dst, copy_function=link_or_copy)
    P = tb.p10_paths(dst)
    if os.path.exists(P["h10p"]):
        os.remove(P["h10p"])
    write_text(P["results_md"], "# FAKE PREREG_10 results: H10P not run (B through PREREG_13; tools/t10_b_fake.py)\n")
    ap = t6.read_json(P["apply"])
    ap.update(score_allowed=False, score_blocked_reasons=["fake: the mixed-colour pool gate failed (as the real one)"])
    fresh_write(P["apply"], lambda p: write_json(p, ap))
    Q = tb.p13_paths(d13)
    os.makedirs(os.path.dirname(Q["view"]), exist_ok=True)
    shutil.copyfile(os.path.join(tb.P06_PLAYER, "player_val_top5.csv"), Q["view_a"])
    shutil.copyfile(P["val_top5"], Q["view_b"])
    fc = t6.read_json(P["frozen"])
    n_view = 263
    ins = {"p1_lda.npz": t6.sha_raw(P["lda"]), "frozen_config.json": t6.sha_lf(P["frozen"]),
           "val_report.json": t6.sha_lf(P["val_report"]), "p10_apply_report.json": t6.sha_lf(P["apply"])}
    outs = {tb.P13_OUT_A: t6.sha_lf(Q["view_a"]), tb.P13_OUT_B: t6.sha_lf(Q["view_b"])}
    view = {"format": tb.P13_FORMATS["view"], "fake": True, "preview": False,
            "what": "FAKE view report (tools/t10_b_fake.py fake-p13): nothing was computed on a view",
            "view": {"n_view": n_view}, "selected": "V2", "w_star": float(fc["w_star"]), "k_star": 10,
            "gates": {"V1": {"passed": True}, "V2": {"passed": True}, "V3": {"passed": True}},
            "margin": {"n_view": n_view, "ci95_lower_must_exceed": tb.H13P_LOW_PER_400 * n_view / 400},
            "inputs_sha256": ins, "outputs_sha256": outs, "score_allowed": "pending_audit",
            "score_blocked_reasons": ["pending_audit"]}
    write_json(Q["view"], view)
    pre = t6.sha_lf(Q["view"])
    audit = {"format": tb.P13_FORMATS["audit"], "fake": True, "program_errors_affecting_view": [],
             "independent_recompute_identical": True,
             "outputs_sha256": dict(outs, **{"view_report.json": pre if a.kind != "audit_unbound" else "0" * 64}),
             "inputs_sha256": {k: ins[k] for k in tb.P13_AUDIT_INPUTS}}
    write_json(Q["audit"], audit)
    view.update(score_allowed=True, score_blocked_reasons=[], finalized="fake (tools/t10_b_fake.py)",
                audit={"sha256": t6.sha_lf(Q["audit"]), "commit": None, "view_report_sha256_at_audit": pre})
    write_json(Q["view"], view)
    if a.kind in ("adopted", "audit_unbound"):
        low, word = -0.004, "採用"
    else:                                   # below -0.010 * 263 / 400 = -0.006575: not adopted under PREREG_13 6.3
        low, word = -0.0066, ("不採用" if a.kind == "not_adopted" else "採用")
    mean = 0.001
    write_json(Q["h13p"], {"fake": True, "A_mean": 0.95, "B_mean": 0.95 + mean,
                           "B_minus_A": {"n": 400, "strata": 1, "mean_diff": mean, "ci95": [low, 0.006]},
                           "what": "FAKE score record for the 10 dry runs (tools/score.py was not run)"})
    write_text(Q["results_md13"], f"# FAKE PREREG_13 results (tools/t10_b_fake.py, dry runs only)\n\nH13P：B′ **{word}**\n")
    print(json.dumps({"q10": dst, "p13": d13, "kind": a.kind}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fake-p10")
    p.add_argument("--root", required=True)
    p.add_argument("--name", default="p10")
    p = sub.add_parser("drill")
    p.add_argument("--p10", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", required=True, choices=("seam_perturbed", "not_adopted", "inconsistent"))
    p = sub.add_parser("fake-p13")
    p.add_argument("--p10", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="adopted", choices=("adopted", "not_adopted", "inconsistent", "audit_unbound"))
    a = ap.parse_args(argv)
    return {"fake-p10": cmd_fake_p10, "drill": cmd_drill, "fake-p13": cmd_fake_p13}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
