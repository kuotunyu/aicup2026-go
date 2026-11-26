"""A fake test04 notebook output for dry runs, from the committed prereg03 validation dumps (spec
results/test04/design/spec.md 3.8; the record formats of section 2.3).

    PY tools/t04_fake_tests.py from-val --root S [--zip-out Z] [--rerun]

S must already hold the tree of `PY tools/t02_fake_tests.py from-val --root S` (S/FAKE_TREE.json, S/data/test,
S/results/test02/rank/config.json) and must be outside the repository. Written:
    S/drive/sw/                      the notebook's result folder (what Drive would hold)
      config.json run_log.json env.json timings.json seam_check.json readonly_before.json readonly_after.json
      seam_val/dumps/  byte copies of results/prereg03/rank_a/dumps/{val_keys, val_p03a_d4, val_p03a_s0_swap}.npz,
                       rank_b/dumps/{val_p03b_d4, val_p03b_s0_swap}.npz and their _DONE_ markers; _INPUT.json
      public/dumps/, private/dumps/   row slices of the same dumps for validation questions 1-400 / 401-800 (the
                       rank rows of t02_fake_tests: RANK_SLICES), question_id re-prefixed val_ -> pub_ / priv_
                       (t02_fake_tests._rekey_array), logits and weights byte-identical; regenerated markers
                       (t02_fake_tests._marker, environment fields of the prereg03 markers); _INPUT.json
    S/drive/test04_rank_results.zip  (or --zip-out) members from sw/, fixed member times
    S/FAKE_TREE_T04.json             what was built from which inputs (SHA-256) and the rule below.
The records say smoke false, rehearsal true, kind 'rehearsal'; notebook_sha256 is the committed
colab/test04_rank.ipynb's (64 zeros while it does not exist; dry runs only record it); run_log one completed run;
seam_check passed with tol 1e-4 and zero differences (the copies are the references; t04_rank.py apply recomputes).
Rule: the games are validation games; only the replay (t04_rank.py opp) and apply may use this tree, never a content
statistic. Nothing is written inside the repository; no label file is opened.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_fake_tests as tf  # noqa: E402
import t02_rank as t2  # noqa: E402
import t02_test_bundle as tb  # noqa: E402

P03 = os.path.join(REPO, "results", "prereg03")
LOCAL = os.path.join(REPO, "data", "local")
RANK_VAL = "rank_val_questions.csv"
NOTEBOOK04 = os.path.join(REPO, "colab", "test04_rank.ipynb")
BASE_NOTEBOOKS = ("colab/test02_rank.ipynb", "colab/prereg03_rank_a.ipynb")
SETS = ("seam_val", "public", "private")
ARMS = ("a", "b")
SW_WEIGHTS = {"a": "790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac",
              "b": "8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42"}
W_LABEL = {x: "prereg03/rank_%s/ckpt/final.pth" % x for x in ARMS}
ZIP_NAME = "test04_rank_results.zip"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)
NB_CONFIG_FORMAT = "c10-test04 rank notebook config v1"
FAKE_FORMAT = "c10-test04 fake tree v1"
FAKE_NOTE = "t04_fake_tests from-val: no notebook ran; dumps copied or row-sliced from results/prereg03"
RULE = "validation games: only replay and apply; never run content statistics"


class FakeError04(SystemExit):
    def __init__(self, message):
        super().__init__(f"t04_fake_tests: {message}")


def stems(x):
    return (f"val_p03{x}_d4", f"val_p03{x}_s0_swap")


def arm_dir(x):
    return os.path.join(P03, f"rank_{x}", "dumps")


def write_json(path, obj):
    return tb.write_json(path, obj)


def slice_bounds(keys, local_dir):
    """Row bounds of each validation question in val_keys order, checked against the question file's order."""
    qids = [str(x) for x in keys["question_id"].tolist()]
    order, first = [], {}
    for i, q in enumerate(qids):
        if q not in first:
            first[q] = i
            order.append(q)
        elif order[-1] != q:
            raise FakeError04(f"val_keys: question {q} is not contiguous")
    if order != tf._question_ids(os.path.join(local_dir, RANK_VAL)):
        raise FakeError04("prereg03 val_keys question order differs from rank_val_questions.csv")
    return order, [first[q] for q in order] + [len(qids)]


def notebook_sha():
    if not os.path.exists(NOTEBOOK04):
        return "0" * 64, "colab/test04_rank.ipynb absent: 64 zeros (dry runs record the identity only)"
    _rec, recomputed = t2.notebook_source_sha256(NOTEBOOK04)
    return recomputed or "0" * 64, "recomputed from colab/test04_rank.ipynb (t02_rank.notebook_source_sha256)"


def write_zip(sw, zip_out):
    tmp = zip_out + ".tmp"
    members = []
    for root, dirs, files in os.walk(sw):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            members.append((os.path.relpath(p, os.path.dirname(sw)).replace(os.sep, "/"), p))
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in members:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(p, "rb") as f:
                z.writestr(info, f.read())
    os.replace(tmp, zip_out)
    return len(members)


def from_val(root, zip_out=None, rerun=False, local_dir=LOCAL, p03_dir=None, log=print):
    root = os.path.abspath(root or "")
    if not root or tb.inside(root, REPO):
        raise FakeError04(f"--root {root!r} must be a scratch folder outside the repository")
    tree = os.path.join(root, "FAKE_TREE.json")
    tests = os.path.join(root, "data", "test")
    cfg2_path = os.path.join(root, "results", "test02", "rank", "config.json")
    for p in (tree, tests, cfg2_path):
        if not os.path.exists(p):
            raise FakeError04(f"{p} is missing: run PY tools/t02_fake_tests.py from-val --root {root} first")
    ft = tb.read_json(tree)
    if ft.get("format") != "c10-test02 fake tree v1" or ft.get("kind") != "from-val":
        raise FakeError04(f"{tree}: not a t02_fake_tests from-val tree")
    drive = os.path.join(root, "drive")
    sw = os.path.join(drive, "sw")
    zip_out = os.path.abspath(zip_out or os.path.join(drive, ZIP_NAME))
    if tb.inside(zip_out, REPO):
        raise FakeError04(f"--zip-out {zip_out!r} is inside the repository")
    rec_path = os.path.join(root, "FAKE_TREE_T04.json")
    existing = [p for p in (sw, zip_out, rec_path) if os.path.exists(p)]
    if existing and not rerun:
        raise FakeError04(f"{existing} exist; pass --rerun to rebuild them")
    for p in existing:
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    p03 = p03_dir or P03
    src = {x: os.path.join(p03, f"rank_{x}", "dumps") for x in ARMS}
    keys = tf._npz(os.path.join(src["a"], "val_keys.npz"))
    if tb.sha256_file(os.path.join(src["a"], "val_keys.npz")) != tb.sha256_file(os.path.join(src["b"], "val_keys.npz")):
        raise FakeError04("the two prereg03 arms have different val_keys.npz")
    order, bounds = slice_bounds(keys, local_dir)
    dumps = {x: {st: tf._npz(os.path.join(src[x], f"{st}.npz")) for st in stems(x)} for x in ARMS}
    markers = {x: {st: tb.read_json(os.path.join(src[x], f"_DONE_{st}.json")) for st in stems(x)} for x in ARMS}
    key_marker = tb.read_json(os.path.join(src["a"], "_DONE_val_keys.json"))
    m0 = markers["a"][stems("a")[0]]
    gpu, torch_v, cudnn, device = m0.get("gpu_name"), m0.get("torch"), m0.get("cudnn"), m0.get("device")
    sets, dump_log, inputs = {}, {}, {}
    # ---- seam_val: byte copies
    d = os.path.join(sw, "seam_val", "dumps")
    os.makedirs(d)
    copies = [(src["a"], "val_keys")] + [(src[x], st) for x in ARMS for st in stems(x)]
    for sdir, st in copies:
        for f in (f"{st}.npz", f"_DONE_{st}.json"):
            shutil.copyfile(os.path.join(sdir, f), tb.check_writable(os.path.join(d, f)))
            inputs[os.path.relpath(os.path.join(sdir, f), REPO).replace(os.sep, "/")] = \
                tb.sha256_file(os.path.join(sdir, f))
    vq = os.path.join(local_dir, RANK_VAL)
    vq_sha = tb.sha256_file(vq)
    write_json(os.path.join(d, "_INPUT.json"), {"set": "seam_val", "question_file": f"bundle local/{RANK_VAL}",
                                               "sha256": vq_sha})
    sets["seam_val"] = {"questions": f"bundle local/{RANK_VAL}", "questions_sha256": vq_sha,
                        "questions_count": len(order), "rows": len(keys["question_id"])}
    inputs[f"data/local/{RANK_VAL}"] = vq_sha
    # ---- public, private: row slices of the same dumps
    for pool in ("public", "private"):
        lo, hi = tf.RANK_SLICES[pool]
        r0, r1 = bounds[lo], bounds[hi]
        d = os.path.join(sw, pool, "dumps")
        os.makedirs(d)
        k2 = {n: (tf._rekey_array(a[r0:r1], tf.PREFIX[pool]) if n == "question_id" else a[r0:r1])
              for n, a in keys.items()}
        paths = {"val_keys": tf._save_npz(os.path.join(d, "val_keys.npz"), k2)}
        write_json(os.path.join(d, "_DONE_val_keys.json"), tf._marker(key_marker, paths["val_keys"], r1 - r0))
        for x in ARMS:
            for st in stems(x):
                z = dumps[x][st]
                z2 = {n: (a[r0:r1] if n == "logits" else a) for n, a in z.items()}
                if z2["logits"].tobytes() != z["logits"][r0:r1].tobytes():
                    raise FakeError04("logit slice changed")
                p = tf._save_npz(os.path.join(d, f"{st}.npz"), z2)
                write_json(os.path.join(d, f"_DONE_{st}.json"), tf._marker(markers[x][st], p, r1 - r0))
        qp = os.path.join(tests, tb.RANK_FILES[pool])
        if tf._question_ids(qp) != [tf.PREFIX[pool] + q[len("val_"):] for q in order[lo:hi]]:
            raise FakeError04(f"{qp}: question order differs from the dump slice")
        qsha = tb.sha256_file(qp)
        write_json(os.path.join(d, "_INPUT.json"), {"set": pool, "question_file": f"tests/{tb.RANK_FILES[pool]}",
                                                   "sha256": qsha})
        sets[pool] = {"questions": f"tests/{tb.RANK_FILES[pool]}", "questions_sha256": qsha,
                      "questions_count": hi - lo, "rows": r1 - r0}
    for s in SETS:
        dump_log[s] = {"questions_path": sets[s]["questions"], "rows": sets[s]["rows"],
                       **{x: {st: sets[s]["rows"] for st in stems(x)} for x in ARMS}, "fake": FAKE_NOTE}
    # ---- records (spec 2.3)
    cfg2 = tb.read_json(cfg2_path)
    log2_path = os.path.join(root, "results", "test02", "rank", "run_log.json")
    log2 = tb.read_json(log2_path) if os.path.exists(log2_path) else {}
    nb_sha, nb_note = notebook_sha()
    ref = {x: {f"{st}.npz": tb.sha256_file(os.path.join(src[x], f"{st}.npz"))
               for st in ("val_keys",) + stems(x)} for x in ARMS}
    now = tb.now_iso()
    tf32 = {"cudnn": True, "matmul": False}
    config = {"format": NB_CONFIG_FORMAT, "fake": FAKE_NOTE, "smoke": False, "rehearsal": True, "kind": "rehearsal",
              "gpu": gpu, "torch": torch_v, "cudnn": cudnn, "device": device,
              "notebook": "colab/test04_rank.ipynb", "notebook_sha256": nb_sha, "notebook_sha256_note": nb_note,
              "base_notebooks": {n: (tb.sha256_text_file(os.path.join(REPO, *n.split("/")))
                                     if os.path.exists(os.path.join(REPO, *n.split("/"))) else None)
                                 for n in BASE_NOTEBOOKS},
              "weights": {x: {"label": W_LABEL[x], "sha256": SW_WEIGHTS[x]} for x in ARMS},
              "seam_reference": {"a": ref["a"], "b": ref["b"], "gpu": gpu},
              "test_bundle": cfg2.get("test_bundle"),
              "test02_record": {"config_sha256": tb.sha256_file(cfg2_path),
                                "manifest_sha256": (cfg2.get("test_bundle") or {}).get("manifest_sha256"),
                                "run_log_status": log2.get("status")},
              "sets": sets,
              "tf32": {"recorded_03": tf32, "default": tf32, "used": tf32, "benchmark_default": False,
                       "benchmark_used": False},
              "dumps": dump_log, "prereg": "docs/PREREG_04_OPPONENT.md rev 3 section 11", "status": "completed"}
    run = {"run": 1, "status": "completed", "start": now, "end": now, "smoke": False, "rehearsal": True, "gpu": gpu,
           "fake": FAKE_NOTE}
    arm_seam = {"c0_s0_max_abs_diff": 0.0, "c0_swap_max_abs_diff": 0.0, "passed": True, "bitwise_equal_d4": True,
                "bitwise_equal_swap": True, "file_sha_equal_d4": True, "file_sha_equal_swap": True,
                "max_abs_logit_diff_per_sym": [0.0] * 8, "max_abs_logit_diff_swap": 0.0, "max_abs_e_diff_s0": 0.0,
                "max_abs_e_diff_swap": 0.0, "rule_b_differences_s0_record_only": 0}
    env = {k: {"gpu_name": gpu, "torch": torch_v, "cudnn": cudnn}
           for k in [f"{s}/{st}" for s in SETS for st in ("val_keys",) + stems("a") + stems("b")]}
    seam = {"fake": FAKE_NOTE, "passed": True, "tol": 1e-4, "seam_dump_gpu": gpu, "reference_gpu": gpu,
            "same_gpu": True, "keys_equal": {k: True for k in ("question_id", "slot", "color", "parsed")},
            "weights_ok": True, "env_same": True, "env": env, "arms": {x: dict(arm_seam) for x in ARMS},
            "bitwise_equal": True, "created": now, "earlier_runs": []}
    ro = {"fake": FAKE_NOTE, "roots": {r: {"n_files": 0, "files": {}} for r in ("prereg01", "prereg02", "prereg03",
                                                                               "test02/rank")},
          "same_as_before": True}
    records = {"config.json": config, "run_log.json": {"status": "completed", "runs": [run], "resume_count": 0},
               "env.json": {"runs": [{"run": 1, "torch": torch_v, "cudnn": cudnn,
                                      "gpu": {"nvidia_smi": [gpu] if gpu else []}, "fake": FAKE_NOTE}]},
               "timings.json": {"runs": [{"run": 1, "seconds": {}}], "fake": FAKE_NOTE},
               "seam_check.json": seam, "readonly_before.json": ro, "readonly_after.json": ro}
    for name, obj in records.items():
        write_json(os.path.join(sw, name), obj)
    os.makedirs(os.path.dirname(zip_out), exist_ok=True)
    n_members = write_zip(sw, zip_out)
    zsha = tb.sha256_file(zip_out)
    rec = {"format": FAKE_FORMAT, "kind": "from-val", "created": now, "rule": RULE,
           "t02_fake_tree": {"path": tree, "sha256": tb.sha256_file(tree)},
           "zip": zip_out, "zip_sha256": zsha, "members": n_members, "sw": sw,
           "id_mapping": {"public": "prereg03 val rows of questions 1-400, val_q_ -> pub_q_",
                          "private": "prereg03 val rows of questions 401-800, val_q_ -> priv_q_",
                          "seam_val": "byte copies"},
           "sets": sets, "inputs_sha256": inputs, "notebook_sha256": nb_sha, "tool": tb.tool_record()}
    rec["tool"]["tool"] = "tools/t04_fake_tests.py"
    rec["tool"]["tool_sha256"] = tb.sha256_text_file(os.path.abspath(__file__))
    write_json(rec_path, rec)
    log(f"t04 from-val: {n_members} members -> {zip_out} (sha256 {zsha[:12]}); records in {sw}; {rec_path}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="t04_fake_tests.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-val", help="fake test04 notebook output from the prereg03 validation dumps")
    p.add_argument("--root", required=True)
    p.add_argument("--zip-out")
    p.add_argument("--rerun", action="store_true")
    a = ap.parse_args(argv)
    from_val(a.root, zip_out=a.zip_out, rerun=a.rerun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
