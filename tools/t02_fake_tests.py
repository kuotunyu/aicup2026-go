"""Fake test-day trees for dry runs without any label file (results/test02/design/spec.md 5.1).

Uses csv, numpy and hashlib (+ tools/t02_test_bundle.py for MANIFEST and zip, tools/c10data.py for
from-train); it imports no p02 model tool. --root must be outside the repository.

    PY tools/t02_fake_tests.py from-val   --root S  [--zip-kind rehearsal|smoke] [--zip-out PATH] [--rerun]
    PY tools/t02_fake_tests.py from-train --root S2 [--shape mixed|official] [--slot-colours empty|filled] [--rerun]

from-val: the pipeline dry-run tree, built from the local validation question/candidate files and the
committed prereg02 dumps by re-prefixing ids only (val_ -> pub_ / priv_):
    S/data/test/<6 official names>   rank public = validation questions 1-400, private = 401-800;
                                      player questions: all 400 in both pools (val_pq_ -> pub_pq_/priv_pq_);
                                      candidates: val_p_<hex> -> pub_p_<hex>/priv_p_<hex>, val_g_N -> pub_g_N/priv_g_N.
                                      Every other byte is identical (the local files keep their CRLF record
                                      ends; t02_checks T1 accepts uniform CRLF or LF as is).
    S/data/test/MANIFEST.json         t02_test_bundle manifest (kind = --zip-kind; t1 = the structure-only
                                      quick check, T1.1-T1.3 subset)
    S/data/colab_bundle/C10_test_v1.zip   (or --zip-out) members tests/<6 names> + tests/MANIFEST.json
    S/results/test02/rank/<set>/dumps     seam_val = byte copies of results/prereg02/rank/dumps/{val_keys,
                                      val_p02e9_d4}.npz and their _DONE markers; public/private = row slices
                                      with only question_id re-keyed (logits, weights_sha256 byte-identical),
                                      regenerated markers; _INPUT.json in every set
    S/results/test02/player/<pool>/dumps  seam_val = byte copies of the prereg02 cand_games/query_games.npz;
                                      public/private = copies with game_id/player_id/question_id re-keyed
                                      (emb_sum, n_windows and the rest byte-identical); _INPUT.json
    S/results/test02/{rank,player}/*.json fake notebook records with the shapes the test02 notebooks write
                                      (config.json smoke false / rehearsal true / test SHAs, seam_check and
                                      repro_check passed, seam_p02 bitwise, run_log completed, ...)
    S/FAKE_TREE.json                  what was built, from which inputs (SHA-256), and the rule below.
Content statistics (T1.4-T5) must never run on this tree: its games are validation games. Only the apply
pipeline and T1.1-T1.3 (structure) may use it.

from-train: the checks dry-run tree, from training players only (c10data.load_rows part fit / inner):
    S2/data/test/<6 official names>  player public = POS-A, private = POS-B: 400 fit players each (40 per
                                      level where possible), 100 candidate games each + one question of
                                      5-20 of their other games (positive control); rank public/private =
                                      400 questions each from inner players (with --index-part fit: 0 copies).
                                      Ids are synthesized: pub_p_<sha8>, pub_g_<n>, pub_q_<n> (priv_ alike).
    S2/neg/data/test/player_identification_{candidates,test}_public.csv
                                      NEG pool: every inner player with >= 100 games (226 in split v1),
                                      100 games each; a question for those with >= 105 games (negative
                                      control; T3 threshold ceil(0.1 * n_players)). Not official size by
                                      design (T1.2 FAIL_STRUCT there is expected).
    S2/FAKE_TREE.json                 roles, player counts and the expected check outcomes.
  --shape official (PREREG_02 rev 8; the official structure of the AIdea data page, DATA.md 2026-09-28):
    player pools: 400 fit players with >= 120 black games each, candidates = 100 BLACK games, one question of
                  their next 5 / 10 / 15 / 20 black games (100 questions each), all colours B;
    rank files:   400 questions of inner players, 5 / 10 / 15 / 20 games (100 each), one colour per question
                  (B or W by hash), the games of that colour;
    NEG pool:     inner players with >= 100 black games, black games only;
    --slot-colours empty (default): color_i after num_games is empty; filled: it holds the question colour (the
                  reading of 'color_1..color_20 are always B' that T1.3 normalizes with
                  clear_colour_after_num_games_if_sgf_empty since rev 8).
"""

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_test_bundle as tb  # noqa: E402  (stdlib manifest / zip / structure helpers)

LOCAL = os.path.join(REPO, "data", "local")
P02 = os.path.join(REPO, "results", "prereg02")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
RANK_VAL = "rank_val_questions.csv"
PLAYER_VAL_Q = "player_val_questions.csv"
PLAYER_VAL_C = "player_val_candidates.csv"
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_SLICES = {"public": (0, 400), "private": (400, 800)}
SETS = ("seam_val", "public", "private")
RANK_DUMPS = ("val_keys", "val_p02e9_d4")
PLAYER_DUMPS = ("cand_games.npz", "query_games.npz")
RANK_WEIGHTS_PATH = "prereg02/rank/ckpt/epoch_9.pth"
PLAYER_WEIGHTS_PATH = "prereg01/player/ckpt/best_model.pth"
FAKE_NOTE = "t02_fake_tests from-val: no notebook ran; dumps copied or re-keyed from results/prereg02"
SALT = "c10-t02fake"
SHAPES = ("mixed", "official")
SLOT_COLOURS = ("empty", "filled")
OFFICIAL_KS = (5, 10, 15, 20)       # AIdea data page: 100 questions of each size per file


class FakeError(SystemExit):
    def __init__(self, message):
        super().__init__(f"t02_fake_tests: {message}")


def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def prepare_root(root, subdirs, rerun):
    if root is None:
        raise FakeError("--root is required")
    if tb.inside(root, REPO):
        raise FakeError(f"--root {root!r} is inside the repository; fake trees live in scratch")
    existing = [d for d in subdirs if os.path.exists(os.path.join(root, d))]
    if existing and not rerun:
        raise FakeError(f"{root} already has {existing}; pass --rerun to rebuild them")
    for d in existing:
        p = os.path.join(root, d)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(root, exist_ok=True)


# ================================================================ from-val: test files
def _split_records(raw, path):
    """(header line, [record lines], line end) of a CSV whose records are single lines (checked against csv)."""
    term = b"\r\n" if b"\r\n" in raw[:200000] else b"\n"
    lines = raw.split(term)
    if lines[-1] != b"":
        raise FakeError(f"{path}: no final line end")
    lines = lines[:-1]
    if any(b"\r" in ln or b"\n" in ln for ln in lines):
        raise FakeError(f"{path}: mixed line ends or multi-line fields")
    csv.field_size_limit(10**9)
    n_csv = sum(1 for _ in csv.reader(io.StringIO(raw.decode("utf-8"), newline=""))) - 1
    if n_csv != len(lines) - 1:
        raise FakeError(f"{path}: {n_csv} csv records but {len(lines) - 1} lines")
    return lines[0], lines[1:], term


def _rekey_line(line, n_fields, patterns, prefix):
    """Replace the leading 'val_' of the first n_fields fields; each field must match its pattern."""
    parts = line.split(b",", n_fields)
    for i in range(n_fields):
        if not re.fullmatch(patterns[i], parts[i]):
            raise FakeError(f"unexpected id field {parts[i][:40]!r}")
        parts[i] = prefix.encode() + parts[i][len(b"val_"):]
    return b",".join(parts)


def _csv_rows(data):
    csv.field_size_limit(10**9)
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


def make_test_files(local_dir, out_dir):
    """The six official files, re-prefixed from the local validation files; returns {name: derivation}."""
    os.makedirs(out_dir, exist_ok=True)
    srcs = {k: os.path.join(local_dir, k) for k in (RANK_VAL, PLAYER_VAL_Q, PLAYER_VAL_C)}
    raw = {}
    for k, p in srcs.items():
        with open(tb.refuse_forbidden(p), "rb") as f:
            raw[k] = f.read()
    q_pat = {RANK_VAL: [rb"val_q_\d+"], PLAYER_VAL_Q: [rb"val_pq_\d+"],
             PLAYER_VAL_C: [rb"val_g_\d+", rb"val_p_[0-9a-f]+"]}
    plan = []   # (official name, source key, (lo, hi) or None, pool)
    for pool in ("public", "private"):
        plan.append((tb.RANK_FILES[pool], RANK_VAL, RANK_SLICES[pool], pool))
        plan.append((tb.PLAYER_Q_FILES[pool], PLAYER_VAL_Q, None, pool))
        plan.append((tb.PLAYER_C_FILES[pool], PLAYER_VAL_C, None, pool))
    split = {k: _split_records(raw[k], srcs[k]) for k in raw}
    parsed = {k: _csv_rows(raw[k]) for k in raw}
    derived = {}
    for name, key, sl, pool in plan:
        header, recs, term = split[key]
        lo, hi = sl if sl else (0, len(recs))
        if hi > len(recs):
            raise FakeError(f"{srcs[key]} has {len(recs)} rows; need {hi}")
        body = [_rekey_line(r, len(q_pat[key]), q_pat[key], PREFIX[pool]) for r in recs[lo:hi]]
        data = term.join([header] + body) + term
        # the only change: ids re-prefixed
        src_rows = parsed[key]
        want = [src_rows[0]] + [[(PREFIX[pool] + v[len("val_"):]) if j < len(q_pat[key]) else v
                                 for j, v in enumerate(r)] for r in src_rows[1 + lo:1 + hi]]
        if _csv_rows(data) != want:
            raise FakeError(f"{name}: re-prefixed file differs from the source beyond the ids")
        tb.write_bytes(os.path.join(out_dir, name), data)
        derived[name] = {"derived_from": f"data/local/{key}", "derived_from_sha256": tb.sha256_bytes(raw[key]),
                         "source_rows": f"{lo + 1}-{hi}", "id_prefix": f"val_ -> {PREFIX[pool]}"}
    return derived


# ================================================================ from-val: dumps
def _npz(path):
    with np.load(tb.refuse_forbidden(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_npz(path, arrays):
    tb.check_writable(path)
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrays)
    os.replace(path + ".tmp", path)
    return path


def _rekey_array(a, prefix):
    vals = [str(x) for x in np.asarray(a).tolist()]
    if any(not v.startswith("val_") for v in vals):
        raise FakeError("an id in a dump does not start with val_")
    return np.array([prefix + v[len("val_"):] for v in vals])


def _question_ids(path):
    csv.field_size_limit(10**9)
    with open(tb.refuse_forbidden(path), newline="", encoding="utf-8") as f:
        return [r["question_id"] for r in csv.DictReader(f)]


def _cand_ids(path):
    csv.field_size_limit(10**9)
    with open(tb.refuse_forbidden(path), newline="", encoding="utf-8") as f:
        return [(r["game_id"], r["player_id"]) for r in csv.DictReader(f)]


def _marker(base, path, rows):
    m = dict(base)
    m.update({"rows": rows, "bytes": os.path.getsize(path), "sha256": tb.sha256_file(path),
              "time": tb.now_iso(), "fake": FAKE_NOTE})
    return m


def make_rank_dumps(local_dir, p02_rank, tests_dir, res_rank):
    src = os.path.join(p02_rank, "dumps")
    keys = _npz(os.path.join(src, "val_keys.npz"))
    dump = _npz(os.path.join(src, "val_p02e9_d4.npz"))
    markers = {s: tb.read_json(os.path.join(src, f"_DONE_{s}.json")) for s in RANK_DUMPS}
    qfile = _question_ids(os.path.join(local_dir, RANK_VAL))
    qids = [str(x) for x in keys["question_id"].tolist()]
    order, first = [], {}
    for i, q in enumerate(qids):
        if q not in first:
            first[q] = i
            order.append(q)
        elif order[-1] != q:
            raise FakeError(f"val_keys: question {q} is not contiguous")
    if order != qfile:
        raise FakeError("val_keys question order differs from rank_val_questions.csv")
    bounds = [first[q] for q in order] + [len(qids)]
    info = {}
    d = os.path.join(res_rank, "seam_val", "dumps")
    os.makedirs(d, exist_ok=True)
    for s in RANK_DUMPS:
        for f in (f"{s}.npz", f"_DONE_{s}.json"):
            shutil.copyfile(os.path.join(src, f), tb.check_writable(os.path.join(d, f)))
    vq = os.path.join(local_dir, RANK_VAL)
    tb.write_json(os.path.join(d, "_INPUT.json"), {"set": "seam_val", "question_file": f"bundle local/{RANK_VAL}",
                                                   "sha256": tb.sha256_file(vq)})
    info["seam_val"] = {"questions_sha256": tb.sha256_file(vq), "rows": len(order), "games": len(qids)}
    for pool in ("public", "private"):
        lo, hi = RANK_SLICES[pool]
        r0, r1 = bounds[lo], bounds[hi]
        d = os.path.join(res_rank, pool, "dumps")
        os.makedirs(d, exist_ok=True)
        k2 = {}
        for name, a in keys.items():
            k2[name] = _rekey_array(a[r0:r1], PREFIX[pool]) if name == "question_id" else a[r0:r1]
        d2 = {name: (a[r0:r1] if name == "logits" else a) for name, a in dump.items()}
        if d2["logits"].tobytes() != dump["logits"][r0:r1].tobytes():
            raise FakeError("logit slice changed")
        paths = {"val_keys": _save_npz(os.path.join(d, "val_keys.npz"), k2),
                 "val_p02e9_d4": _save_npz(os.path.join(d, "val_p02e9_d4.npz"), d2)}
        for s in RANK_DUMPS:
            tb.write_json(os.path.join(d, f"_DONE_{s}.json"), _marker(markers[s], paths[s], r1 - r0))
        qp = os.path.join(tests_dir, tb.RANK_FILES[pool])
        if _question_ids(qp) != [PREFIX[pool] + q[len("val_"):] for q in order[lo:hi]]:
            raise FakeError(f"{qp}: question order differs from the dump slice")
        tb.write_json(os.path.join(d, "_INPUT.json"), {"set": pool, "question_file": f"tests/{tb.RANK_FILES[pool]}",
                                                       "sha256": tb.sha256_file(qp)})
        info[pool] = {"questions_sha256": tb.sha256_file(qp), "rows": hi - lo, "games": r1 - r0}
    return info, markers["val_p02e9_d4"]


def make_player_dumps(local_dir, p02_player, tests_dir, res_player):
    src = os.path.join(p02_player, "dumps")
    info = {}
    d = os.path.join(res_player, "seam_val", "dumps")
    os.makedirs(d, exist_ok=True)
    for f in PLAYER_DUMPS:
        shutil.copyfile(os.path.join(src, f), tb.check_writable(os.path.join(d, f)))
    vq, vc = os.path.join(local_dir, PLAYER_VAL_Q), os.path.join(local_dir, PLAYER_VAL_C)
    tb.write_json(os.path.join(d, "_INPUT.json"),
                  {"set": "seam_val", "question_file": PLAYER_VAL_Q, "sha256": tb.sha256_file(vq),
                   "candidates_file": PLAYER_VAL_C, "candidates_sha256": tb.sha256_file(vc)})
    cand = _npz(os.path.join(src, "cand_games.npz"))
    query = _npz(os.path.join(src, "query_games.npz"))
    info["seam_val"] = {"questions_file": PLAYER_VAL_Q, "questions_sha256": tb.sha256_file(vq),
                        "candidates_file": PLAYER_VAL_C, "candidates_sha256": tb.sha256_file(vc),
                        "rows": len(set(query["question_id"].tolist())), "games": int(len(query["question_id"])),
                        "candidate_rows": int(len(cand["game_id"]))}
    for pool in ("public", "private"):
        d = os.path.join(res_player, pool, "dumps")
        os.makedirs(d, exist_ok=True)
        c2 = {k: (_rekey_array(a, PREFIX[pool]) if k in ("game_id", "player_id") else a) for k, a in cand.items()}
        q2 = {k: (_rekey_array(a, PREFIX[pool]) if k == "question_id" else a) for k, a in query.items()}
        qp, cp = (os.path.join(tests_dir, tb.PLAYER_Q_FILES[pool]), os.path.join(tests_dir, tb.PLAYER_C_FILES[pool]))
        if _cand_ids(cp) != list(zip(c2["game_id"].tolist(), c2["player_id"].tolist())):
            raise FakeError(f"{cp}: candidate order differs from cand_games.npz")
        qorder = list(dict.fromkeys(q2["question_id"].tolist()))
        if _question_ids(qp) != qorder:
            raise FakeError(f"{qp}: question order differs from query_games.npz")
        _save_npz(os.path.join(d, "cand_games.npz"), c2)
        _save_npz(os.path.join(d, "query_games.npz"), q2)
        tb.write_json(os.path.join(d, "_INPUT.json"),
                      {"set": pool, "question_file": tb.PLAYER_Q_FILES[pool], "sha256": tb.sha256_file(qp),
                       "candidates_file": tb.PLAYER_C_FILES[pool], "candidates_sha256": tb.sha256_file(cp)})
        info[pool] = {"questions_file": tb.PLAYER_Q_FILES[pool], "questions_sha256": tb.sha256_file(qp),
                      "candidates_file": tb.PLAYER_C_FILES[pool], "candidates_sha256": tb.sha256_file(cp),
                      "rows": len(qorder), "games": int(len(q2["question_id"])), "candidate_rows": int(len(c2["game_id"]))}
    return info, str(cand["weights_sha256"])


# ================================================================ from-val: notebook records
def _records(res, comp, test_bundle, sets, weights, gpu, torch_v, cudnn, extra):
    config = {"fake": FAKE_NOTE, "smoke": False, "rehearsal": True, "notebook": f"colab/test02_{comp}.ipynb",
              "notebook_sha256": None, "gpu": gpu, "torch": torch_v, "cudnn": cudnn, "weights": weights,
              "test_bundle": test_bundle, "sets": sets, "changes_from_prereg02": []}
    run = {"run": 1, "status": "completed", "smoke": False, "rehearsal": True, "gpu": gpu, "fake": FAKE_NOTE}
    ro = {"fake": FAKE_NOTE, "roots": {"prereg01": {"n_files": 0, "files": {}},
                                       "prereg02": {"n_files": 0, "files": {}}}, "same_as_before": True}
    files = {"config.json": config, "run_log.json": {"status": "completed", "runs": [run]},
             "env.json": {"runs": [{"run": 1, "torch": torch_v, "cudnn": cudnn,
                                    "gpu": {"nvidia_smi": [gpu] if gpu else []}, "fake": FAKE_NOTE}]},
             "timings.json": {"runs": [{"run": 1, "seconds": {}}], "fake": FAKE_NOTE},
             "readonly_before.json": ro, "readonly_after.json": ro}
    files.update(extra)
    for name, obj in files.items():
        tb.write_json(os.path.join(res, name), obj)


def from_val(root, zip_kind="rehearsal", zip_out=None, rerun=False, local_dir=LOCAL, p02_dir=P02, log=print):
    if zip_kind not in ("rehearsal", "smoke"):
        raise FakeError(f"--zip-kind {zip_kind!r}: rehearsal or smoke")
    zip_out = zip_out or os.path.join(root, "data", "colab_bundle", tb.ZIP_NAME)
    if tb.inside(zip_out, REPO):
        raise FakeError(f"--zip-out {zip_out!r} is inside the repository")
    prepare_root(root, [os.path.join("data", "test"), os.path.join("results", "test02"), "FAKE_TREE.json"], rerun)
    if os.path.exists(zip_out):
        if not rerun:
            raise FakeError(f"{zip_out} exists; pass --rerun")
        os.remove(zip_out)
    tests = os.path.join(root, "data", "test")
    res = os.path.join(root, "results", "test02")

    derived = make_test_files(local_dir, tests)
    q = {c: tb.quick_structure(tests, c) for c in ("rank", "player")}
    if q["rank"] or q["player"]:
        raise FakeError(f"quick structure check failed on the fake files: {q}")
    files = {n: tb.file_record(os.path.join(tests, n), **derived[n]) for n in tb.ALL_NAMES}
    mpath, _man = tb.write_manifest(
        tests, zip_kind, files, {"rank": "PASS", "player": "PASS"},
        t1_basis="t02_fake_tests quick structure check (T1.1-T1.3 subset); t02_checks was not run",
        fake=FAKE_NOTE)
    zsha, zbytes = tb.write_zip(zip_out, tests)
    test_bundle = {"zip_sha256": zsha, "manifest_sha256": tb.sha256_file(mpath), "kind": zip_kind,
                   "t1": {"rank": "PASS", "player": "PASS"}, "files": {n: files[n]["sha256"] for n in tb.ALL_NAMES}}
    log(f"from-val: test files and MANIFEST in {tests}; zip {zip_out} ({zbytes:,d} bytes, sha256 {zsha[:12]})")

    rank_sets, rank_marker = make_rank_dumps(local_dir, os.path.join(p02_dir, "rank"), tests,
                                             os.path.join(res, "rank"))
    gpu, torch_v, cudnn = rank_marker.get("gpu_name"), rank_marker.get("torch"), rank_marker.get("cudnn")
    seam = {"fake": FAKE_NOTE, "keys_equal": {k: True for k in ("question_id", "slot", "color", "parsed")},
            "same_gpu_as_prereg02": True, "tol": 1e-4, "bitwise_equal": True, "max_abs_logit_diff": 0.0,
            "max_abs_c0_diff": 0.0, "problems": [], "questions": rank_sets["seam_val"]["rows"],
            "games": rank_sets["seam_val"]["games"],
            "rule_b_diff": {"questions": 0, "allowed": 4, "question_ids": []}, "passed": True,
            "gpu": gpu, "p02_gpu": gpu, "torch": torch_v, "p02_torch": torch_v, "cudnn": cudnn, "p02_cudnn": cudnn}
    _records(os.path.join(res, "rank"), "rank", test_bundle, rank_sets,
             {"path": RANK_WEIGHTS_PATH, "sha256": rank_marker.get("weights_sha256")}, gpu, torch_v, cudnn,
             {"seam_check.json": seam})
    log(f"from-val: rank dumps for {list(rank_sets)} in {os.path.join(res, 'rank')}")

    player_sets, wsha = make_player_dumps(local_dir, os.path.join(p02_dir, "player"), tests,
                                          os.path.join(res, "player"))
    repro_path = os.path.join(p02_dir, "player", "repro_check.json")
    repro = tb.read_json(repro_path) if os.path.exists(repro_path) else {"passed": True}
    repro = dict(repro, fake=FAKE_NOTE + " (repro_check.json copied from prereg02)")
    seam_p02 = {"fake": FAKE_NOTE, "smoke": False,
                "files": {f: {"bitwise_equal": True, "max_abs_emb_diff": 0.0,
                              "sha256": tb.sha256_file(os.path.join(res, "player", "seam_val", "dumps", f)),
                              "reference_sha256": tb.sha256_file(os.path.join(p02_dir, "player", "dumps", f))}
                          for f in PLAYER_DUMPS},
                "bitwise_equal": True, "max_abs_emb_diff": 0.0,
                "rule": "record only; the local seam (t02_player.py run --pool seam_val) decides"}
    _records(os.path.join(res, "player"), "player", test_bundle, player_sets,
             {"path": PLAYER_WEIGHTS_PATH, "sha256": wsha}, repro.get("gpu", gpu), repro.get("torch", torch_v),
             cudnn, {"repro_check.json": repro, "seam_p02.json": seam_p02})
    log(f"from-val: player dumps for {list(player_sets)} in {os.path.join(res, 'player')}")

    inputs = {f"data/local/{k}": tb.sha256_file(os.path.join(local_dir, k)) for k in (RANK_VAL, PLAYER_VAL_Q, PLAYER_VAL_C)}
    for comp, names in (("rank", [f"{s}.npz" for s in RANK_DUMPS] + [f"_DONE_{s}.json" for s in RANK_DUMPS]),
                        ("player", list(PLAYER_DUMPS))):
        for n in names:
            inputs[f"results/prereg02/{comp}/dumps/{n}"] = tb.sha256_file(os.path.join(p02_dir, comp, "dumps", n))
    tree = {"format": "c10-test02 fake tree v1", "kind": "from-val", "created": tb.now_iso(),
            "rule": "validation-derived: never run content statistics (T1.4-T5) on this tree; only the apply "
                    "pipeline and T1.1-T1.3 (structure)",
            "zip_kind": zip_kind, "zip": os.path.abspath(zip_out), "zip_sha256": zsha,
            "manifest_sha256": test_bundle["manifest_sha256"], "files": test_bundle["files"],
            "id_mapping": {"rank public": "val_q_ rows 1-400 -> pub_q_", "rank private": "val_q_ rows 401-800 -> priv_q_",
                           "player questions": "val_pq_ -> pub_pq_ / priv_pq_ (all 400 in both pools)",
                           "candidates": "val_p_<hex> -> pub_p_/priv_p_, val_g_N -> pub_g_N/priv_g_N"},
            "sets": {"rank": rank_sets, "player": player_sets}, "inputs_sha256": inputs, "tool": tb.tool_record()}
    tb.write_json(os.path.join(root, "FAKE_TREE.json"), tree)
    log(f"from-val: done ({root}). Only t1 --only T1.1,T1.2,T1.3 may run on this tree.")
    return tree


# ================================================================ from-train
def _write_csv(path, header, rows):
    tb.check_writable(path)
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".tmp", path)
    return path


def _question_row(qid, games, fill=""):
    """One question row; fill = the colour written into color_i after num_games ('' leaves them empty)."""
    k = len(games)
    return ([qid, k] + [g["sgf_content"] for g in games] + [""] * (tb.N_SLOTS - k)
            + [g["color"] for g in games] + [fill] * (tb.N_SLOTS - k))


def _pick_players(eligible, levels, per_pool, n_pools, tag):
    """n_pools disjoint lists of per_pool players, per_pool/len(levels) per level where possible."""
    by_level = defaultdict(list)
    for pid, lv in eligible.items():
        by_level[lv].append(pid)
    for lv in by_level:
        by_level[lv].sort(key=lambda p: _h(SALT, tag, p))
    quota = per_pool // len(levels)
    pools = [[] for _ in range(n_pools)]
    used = set()
    for lv in levels:
        for k in range(n_pools):
            take = by_level[lv][k * quota:(k + 1) * quota]
            pools[k] += take
            used.update(take)
    rest = sorted((p for p in eligible if p not in used), key=lambda p: _h(SALT, tag, "fill", p))
    for k in range(n_pools):
        need = per_pool - len(pools[k])
        pools[k] += rest[:need]
        rest = rest[need:]
        if len(pools[k]) != per_pool:
            raise FakeError(f"not enough eligible players for {n_pools} pools of {per_pool}")
    return pools


def _pool_files(players, games_of, prefix, games_per_player, tag, questions=True):
    """(candidate rows, question rows, info) for one pool; players in hash order, ids synthesized."""
    order = sorted(players, key=lambda p: _h(SALT, tag, "order", p))
    cand, qsrc, pmap = [], [], {}
    n = 0
    for pid in order:
        sid = f"{prefix}p_{_h(SALT, tag, 'pid', pid)[:8]}"
        pmap[pid] = sid
        gs = sorted(games_of[pid], key=lambda g: _h(SALT, tag, "game", pid, g["game_id"]))
        for g in gs[:games_per_player]:
            n += 1
            cand.append([f"{prefix}g_{n:06d}", sid, g["sgf_content"], g["color"]])
        other = gs[games_per_player:]
        k = min(5 + int(_h(SALT, tag, "k", pid)[:8], 16) % 16, len(other))
        if questions and k >= 5:
            qsrc.append((pid, other[:k]))
    if len(set(pmap.values())) != len(pmap):
        raise FakeError("synthesized player ids collide")
    qsrc.sort(key=lambda x: _h(SALT, tag, "q", x[0]))
    qrows = [_question_row(f"{prefix}q_{i:04d}", gs) for i, (_pid, gs) in enumerate(qsrc, 1)]
    return cand, qrows, {"players": len(order), "candidate_rows": len(cand), "questions": len(qrows)}


def _rank_questions(games_of, players, prefix, n_questions, tag, used):
    """n_questions questions of 5-20 games, one player each (players may repeat, games never)."""
    order = sorted(players, key=lambda p: _h(SALT, tag, "order", p))
    shuffled = {p: sorted(games_of[p], key=lambda g: _h(SALT, tag, "game", p, g["game_id"])) for p in order}
    rows, i = [], 0
    while len(rows) < n_questions:
        if i > 50 * len(order) + n_questions:
            raise FakeError("not enough inner games for the rank questions")
        pid = order[i % len(order)]
        i += 1
        k = 5 + int(_h(SALT, tag, "k", pid, i)[:8], 16) % 16
        free = [g for g in shuffled[pid] if (pid, g["game_id"]) not in used]
        if len(free) < k:
            continue
        gs = free[:k]
        used.update((pid, g["game_id"]) for g in gs)
        rows.append(_question_row(f"{prefix}q_{len(rows) + 1:04d}", gs))
    return rows


def _blacks(games_of, pid, tag):
    return sorted((g for g in games_of[pid] if g["color"] == "B"),
                  key=lambda g: _h(SALT, tag, "game", pid, g["game_id"]))


def _pool_files_official(players, games_of, prefix, games_per_player, tag, fill):
    """Official-shape pool: candidates = 100 black games each; one black question per player of its next k black
    games, k = 5 / 10 / 15 / 20 in turn over the players' hash order (the largest official k that fits when a
    player has fewer; none below 5); fill = the colour after num_games ('' or 'B')."""
    order = sorted(players, key=lambda p: _h(SALT, tag, "order", p))
    cand, qsrc, pmap = [], [], {}
    n = 0
    for j, pid in enumerate(order):
        sid = f"{prefix}p_{_h(SALT, tag, 'pid', pid)[:8]}"
        pmap[pid] = sid
        gs = _blacks(games_of, pid, tag)
        if len(gs) < games_per_player:
            raise FakeError(f"player {pid} has {len(gs)} black games; the official shape needs {games_per_player}")
        for g in gs[:games_per_player]:
            n += 1
            cand.append([f"{prefix}g_{n:06d}", sid, g["sgf_content"], "B"])
        other = gs[games_per_player:]
        fits = [k for k in OFFICIAL_KS if k <= min(OFFICIAL_KS[j % len(OFFICIAL_KS)], len(other))]
        if fits:
            qsrc.append((pid, other[:fits[-1]]))
    if len(set(pmap.values())) != len(pmap):
        raise FakeError("synthesized player ids collide")
    qsrc.sort(key=lambda x: _h(SALT, tag, "q", x[0]))
    qrows = [_question_row(f"{prefix}q_{i:04d}", gs, fill) for i, (_pid, gs) in enumerate(qsrc, 1)]
    ks = defaultdict(int)
    for _pid, gs in qsrc:
        ks[len(gs)] += 1
    return cand, qrows, {"players": len(order), "candidate_rows": len(cand), "questions": len(qrows),
                         "num_games": {str(k): ks[k] for k in sorted(ks)}}


def _rank_questions_official(games_of, players, prefix, n_questions, tag, used, filled):
    """n_questions single-colour questions (k = 5 / 10 / 15 / 20, a quarter each, in hash order; colour B or W by
    hash), one player each (players may repeat, games never); filled: color_i after num_games = that colour."""
    order = sorted(players, key=lambda p: _h(SALT, tag, "order", p))
    by_col = {c: {p: sorted((g for g in games_of[p] if g["color"] == c),
                            key=lambda g: _h(SALT, tag, "game", p, g["game_id"])) for p in order} for c in "BW"}
    plan = sorted(range(n_questions), key=lambda i: _h(SALT, tag, "plan", i))
    rows, j = [], 0
    for q, i in enumerate(plan, 1):
        k = OFFICIAL_KS[i % len(OFFICIAL_KS)]
        colour = "B" if int(_h(SALT, tag, "colour", i)[:8], 16) % 2 == 0 else "W"
        free = []
        for _ in range(len(order)):
            pid = order[j % len(order)]
            j += 1
            free = [g for g in by_col[colour][pid] if (pid, g["game_id"]) not in used]
            if len(free) >= k:
                break
        if len(free) < k:
            raise FakeError(f"not enough inner games of colour {colour} for a {k}-game rank question")
        gs = free[:k]
        used.update((pid, g["game_id"]) for g in gs)
        rows.append(_question_row(f"{prefix}q_{q:04d}", gs, colour if filled else ""))
    return rows


def _official_quick_structure(tests, slot_colours):
    """quick_structure of an official-shape tree: clean for 'empty'; for 'filled' only the expected 'slot >
    num_games is not empty' problems (the raw files need the rev 8 normalization first)."""
    q = {c: tb.quick_structure(tests, c) for c in ("rank", "player")}
    if slot_colours == "empty":
        if q["rank"] or q["player"]:
            raise FakeError(f"quick structure check failed on the from-train files: {q}")
        return "PASS"
    other = [e for c in q for e in q[c] if not e.endswith("> num_games is not empty")]
    if other or not (q["rank"] and q["player"]):
        raise FakeError(f"filled slot colours: unexpected quick structure result {q}")
    return ("FAIL on the raw files as intended (colour letters after num_games; %d / %d problems listed); the "
            "normalized copies of t02_test_bundle build pass" % (len(q["rank"]), len(q["player"])))


def from_train(root, rerun=False, rows_fit=None, rows_inner=None, per_pool=400, games=100, n_rank=400,
               log=print, shape="mixed", slot_colours="empty"):
    if shape not in SHAPES or slot_colours not in SLOT_COLOURS:
        raise FakeError(f"shape {shape!r} / slot colours {slot_colours!r}: shape in {SHAPES}, slot colours in "
                        f"{SLOT_COLOURS}")
    if shape == "mixed" and slot_colours != "empty":
        raise FakeError("--slot-colours filled is for --shape official")
    prepare_root(root, [os.path.join("data", "test"), "neg", "FAKE_TREE.json"], rerun)
    if shape == "official":
        return _from_train_official(root, rows_fit, rows_inner, per_pool, games, n_rank, log, slot_colours)
    if rows_fit is None or rows_inner is None:
        import c10data   # training players only; validation players are never returned
        rows_fit = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=LOCAL)
        rows_inner = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=LOCAL)
        levels = c10data.LEVELS
    else:
        levels = sorted({r["rank"] for r in rows_fit})
    fit_games, fit_level = defaultdict(list), {}
    for r in rows_fit:
        fit_games[r["player_id"]].append(r)
        fit_level[r["player_id"]] = r["rank"]
    inner_games = defaultdict(list)
    for r in rows_inner:
        inner_games[r["player_id"]].append(r)

    tests = os.path.join(root, "data", "test")
    os.makedirs(tests, exist_ok=True)
    eligible = {p: fit_level[p] for p, gs in fit_games.items() if len(gs) >= games + 5}
    pos = _pick_players(eligible, levels, per_pool, 2, "pos")
    info = {}
    for pool, players, tag in (("public", pos[0], "posA"), ("private", pos[1], "posB")):
        cand, qrows, inf = _pool_files(players, fit_games, PREFIX[pool], games, tag)
        _write_csv(os.path.join(tests, tb.PLAYER_C_FILES[pool]), tb.CANDIDATE_COLUMNS, cand)
        _write_csv(os.path.join(tests, tb.PLAYER_Q_FILES[pool]), tb.QUESTION_COLUMNS, qrows)
        info[f"player {pool} (POS)"] = dict(inf, source="fit", tag=tag)
    used = set()
    inner_players = sorted(inner_games)
    for pool in ("public", "private"):
        qrows = _rank_questions(inner_games, inner_players, PREFIX[pool], n_rank, f"rank{pool}", used)
        _write_csv(os.path.join(tests, tb.RANK_FILES[pool]), tb.QUESTION_COLUMNS, qrows)
        info[f"rank {pool}"] = {"questions": len(qrows), "source": "inner"}

    neg_dir = os.path.join(root, "neg", "data", "test")
    os.makedirs(neg_dir, exist_ok=True)
    neg_players = [p for p, gs in inner_games.items() if len(gs) >= games]
    cand, qrows, inf = _pool_files(neg_players, inner_games, PREFIX["public"], games, "neg")
    _write_csv(os.path.join(neg_dir, tb.PLAYER_C_FILES["public"]), tb.CANDIDATE_COLUMNS, cand)
    _write_csv(os.path.join(neg_dir, tb.PLAYER_Q_FILES["public"]), tb.QUESTION_COLUMNS, qrows)
    info["neg player public (NEG)"] = dict(inf, source="inner", t3_threshold=-(-len(neg_players) // 10))

    official = per_pool == tb.N_PLAYERS and games == tb.GAMES_PER_PLAYER and n_rank == tb.N_QUESTIONS
    struct = {c: tb.quick_structure(tests, c) for c in ("rank", "player")} if official else None
    if struct and (struct["rank"] or struct["player"]):
        raise FakeError(f"quick structure check failed on the from-train files: {struct}")
    tree = {"format": "c10-test02 fake tree v1", "kind": "from-train", "shape": "mixed", "created": tb.now_iso(),
            "rule": "training players only (c10data.load_rows fit / inner); validation players never used",
            "pools": info, "files": {n: tb.sha256_file(os.path.join(tests, n)) for n in tb.ALL_NAMES},
            "neg_files": {n: tb.sha256_file(os.path.join(neg_dir, n))
                          for n in (tb.PLAYER_C_FILES["public"], tb.PLAYER_Q_FILES["public"])},
            "quick_structure": "PASS" if official else "not run (non-official sizes)",
            "expected": {"T1 (data/test)": "PASS for rank and player",
                         "T1 (neg/data/test)": "FAIL_STRUCT by design (not 400 players, public pool only)",
                         "T2(a)": "FAIL on the player files (their games are fit games); rank files FAIL with the "
                                  "default train index (inner games) and give 0 copies with --index-part fit",
                         "T3": "POS pools (data/test) declared overlap; NEG pool (neg/) not declared",
                         "T4": "inside the training null ranges"},
            "tool": tb.tool_record()}
    tb.write_json(os.path.join(root, "FAKE_TREE.json"), tree)
    log(f"from-train: {root} ({', '.join(f'{k}: {v}' for k, v in info.items())})")
    return tree


def _from_train_official(root, rows_fit, rows_inner, per_pool, games, n_rank, log, slot_colours):
    """from-train --shape official (see the module docstring); root is prepared by from_train."""
    if rows_fit is None or rows_inner is None:
        import c10data   # training players only; validation players are never returned
        rows_fit = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=LOCAL)
        rows_inner = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=LOCAL)
        levels = c10data.LEVELS
    else:
        levels = sorted({r["rank"] for r in rows_fit})
    filled = slot_colours == "filled"
    fit_games, fit_level = defaultdict(list), {}
    for r in rows_fit:
        fit_games[r["player_id"]].append(r)
        fit_level[r["player_id"]] = r["rank"]
    inner_games = defaultdict(list)
    for r in rows_inner:
        inner_games[r["player_id"]].append(r)
    n_black = {p: sum(1 for r in gs if r["color"] == "B") for p, gs in fit_games.items()}
    tests = os.path.join(root, "data", "test")
    os.makedirs(tests, exist_ok=True)
    eligible = {p: fit_level[p] for p in fit_games if n_black[p] >= games + max(OFFICIAL_KS)}
    pos = _pick_players(eligible, levels, per_pool, 2, "pos")
    info = {}
    for pool, players, tag in (("public", pos[0], "posA"), ("private", pos[1], "posB")):
        cand, qrows, inf = _pool_files_official(players, fit_games, PREFIX[pool], games, tag, "B" if filled else "")
        _write_csv(os.path.join(tests, tb.PLAYER_C_FILES[pool]), tb.CANDIDATE_COLUMNS, cand)
        _write_csv(os.path.join(tests, tb.PLAYER_Q_FILES[pool]), tb.QUESTION_COLUMNS, qrows)
        info[f"player {pool} (POS)"] = dict(inf, source="fit, black games only", tag=tag,
                                            eligible_players=len(eligible))
    used = set()
    inner_players = sorted(inner_games)
    for pool in ("public", "private"):
        qrows = _rank_questions_official(inner_games, inner_players, PREFIX[pool], n_rank, f"rank{pool}", used, filled)
        _write_csv(os.path.join(tests, tb.RANK_FILES[pool]), tb.QUESTION_COLUMNS, qrows)
        info[f"rank {pool}"] = {"questions": len(qrows), "source": "inner, one colour per question",
                                "questions_black": sum(1 for r in qrows if r[2 + tb.N_SLOTS] == "B")}
    neg_dir = os.path.join(root, "neg", "data", "test")
    os.makedirs(neg_dir, exist_ok=True)
    neg_players = [p for p, gs in inner_games.items() if sum(1 for r in gs if r["color"] == "B") >= games]
    cand, qrows, inf = _pool_files_official(neg_players, inner_games, PREFIX["public"], games, "neg",
                                            "B" if filled else "")
    _write_csv(os.path.join(neg_dir, tb.PLAYER_C_FILES["public"]), tb.CANDIDATE_COLUMNS, cand)
    _write_csv(os.path.join(neg_dir, tb.PLAYER_Q_FILES["public"]), tb.QUESTION_COLUMNS, qrows)
    info["neg player public (NEG)"] = dict(inf, source="inner, black games only",
                                           t3_threshold=-(-len(neg_players) // 10))
    official = per_pool == tb.N_PLAYERS and games == tb.GAMES_PER_PLAYER and n_rank == tb.N_QUESTIONS
    qs = _official_quick_structure(tests, slot_colours) if official else "not run (non-official sizes)"
    t1_exp = ("PASS for rank and player" if not filled else
              "NORMALIZED for rank and player: clear_colour_after_num_games_if_sgf_empty on the four question files "
              "(PREREG_02 rev 8; FAIL_STRUCT before rev 8); t02_test_bundle build then writes copies whose T1.1-T1.3 "
              "re-check is PASS")
    tree = {"format": "c10-test02 fake tree v1", "kind": "from-train", "shape": "official",
            "slot_colours": slot_colours, "created": tb.now_iso(),
            "rule": "training players only (c10data.load_rows fit / inner); validation players never used",
            "official_structure": "competition 1: candidates and questions black only, 400 x 100 candidate games, "
                                  "questions 5/10/15/20 games x 100; competition 2: questions 5/10/15/20 x 100, one "
                                  "colour per question (AIdea data page, docs/DATA.md 2026-09-28)",
            "pools": info, "files": {n: tb.sha256_file(os.path.join(tests, n)) for n in tb.ALL_NAMES},
            "neg_files": {n: tb.sha256_file(os.path.join(neg_dir, n))
                          for n in (tb.PLAYER_C_FILES["public"], tb.PLAYER_Q_FILES["public"])},
            "quick_structure": qs,
            "expected": {"T1 (data/test)": t1_exp,
                         "T1 (neg/data/test)": "FAIL_STRUCT by design (not 400 players, public pool only)",
                         "T2(a)": "FAIL on the player files (fit games), every copy same_side; rank files: copies "
                                  "same_side with the default train index, 0 copies with --index-part fit",
                         "T3": "POS pools (b_c = 1): variant T3b, overlap declared; NEG pool (inner, b_c = 1): variant "
                               "T3b, not declared",
                         "T4": "player files black only: mixed-colour null ranges 'not applicable'; b_c = b_q = 1 -> "
                               "PASS",
                         "T5": "official_design_match true for the four question files"},
            "tool": tb.tool_record()}
    tb.write_json(os.path.join(root, "FAKE_TREE.json"), tree)
    log(f"from-train (official, slot colours {slot_colours}): {root} "
        f"({', '.join(f'{k}: {v}' for k, v in info.items())})")
    return tree


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="t02_fake_tests.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-val", help="pipeline dry-run tree from validation files (re-prefixed ids)")
    p.add_argument("--root", required=True)
    p.add_argument("--zip-kind", default="rehearsal", choices=("rehearsal", "smoke"))
    p.add_argument("--zip-out")
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("from-train", help="checks dry-run tree from training players")
    p.add_argument("--root", required=True)
    p.add_argument("--shape", choices=SHAPES, default="mixed",
                   help="mixed (default; the original dry-run tree) or official (PREREG_02 rev 8 structure)")
    p.add_argument("--slot-colours", choices=SLOT_COLOURS, default="empty",
                   help="official shape: colour cells after num_games empty (default) or filled with the question "
                        "colour")
    p.add_argument("--rerun", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "from-val":
        return from_val(args.root, args.zip_kind, args.zip_out, args.rerun)
    return from_train(args.root, args.rerun, shape=args.shape, slot_colours=args.slot_colours)


if __name__ == "__main__":
    main()
