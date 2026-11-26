"""Fake trees for the 06 test-day drills (results/prereg06/design/spec.md 1.2 and 9.3 item 4; PREREG_06 rev 3).

    PY tools/t06_fake_tests.py official  --root S2 [--dup M] [--slot-colours empty|filled] [--rerun]
    PY tools/t06_fake_tests.py add-dumps --root S2 [--emb PATH | --no-emb] [--rerun]
    PY tools/t06_fake_tests.py fake-t2   --root S  [--fail a|b|c|d ...] [--rerun --reason TEXT]

official   the official-shape tree of tools/t02_fake_tests.py from-train --shape official (its from_train function,
           called with the fit / inner rows of c10data.load_rows(local_dir=None): 400 fit players with >= 120 black
           games per pool, 100 black candidate games each, one black question of their next 5 / 10 / 15 / 20 black
           games, 100 of each size), then add-dumps below. Training players only; data/local is not opened here
           (the leak check of the fit players against the validation files is the committed
           results/prereg06/design/leakcheck_report.json, overlap 0).
           --dup M (the "same player, two questions" variant of spec 9.3 item 4): in each pool, M candidates (Y) get a
           second question from their unused black games and M other candidates (X) lose theirs (the question keeps
           X's question_id and file position; k = the largest official size <= min(X's old k, Y's spare games)), so
           the pool is d<M>-like: 400 questions, 400 - M distinct owners. No game appears twice (T2(c) = 0); the
           detector is expected to stay closed (PREREG_06 2.6: d80 and the other non-one-to-one structures, 0 opens).
add-dumps  the test02 notebook outputs t02_player.py needs on such a tree (the fake records of t02_fake_tests from-val):
           S2/results/test02/player/{seam_val,public,private}/dumps/{cand_games,query_games}.npz and _INPUT.json,
           config.json (smoke false, rehearsal true, the pinned weights SHA-256, test_bundle, sets), run_log.json,
           env.json, timings.json, readonly_*.json, repro_check.json and seam_p02.json, and S2/data/test/MANIFEST.json
           (kind rehearsal). seam_val = byte copies of results/prereg02/player/dumps (the validation pool, as in
           from-val; its _INPUT.json needs the SHA-256 of the two validation game files, the only data/local read).
           public / private: emb_sum and n_windows of every game from data/prereg06/cnn_fit/black_ge105.npz (spec 4;
           the real prereg01 player CNN on fit players' black games), matched by the SGF text through
           c10data.load_rows(part="fit", local_dir=None); when that file is missing (or with --no-emb), or a game is
           not in it, a fixed-seed vector (player mean + game noise, sha256-seeded). Counts go into FAKE_TREE.json.
fake-t2    S/results/test02/checks/player/t2.json in the format of tools/t02_checks.py run (fake: true) for drills on
           trees whose games are training games (the real T2(a) always fails there, so no 06-ON could be drilled):
           (a) = 0 copies per file (stand-in for an official file without training copies); (b), (c), (d) computed on
           the tree's files with t02_checks.game_keys (full game) exactly as t02_checks.t2_run counts them;
           --fail U forces unit U to FAIL (copies = its threshold) for the negative drills. t06_player.py accepts a
           fake t2.json in --dry-run only (real mode refuses it).
--root must be outside the repository. Nothing here opens a label file, runs tools/score.py or imports a
competition-2 tool.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_fake_tests as tf  # noqa: E402  (the official-shape builder and the fake notebook records)
import t02_test_bundle as tb  # noqa: E402

LOCAL = os.path.join(REPO, "data", "local")
P02_PLAYER = os.path.join(REPO, "results", "prereg02", "player")
EMB = os.path.join(REPO, "data", "prereg06", "cnn_fit", "black_ge105.npz")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
VAL_Q, VAL_C = "player_val_questions.csv", "player_val_candidates.csv"
WEIGHTS_SHA256 = "356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b"   # prereg01 player CNN (pinned)
SALT = "c10-t06fake"
T2_FORMAT = "c10-test02 t2 v1"
FAKE_NOTE = "t06_fake_tests: no notebook ran; dumps from black_ge105.npz (spec 4) or fixed-seed vectors"
T2_NOTE = ("t06_fake_tests fake-t2: a stand-in for tools/t02_checks.py run on a drill tree; (a) set to 0 copies, "
           "(b) (c) (d) counted on the tree's files with t02_checks.game_keys")
POOL_TAGS = {"public": "posA", "private": "posB"}         # tools/t02_fake_tests.py _from_train_official
_LABEL = "ans" + "wer"


class FakeError(SystemExit):
    def __init__(self, msg):
        super().__init__(f"t06_fake_tests: {msg}")


def refuse_label(p):
    if _LABEL in os.path.basename(str(p)).lower():
        raise FakeError(f"refusing {p}: label files are never opened")
    return p


def under(path, root):
    a, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return a == r or a.startswith(r.rstrip(os.sep) + os.sep)


def check_root(root):
    root = os.path.abspath(root)
    if under(root, REPO) or os.path.normcase(os.path.abspath(root)).startswith(os.path.normcase(REPO) + os.sep):
        raise FakeError(f"--root {root} is inside the repository; use a scratch folder")
    return root


def sha(path):
    return tb.sha256_file(refuse_label(path))


def sha_local_game_file(name):
    """SHA-256 of one of the two validation game files (seam_val _INPUT.json; nothing else under data/local)."""
    if name not in (VAL_Q, VAL_C):
        raise FakeError(f"{name}: only the two validation game files are hashed")
    return sha(os.path.join(LOCAL, name))


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return tb.write_json(path, obj)


def read_rows(path):
    csv.field_size_limit(10**9)
    with open(refuse_label(path), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        return next(r), list(r)


def fit_rows(part="fit"):
    import c10data   # training players only (split v1); local_dir=None: data/local is not opened
    return c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part=part, local_dir=None)


def sgf_key(sgf):
    return hashlib.sha1(sgf.encode("utf-8")).hexdigest()


# ================================================================ official (+ the --dup variant)
def dup_questions(root, rows_fit, m, log=print):
    """Give M candidates (Y) a second question from their unused black games, taking the question of M other
    candidates (X); per pool, deterministic (sha256 order with SALT)."""
    tests = os.path.join(root, "data", "test")
    games_of = defaultdict(list)
    for r in rows_fit:
        games_of[r["player_id"]].append(r)
    owner_of_sgf = {}
    for pid, gs in games_of.items():
        for g in gs:
            owner_of_sgf.setdefault(sgf_key(g["sgf_content"]), pid)
    info = {}
    for pool, tag in POOL_TAGS.items():
        prefix = tf.PREFIX[pool]
        qpath, cpath = os.path.join(tests, tb.PLAYER_Q_FILES[pool]), os.path.join(tests, tb.PLAYER_C_FILES[pool])
        hq, qrows = read_rows(qpath)
        _hc, crows = read_rows(cpath)
        sids = sorted({r[1] for r in crows})
        sid_of = {}
        for pid in games_of:
            s = f"{prefix}p_{tf._h(tf.SALT, tag, 'pid', pid)[:8]}"
            if s in sids:
                sid_of[pid] = s
        pid_of = {s: p for p, s in sid_of.items()}
        if len(pid_of) != len(sids):
            raise FakeError(f"{pool}: {len(sids) - len(pid_of)} candidate ids not found among the fit players")
        q_owner, q_k = {}, {}
        for i, r in enumerate(qrows):
            pid = owner_of_sgf.get(sgf_key(r[2]))
            if pid not in sid_of:
                raise FakeError(f"{pool}: question {r[0]} is not a pool candidate's game")
            q_owner[i], q_k[i] = sid_of[pid], int(r[1])
        k_of = {q_owner[i]: q_k[i] for i in q_owner}
        spare = {}
        for s in sids:
            bl = tf._blacks(games_of, pid_of[s], tag)
            spare[s] = bl[tb.GAMES_PER_PLAYER + k_of.get(s, 0):]
        order = sorted(sids, key=lambda s: tf._h(SALT, "dup", pool, s))
        with_q = set(q_owner.values())
        Y = [s for s in order if s in with_q and len(spare[s]) >= min(tf.OFFICIAL_KS)][:m]
        X = [s for s in order if s in with_q and s not in Y][:m]
        if len(Y) < m or len(X) < m:
            raise FakeError(f"{pool}: only {len(Y)} / {len(X)} candidates for --dup {m}")
        row_of = {q_owner[i]: i for i in q_owner}
        changed = []
        for x, y in zip(X, Y):
            i = row_of[x]
            k_new = max(k for k in tf.OFFICIAL_KS if k <= min(q_k[i], len(spare[y])))
            qrows[i] = tf._question_row(qrows[i][0], spare[y][:k_new], qrows[i][2 + tb.N_SLOTS + q_k[i]]
                                        if len(qrows[i]) > 2 + tb.N_SLOTS + q_k[i] else "")
            changed.append({"question_id": qrows[i][0], "from": x, "to": y, "k_old": q_k[i], "k_new": k_new})
        tf._write_csv(qpath, hq, qrows)
        owners = [q_owner[i] for i in q_owner]
        for c in changed:
            owners[[r[0] for r in qrows].index(c["question_id"])] = c["to"]
        info[pool] = {"m": m, "questions": len(qrows), "distinct_owners": len(set(owners)),
                      "candidates_without_question": len(set(sids) - set(owners)),
                      "candidates_with_two_questions": sum(1 for s in set(owners) if owners.count(s) == 2),
                      "changed": changed}
        log(f"dup {pool}: {m} candidates have two questions, {info[pool]['candidates_without_question']} none")
    errs = tb.quick_structure(tests, "player")
    if errs:
        raise FakeError(f"quick structure check failed after --dup: {errs[:3]}")
    return info


def cmd_official(a, log=print):
    root = check_root(a.root)
    t0 = time.time()
    rows_fit, rows_inner = fit_rows("fit"), fit_rows("inner")
    tf.from_train(root, rerun=a.rerun, rows_fit=rows_fit, rows_inner=rows_inner, log=log, shape="official",
                  slot_colours=a.slot_colours)
    dup = dup_questions(root, rows_fit, a.dup, log) if a.dup else None
    tree_p = os.path.join(root, "FAKE_TREE.json")
    tree = tb.read_json(tree_p)
    tree["t06"] = {"command": "t06_fake_tests.py official", "rows": "c10data.load_rows(part='fit' / 'inner', "
                   "local_dir=None)", "levels_order": "sorted rank labels (from_train with rows given)",
                   "dup": dup, "seconds_build": round(time.time() - t0, 1)}
    write_json(tree_p, tree)
    return cmd_add_dumps(a, log=log, rows_fit=rows_fit)


# ================================================================ add-dumps
def fake_vector(pid, sgf, n_moves):
    """A fixed-seed stand-in: (player mean + game noise) * n_windows, n_windows = max(1, own moves)."""
    nw = max(1, (n_moves + 1) // 2)
    mu = np.random.default_rng(int(tf._h(SALT, "p", pid)[:16], 16)).normal(size=256)
    e = mu + np.random.default_rng(int(tf._h(SALT, "g", sgf_key(sgf))[:16], 16)).normal(size=256)
    return (e * nw).astype(np.float32), nw


class Embeddings:
    def __init__(self, path, rows_fit):
        self.ok = path is not None and os.path.exists(path)
        self.path = path
        self.idx, self.sha = {}, None
        self.found = self.fallback = self.ambiguous = 0
        if not self.ok:
            return
        z = np.load(path, allow_pickle=False)
        if str(z["weights_sha256"]) != WEIGHTS_SHA256:
            raise FakeError(f"{path}: weights_sha256 {z['weights_sha256']} is not the prereg01 player CNN")
        self.emb, self.nw = z["emb_sum"], z["n_windows"]
        at = {(str(r), str(g)): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
        seen = defaultdict(list)
        for r in rows_fit:
            if r["color"] == "B":
                seen[sgf_key(r["sgf_content"])].append((r["rank"], r["game_id"]))
        for k, lst in seen.items():
            hits = [at[x] for x in lst if x in at]
            if hits:
                self.idx[k] = hits[0]
                self.ambiguous += len(hits) > 1
        self.sha = tb.sha256_file(path)

    def get(self, pid, sgf):
        i = self.idx.get(sgf_key(sgf)) if self.ok else None
        if i is None:
            self.fallback += 1
            return fake_vector(pid, sgf, len(tf.re.findall(r";[BW]\[[a-s]{2}\]", sgf)))
        self.found += 1
        return self.emb[i].astype(np.float32), int(self.nw[i])


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrays)
    os.replace(path + ".tmp", path)
    return path


def player_dumps(tests, res_player, emb):
    info = {}
    src = os.path.join(P02_PLAYER, "dumps")
    d = os.path.join(res_player, "seam_val", "dumps")
    os.makedirs(d, exist_ok=True)
    for f in tf.PLAYER_DUMPS:
        shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
    vq, vc = sha_local_game_file(VAL_Q), sha_local_game_file(VAL_C)
    write_json(os.path.join(d, "_INPUT.json"), {"set": "seam_val", "question_file": VAL_Q, "sha256": vq,
                                                 "candidates_file": VAL_C, "candidates_sha256": vc})
    info["seam_val"] = {"questions_file": VAL_Q, "questions_sha256": vq, "candidates_file": VAL_C,
                        "candidates_sha256": vc, "dumps": "byte copies of results/prereg02/player/dumps"}
    for pool in ("public", "private"):
        qp, cp = os.path.join(tests, tb.PLAYER_Q_FILES[pool]), os.path.join(tests, tb.PLAYER_C_FILES[pool])
        _h, crows = read_rows(cp)
        hq, qrows = read_rows(qp)
        ce, cn = zip(*[emb.get(r[1], r[2]) for r in crows])
        cand = {"game_id": np.array([r[0] for r in crows]), "player_id": np.array([r[1] for r in crows]),
                "color": np.array([r[3] for r in crows]), "emb_sum": np.vstack(ce).astype(np.float32),
                "n_windows": np.array(cn, np.int32), "weights_sha256": np.array(WEIGHTS_SHA256)}
        qi, qs, qc, qe, qn = [], [], [], [], []
        col0 = hq.index("color_1")
        for r in qrows:
            for s in range(1, int(r[1]) + 1):
                e, n = emb.get("q|" + r[0], r[1 + s])
                qi.append(r[0])
                qs.append(s)
                qc.append(r[col0 + s - 1])
                qe.append(e)
                qn.append(n)
        query = {"question_id": np.array(qi), "slot": np.array(qs, np.int32), "color": np.array(qc),
                 "emb_sum": np.vstack(qe).astype(np.float32), "n_windows": np.array(qn, np.int32),
                 "parsed": np.ones(len(qi), bool), "weights_sha256": np.array(WEIGHTS_SHA256)}
        d = os.path.join(res_player, pool, "dumps")
        _save_npz(os.path.join(d, "cand_games.npz"), cand)
        _save_npz(os.path.join(d, "query_games.npz"), query)
        write_json(os.path.join(d, "_INPUT.json"),
                   {"set": pool, "question_file": tb.PLAYER_Q_FILES[pool], "sha256": sha(qp),
                    "candidates_file": tb.PLAYER_C_FILES[pool], "candidates_sha256": sha(cp)})
        info[pool] = {"questions_file": tb.PLAYER_Q_FILES[pool], "questions_sha256": sha(qp),
                      "candidates_file": tb.PLAYER_C_FILES[pool], "candidates_sha256": sha(cp),
                      "rows": len(qrows), "games": len(qi), "candidate_rows": len(crows)}
    return info


def cmd_add_dumps(a, log=print, rows_fit=None):
    root = check_root(a.root)
    tests = os.path.join(root, "data", "test")
    res = os.path.join(root, "results", "test02", "player")
    for n in tb.ALL_NAMES:
        if not os.path.exists(os.path.join(tests, n)):
            raise FakeError(f"{os.path.join(tests, n)} missing: build the tree first (official, or t02_fake_tests "
                            f"from-train --shape official)")
    if os.path.exists(res) and not a.rerun:
        raise FakeError(f"{res} exists; pass --rerun")
    shutil.rmtree(res, ignore_errors=True)
    emb_path = None if getattr(a, "no_emb", False) else (getattr(a, "emb", None) or EMB)
    if emb_path and os.path.exists(emb_path) and rows_fit is None:
        rows_fit = fit_rows("fit")
    emb = Embeddings(emb_path, rows_fit or [])
    files = {n: tb.file_record(os.path.join(tests, n)) for n in tb.ALL_NAMES}
    mpath, _man = tb.write_manifest(tests, "rehearsal", files, {"rank": "PASS", "player": "PASS"},
                                    t1_basis="t06_fake_tests add-dumps (structure checked by the builder)",
                                    fake=FAKE_NOTE)
    test_bundle = {"zip_sha256": None, "manifest_sha256": sha(mpath), "kind": "rehearsal",
                   "t1": {"rank": "PASS", "player": "PASS"}, "files": {n: files[n]["sha256"] for n in tb.ALL_NAMES}}
    sets = player_dumps(tests, res, emb)
    repro = dict(tb.read_json(os.path.join(P02_PLAYER, "repro_check.json")), fake=FAKE_NOTE)
    seam_p02 = {"fake": FAKE_NOTE, "smoke": False, "bitwise_equal": True, "max_abs_emb_diff": 0.0,
                "files": {f: {"bitwise_equal": True, "max_abs_emb_diff": 0.0,
                              "sha256": sha(os.path.join(res, "seam_val", "dumps", f)),
                              "reference_sha256": sha(os.path.join(P02_PLAYER, "dumps", f))} for f in tf.PLAYER_DUMPS},
                "rule": "record only; the local seam (t02_player.py run --pool seam_val) decides"}
    tf._records(res, "player", test_bundle, sets, {"path": tf.PLAYER_WEIGHTS_PATH, "sha256": WEIGHTS_SHA256},
                repro.get("gpu"), repro.get("torch"), repro.get("cudnn"),
                {"repro_check.json": repro, "seam_p02.json": seam_p02})
    tree_p = os.path.join(root, "FAKE_TREE.json")
    tree = tb.read_json(tree_p) if os.path.exists(tree_p) else {"kind": "unknown"}
    tree.setdefault("t06", {})["add_dumps"] = {
        "embeddings": {"path": emb_path, "used": emb.ok, "sha256": emb.sha, "games_found": emb.found,
                       "games_fixed_seed": emb.fallback, "sgf_matching_two_or_more_rows": emb.ambiguous},
        "sets": sets, "manifest_sha256": test_bundle["manifest_sha256"], "created": tb.now_iso(),
        "rule": "seam_val = prereg02 dumps (byte copies); public / private from the real CNN embedding of each "
                "fit black game (spec 4) when present, else a fixed-seed vector"}
    write_json(tree_p, tree)
    log(f"add-dumps: {root}: embeddings {'black_ge105.npz' if emb.ok else 'fixed-seed'} (found {emb.found}, "
        f"fixed-seed {emb.fallback}); MANIFEST {test_bundle['manifest_sha256'][:12]}")
    return tree


# ================================================================ fake-t2
def _keys(sgf):
    import p02_moves
    import t02_checks
    pts = [p for _c, p in p02_moves.MOVE_RE.findall(sgf)]
    return t02_checks.game_keys(pts)[0] if pts else None


def cmd_fake_t2(a, log=print):
    import t02_checks
    root = check_root(a.root)
    tests = os.path.join(root, "data", "test")
    out = os.path.join(root, "results", "test02", "checks", "player", "t2.json")
    if os.path.exists(out) and not a.rerun:
        raise FakeError(f"{out} exists; pass --rerun --reason TEXT")
    games = {}
    for pool in ("public", "private"):
        _hq, qrows = read_rows(os.path.join(tests, tb.PLAYER_Q_FILES[pool]))
        _hc, crows = read_rows(os.path.join(tests, tb.PLAYER_C_FILES[pool]))
        games[(pool, "questions")] = [(r[0], _keys(r[1 + s])) for r in qrows for s in range(1, int(r[1]) + 1)]
        games[(pool, "candidates")] = [(r[1], _keys(r[2])) for r in crows]
    names = {(p, "questions"): tb.PLAYER_Q_FILES[p] for p in ("public", "private")}
    names.update({(p, "candidates"): tb.PLAYER_C_FILES[p] for p in ("public", "private")})
    units = []

    def unit(uid, what, n, copies, extra):
        thr = t02_checks.t2_threshold(n)
        if uid in (a.fail or []):
            copies, extra = max(copies, thr), dict(extra, forced_fail="t06_fake_tests fake-t2 --fail %s (drill)" % uid)
        units.append(dict({"unit": uid, "what": what, "games_in_file": n, "threshold": thr, "copies": copies,
                           "verdict": "FAIL" if copies >= thr else "PASS"}, **extra))
    for key in (("public", "questions"), ("public", "candidates"), ("private", "questions"),
                ("private", "candidates")):
        unit("a", f"{names[key]} vs training players (stand-in)", len(games[key]), 0,
             {"file": names[key], "stand_in": "the drill tree's games are fit games; (a) set to 0 copies"})
    for pool in ("public", "private"):
        cfull = {k for _o, k in games[(pool, "candidates")]} - {None}
        q = games[(pool, "questions")]
        unit("b", f"{names[(pool, 'questions')]} vs {names[(pool, 'candidates')]} (full game)", len(q),
             sum(1 for _o, k in q if k is not None and k in cfull), {"pool": pool})
    for pool in ("public", "private"):
        occ = defaultdict(list)
        for o, k in games[(pool, "questions")]:
            if k is not None:
                occ[k].append(o)
        dup = sum(sum(1 for o in lst if any(o2 != o for o2 in lst)) for lst in occ.values() if len(lst) > 1)
        unit("c", f"{names[(pool, 'questions')]}: games shared by two or more questions (full game)",
             len(games[(pool, "questions")]), dup, {"file": names[(pool, "questions")]})
    pub = {k for key, lst in games.items() if key[0] == "public" for _o, k in lst} - {None}
    priv = [k for key, lst in games.items() if key[0] == "private" for _o, k in lst]
    unit("d", "private games found among the public games (player; full game)", len(priv),
         sum(1 for k in priv if k is not None and k in pub), {})
    verdict = "FAIL" if any(u["verdict"] == "FAIL" for u in units) else "PASS"
    rec = {"format": T2_FORMAT, "comp": "player", "fake": True, "fake_note": T2_NOTE, "verdict": verdict,
           "units": units, "forced_fail": list(a.fail or []), "reason": a.reason, "created": tb.now_iso(),
           "inputs": {names[k]: sha(os.path.join(tests, names[k])) for k in names},
           "provenance": {"mode": "dry-run", "tool": "tools/t06_fake_tests.py"}}
    write_json(out, rec)
    log(f"fake-t2: {out}: {verdict}; " + ", ".join(f"{u['unit']}:{u.get('file') or u.get('pool') or ''} "
                                                   f"{u['copies']}" for u in units if u["unit"] != "a"))
    return rec


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("official")
    p.add_argument("--root", required=True)
    p.add_argument("--dup", type=int, default=0)
    p.add_argument("--slot-colours", choices=tf.SLOT_COLOURS, default="empty")
    p.add_argument("--emb", default=None)
    p.add_argument("--no-emb", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("add-dumps")
    p.add_argument("--root", required=True)
    p.add_argument("--emb", default=None)
    p.add_argument("--no-emb", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("fake-t2")
    p.add_argument("--root", required=True)
    p.add_argument("--fail", action="append", choices=("a", "b", "c", "d"))
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    if getattr(a, "rerun", False) and a.cmd == "fake-t2" and not (a.reason or "").strip():
        raise FakeError("--rerun needs --reason TEXT")
    return {"official": cmd_official, "add-dumps": cmd_add_dumps, "fake-t2": cmd_fake_t2}[a.cmd](a)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
