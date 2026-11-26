"""Fake trees and drills for the 10 test-day kit (docs/PREREG_10_PLAYER_GUARD_P1.md 4.2; competition 1 only).

    PY tools/t10_fake_tests.py nov4  --root S [--nov4-pools public,private] [--white-question POOL] [--rerun]
    PY tools/t10_fake_tests.py drill --root S --kind level0|fail_opening|t06_seam_failed --name N
                                     [--pool public|private]

nov4   the official-shape tree of tools/t06_fake_tests.py official (tools/t02_fake_tests.py from-train --shape official
       on the fit / inner rows of c10data.load_rows(local_dir=None): 400 fit players with >= 120 black games per pool,
       100 black candidate games each, one black question each of 5 / 10 / 15 / 20 games), then, in each pool named by
       --nov4-pools (default both), every question is rebuilt "nov4" style (PREREG_10 0 / 2.2, the exploration's
       x1_scenarios rule for one question per candidate): the owner's black games after the 100 candidate games
       whose first 4 plies, canonicalised under the 8 board symmetries (p02_moves.canon on 4 plies, the key of
       results/prereg10/explore/x0_open_keys.py), never occur among the owner's 100 candidate games, in the same
       hash order; k = the largest official size <= min(the question's k, the owner's novel games); an owner with
       fewer than 5 novel games keeps the original question (recorded). All games stay black and fit, so test02's
       b_c and b_q stay exactly 1 and its rho falls below theta2 (guard level 3 by rho): the pool A of PREREG_10 1.1
       is meant for. --white-question POOL (the negative drill "b_q != 1"): in that pool, the last game of the first
       rebuilt question (hash order) is replaced by one of the owner's white fit games outside the candidate games,
       colour W, so b_q < 1 while the pool still falls to level 3 by rho. Then tools/t06_fake_tests.py add-dumps (the
       real CNN embedding of each black fit game from data/prereg06/cnn_fit/black_ge105.npz; the white game gets the
       fixed-seed stand-in, counted in FAKE_TREE.json). FAKE_TREE.json gets a "t10" record.
drill  scratch copies for the negative drills of PREREG_10 4.2 on a tree where test02 and 06 already ran (the
       layout of tests/test_t06_dryrun.py: S/out/player = test02's player outputs, S/results/test02/checks, S/o6 = the
       06 outputs): S/drills/<name>/{o2,checks,o6} with
         level0          test02's apply_report.json of --pool set to a CNN-only selection (Q00 effective and applied,
                         guard level 0, reason 'exempt: CNN-only combination'); the 06 apply has to run again on it
         fail_opening    t1.json verdict FAIL_OPENING
         t06_seam_failed t06_seam.json verdict not_used_dumps_differ, passed false (the registered "06 not used")
       and prints the folders as JSON. b_q != 1 is a real tree (nov4 --white-question); the B GPU seam drill is the
       stub t10_player.py p1 (B not adopted: exit 3) until PREREG_10 B is adopted.
--root must be outside the repository. Nothing here opens a label file, runs tools/score.py or imports a
competition-2 tool.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t02_fake_tests as tf  # noqa: E402
import t02_test_bundle as tb  # noqa: E402
import t06_fake_tests as t6f  # noqa: E402
import p02_moves as pm  # noqa: E402

SALT = "c10-t10fake"
NOV_PLIES = 4
POOL_TAGS = t6f.POOL_TAGS                       # public posA, private posB (tools/t02_fake_tests.py)
DRILLS = ("level0", "fail_opening", "t06_seam_failed")
_LABEL = "ans" + "wer"


class FakeError(SystemExit):
    def __init__(self, msg):
        super().__init__(f"t10_fake_tests: {msg}")


def open_key(sgf):
    """First NOV_PLIES plies canonicalised under the 8 symmetries (x0_open_keys.py 'c4')."""
    mv = pm.parse_moves(sgf, 12)
    return pm.points_string(pm.canon(mv[:NOV_PLIES], NOV_PLIES))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


# ================================================================ nov4
def nov4_questions(root, rows_fit, pools, white_pool=None, log=print):
    """Rebuild the question files of `pools` nov4 style (module docstring). -> {pool: info}."""
    tests = os.path.join(root, "data", "test")
    games_of = defaultdict(list)
    for r in rows_fit:
        games_of[r["player_id"]].append(r)
    owner_of_sgf = {}
    for pid, gs in games_of.items():
        for g in gs:
            owner_of_sgf.setdefault(t6f.sgf_key(g["sgf_content"]), pid)
    info = {}
    for pool in pools:
        tag = POOL_TAGS[pool]
        prefix = tf.PREFIX[pool]
        qpath, cpath = os.path.join(tests, tb.PLAYER_Q_FILES[pool]), os.path.join(tests, tb.PLAYER_C_FILES[pool])
        hq, qrows = t6f.read_rows(qpath)
        _hc, crows = t6f.read_rows(cpath)
        sids = sorted({r[1] for r in crows})
        sid_of = {}
        for pid in games_of:
            s = f"{prefix}p_{tf._h(tf.SALT, tag, 'pid', pid)[:8]}"
            if s in sids:
                sid_of[pid] = s
        pid_of = {s: p for p, s in sid_of.items()}
        if len(pid_of) != len(sids):
            raise FakeError(f"{pool}: {len(sids) - len(pid_of)} candidate ids not found among the fit players")
        cand_keys = defaultdict(set)
        for r in crows:
            cand_keys[r[1]].add(open_key(r[2]))
        col0 = hq.index("color_1")
        n_new, n_kept, ks_new, changed = 0, 0, defaultdict(int), []
        for i, r in enumerate(qrows):
            pid = owner_of_sgf.get(t6f.sgf_key(r[2]))
            if pid not in sid_of:
                raise FakeError(f"{pool}: question {r[0]} is not a pool candidate's game")
            s = sid_of[pid]
            k = int(r[1])
            blacks = tf._blacks(games_of, pid, tag)
            rest = blacks[tb.GAMES_PER_PLAYER:]
            nov = [g for g in rest if open_key(g["sgf_content"]) not in cand_keys[s]]
            fits = [x for x in tf.OFFICIAL_KS if x <= min(k, len(nov))]
            if not fits:
                n_kept += 1
                ks_new[k] += 1
                continue
            qrows[i] = tf._question_row(r[0], nov[:fits[-1]], "")       # slot colours after num_games: empty
            n_new += 1
            ks_new[fits[-1]] += 1
            changed.append({"question_id": r[0], "owner": s, "k_old": k, "k_new": fits[-1], "novel_games": len(nov)})
        white = None
        if pool == white_pool:
            order = sorted(changed, key=lambda c: tf._h(SALT, "white", pool, c["question_id"]))
            for c in order:
                pid = pid_of[c["owner"]]
                used = {g["game_id"] for g in tf._blacks(games_of, pid, tag)[:tb.GAMES_PER_PLAYER]}
                whites = sorted((g for g in games_of[pid] if g["color"] == "W" and g["game_id"] not in used),
                                key=lambda g: tf._h(SALT, "wg", pid, g["game_id"]))
                if not whites:
                    continue
                i = [r[0] for r in qrows].index(c["question_id"])
                k = int(qrows[i][1])
                qrows[i][1 + k] = whites[0]["sgf_content"]
                qrows[i][col0 + k - 1] = "W"
                white = {"question_id": c["question_id"], "owner": c["owner"], "slot": k,
                         "white_game_id": whites[0]["game_id"]}
                break
            if white is None:
                raise FakeError(f"{pool}: no rebuilt question whose owner has a spare white game")
        tf._write_csv(qpath, hq, qrows)
        n_slots = sum(int(r[1]) for r in qrows)
        info[pool] = {"rule": "nov4: owner's black games after the 100 candidates whose canonical first 4 plies never "
                              "occur among the owner's candidates; k = largest official size <= min(k, novel games)",
                      "questions": len(qrows), "rebuilt": n_new, "kept_original_fewer_than_5_novel": n_kept,
                      "num_games": {str(x): ks_new[x] for x in sorted(ks_new)}, "question_games": n_slots,
                      "white_question": white,
                      "expected_b_q": 1.0 if white is None else (n_slots - 1) / n_slots}
        log(f"nov4 {pool}: {n_new} questions rebuilt, {n_kept} kept (fewer than 5 novel games), k {info[pool]['num_games']}"
            + (f", white game in {white['question_id']} slot {white['slot']}" if white else ""))
    errs = tb.quick_structure(tests, "player")
    if errs:
        raise FakeError(f"quick structure check failed after nov4: {errs[:3]}")
    return info


def cmd_nov4(a, log=print):
    root = t6f.check_root(a.root)
    pools = [p.strip() for p in (a.nov4_pools or "public,private").split(",") if p.strip()]
    if not pools or set(pools) - set(POOL_TAGS):
        raise FakeError(f"--nov4-pools {a.nov4_pools!r}: public and / or private")
    if a.white_question and a.white_question not in pools:
        raise FakeError("--white-question names a pool that is not rebuilt nov4")
    t0 = time.time()
    rows_fit, rows_inner = t6f.fit_rows("fit"), t6f.fit_rows("inner")
    tf.from_train(root, rerun=a.rerun, rows_fit=rows_fit, rows_inner=rows_inner, log=log, shape="official",
                  slot_colours="empty")
    info = nov4_questions(root, rows_fit, pools, a.white_question, log)
    tree_p = os.path.join(root, "FAKE_TREE.json")
    tree = tb.read_json(tree_p)
    tree["t06"] = {"command": "t10_fake_tests.py nov4 (t06_fake_tests official, then the nov4 rebuild)",
                   "rows": "c10data.load_rows(part='fit' / 'inner', local_dir=None)", "dup": None}
    tree["t10"] = {"command": "t10_fake_tests.py nov4", "nov4_pools": pools, "white_question_pool": a.white_question,
                   "pools": info, "seconds_build": round(time.time() - t0, 1),
                   "expected": "test02: b_c = 1 in both pools; b_q = 1 in the nov4 pools without a white game; rho "
                               "below theta2 -> guard level 3 by rho (PREREG_10 1.1)"}
    write_json(tree_p, tree)
    ns = argparse.Namespace(root=root, emb=None, no_emb=False, rerun=True)
    return t6f.cmd_add_dumps(ns, log=log, rows_fit=rows_fit)


# ================================================================ drill
def cmd_drill(a, log=print):
    root = t6f.check_root(a.root)
    if a.kind not in DRILLS:
        raise FakeError(f"--kind {a.kind!r}: one of {DRILLS}")
    base = os.path.join(root, "drills", a.name)
    shutil.rmtree(base, ignore_errors=True)
    out = {"o2": os.path.join(base, "o2"), "checks": os.path.join(base, "checks"), "o6": os.path.join(base, "o6")}
    shutil.copytree(os.path.join(root, "out", "player"), out["o2"])
    shutil.copytree(os.path.join(root, "results", "test02", "checks"), out["checks"])
    shutil.copytree(os.path.join(root, "o6"), out["o6"])
    if a.kind == "level0":
        import p02_player_fuse as pf
        p = os.path.join(out["o2"], a.pool, "apply_report.json")
        rep = read_json(p)
        q00 = dict(pf.COMBO_BY_ID["Q00"])
        rep.update(selected=q00, effective=q00, applied=q00, guard_level=0,
                   guard_reason="exempt: CNN-only combination", guard_changed_combination=False)
        write_json(p, rep)
        for pool in ("public", "private"):           # the 06 apply runs again on this drill (its inputs changed)
            for n in ("t06_pool.json", "top5_off.csv", "top5_on.csv"):
                f = os.path.join(out["o6"], pool, n)
                if os.path.exists(f):
                    os.remove(f)
        for n in ("off", "on", "not_used.json"):
            f = os.path.join(out["o6"], n)
            if os.path.isdir(f):
                shutil.rmtree(f)
            elif os.path.exists(f):
                os.remove(f)
    elif a.kind == "fail_opening":
        p = os.path.join(out["checks"], "player", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_json(p, t1)
    else:
        p = os.path.join(out["o6"], "seam_val", "t06_seam.json")
        s = read_json(p)
        s.update(verdict="not_used_dumps_differ", passed=False)
        write_json(p, s)
    rec = dict(out, kind=a.kind, pool=a.pool if a.kind == "level0" else None)
    write_json(os.path.join(base, "DRILL.json"), rec)
    log(json.dumps(rec, ensure_ascii=False))
    return rec


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nov4")
    p.add_argument("--root", required=True)
    p.add_argument("--nov4-pools", default="public,private")
    p.add_argument("--white-question", choices=("public", "private"), default=None)
    p.add_argument("--rerun", action="store_true")
    p = sub.add_parser("drill")
    p.add_argument("--root", required=True)
    p.add_argument("--kind", required=True, choices=DRILLS)
    p.add_argument("--name", required=True)
    p.add_argument("--pool", choices=("public", "private"), default="public")
    a = ap.parse_args(argv)
    if _LABEL in os.path.abspath(a.root).lower():
        raise FakeError("refusing a root whose path holds the label-file word")
    return {"nov4": cmd_nov4, "drill": cmd_drill}[a.cmd](a)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
