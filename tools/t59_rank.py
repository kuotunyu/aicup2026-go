"""C10 test59, competition 2 (rank, TEAM_11248): the 59 test-day local tool (Q59 = Q25's seven terms + OPP2, the
opponent term of prereg59). Registration: docs/PREREG_59_OPPONENT_IDENTITY.md section 4. CPU only, .venv, numpy.

    PY tools/t59_rank.py apply [--out DIR] [--rerun --reason TEXT]
    (PY = .venv/Scripts/python.exe; Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL
     thread variable)

--out DIR  a rehearsal: DIR outside the repository, git state not checked. Without it: results/test59 (git clean).

tools/t25_rank.py (and through it t19 / t09 / t08 / t07 / t04 / t02) and results/prereg59/explore/p59_lib.py are
imported, never edited. Like t25 this tool binds the committed official runs and never opens, lists or hashes anything
under data/local; it never imports torch, never fits anything, never runs tools/score.py, never reads competition 1,
never uploads and never calls tools/official_log.py.

apply, in this order (the first failing item decides):
  prerequisites: git clean (real); no BLAS thread variable; K6-59 (2026-11-23 12:00 +08:00); the prereg59 files equal
      their pinned SHA-256s; the prereg25 pins / adoption / .venv versions, 19's official run and the frozen 09 / 07 /
      04 readers exactly as t25 checks them (t25's own K6 is not consulted: 25's run exists and is bound below); 25's
      official run (results/test25/rank/t25_apply_report.json: real, exit 0, usable, committed, outputs unchanged).
  1  the seven terms of public / private recomputed with t25's functions.
     seam A: Q25 recomputed = results/test25/rank/rank_test_probs.csv byte for byte (else 4).
  2  the opponent-side S0 vectors of the test games (data/prereg58/emb_opp_rank/<tag>/pool_<set>.npz): bound to the
     official raw question files (seam O: the embedding's source file has the raw file's SHA-256 and no candidate row;
     5,000 games; S0's checkpoint; every game's sample count = the opponent's non-pass moves after the first 7 plies;
     the total number of moves; else 4); HD group and the player's colour from the raw file (colour = val_keys).
  3  OPP2 per game and per question (p59_lib; kernel.json's M, tau1, tau2, b; no class left out).
     seam B: the frozen Q25 theta with weight 0 for OPP2 through the 8-term path gives
     results/test25/rank/submission_rank.csv byte for byte (else 4).
  4  Q59 with the frozen 8-term theta -> rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json (decide.py's
     file plus the key "prereg59_record"), submission_rank.csv (= pred_B.csv), t59_apply_report.json;
     check_submission --official (else 4, outputs removed).
Exit codes: 0 done; 2 refused or error; 4 a seam or check failed (stop and find out why).
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
P59_EXPLORE = os.path.join(REPO, "results", "prereg59", "explore")
for _p in (HERE, P59_EXPLORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t25_rank as t25  # noqa: E402
import p59_lib as K59  # noqa: E402  (numpy only; imports p58_lib)

t19, t9 = t25.t19, t25.t9
t8, t7, t4, pf7, r8, rf, t2 = t25.t8, t25.t7, t25.t4, t25.pf7, t25.r8, t25.rf, t25.t2
decide, check_submission = t25.decide, t25.check_submission

REAL_R59 = os.path.join(REPO, "results", "test59")
TAG = "V2_step_320000_raw_tta1"
OPP_DIR = os.path.join(REPO, "data", "prereg58", "emb_opp_rank", TAG)
SRC_DIR = os.path.join(REPO, "data", "prereg56", "src_rank")
RAW_DIR = os.path.join(REPO, "data", "test_v2_download", "rank")
PINS = {  # .json: SHA-256 with CRLF folded to LF; .npz: raw bytes
    "results/prereg59/rank/frozen_config.json": "c5e68b3f3fc5406fd4c4da9fa4ac1a2f6e588a5acb8425daab1bbf4dbf4bff71",
    "results/prereg59/rank/prototypes.npz": "405ca18344bcc10254a04100016825a5e2d53a0f3f56adfd5eab176ef6bf65aa",
    "results/prereg59/rank/s0_head.npz": "66bb7d35fa187da47609f05cc8a8bb4c8544f75f2be1f0be0cab3f298f4ab723",
    "results/prereg59/explore/kernel.json": "630afc9dda5b145e7d97c9f79b2cabb263114c73f83044b350e1895068a94b93"}
TERMS8 = t25.TERMS + ["OPP2"]
K6_59 = dt.datetime(2026, 11, 23, 12, 0, 0, tzinfo=t25.TW)
INNER_MEAN_MAX_Q = 0.5803                     # eval.json term.mean_max_q_record (4 decimals), the shift reference
REVIEW_MAX_Q = 0.05                           # PREREG_59 section 4: review lines (not automatic)
REVIEW_CHANGED = 100
TEST_SETS = t25.TEST_SETS
OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
           "t59_apply_report.json")
_HD = re.compile(r"HD\[(\d+)\]")
_MOVE = re.compile(r";([BW])\[([a-s]{2})\]")


class T59Error(RuntimeError):
    def __init__(self, message, code=2, details=None):
        super().__init__(message)
        self.code, self.details = code, details


def refuse(msg):
    raise T59Error(msg, 2)


def drift(msg, details=None):
    raise T59Error(msg, 4, details)


def no_local(path):
    if pf7.under_data_local(os.path.abspath(path)):
        refuse(f"refusing {path}: data/local is never opened")
    return path


def sha_raw(path):
    return t2.sha256_file(no_local(path))


def sha_lf(path):
    return t2.sha256_text(no_local(path))


def path_of(rel):
    return os.path.join(REPO, *rel.split("/"))


def read_json(path):
    with open(no_local(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t4.jsonable(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


# ================================================================ prerequisites
def check_pins():
    got = {r_: ((sha_lf if r_.endswith(".json") else sha_raw)(path_of(r_)) if os.path.exists(path_of(r_)) else None)
           for r_ in PINS}
    bad = sorted(r_ for r_ in PINS if got[r_] != PINS[r_])
    if bad:
        refuse(f"prereg59 files differ from the pinned SHA-256s: {bad}")
    fc = read_json(path_of("results/prereg59/rank/frozen_config.json"))
    par = fc.get("params") or {}
    if fc.get("format") != "c10-prereg59 rank frozen_config v1" or par.get("terms") != TERMS8 or \
            len(par.get("theta") or []) != 2 + len(TERMS8) + rf.N_LEVELS - 1:
        refuse("results/prereg59/rank/frozen_config.json is not the frozen Q59 configuration")
    k = read_json(path_of("results/prereg59/explore/kernel.json"))
    term = fc["term"]
    if (term["tau1"], term["tau2"], term["b"]) != (k["tau1"], k["tau2"], k["b"]) or term["scale"] != K59.SCALE or \
            term["kernel_sha256_lf"] != PINS["results/prereg59/explore/kernel.json"]:
        refuse("frozen_config.json's term is not kernel.json's")
    M = np.asarray(k["M"], np.float64)
    if M.shape != (K59.N_HD, K59.N_COL, K59.L, K59.L) or not np.allclose(M.sum(axis=3), 1.0):
        refuse("kernel.json: not a [3, 2, 10, 10] table")
    return fc, M, (int(k["tau1"]), int(k["tau2"]), int(k["b"])), got


def t25_state(ctx, real):
    """25's official run: real, exit 0, usable, Q25, committed, its outputs unchanged."""
    rank25 = ctx["rank25"]
    rp = os.path.join(rank25, "t25_apply_report.json")
    if not os.path.exists(rp):
        refuse("test25 apply has not finished (no t25_apply_report.json)")
    rep = read_json(rp)
    if rep.get("format") != t25.FORMATS["apply"] or rep.get("mode") != "real" or rep.get("exit_code") != 0 or \
            rep.get("usable") is not True or (rep.get("combo") or {}).get("id") != "Q25":
        refuse(f"{rp}: not a usable real Q25 run")
    files = [rp] + [os.path.join(rank25, n) for n in t25.TEST_OUTPUTS]
    if real:
        t25.require_committed(files, "results/test25")
    want = rep.get("outputs_sha256") or {}
    changed = [n for n in t25.TEST_OUTPUTS if sha_raw(os.path.join(rank25, n)) != want.get(n)]
    if changed:
        refuse(f"test25 outputs differ from its report: {changed}")
    return rep, {"t25_apply_report_sha256_lf": sha_lf(rp), "submission_sha256": want.get("submission_rank.csv"),
                 "probs_sha256": want.get("rank_test_probs.csv")}


# ================================================================ the seven terms (t25's step 2, same calls)
def seven_terms(ctx, a, fc25, models25, readers, rep19, t0):
    fc9, mp, fc07, fc04, model07 = readers
    call = t25.call
    rep8 = t25.read_json(os.path.join(ctx["rank"], "t08_apply_report.json"))
    ctx9 = dict(ctx, mode=rep19["mode"], real=rep19["mode"] == "real")
    a9 = t9.Apply(ctx9, a, t0)
    p1 = call(a9.p1_sets, rep8)
    for s in t25.SETS:
        want = (rep19.get("p1_sets") or {}).get(s) or {}
        got = a9.rec["p1_sets"][s]
        if (want.get("report_sha256"), want.get("npz_sha256")) != (got["report_sha256"], got["npz_sha256"]):
            refuse(f"{s}: 08's P1 set differs from the one 19's report records")
    qfiles = {s: os.path.join(ctx["T"], t2.TEST_FILES[s]) for s in TEST_SETS}
    opening = t25.load_opening(ctx, qfiles)
    vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in TEST_SETS}
    sw = {s: call(t7.load_sw, ctx, s, vk[s]) for s in TEST_SETS}
    o4z = {s: t25.load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in TEST_SETS}
    c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
    PF = pf7.import_pilot_fuse()
    hs = {s: t25.load_npz(os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")) for s in TEST_SETS}
    out = {}
    for s in TEST_SETS:
        ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
        c5 = call(t7.t4call, t4.load_c5, s, ddir, vk[s], c5w)
        f = call(t7.t4call, t4.fuse_set, s, vk[s], qfiles[s], c5, opening, o4z[s], sw[s], fc04, False)
        f["c5_sha256"] = c5["sha256"]
        f.update(t7.port_c5(vk[s], c5, fc04, f))
        if not f["port_equal"]:
            drift(f"{s}: the ported C5 lines do not give fuse_set's Q02 bit for bit")
        z = p1[s]["z"]
        if not (list(map(str, z["question_id"])) == vk[s]["question_id"]
                and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"]
                and list(map(str, hs[s]["question_id"])) == vk[s]["question_id"]
                and [int(x) for x in hs[s]["slot"]] == vk[s]["slot"]):
            refuse(f"{s}: 08's P1 / 07's HS keys differ from test02's val_keys.npz")
        h = call(t7.q02h_set, f, vk[s], hs[s], model07, fc07, PF)
        lpE, okE = t9.set_features(p1[s], mp)
        us = np.asarray(vk[s]["parsed"], dtype=bool) & okE
        q9 = call(t8.q08_core, f, vk[s], h, lpE, okE, us, fc9)
        E25, _lp25, has25 = call(t19.set_term_of, f, vk[s], p1[s], us, models25, mp)
        res25 = call(t19.q19_core, f, vk[s], h, q9, E25, has25, fc25)
        Es7 = [f["E"][t] for t in r8.Q02_TERMS] + [h["E_hs"], q9["E_p1"], E25]
        out[s] = {"fz": f, "vk": vk[s], "Es7": Es7, "res25": res25}
    return out, qfiles


# ================================================================ the opponent term on the test games
def raw_games(s):
    """The official raw question file of one set -> (path, [question ids], per game: question index, HD, colour,
    number of the opponent's non-pass moves, number of all non-pass moves), games in file order."""
    path = os.path.join(RAW_DIR, t2.TEST_FILES[s])
    csv.field_size_limit(10 ** 9)
    qids, qi, hd, col, nopp, nall = [], [], [], [], [], []
    with open(no_local(path), newline="", encoding="utf-8-sig") as f:
        for k, r in enumerate(csv.DictReader(f)):
            qids.append(r["question_id"])
            for i in range(1, int(r["num_games"]) + 1):
                sgf, c = r[f"sgf_{i}"], r[f"color_{i}"].strip()
                if c not in ("B", "W"):
                    refuse(f"{path}: colour {c!r}")
                root = sgf.split(";", 2)[1] if sgf.count(";") >= 2 else ""
                m = _HD.findall(root)
                if len(m) != 1:
                    refuse(f"{path}: question {r['question_id']} game {i}: root node without exactly one HD[n]")
                mv = _MOVE.findall(sgf)
                qi.append(k), hd.append(int(m[0])), col.append(c)
                nopp.append(sum(1 for x, _ in mv if x != c)), nall.append(len(mv))
    return path, qids, np.array(qi), np.array(hd), np.array(col), np.array(nopp), np.array(nall)


def opp_term(s, vk, fz, Wn, wr, proto, M, par, head_ckpt):
    """E_OPP2 [Q, 10] of one set and its record (seam O inside)."""
    raw_path, qids, qi, hd, col, nopp, nall = raw_games(s)
    src_q = os.path.join(SRC_DIR, f"player_identification_test2_{s}.csv")
    src_c = os.path.join(SRC_DIR, f"player_identification_candidates2_{s}.csv")
    with open(no_local(src_c), encoding="utf-8-sig") as f:
        cand_lines = [x for x in f.read().splitlines() if x.strip()]
    npz = os.path.join(OPP_DIR, f"pool_{s}.npz")
    with np.load(no_local(npz), allow_pickle=False) as z:
        emb, count = np.asarray(z["emb"]), np.asarray(z["count"])
        ckpt, cfg = str(z["ckpt_sha256"]), json.loads(str(z["embed_cfg"]))
    groups = vk["groups"]
    rows = np.array([i for _q, rws in groups for i in rws], np.int64)
    expect = nopp - np.where(col == "W", 4, 3)          # the first 7 plies are skipped (prereg36's sampler)
    seam = {"source_question_file_sha256_equal_raw": sha_raw(src_q) == sha_raw(raw_path),
            "source_candidate_rows": len(cand_lines) - 1,
            "games": [int(len(emb)), int(len(qi)), int(vk["n"])],
            "checkpoint_is_s0": ckpt == head_ckpt, "weights": cfg.get("weights"), "tta": cfg.get("tta"),
            "mode": cfg.get("mode"), "moves": [int(cfg.get("moves", -1)), int(nall.sum())],
            "sample_count_equals_opponent_moves_minus_first_7_plies": bool(len(count) == len(expect)
                                                                           and np.array_equal(count, expect)),
            "question_ids_equal_val_keys": qids == [q for q, _ in groups],
            "games_per_question_equal": [len(r_) for _q, r_ in groups] == np.bincount(qi, minlength=len(qids)).tolist(),
            "colour_equal_val_keys": bool(len(rows) == len(col) and [vk["color"][i] for i in rows] == col.tolist()),
            "finite": bool(np.isfinite(emb).all())}
    seam["ok"] = bool(seam["source_question_file_sha256_equal_raw"] and seam["source_candidate_rows"] == 0
                      and len(set(seam["games"])) == 1 and seam["checkpoint_is_s0"] and seam["weights"] == "raw"
                      and seam["tta"] == 1 and seam["mode"] is None and seam["moves"][0] == seam["moves"][1]
                      and seam["sample_count_equals_opponent_moves_minus_first_7_plies"]
                      and seam["question_ids_equal_val_keys"] and seam["games_per_question_equal"]
                      and seam["colour_equal_val_keys"] and seam["finite"])
    if not seam["ok"]:
        drift(f"{s}: seam O failed (the opponent-side vectors are not bound to the question file)", seam)
    h_g, c_g = np.minimum(hd, 2), np.where(col == "B", 0, 1)
    parts = K59.class_parts(emb, Wn, wr, proto["proto"], proto["rank"], None, tau1=(par[0],), tau2=(par[1],))
    q = K59.combine(parts, *par)
    lg = K59.game_ll(q, M, h_g, c_g)
    has = count > 0
    lg[~has] = 0.0
    lg_rows = np.zeros((vk["n"], K59.L))
    lg_rows[rows] = lg
    E = K59.SCALE * np.asarray(fz["S"] @ lg_rows)
    if not np.isfinite(E).all():
        drift(f"{s}: the term is not finite")
    parsed = np.asarray(vk["parsed"], bool)[rows]
    qcol = [col[qi == k][0] for k in range(len(qids))]
    qhd1 = np.array([hd[qi == k].mean() for k in range(len(qids))])
    rec = {"seam_O": seam, "vectors": {"file": t2.shown(npz), "sha256": sha_raw(npz)},
           "raw_question_file": {"file": t2.shown(raw_path), "sha256": sha_raw(raw_path)},
           "games": int(len(qi)), "games_without_sample": int((~has).sum()), "games_not_parsed": int((~parsed).sum()),
           "mean_max_q": float(q[has].max(axis=1).mean()),
           "mean_max_q_by_colour": {c: float(q[has & (col == c)].max(axis=1).mean()) for c in ("B", "W")},
           "mean_q_by_rank_class": q[has].mean(axis=0).tolist(),
           "games_share_by_hd_group_x_colour": {f"HD{h}_{c}": float(((h_g == h) & (col == c)).mean())
                                                for h in range(K59.N_HD) for c in ("B", "W")},
           "questions_share_by_colour": {c: float(np.mean([x == c for x in qcol])) for c in ("B", "W")},
           "single_colour_questions": int(sum(len(set(col[qi == k].tolist())) == 1 for k in range(len(qids)))),
           "questions": len(qids), "question_mean_share_of_HD1_games": float(qhd1.mean())}
    return E, rec


# ================================================================ fusion and outputs
def rows_of(vk, P, ok, n):
    return [[q, len(rws), int(n[k])] + ([repr(float(v)) for v in P[k]] if ok[k] else [""] * rf.N_LEVELS)
            for k, (q, rws) in enumerate(vk["groups"])]


def fuse_rows(base, E8, theta):
    rows, P = [], {}
    for s in TEST_SETS:
        fz = base[s]["fz"]
        P[s] = r8.fuse_probs(theta, fz["m"], base[s]["Es7"] + [E8[s]], fz["n"], fz["ok"])
        rows += rows_of(base[s]["vk"], P[s], fz["ok"], fz["n"])
    return rows, P


def bytes_of(path):
    with open(path, "rb") as f:
        return f.read()


def expected_level(P):
    return P @ np.arange(rf.N_LEVELS)


def apply(a):
    t0 = time.time()
    real = not a.out
    R59 = REAL_R59 if real else os.path.abspath(a.out)
    if not real and t2.is_inside(R59, REPO):
        refuse("--out must be outside the repository (a rehearsal)")
    no_local(R59)
    rank = os.path.join(R59, "rank")
    existing = [n for n in OUTPUTS if os.path.exists(os.path.join(rank, n))]
    if existing and not (a.rerun and a.reason):
        refuse(f"outputs already exist ({existing}); never overwritten without --rerun --reason TEXT")
    pre = {}
    if real:
        rc, st = t25.git("status", "--porcelain", "--untracked-files=no")
        if rc != 0 or st:
            refuse(f"commit or restore the tracked changes first ({st.splitlines()[:6]})")
        pre["git"] = {"clean": True, "head": t25.git_head(),
                      "committed": t25.require_committed([os.path.abspath(__file__)]
                                                         + [path_of(r_) for r_ in PINS], "the test59 files")}
    pre["blas_env"] = t25.blas_env_check()
    now = dt.datetime.now(t25.TW)
    if now >= K6_59:
        refuse(f"K6-59 {K6_59.isoformat()} has passed (PREREG_59 section 4): nothing is produced or rerun")
    pre["k6_59"] = {"now": now.isoformat(), "k6_59": K6_59.isoformat()}
    try:
        fc59, M, par, pre["prereg59_sha256"] = check_pins()
        ns = argparse.Namespace(dry_run=False, results_dir=None, now=None, rerun=False, reason=None, bug_revision=None,
                                **{f.lstrip("-").replace("-", "_"): None for f in t25.DIR_FLAGS.values()})
        ctx = t25.resolve(ns, "apply")
        fc25, models25, pre["prereg25"] = t25.check_p25_pins()
        pre["venv"] = t25.check_venv(fc25)
        rep19, pre["t19_apply_report"] = t25.t19_state(ctx)
        readers = t25.frozen_readers(ctx)
        pre["frozen_readers"] = readers[-1]
        rep25, pre["t25_apply_report"] = t25_state(ctx, real)
        base, qfiles = seven_terms(ctx, ns, fc25, models25, readers[:-1], rep19, t0)
    except t25.T25Error as e:
        raise T59Error(f"(t25_rank) {e}", 4 if e.code == 4 else 2, getattr(e, "details", None))
    if list(r8.Q02_TERMS) + ["HS", "P1TO", "SET"] != t25.TERMS:
        refuse("the term order of t25 changed")
    # ---- seam A: Q25 recomputed = test25's probabilities, byte for byte
    rows25 = base["public"]["res25"]["rows"] + base["private"]["res25"]["rows"]
    seam_a = t25.rows_compare(rows25, os.path.join(ctx["rank25"], "rank_test_probs.csv"))
    print(f"seam A (Q25 recomputed vs test25): byte equal {seam_a['byte_equal']}, max |dP| {seam_a['max_abs_diff']:.3g}",
          flush=True)
    if not (seam_a["equal"] and seam_a["byte_equal"]):
        drift("seam A failed: Q25 recomputed differs from results/test25/rank/rank_test_probs.csv", seam_a)
    # ---- the opponent term
    Wn, wr, hrec = K59.K58.load_head(path_of("results/prereg59/rank/s0_head.npz"))
    with np.load(path_of("results/prereg59/rank/prototypes.npz"), allow_pickle=False) as z:
        proto = {"proto": np.asarray(z["proto"], np.float64), "rank": np.asarray(z["rank"], np.int64),
                 "player_id": np.asarray(z["player_id"]).astype(str), "ckpt_sha256": str(z["ckpt_sha256"])}
    if proto["ckpt_sha256"] != hrec["ckpt_sha256"]:
        refuse("prototypes.npz and s0_head.npz are of different checkpoints")
    E8, orec = {}, {}
    for s in TEST_SETS:
        E8[s], orec[s] = opp_term(s, base[s]["vk"], base[s]["fz"], Wn, wr, proto, M, par, hrec["ckpt_sha256"])
        print(f"{s}: seam O ok; mean max_s q {orec[s]['mean_max_q']:.4f} (inner {INNER_MEAN_MAX_Q})", flush=True)
    # ---- seam B: weight 0 through the 8-term path = 25's submission, byte for byte
    os.makedirs(rank, exist_ok=True)
    th25 = np.asarray(fc25["params"]["theta"], np.float64)
    j = 2 + len(t25.TERMS)
    th25z = np.concatenate([th25[:j], [0.0], th25[j:]])
    tmp = os.path.join(R59, "tmp_seam_b")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rows_z, _P = fuse_rows(base, E8, th25z)
    t2.write_probs(os.path.join(tmp, "rank_test_probs.csv"), rows_z)
    decide.run(os.path.join(tmp, "rank_test_probs.csv"), tmp)
    seam_b = {"probs_byte_equal": bytes_of(os.path.join(tmp, "rank_test_probs.csv"))
              == bytes_of(os.path.join(ctx["rank25"], "rank_test_probs.csv")),
              "submission_byte_equal": bytes_of(os.path.join(tmp, "pred_B.csv"))
              == bytes_of(os.path.join(ctx["rank25"], "submission_rank.csv"))}
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"seam B (weight 0 = 25): {seam_b}", flush=True)
    if not all(seam_b.values()):
        drift("seam B failed: weight 0 does not give 25's files byte for byte", seam_b)
    # ---- Q59
    for n in OUTPUTS:
        p = os.path.join(rank, n)
        if os.path.exists(p):
            os.remove(p)
    th59 = np.asarray(fc59["params"]["theta"], np.float64)
    rows59, P59 = fuse_rows(base, E8, th59)
    probs_csv = os.path.join(rank, "rank_test_probs.csv")
    t2.write_probs(probs_csv, rows59)
    dsum = decide.run(probs_csv, rank)
    sub = os.path.join(rank, "submission_rank.csv")
    t2.atomic_copy(os.path.join(rank, "pred_B.csv"), sub)
    if bytes_of(sub) != bytes_of(os.path.join(rank, "pred_B.csv")):
        refuse("submission_rank.csv is not a byte copy of pred_B.csv")
    check_q = [qfiles["public"], qfiles["private"]]
    errors = check_submission.check("rank", sub, check_q, official=True)
    if errors:
        for n in OUTPUTS:
            if os.path.exists(os.path.join(rank, n)):
                os.remove(os.path.join(rank, n))
        drift(f"check_submission --official failed: {errors[:5]}", {"errors": errors[:50]})
    # ---- record
    p59 = dict(t2.read_pred(sub))
    p25 = dict(t2.read_pred(os.path.join(ctx["rank25"], "submission_rank.csv")))
    lv = {x: i for i, x in enumerate(K59.RANKS)}
    per = {}
    for s in TEST_SETS:
        fz, vk = base[s]["fz"], base[s]["vk"]
        qids = [q for q, _ in vk["groups"]]
        d = np.array([lv[p59[q]] - lv[p25[q]] for q in qids])
        nq = np.asarray(fz["n"]).astype(int)
        rho, rho_n = pf7.hs_rho(E8[s], base[s]["res25"]["P"], fz["ok"])
        P25 = base[s]["res25"]["P"]
        per[s] = {"questions": len(qids), "changed_vs_25": int((d != 0).sum()),
                  "changed_by_one_level": int((np.abs(d) == 1).sum()), "changed_by_two_or_more": int((np.abs(d) >= 2).sum()),
                  "moved_up": int((d > 0).sum()), "moved_down": int((d < 0).sum()),
                  "changed_by_number_of_games": {str(k): [int(((d != 0) & (nq == k)).sum()), int((nq == k).sum())]
                                                 for k in sorted(set(nq.tolist()))},
                  "predicted_25": {x: sum(1 for q in qids if p25[q] == x) for x in K59.RANKS},
                  "predicted_59": {x: sum(1 for q in qids if p59[q] == x) for x in K59.RANKS},
                  "mean_expected_level_25": float(expected_level(P25[fz["ok"]]).mean()),
                  "mean_expected_level_59": float(expected_level(P59[s][fz["ok"]]).mean()),
                  "mean_max_probability_25": float(P25[fz["ok"]].max(axis=1).mean()),
                  "mean_max_probability_59": float(P59[s][fz["ok"]].max(axis=1).mean()),
                  "opp2_vs_q25_spearman_expected_level": {"rho": rho, "questions": rho_n},
                  "mean_max_q": orec[s]["mean_max_q"], "mean_max_q_minus_inner": orec[s]["mean_max_q"] - INNER_MEAN_MAX_Q,
                  "term": orec[s]}
        per[s]["review"] = {"max_q_shift_over_line": bool(abs(per[s]["mean_max_q_minus_inner"]) > REVIEW_MAX_Q),
                            "changed_over_line": bool(per[s]["changed_vs_25"] > REVIEW_CHANGED)}
    record = {"what": "record only (PREREG_59 section 4): Q59 vs the uploaded 25, and the shift check of the term",
              "inner_reference": {"mean_max_q": INNER_MEAN_MAX_Q,
                                  "hd_groups_of_question_games": fc59["term"]["inner_hd_groups_of_question_games"],
                                  "player_colour_of_question_games": fc59["term"]["inner_player_colour_of_question_games"]},
              "review_lines": {"abs_mean_max_q_minus_inner": REVIEW_MAX_Q, "changed_questions_per_pool": REVIEW_CHANGED},
              "review_needed": bool(any(any(per[s]["review"].values()) for s in TEST_SETS)),
              "sets": {s: {k: v for k, v in per[s].items() if k != "term"} for s in TEST_SETS}}
    ds_path = os.path.join(rank, "decide_summary.json")
    ds = read_json(ds_path)
    ds["prereg59_record"] = record
    write_json(ds_path, ds)
    outs = [n for n in OUTPUTS if n != "t59_apply_report.json"]
    rep = {"format": "c10-test59 rank apply v1", "mode": "real" if real else "rehearsal",
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "exit_code": 0,
           "what": "test59 competition 2 apply (PREREG_59 section 4): frozen Q59 = Q25's seven terms + OPP2 per set, "
                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
           "combo": {"id": "Q59", "terms": TERMS8}, "theta": fc59["params"]["theta"], "params_w": fc59["params"]["w"],
           "term": {"tau1": par[0], "tau2": par[1], "b": par[2], "scale": K59.SCALE,
                    "classes": [int(len(wr)), int(len(proto["rank"]))], "head": hrec},
           "seam_A_q25_recomputed_vs_test25": seam_a, "seam_B_weight_0_equals_25": seam_b,
           "seam_O": {s: orec[s]["seam_O"] for s in TEST_SETS},
           "sets": per, "record": record,
           "check_submission": {"mode": "official", "questions": [t2.shown(p) for p in check_q], "errors": errors},
           "outputs_sha256": {n: sha_raw(os.path.join(rank, n)) for n in outs},
           "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "n_A_ne_B", "distribution_B")},
           "test25": pre["t25_apply_report"], "checks": pre, "rerun": {"reason": a.reason, "replaced": existing},
           "git_head": t25.git_head(), "data_local": "never opened, listed or hashed",
           "not_done_here": "no upload, no official_log.py record (the main session decides, PREREG_59 section 5)",
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(rank, "t59_apply_report.json"), rep)
    print(f"apply Q59 -> {t2.shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); "
          f"check_submission official: OK; changed vs 25 public / private: {per['public']['changed_vs_25']} / "
          f"{per['private']['changed_vs_25']}; review needed: {record['review_needed']}")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="t59_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--out")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    a = ap.parse_args(argv)
    try:
        return apply(a)
    except T59Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'DRIFT'} (exit {e.code}): {e}"
              + (f"\n{json.dumps(t4.jsonable(e.details), ensure_ascii=False)[:3000]}" if e.details else ""),
              file=sys.stderr)
        return e.code
    except t25.T25Error as e:
        code = 4 if e.code == 4 else 2
        print(f"{'DRIFT' if code == 4 else 'REFUSED'} (exit {code}): (t25_rank) {e}", file=sys.stderr)
        return code
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t59_rank.py", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
