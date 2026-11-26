"""C10 test64, competition 2 (rank, TEAM_11248): the 64 test-day local tool. Q64 = w_A * log_softmax(Q25) +
w_O * OPP3 (the c49 opponent term of prereg64), then the same-player soft merge inside each test pool (step 4, when
the frozen config says so). Registration: docs/PREREG_64_OPPONENT_C49.md. CPU only, .venv, numpy.

    PY tools/t64_rank.py apply [--out DIR] [--rerun --reason TEXT] [--no-merge]
    (PY = .venv/Scripts/python.exe; Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL
     thread variable)

--out DIR    a rehearsal: DIR outside the repository, git state not checked. Without it: results/test64 (git clean).
--no-merge   rehearsal only (needs --out): the step 1-3 system without the merge (seam C).

tools/t59_rank.py (and through it t25 / t19 / t09 / t08 / t07 / t04 / t02), results/prereg59/explore/p59_lib.py and
results/prereg64/explore/p64_lib.py are imported, never edited (t59's module constant OPP_DIR is pointed at the c49
opponent-side vectors for this process only). Never opens, lists or hashes anything under data/local; never imports
torch, never fits anything, never runs tools/score.py, never uploads, never calls tools/official_log.py.

apply, in this order (the first failing item decides):
  prerequisites: git clean (real); no BLAS thread variable; K6-64; the prereg64 files equal their pinned SHA-256s;
      t25's prerequisites and 25's official run exactly as t59 checks them.
  1  the seven terms of public / private recomputed with t25's functions (t59.seven_terms).
     seam A: Q25 recomputed = results/test25/rank/rank_test_probs.csv byte for byte (else 4).
  2  OPP3 per question: c49 opponent-side vectors (data/prereg56/emb_opp_rank/<tag>/pool_<set>.npz) through t59's
     opp_term (seam O: bound to the raw question file, c49 checkpoint, sample counts) with the c49 head, the prereg64
     prototypes, kernel.json's M and (tau1, tau2, b) = (50, 30, 1); nothing left out.
     seam B: w_A = 1, w_O = 0, no merge gives results/test25/rank/submission_rank.csv byte for byte (else 4).
  3  Q64 steps 1-3 with the frozen (w_A, w_O); step 4 merge per pool (c49 player-side vectors
     data/prereg56/emb_rank/<tag>/pool_<set>.npz, seam M: c49, sample counts = the player's moves after the first
     7 plies) -> rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json (+ "prereg64_record": changes vs
     59 and 25 per pool, merge groups with sizes), submission_rank.csv (= pred_B.csv), t64_apply_report.json;
     check_submission --official (else 4, outputs removed).
Exit codes: 0 done; 2 refused or error; 4 a seam or check failed (stop and find out why).
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
P64_EXPLORE = os.path.join(REPO, "results", "prereg64", "explore")
for _p in (HERE, P64_EXPLORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t59_rank as t59  # noqa: E402
import p64_lib as K64  # noqa: E402  (numpy / scipy; imports p58_lib, p59_lib)

t25, t2, t4, rf, r8, decide, check_submission = t59.t25, t59.t2, t59.t4, t59.rf, t59.r8, t59.decide, t59.check_submission
K59 = t59.K59
TAG = t59.TAG
OPP_DIR_C49 = os.path.join(REPO, "data", "prereg56", "emb_opp_rank", TAG)
OWN_DIR_C49 = os.path.join(REPO, "data", "prereg56", "emb_rank", TAG)
REAL_R64 = os.path.join(REPO, "results", "test64")
RANK59 = os.path.join(REPO, "results", "test59", "rank")
PINS = {  # .json: SHA-256 with CRLF folded to LF; .npz: raw bytes
    "results/prereg64/rank/frozen_config.json": "35c0b96fcb554974d6406436dd19c12331c2c45d029b2d4ecc00b59bf48a2bea",
    "results/prereg64/rank/c49_head.npz": "ea7408d96181e5f6012bb7aafd1f4c2c31a66be77bf72b8220d688a41d0f0a6a",
    "results/prereg64/rank/prototypes.npz": "6ffe6682b6584ee08c544247ef2f6ec9371ab08c59fb726b741b6fe5ff5565f7",
    "results/prereg64/explore/kernel.json": "ad0c95b5711fbe5afb484d3198515232cc4e8d2dee3310cb825041b5e103b0f2"}
K6_64 = dt.datetime(2026, 11, 24, 23, 0, 0, tzinfo=t25.TW)
TEST_SETS = t25.TEST_SETS
OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
           "t64_apply_report.json")
refuse, drift, no_local, sha_raw, sha_lf, path_of, read_json, write_json, bytes_of = (
    t59.refuse, t59.drift, t59.no_local, t59.sha_raw, t59.sha_lf, t59.path_of, t59.read_json, t59.write_json,
    t59.bytes_of)
T64Error = t59.T59Error


def check_pins():
    got = {r_: ((sha_lf if r_.endswith(".json") else sha_raw)(path_of(r_)) if os.path.exists(path_of(r_)) else None)
           for r_ in PINS}
    bad = sorted(r_ for r_ in PINS if got[r_] != PINS[r_])
    if bad:
        refuse(f"prereg64 files differ from the pinned SHA-256s: {bad}")
    fc = read_json(path_of("results/prereg64/rank/frozen_config.json"))
    if fc.get("format") != "c10-prereg64 rank frozen_config v1":
        refuse("results/prereg64/rank/frozen_config.json is not the frozen Q64 configuration")
    t = fc["term"]
    if (t["tau1"], t["tau2"], t["b"], t["scale"]) != (K64.TAU1, K64.TAU2, K64.BIAS, K64.SCALE) or \
            t["kernel_sha256_lf"] != PINS["results/prereg64/explore/kernel.json"]:
        refuse("frozen_config.json's term is not kernel.json's / p64_lib's")
    k = read_json(path_of("results/prereg64/explore/kernel.json"))
    M = np.asarray(k["M"], np.float64)
    if M.shape != (3, 2, 10, 10) or not np.allclose(M.sum(axis=3), 1.0):
        refuse("kernel.json: not a [3, 2, 10, 10] table")
    return fc, M, got


def own_vectors(s, vk, qi, col, nall, nopp):
    """c49 player-side vectors of the questions' games (seam M) -> unit rows in raw-file game order."""
    npz = os.path.join(OWN_DIR_C49, f"pool_{s}.npz")
    with np.load(no_local(npz), allow_pickle=False) as z:
        emb, count = np.asarray(z["emb"]), np.asarray(z["count"])
        ckpt, cfg = str(z["ckpt_sha256"]), json.loads(str(z["embed_cfg"]))
    nown = nall - nopp
    expect = nown - np.where(col == "B", 4, 3)         # the player's moves after the first 7 plies
    seam = {"games": [int(len(emb)), int(len(qi)), int(vk["n"])], "checkpoint_is_c49": ckpt == K64.C49_CKPT_SHA,
            "weights": cfg.get("weights"), "tta": cfg.get("tta"), "mode": cfg.get("mode"),
            "moves": [int(cfg.get("moves", -1)), int(nall.sum())],
            "sample_count_equals_player_moves_minus_first_7_plies": bool(len(count) == len(expect)
                                                                         and np.array_equal(count, expect)),
            "finite": bool(np.isfinite(emb).all())}
    seam["ok"] = bool(len(set(seam["games"])) == 1 and seam["checkpoint_is_c49"] and seam["weights"] == "raw"
                      and seam["tta"] == 1 and seam["mode"] is None and seam["moves"][0] == seam["moves"][1]
                      and seam["sample_count_equals_player_moves_minus_first_7_plies"] and seam["finite"])
    if not seam["ok"]:
        drift(f"{s}: seam M failed (the player-side vectors are not bound to the question file)", seam)
    return K64.l2(emb), {"seam_M": seam, "file": t2.shown(npz), "sha256": sha_raw(npz)}


def anchor_logp(theta25, fz, Es7):
    ok = np.asarray(fz["ok"], bool)
    z, _t = r8.p4.gen_logits(np.asarray(theta25, np.float64), fz["m"][ok], [E[ok] for E in Es7], fz["n"][ok])
    return K64.log_softmax(z), ok


def q64_probs(base, E8, theta25, w_A, w_O, merge_on, own, record_groups=False):
    """-> {set: P [Q, 10] (NaN rows when not ok)}, merge record (the groups the merge finds are recorded when
    record_groups, also when the merge is not applied)."""
    P, mrec = {}, {}
    for s in TEST_SETS:
        fz, vk = base[s]["fz"], base[s]["vk"]
        A, ok = anchor_logp(theta25, fz, base[s]["Es7"])
        Z = w_A * A + w_O * E8[s][ok]
        L0 = K64.log_softmax(Z)
        if merge_on or record_groups:
            emb, qi, qcol, nq = own[s]
            okq = np.flatnonzero(ok)
            groups_g = [np.flatnonzero(qi == k).tolist() for k in okq]
            S = K64.cc_whiten_sim(emb, groups_g, [qcol[k] for k in okq])
            Lm, W_adj = K64.merge(L0, S, nq[okq])
            comps = K64.groups(W_adj)
            qids = [q for q, _ in vk["groups"]]
            mrec[s] = {"applied": bool(merge_on), "edges": int((np.triu(W_adj, 1) > 0).sum()), "groups": len(comps),
                       "group_sizes": sorted([len(c) for c in comps], reverse=True),
                       "groups_question_ids": [[qids[okq[i]] for i in c] for c in comps],
                       "max_similarity": float(S.max())}
            mrec[s]["questions_the_merge_would_change"] = int(
                (rf.rule_b_np(rf.softmax(Lm)) != rf.rule_b_np(rf.softmax(L0))).sum())
            if merge_on:
                L0 = Lm
        Pq = np.full((len(ok), rf.N_LEVELS), np.nan)
        Pq[ok] = rf.softmax(L0)
        P[s] = Pq
    return P, mrec


def write_rows(base, P, path):
    rows = []
    for s in TEST_SETS:
        fz = base[s]["fz"]
        rows += t59.rows_of(base[s]["vk"], P[s], fz["ok"], fz["n"])
    t2.write_probs(path, rows)


def apply(a):
    t0 = time.time()
    real = not a.out
    if a.no_merge and real:
        refuse("--no-merge is for a rehearsal (--out) only")
    R64 = REAL_R64 if real else os.path.abspath(a.out)
    if not real and t2.is_inside(R64, REPO):
        refuse("--out must be outside the repository (a rehearsal)")
    no_local(R64)
    rank = os.path.join(R64, "rank")
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
                                                         + [path_of(r_) for r_ in PINS], "the test64 files")}
    pre["blas_env"] = t25.blas_env_check()
    now = dt.datetime.now(t25.TW)
    if now >= K6_64:
        refuse(f"K6-64 {K6_64.isoformat()} has passed: nothing is produced or rerun")
    pre["k6_64"] = {"now": now.isoformat(), "k6_64": K6_64.isoformat()}
    try:
        fc64, M, pre["prereg64_sha256"] = check_pins()
        ns = argparse.Namespace(dry_run=False, results_dir=None, now=None, rerun=False, reason=None, bug_revision=None,
                                **{f.lstrip("-").replace("-", "_"): None for f in t25.DIR_FLAGS.values()})
        ctx = t25.resolve(ns, "apply")
        fc25, models25, pre["prereg25"] = t25.check_p25_pins()
        pre["venv"] = t25.check_venv(fc25)
        rep19, pre["t19_apply_report"] = t25.t19_state(ctx)
        readers = t25.frozen_readers(ctx)
        pre["frozen_readers"] = readers[-1]
        rep25, pre["t25_apply_report"] = t59.t25_state(ctx, real)
        base, qfiles = t59.seven_terms(ctx, ns, fc25, models25, readers[:-1], rep19, t0)
    except t25.T25Error as e:
        raise T64Error(f"(t25_rank) {e}", 4 if e.code == 4 else 2, getattr(e, "details", None))
    theta25 = np.asarray(fc25["params"]["theta"], np.float64)
    if fc64["q25_theta"] != [float(x) for x in theta25]:
        refuse("frozen_config.json's Q25 theta is not results/prereg25/rank's")
    # ---- seam A
    rows25 = base["public"]["res25"]["rows"] + base["private"]["res25"]["rows"]
    seam_a = t25.rows_compare(rows25, os.path.join(ctx["rank25"], "rank_test_probs.csv"))
    print(f"seam A (Q25 recomputed vs test25): byte equal {seam_a['byte_equal']}", flush=True)
    if not (seam_a["equal"] and seam_a["byte_equal"]):
        drift("seam A failed: Q25 recomputed differs from results/test25/rank/rank_test_probs.csv", seam_a)
    # ---- OPP3 (t59.opp_term with the c49 vectors, head, prototypes and M)
    Wn, hid, wr = K64.load_head(path_of("results/prereg64/rank/c49_head.npz"))
    with np.load(path_of("results/prereg64/rank/prototypes.npz"), allow_pickle=False) as z:
        proto = {"proto": np.asarray(z["proto"], np.float64), "rank": np.asarray(z["rank"], np.int64),
                 "player_id": np.asarray(z["player_id"]).astype(str), "ckpt_sha256": str(z["ckpt_sha256"])}
    if proto["ckpt_sha256"] != K64.C49_CKPT_SHA:
        refuse("prototypes.npz is not of the c49 checkpoint")
    t59.OPP_DIR = OPP_DIR_C49
    par = (K64.TAU1, K64.TAU2, K64.BIAS)
    E8, orec, own = {}, {}, {}
    for s in TEST_SETS:
        E8[s], orec[s] = t59.opp_term(s, base[s]["vk"], base[s]["fz"], Wn, wr, proto, M, par, K64.C49_CKPT_SHA)
        _raw, qids, qi, hd, col, nopp, nall = t59.raw_games(s)
        emb, orec[s]["own_vectors"] = own_vectors(s, base[s]["vk"], qi, col, nall, nopp)
        qcol = [str(col[qi == k][0]) for k in range(len(qids))]
        own[s] = (emb, qi, qcol, np.bincount(qi, minlength=len(qids)))
        print(f"{s}: seams O and M ok; mean max_s q {orec[s]['mean_max_q']:.4f} (inner "
              f"{fc64['term']['inner_mean_max_q']:.4f})", flush=True)
    # ---- seam B: w_A 1, w_O 0, no merge = 25's submission
    os.makedirs(rank, exist_ok=True)
    tmp = os.path.join(R64, "tmp_seam_b")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    Pz, _m = q64_probs(base, E8, theta25, 1.0, 0.0, False, own)
    write_rows(base, Pz, os.path.join(tmp, "rank_test_probs.csv"))
    decide.run(os.path.join(tmp, "rank_test_probs.csv"), tmp)
    seam_b = {"submission_byte_equal": bytes_of(os.path.join(tmp, "pred_B.csv"))
              == bytes_of(os.path.join(ctx["rank25"], "submission_rank.csv")),
              "record_probs_byte_equal": bytes_of(os.path.join(tmp, "rank_test_probs.csv"))
              == bytes_of(os.path.join(ctx["rank25"], "rank_test_probs.csv"))}
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"seam B (w_O 0, no merge = 25): {seam_b}", flush=True)
    if not seam_b["submission_byte_equal"]:
        drift("seam B failed: w_A 1, w_O 0 without merge does not give 25's submission byte for byte", seam_b)
    # ---- Q64
    for n in OUTPUTS:
        p = os.path.join(rank, n)
        if os.path.exists(p):
            os.remove(p)
    merge_on = bool(fc64["merge"]["apply"]) and not a.no_merge
    w_A, w_O = float(fc64["w_A"]), float(fc64["w_O"])
    P13, _m = q64_probs(base, E8, theta25, w_A, w_O, False, own)
    P64, mrec = q64_probs(base, E8, theta25, w_A, w_O, merge_on, own, record_groups=True)
    probs_csv = os.path.join(rank, "rank_test_probs.csv")
    write_rows(base, P64, probs_csv)
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
    p64 = dict(t2.read_pred(sub))
    p25 = dict(t2.read_pred(os.path.join(ctx["rank25"], "submission_rank.csv")))
    p59 = dict(t2.read_pred(os.path.join(RANK59, "submission_rank.csv")))
    lv = {x: i for i, x in enumerate(K59.RANKS)}
    per = {}
    for s in TEST_SETS:
        fz, vk = base[s]["fz"], base[s]["vk"]
        qids = [q for q, _ in vk["groups"]]
        ok = np.asarray(fz["ok"], bool)
        d25 = np.array([lv[p64[q]] - lv[p25[q]] for q in qids])
        d59 = np.array([lv[p64[q]] - lv[p59[q]] for q in qids])
        pr13 = rf.rule_b_np(np.where(np.isnan(P13[s]), 0.1, P13[s]))
        pr64 = rf.rule_b_np(np.where(np.isnan(P64[s]), 0.1, P64[s]))
        nq = np.asarray(fz["n"]).astype(int)
        per[s] = {"questions": len(qids), "ok_questions": int(ok.sum()),
                  "changed_vs_25": int((d25 != 0).sum()), "changed_vs_59": int((d59 != 0).sum()),
                  "changed_vs_59_by_two_or_more": int((np.abs(d59) >= 2).sum()),
                  "changed_by_merge": int(((pr13 != pr64) & ok).sum()),
                  "changed_vs_59_by_number_of_games": {str(k): [int(((d59 != 0) & (nq == k)).sum()), int((nq == k).sum())]
                                                       for k in sorted(set(nq.tolist()))},
                  "predicted_59": {x: sum(1 for q in qids if p59[q] == x) for x in K59.RANKS},
                  "predicted_64": {x: sum(1 for q in qids if p64[q] == x) for x in K59.RANKS},
                  "mean_max_q": orec[s]["mean_max_q"], "merge": mrec.get(s, {"applied": False})}
    record = {"what": "record only: Q64 vs the uploaded 59 and 25; merge groups per pool",
              "merge_applied": merge_on, "w_A": w_A, "w_O": w_O, "sets": per}
    ds_path = os.path.join(rank, "decide_summary.json")
    ds = read_json(ds_path)
    ds["prereg64_record"] = record
    write_json(ds_path, ds)
    outs = [n for n in OUTPUTS if n != "t64_apply_report.json"]
    rep = {"format": "c10-test64 rank apply v1", "mode": "real" if real else "rehearsal",
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "exit_code": 0,
           "what": "test64 competition 2 apply: frozen Q64 = w_A * log_softmax(Q25) + w_O * OPP3 per set, then the "
                   "same-player merge per pool (when frozen), rule B, submission_rank.csv = pred_B.csv (public then "
                   "private)",
           "combo": {"id": "Q64", "merge": merge_on}, "w_A": w_A, "w_O": w_O,
           "term": {"tau1": par[0], "tau2": par[1], "b": par[2], "scale": K64.SCALE,
                    "classes": [int(len(wr)), int(len(proto["rank"]))]},
           "seam_A_q25_recomputed_vs_test25": seam_a, "seam_B_weight_0_equals_25": seam_b,
           "seams_O_M": {s: {"O": orec[s]["seam_O"], "M": orec[s]["own_vectors"]["seam_M"]} for s in TEST_SETS},
           "sets": per, "terms": {s: {k: v for k, v in orec[s].items() if k not in ("seam_O",)} for s in TEST_SETS},
           "check_submission": {"mode": "official", "questions": [t2.shown(p) for p in check_q], "errors": errors},
           "outputs_sha256": {n: sha_raw(os.path.join(rank, n)) for n in outs},
           "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "n_A_ne_B", "distribution_B")},
           "test25": pre["t25_apply_report"], "checks": pre, "rerun": {"reason": a.reason, "replaced": existing},
           "git_head": t25.git_head(), "data_local": "never opened, listed or hashed",
           "not_done_here": "no upload, no official_log.py record (the main session decides)",
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(rank, "t64_apply_report.json"), rep)
    print(f"apply Q64 -> {t2.shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission "
          f"official: OK; changed vs 59 public / private: {per['public']['changed_vs_59']} / "
          f"{per['private']['changed_vs_59']}; vs 25: {per['public']['changed_vs_25']} / {per['private']['changed_vs_25']}; "
          f"merge: {merge_on}")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="t64_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--out")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--no-merge", action="store_true")
    a = ap.parse_args(argv)
    try:
        return apply(a)
    except T64Error as e:
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
        print("ERROR (exit 2): unexpected internal error of tools/t64_rank.py", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
