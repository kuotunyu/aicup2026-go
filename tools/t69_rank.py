"""C10 test69, competition 2 (rank, TEAM_11248): 64 + prereg65's safe same-player merge M8 (prereg69). Q64 steps 1-3
exactly as tools/t64_rank.py computes them, then M8 inside each test pool. Registration: results/prereg69/explore/
PLAN.md, NOTES.md, frozen.json; docs/PREREG_69_M8_ON_64.md. CPU only, .venv, numpy.

    PY tools/t69_rank.py apply [--out DIR] [--rerun --reason TEXT]
    (PY = .venv/Scripts/python.exe; Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL
     thread variable)

--out DIR  a rehearsal: DIR outside the repository, git state not checked. Without it: results/test69 (git clean).

tools/t64_rank.py (and through it t59 / t25 / ...), results/prereg64/explore/p64_lib.py, results/prereg65/explore/
p65_lib.py and results/prereg69/explore/p69_lib.py are imported, never edited. Never opens, lists or hashes anything
under data/local; never imports torch, never fits anything, never runs tools/score.py, never uploads, never calls
tools/official_log.py.

apply, in this order (the first failing item decides):
  prerequisites: as t64 (git clean (real), BLAS variables, K6-64, prereg64 pins, 25's official run); the prereg69
      merge simulation passed (results/prereg69/explore/merge_sim.json gate.passed).
  1  Q25's seven terms and seam A (Q25 recomputed = results/test25 byte for byte); OPP3 per set (seams O, M) - t64's
     functions.
  2  seam T: Q64 with the merge off -> rank_test_probs.csv and pred_B.csv = results/test64/rank/rank_test_probs.csv
     and submission_rank.csv byte for byte (else 4).
  3  M8 per pool (p69_lib.m8_rows: M8's pred0 = rule B of 64; a changed question gets softmax(L_new) and its rule B
     = M8's prediction; other rows keep 64's values) -> rank_test_probs.csv, pred_A.csv, pred_B.csv,
     decide_summary.json (+ "prereg69_record": changed questions per pool with old / new class and the M8 group),
     submission_rank.csv (= pred_B.csv), t69_apply_report.json; pred_B = M8's predictions on every question;
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
P69_EXPLORE = os.path.join(REPO, "results", "prereg69", "explore")
for _p in (HERE, P69_EXPLORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t64_rank as t64  # noqa: E402
import p69_lib as K69  # noqa: E402

t59, t25, t2, t4, rf, decide, check_submission = t64.t59, t64.t25, t64.t2, t64.t4, t64.rf, t64.decide, t64.check_submission
K64, K59 = t64.K64, t64.K59
REAL_R69 = os.path.join(REPO, "results", "test69")
RANK64 = os.path.join(REPO, "results", "test64", "rank")
MERGE_SIM = "results/prereg69/explore/merge_sim.json"
TEST_SETS = t64.TEST_SETS
OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
           "t69_apply_report.json")
refuse, drift, no_local, sha_raw, sha_lf, path_of, read_json, write_json, bytes_of = (
    t64.refuse, t64.drift, t64.no_local, t64.sha_raw, t64.sha_lf, t64.path_of, t64.read_json, t64.write_json,
    t64.bytes_of)
T69Error = t64.T64Error


def check_gate():
    p = path_of(MERGE_SIM)
    if not os.path.exists(p):
        refuse("results/prereg69/explore/merge_sim.json does not exist (the gate simulation has not run)")
    ms = read_json(p)
    if ms.get("format") != "c10-prereg69 merge simulation v1" or not ms.get("seam_S", {}).get("ok") \
            or not ms.get("gate", {}).get("passed"):
        refuse("the prereg69 gate did not pass: M8 is not applied")
    return {"file": MERGE_SIM, "sha256_lf": sha_lf(p), "gate": ms["gate"]}


def apply(a):
    t0 = time.time()
    real = not a.out
    R69 = REAL_R69 if real else os.path.abspath(a.out)
    if not real and t2.is_inside(R69, REPO):
        refuse("--out must be outside the repository (a rehearsal)")
    no_local(R69)
    rank = os.path.join(R69, "rank")
    existing = [n for n in OUTPUTS if os.path.exists(os.path.join(rank, n))]
    if existing and not (a.rerun and a.reason):
        refuse(f"outputs already exist ({existing}); never overwritten without --rerun --reason TEXT")
    pre = {}
    if real:
        rc, st = t25.git("status", "--porcelain", "--untracked-files=no")
        if rc != 0 or st:
            refuse(f"commit or restore the tracked changes first ({st.splitlines()[:6]})")
        pre["git"] = {"clean": True, "head": t25.git_head(),
                      "committed": t25.require_committed([os.path.abspath(__file__), os.path.abspath(t64.__file__),
                                                          os.path.abspath(K69.__file__), path_of(MERGE_SIM)]
                                                         + [path_of(r_) for r_ in t64.PINS], "the test69 files")}
    pre["blas_env"] = t25.blas_env_check()
    now = dt.datetime.now(t25.TW)
    if now >= t64.K6_64:
        refuse(f"K6-64 {t64.K6_64.isoformat()} has passed: nothing is produced or rerun")
    pre["k6_64"] = {"now": now.isoformat(), "k6_64": t64.K6_64.isoformat()}
    pre["prereg69_gate"] = check_gate()
    try:
        fc64, M, pre["prereg64_sha256"] = t64.check_pins()
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
        raise T69Error(f"(t25_rank) {e}", 4 if e.code == 4 else 2, getattr(e, "details", None))
    theta25 = np.asarray(fc25["params"]["theta"], np.float64)
    if fc64["q25_theta"] != [float(x) for x in theta25]:
        refuse("frozen_config.json's Q25 theta is not results/prereg25/rank's")
    w_A, w_O = float(fc64["w_A"]), float(fc64["w_O"])
    if (w_A, w_O) != (K69.W_A_64, K69.W_O_64):
        refuse("prereg64's frozen (w_A, w_O) are not the ones prereg69 registered")
    if fc64["merge"]["apply"]:
        refuse("prereg64's frozen config applies its own merge: not the 64 prereg69 registered")
    # ---- seam A
    rows25 = base["public"]["res25"]["rows"] + base["private"]["res25"]["rows"]
    seam_a = t25.rows_compare(rows25, os.path.join(ctx["rank25"], "rank_test_probs.csv"))
    print(f"seam A (Q25 recomputed vs test25): byte equal {seam_a['byte_equal']}", flush=True)
    if not (seam_a["equal"] and seam_a["byte_equal"]):
        drift("seam A failed: Q25 recomputed differs from results/test25/rank/rank_test_probs.csv", seam_a)
    # ---- OPP3 and the player-side vectors (t64)
    Wn, hid, wr = K64.load_head(path_of("results/prereg64/rank/c49_head.npz"))
    with np.load(path_of("results/prereg64/rank/prototypes.npz"), allow_pickle=False) as z:
        proto = {"proto": np.asarray(z["proto"], np.float64), "rank": np.asarray(z["rank"], np.int64),
                 "player_id": np.asarray(z["player_id"]).astype(str), "ckpt_sha256": str(z["ckpt_sha256"])}
    if proto["ckpt_sha256"] != K64.C49_CKPT_SHA:
        refuse("prototypes.npz is not of the c49 checkpoint")
    t59.OPP_DIR = t64.OPP_DIR_C49
    par = (K64.TAU1, K64.TAU2, K64.BIAS)
    E8, orec, own = {}, {}, {}
    for s in TEST_SETS:
        E8[s], orec[s] = t59.opp_term(s, base[s]["vk"], base[s]["fz"], Wn, wr, proto, M, par, K64.C49_CKPT_SHA)
        _raw, qids, qi, hd, col, nopp, nall = t59.raw_games(s)
        emb, orec[s]["own_vectors"] = t64.own_vectors(s, base[s]["vk"], qi, col, nall, nopp)
        qcol = [str(col[qi == k][0]) for k in range(len(qids))]
        own[s] = (emb, qi, qcol, np.bincount(qi, minlength=len(qids)))
    # ---- seam T: merge off = 64 byte for byte
    os.makedirs(rank, exist_ok=True)
    tmp = os.path.join(R69, "tmp_seam_t")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    P64, _m = t64.q64_probs(base, E8, theta25, w_A, w_O, False, own)
    t64.write_rows(base, P64, os.path.join(tmp, "rank_test_probs.csv"))
    decide.run(os.path.join(tmp, "rank_test_probs.csv"), tmp)
    seam_t = {"probs_byte_equal_test64": bytes_of(os.path.join(tmp, "rank_test_probs.csv"))
              == bytes_of(os.path.join(RANK64, "rank_test_probs.csv")),
              "submission_byte_equal_test64": bytes_of(os.path.join(tmp, "pred_B.csv"))
              == bytes_of(os.path.join(RANK64, "submission_rank.csv"))}
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"seam T (merge off = 64): {seam_t}", flush=True)
    if not all(seam_t.values()):
        drift("seam T failed: Q64 recomputed without M8 is not results/test64/rank byte for byte", seam_t)
    # ---- M8 per pool
    for n in OUTPUTS:
        p = os.path.join(rank, n)
        if os.path.exists(p):
            os.remove(p)
    P69, pred69, mrec = {}, {}, {}
    for s in TEST_SETS:
        fz, vk = base[s]["fz"], base[s]["vk"]
        A, ok = t64.anchor_logp(theta25, fz, base[s]["Es7"])
        L0 = K64.log_softmax(w_A * A + w_O * E8[s][ok])
        P0 = P64[s][ok]
        if not np.array_equal(rf.softmax(L0), P0):
            drift(f"{s}: the recomputed 64 rows differ from t64.q64_probs")
        emb, qi, qcol, nq = own[s]
        okq = np.flatnonzero(ok)
        S = K64.cc_whiten_sim(emb, [np.flatnonzero(qi == k).tolist() for k in okq], [qcol[k] for k in okq])
        qc_ok = [qcol[k] for k in okq]
        try:
            P_out, pred_out, info = K69.m8_rows(L0, P0, S, qc_ok, rf.rule_b_np, rf.softmax)
        except ValueError as e:
            drift(f"{s}: {e}")
        adj = K69.m8_adjacency(L0, S, qc_ok)
        comps = K64.groups(adj.astype(np.float64))
        qids = [q for q, _ in vk["groups"]]
        Pq = np.array(P64[s], np.float64, copy=True)
        Pq[ok] = P_out
        P69[s] = Pq
        full_pred = np.full(len(ok), -1, np.int64)
        full_pred[ok] = pred_out
        pred69[s] = full_pred
        changed = []
        for i in info["changed"]:
            nb = np.flatnonzero(adj[i]).tolist()
            changed.append({"question_id": qids[okq[i]], "n_games": int(nq[okq[i]]), "colour": qc_ok[i],
                            "old": K59.RANKS[int(info["pred0"][i])], "new": K59.RANKS[int(pred_out[i])],
                            "group": [qids[okq[j]] for j in [i] + nb],
                            "group_old": [K59.RANKS[int(info["pred0"][j])] for j in [i] + nb],
                            "max_similarity_in_group": float(max(S[i, j] for j in nb))})
        mrec[s] = {"edges": info["edges"], "groups": len(comps), "group_sizes": sorted([len(c) for c in comps], reverse=True),
                   "groups_question_ids": [[qids[okq[i]] for i in c] for c in comps],
                   "max_similarity": float(S.max()), "changed": len(changed), "changed_questions": changed}
        print(f"{s}: M8 edges {info['edges']}, groups {len(comps)}, changed {len(changed)}", flush=True)
    probs_csv = os.path.join(rank, "rank_test_probs.csv")
    t64.write_rows(base, P69, probs_csv)
    dsum = decide.run(probs_csv, rank)
    sub = os.path.join(rank, "submission_rank.csv")
    t2.atomic_copy(os.path.join(rank, "pred_B.csv"), sub)
    if bytes_of(sub) != bytes_of(os.path.join(rank, "pred_B.csv")):
        refuse("submission_rank.csv is not a byte copy of pred_B.csv")
    p69 = dict(t2.read_pred(sub))
    p64 = dict(t2.read_pred(os.path.join(RANK64, "submission_rank.csv")))
    mism = []
    for s in TEST_SETS:
        qids = [q for q, _ in base[s]["vk"]["groups"]]
        for k, q in enumerate(qids):
            if pred69[s][k] >= 0 and p69[q] != K59.RANKS[int(pred69[s][k])]:
                mism.append(q)
            if pred69[s][k] < 0 and p69[q] != p64[q]:
                mism.append(q)
    check_q = [qfiles["public"], qfiles["private"]]
    errors = check_submission.check("rank", sub, check_q, official=True)
    if errors or mism:
        for n in OUTPUTS:
            if os.path.exists(os.path.join(rank, n)):
                os.remove(os.path.join(rank, n))
        drift(f"check failed: check_submission {errors[:5]}; pred_B != M8 on {mism[:5]}",
              {"errors": errors[:50], "mismatch": mism[:50]})
    per = {s: {"questions": len(base[s]["vk"]["groups"]), "ok_questions": int(np.asarray(base[s]["fz"]["ok"]).sum()),
               "changed_vs_64": int(sum(1 for q, _ in base[s]["vk"]["groups"] if p69[q] != p64[q])),
               "merge": mrec[s]} for s in TEST_SETS}
    ds_path = os.path.join(rank, "decide_summary.json")
    ds = read_json(ds_path)
    ds["prereg69_record"] = {"what": "M8 on Q64 per pool: changed questions with old / new class and the M8 group "
                                     "(the question first, then its M8 neighbours)", "w_A": w_A, "w_O": w_O,
                             "sets": per}
    write_json(ds_path, ds)
    outs = [n for n in OUTPUTS if n != "t69_apply_report.json"]
    rep = {"format": "c10-test69 rank apply v1", "mode": "real" if real else "rehearsal",
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "exit_code": 0,
           "what": "test69 competition 2 apply: frozen Q64 (merge off) then prereg65's M8 per pool, rule B, "
                   "submission_rank.csv = pred_B.csv (public then private)",
           "combo": {"id": "Q69", "base": "Q64", "merge": "M8"}, "w_A": w_A, "w_O": w_O,
           "seam_A_q25_recomputed_vs_test25": seam_a, "seam_T_merge_off_equals_64": seam_t,
           "seams_O_M": {s: {"O": orec[s]["seam_O"], "M": orec[s]["own_vectors"]["seam_M"]} for s in TEST_SETS},
           "sets": per,
           "check_submission": {"mode": "official", "questions": [t2.shown(p) for p in check_q], "errors": errors},
           "outputs_sha256": {n: sha_raw(os.path.join(rank, n)) for n in outs},
           "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "n_A_ne_B", "distribution_B")},
           "checks": pre, "rerun": {"reason": a.reason, "replaced": existing},
           "git_head": t25.git_head(), "data_local": "never opened, listed or hashed",
           "not_done_here": "no upload, no official_log.py record (the main session decides)",
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(rank, "t69_apply_report.json"), rep)
    print(f"apply Q69 -> {t2.shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission "
          f"official: OK; changed vs 64 public / private: {per['public']['changed_vs_64']} / "
          f"{per['private']['changed_vs_64']}")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="t69_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--out")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    a = ap.parse_args(argv)
    try:
        return apply(a)
    except T69Error as e:
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
        print("ERROR (exit 2): unexpected internal error of tools/t69_rank.py", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
