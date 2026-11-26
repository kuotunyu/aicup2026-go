"""prereg61 (PLAN.md Judgement / Apply): the one E judgement against v2-7, official pools. CPU only. Over p60_eval.

    PY results/prereg61/explore/p61_eval.py e-use                      # prints the E use number this judgement gets
    PY results/prereg61/explore/p61_eval.py judge --e-use N            # once -> eval.json
    PY results/prereg61/explore/p61_eval.py apply --pool public|private [--seam-only]
    PY results/prereg61/explore/p61_eval.py assemble

Nothing of prereg60 is edited: p60_eval is imported, its module constants (paths, step, tag, hard stop) and the
default arguments of check_vectors / train_record are replaced (configure()), and the judgement is written here
with prereg61's names. System(vectors) = v2-7's pipeline unchanged (p51_eval.e_context / e_scores) - nothing tuned.

judge: A61 = data/prereg61/emb/p60_step_200000_raw_tta1/store_E.npz. Adopt if mean(A61 - v2-7) >= +0.01, pool
  bootstrap (10,000, seed 43) 95% low > 0 and higher in >= 18 of 20 pools (p43_eval.judge_rule). Needs: before
  2026-11-23 12:00; no eval.json yet; prereg61's PLAN.md, p61_run.py, this file, train_record.json and prereg60's
  p60_lib.py, p60_run.py, p60_eval.py committed and unchanged; the training record finished at 200,000 steps with
  the 200,000-step checkpoint whose sha256 is the file's and the vectors'; vectors written after the code's commit,
  of the same games / store rows as c49's file with count - c49's count in 0..7.
  Seam (PLAN): v2-7's E score bit for bit through the judging path - P51.base_seam inside the judgement raises
  unless every pool is identical (mean 0.7035676497986656). Also required: prereg60's committed seam.json ok and
  made with the present p60_lib.py (the widened network equals c49 while the new channel is zero) - NOTES.md 4.
  E use number = 1 + p60_eval.e_uses_so_far (through prereg49 + later eval*.json with "e_pools_use_number", which
  leaves out prereg60's own) + prereg60's eval*.json; prereg61's own files never count.
apply / assemble: p60_eval's, reading prereg61's vectors, writing results/prereg61/apply.
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P60D = os.path.join(REPO, "results", "prereg60", "explore")
if P60D not in sys.path:
    sys.path.insert(0, P60D)
import p60_eval as E60  # noqa: E402

P51, P49, E43, E45, BK, EV, L = E60.P51, E60.P49, E60.E43, E60.E45, E60.BK, E60.EV, E60.L
_rel, read_json = E60._rel, E60.read_json

DATA = os.path.join(REPO, "data", "prereg61")
PLAN = os.path.join(HERE, "PLAN.md")
RUN = os.path.join(HERE, "p61_run.py")
EVAL_JSON, TRAIN_RECORD = (os.path.join(HERE, x) for x in ("eval.json", "train_record.json"))
P60_SEAM_JSON = E60.SEAM_JSON                                   # results/prereg60/explore/seam.json (committed)
P60_CODE = (E60.LIB, E60.RUN, os.path.abspath(E60.__file__))
CODE = (PLAN, RUN, os.path.abspath(__file__)) + P60_CODE
STEP = 200000
CKPT = os.path.join(DATA, "S0", "ckpt", "step_%06d.pt" % STEP)
LR, WARMUP = 1e-3, 2000                                         # PLAN.md Method (= p61_run's; test_p61 compares)
TAG = "p60_step_%06d_raw_tta1" % STEP
VEC_DIR = os.path.join(DATA, "emb", TAG)
OUT_APPLY = os.path.join(REPO, "results", "prereg61", "apply")
LAST = datetime.datetime(2026, 11, 23, 12, 0, 0)               # PLAN.md: hard stop
RAM_NEED_MB = E60.RAM_NEED_MB


def configure():
    """Point p60_eval's helpers (vec_file, check_vectors, train_record, cmd_apply, cmd_assemble) at prereg61."""
    E60.DATA, E60.PLAN, E60.EVAL_JSON, E60.TRAIN_RECORD = DATA, PLAN, EVAL_JSON, TRAIN_RECORD
    E60.CODE, E60.STEP, E60.CKPT, E60.TAG, E60.VEC_DIR = CODE, STEP, CKPT, TAG, VEC_DIR
    E60.OUT_APPLY, E60.LAST = OUT_APPLY, LAST
    E60.check_vectors.__defaults__ = (TAG,)                     # (path, ref, ckpt_sha, tag=TAG)
    E60.train_record.__defaults__ = (TRAIN_RECORD, CKPT)        # (path=TRAIN_RECORD, ckpt=CKPT)
    return E60


# ================================================================ E use number
def e_uses_so_far(repo=REPO):
    """p60_eval.e_uses_so_far (which leaves out prereg60's own files) + prereg60's eval*.json that carry
    "e_pools_use_number"; prereg61's own files never count."""
    u = E60.e_uses_so_far(repo)
    later = [x for x in u["later"] if not x.startswith("prereg61/")]
    for f in sorted(glob.glob(os.path.join(repo, "results", "prereg60", "explore", "eval*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                has = "e_pools_use_number" in json.load(fh)
        except (OSError, ValueError):
            raise SystemExit("cannot read %s" % f)
        if has:
            later.append("prereg60/%s" % os.path.basename(f))
    later.sort()
    return {"through_prereg49": u["through_prereg49"], "later": later,
            "count": len(u["through_prereg49"]) + len(later)}


def schedule_ok(tr):
    """The training record is this run's: prereg61's schedule on prereg60's recipe."""
    c = tr.get("config", {}).get("p60", {})
    got = (c.get("total_steps"), c.get("lr"), c.get("warmup"), c.get("first_batch"), c.get("n_planes"),
           c.get("early_moves"), c.get("action_plane"), c.get("fp16"), c.get("smoke"))
    want = (STEP, LR, WARMUP, 320000, 18, True, True, True, False)
    if got != want or "p61_run.py" not in tr.get("code_sha256", {}):
        raise SystemExit("%s is not prereg61's run: %s != %s" % (_rel(TRAIN_RECORD), got, want))
    for name, sha in tr["code_sha256"].items():
        f = {"p61_run.py": RUN, "p60_lib.py": E60.LIB, "p60_run.py": E60.RUN}.get(name)
        if f and E60.sha_lf(f) != sha:
            raise SystemExit("%s changed since the training run" % _rel(f))


# ================================================================ judge (E, once)
def cmd_judge(a):
    t0 = time.time()
    if datetime.datetime.now() > LAST:
        raise SystemExit("past %s (PLAN.md hard stop): dropped" % LAST)
    if os.path.exists(EVAL_JSON):
        raise SystemExit("%s exists: the judgement is made once" % _rel(EVAL_JSON))
    E60.committed(CODE)
    seam = E60.seam_ok(P60_SEAM_JSON)                            # committed, ok, made with the present p60_lib.py
    tr, sha = E60.train_record(TRAIN_RECORD, CKPT)               # committed, finished, STEP steps, sha of the file
    schedule_ok(tr)
    uses = e_uses_so_far()
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, uses["count"],
                                                                                         uses))
    path = E60.vec_file("E")
    vec = E60.check_vectors(path, P51.c49_file("E"), sha, TAG)
    E43.made_after([path], max(E43.commit_time(f) for f in CODE))
    pools, zs, sets, v26 = P51.e_context()
    base = P51.v27_base()
    bper, binfo = P51.base_seam(base, pools, zs, v26)            # raises unless v2-7 is reproduced bit for bit
    per, byk = P51.e_scores(path, pools, zs, v26)
    j = E43.judge_rule(per, bper)
    rec = {"format": "c10-prereg61 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "e_reused": True, "development_set": None,
           "vectors": vec, "train_record_sha256": E60.sha_lf(TRAIN_RECORD), "train_seconds": tr.get("seconds"),
           "schedule": {k: tr["config"]["p60"][k] for k in ("total_steps", "lr", "warmup", "first_batch")},
           "p60_seam_sha256": E60.sha_lf(P60_SEAM_JSON), "p60_seam_a_device": seam["a_device"],
           "p60_seam_a_max_abs_diff": seam["a_widened_c49_equals_c49"]["max_abs_diff"],
           "base": {"name": "c49 (v2-7)", "eval": base["eval"], "e_mean": base["e_mean"], "vectors_E": binfo},
           "cfg": v26["cfg"], "wP": v26["wP"], "k": v26["k"],
           "seam_base_E_identical": bool(float(np.mean(bper)) == E60.V27_E),
           "git_head": L.git_head(), "p1_sets": sets,
           "means": {"p61": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"p61": per, "base": list(bper)}, "by_k": {"p61": EV.mean_by_k(byk)}}
    rec["p61_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "A61")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = E45.peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(EVAL_JSON, rec)
    print(json.dumps({x: rec[x] for x in ("means", "p61_minus_base", "by_k", "adopt", "e_pools_use_number")},
                     indent=1))
    return 0


def main(argv=None):
    configure()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e-use")
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    p = sub.add_parser("apply")
    p.add_argument("--src", default=E60.SRC_OFFICIAL)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--seam-only", action="store_true", help="only rebuild v2-7 on the pool and compare (no eval.json)")
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", default=E60.SRC_OFFICIAL)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd == "e-use":
        print(e_uses_so_far()["count"] + 1)
        return 0
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"judge": cmd_judge, "apply": cmd_apply, "assemble": E60.cmd_assemble}[a.cmd](a)


def cmd_apply(a):
    return E60.cmd_apply(a)


if __name__ == "__main__":
    sys.exit(main())
