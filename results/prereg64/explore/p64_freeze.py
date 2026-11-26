"""prereg64: freeze the fusion Q64 = w_A * log_softmax(Q25 frozen) + w_O * OPP3 on ALL inner fit questions
(NOTES.md 7: theta = Q25's frozen all-inner-fit parameters, (w_A, w_O) by the same scalar fit as the judgement).
CPU, .venv, 4 cores, <= 3 GB. Run once, after eval.json (steps 1-3 passed) and merge_sim.json are committed.

    PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/prereg64/explore/p64_freeze.py
    -> results/prereg64/rank/frozen_config.json, c49_head.npz, prototypes.npz (byte copies of the judged files)

No inner score is computed. Seam Z: Q25's seven-term fit on all inner fit questions = results/prereg25/rank theta bit
for bit. Seam T: the OPP3 term is the judged one (mean max q and games equal eval.json's record).
"""
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import p64_eval as EV  # noqa: E402
import p64_lib as K  # noqa: E402

E58, K58 = EV.E58, EV.K58
REPO = EV.REPO
OUT = os.path.join(REPO, "results", "prereg64", "rank")
FROZEN = os.path.join(OUT, "frozen_config.json")
Q25_FROZEN = os.path.join(REPO, "results", "prereg25", "rank", "frozen_config.json")
log = EV.log


def main():
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(FROZEN):
        raise SystemExit("frozen_config.json exists (run once)")
    for p in (K.KERNEL, K.SEAM, K.EVAL, K.MERGE_SIM, os.path.abspath(__file__)):
        if not EV.committed_clean(p):
            raise SystemExit(f"{p} is not committed and clean")
    ev = K58.read_json(K.EVAL)
    ms = K58.read_json(K.MERGE_SIM)
    if ev.get("adopt_steps_1_3") is not True:
        raise SystemExit("eval.json: steps 1-3 not passed")
    M, pr, krec = EV.load_kernel()
    if K58.sha_lf(K.KERNEL) != ev["kernel_sha256_lf"]:
        raise SystemExit("kernel.json is not the judged one")
    mods, inner, terms7, Emap, q25, seam = E58.build_q25()
    import p04_rank_fuse as p4
    cs = EV.code_state()
    if cs["prereg64_dirty"] or cs["imported_dirty"]:
        raise SystemExit(f"not committed and clean: {cs['prereg64_dirty']} {cs['imported_dirty']}")
    if not seam["ok"]:
        raise SystemExit("seam Q25 failed")
    mf, Ef7 = q25["aux"]["mf"], q25["aux"]["Ef"]
    Xf = inner.Xf
    fc25 = K58.read_json(Q25_FROZEN)
    th7, rec7 = p4.gen_fit_record(mf, Ef7, Xf.n, Xf.y)
    ref7 = np.asarray(fc25["params"]["theta"], np.float64)
    seam_z = {"theta_bitwise": bool(np.array_equal(th7, ref7)), "loss_bitwise": bool(rec7["loss"] == fc25["refit"]["loss"]),
              "fit_questions": [int(len(Xf)), int(fc25["refit"]["fit_questions"])]}
    seam_z["ok"] = bool(seam_z["theta_bitwise"] and seam_z["loss_bitwise"]
                        and seam_z["fit_questions"][0] == seam_z["fit_questions"][1])
    log(f"seam Z {seam_z}")
    if not seam_z["ok"]:
        raise SystemExit("seam Z failed")
    lg, trec = EV.inner_opp3(inner, M, pr)
    seam_t = {"mean_max_q": [trec["mean_max_q_record"], ev["term"]["mean_max_q_record"]],
              "games_in_questions": [trec["games_in_questions"], ev["term"]["games_in_questions"]]}
    seam_t["ok"] = bool(seam_t["mean_max_q"][0] == seam_t["mean_max_q"][1]
                        and seam_t["games_in_questions"][0] == seam_t["games_in_questions"][1])
    if not seam_t["ok"]:
        raise SystemExit(f"seam T failed {seam_t}")
    O_fit = K.question_term(inner.d["inner"]["S_fit"], lg)
    z, _t = p4.gen_logits(ref7, mf, Ef7, Xf.n)
    ws, frec = K.fit_weights([K.log_softmax(z), O_fit], Xf.y)
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for name, src in (("prototypes.npz", K.PROTO), ("c49_head.npz", K.HEAD)):
        dst = os.path.join(OUT, name)
        shutil.copyfile(src, dst)
        files[name] = {"source": EV.rel(src), "sha256": K58.sha256_file(dst)}
        if files[name]["sha256"] != K58.sha256_file(src):
            raise SystemExit(f"{name}: copy differs")
    if files["prototypes.npz"]["sha256"] != krec["prototypes_file"]["sha256"]:
        raise SystemExit("prototypes.npz is not kernel.json's")
    rep = {"format": "c10-prereg64 rank frozen_config v1", "created": EV.now(), "seconds": round(time.time() - t0, 1),
           "what": "Q64 = w_A * log_softmax(Q25 frozen gen_logits) + w_O * OPP3 (10 x mean over the question's games of "
                   "l_g, flat); (w_A, w_O) fitted once on all inner fit questions (Gemini 04 scalar fit); merge "
                   "(step 4) applied after it inside each test pool when merge_sim passed",
           "code": cs,
           "adopted_on": {"eval": "results/prereg64/explore/eval.json", "eval_sha256_lf": K58.sha_lf(K.EVAL),
                          "gates_vs_Q59": ev["gates"], "inner_diff_vs_Q59": ev["candidate"]["vs_Q59"]["inner_diff"],
                          "inner_ci95_vs_Q59": ev["candidate"]["vs_Q59"]["inner_ci95"],
                          "merge_sim": "results/prereg64/explore/merge_sim.json",
                          "merge_sim_sha256_lf": K58.sha_lf(K.MERGE_SIM), "merge_sim_passed": ms["passed"]},
           "q25_frozen_sha256_lf": K58.sha_lf(Q25_FROZEN), "q25_theta": ref7.tolist(),
           "w_A": float(ws[0]), "w_O": float(ws[1]), "fit": frec, "fit_questions": int(len(Xf)),
           "w_at_bound": [bool(w <= 1e-8 or w >= K.W_BOUND - 1e-8) for w in ws],
           "term": {"name": EV.TERM, "tau1": K.TAU1, "tau2": K.TAU2, "b": K.BIAS, "scale": K.SCALE,
                    "kernel": "results/prereg64/explore/kernel.json", "kernel_sha256_lf": K58.sha_lf(K.KERNEL),
                    "files": files, "classes": krec["classes"]["union"],
                    "inner_mean_max_q": trec["mean_max_q_record"], "c49_ckpt_sha256": K.C49_CKPT_SHA},
           "merge": {"apply": bool(ms["passed"]), "lam": K.MERGE_LAM, "threshold": K.MERGE_THR,
                     "n_div": K.MERGE_NDIV, "beta": K.MERGE_BETA,
                     "vectors": "c49 player side of the test questions' games, data/prereg56/emb_rank/<tag>/pool_<set>.npz"},
           "record_only": {"oof_weights_by_fold": ev["weights_by_fold"]},
           "seams": {"Q25_oof": seam, "Z_seven_term_refit_equals_frozen_Q25": seam_z, "T_term_is_the_judged_one": seam_t},
           "peak_working_set_gb": EV.peak_gb(st)}
    K58.write_json(FROZEN, rep)
    log(f"frozen Q64: w_A {ws[0]:.4f}, w_O {ws[1]:.4f}; {time.time() - t0:.0f}s, peak {EV.peak_gb(st)} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
