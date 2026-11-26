"""prereg59: freeze the 8-term fusion Q59 = Q25's seven terms + OPP2 (docs/PREREG_59_OPPONENT_IDENTITY.md section 2),
the way tools/p25_set.py froze Q25: ONE fit of theta on all inner fit pseudo-questions (p04_rank_fuse.gen_fit_record).
CPU, .venv, 4 cores, <= 3 GB (p58_lib.limits). Run once.

    PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/prereg59/explore/p59_freeze.py
    -> results/prereg59/rank/frozen_config.json, prototypes.npz, s0_head.npz (byte copies of the judged files)

No inner score of Q59 or of any variant is computed here (no OOF run with the new term, no metric of the refit): the
judgement stays eval.json. The Q25 OOF that p58_eval.build_q25 runs is the seam of the judgement (an existing number,
checked bit for bit). Seam Z: the same call on the seven terms gives the frozen Q25 theta bit for bit. Never
data/local, never a test-set file, never tools/score.py.
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

import p59_eval as EV  # noqa: E402
import p59_lib as K  # noqa: E402

E58, K58 = EV.E58, EV.K58
REPO = EV.REPO
OUT = os.path.join(REPO, "results", "prereg59", "rank")
FROZEN = os.path.join(OUT, "frozen_config.json")
Q25_FROZEN = os.path.join(REPO, "results", "prereg25", "rank", "frozen_config.json")
DOC = os.path.join(REPO, "docs", "PREREG_59_OPPONENT_IDENTITY.md")
log = EV.log


def main():
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(FROZEN):
        raise SystemExit("frozen_config.json exists (run once)")
    for p in (K.KERNEL, K.SEAM, K.EVAL, DOC, os.path.abspath(__file__)):
        if not EV.committed_clean(p):
            raise SystemExit(f"{p} is not committed and clean")
    ev = K58.read_json(K.EVAL)
    if ev.get("adopt") is not True or ev.get("smoke"):
        raise SystemExit("eval.json does not say adopt")
    M, par, pr, krec = EV.load_kernel(K.KERNEL, K.PROTO)
    if K58.sha_lf(K.KERNEL) != ev["kernel_sha256_lf"] or krec["prototypes_file"]["sha256"] != ev["prototypes_sha256"]:
        raise SystemExit("kernel.json / prototypes.npz are not the judged files")
    mods, inner, terms7, Emap, q25, seam = E58.build_q25()
    _E41, _E23, r, _K23 = mods
    import p04_rank_fuse as p4
    cs = EV.code_state()
    if cs["prereg59_dirty"] or cs["imported_dirty"]:
        raise SystemExit(f"not committed and clean: {cs['prereg59_dirty']} {cs['imported_dirty']}")
    log(f"seam Q25 {seam['ok']} ({time.time() - t0:.0f}s)")
    if not seam["ok"]:
        raise SystemExit("seam Q25 failed")
    mf, Ef7 = q25["aux"]["mf"], q25["aux"]["Ef"]
    Xf = inner.Xf
    # ---- seam Z: the 7-term fit on all inner fit questions = the frozen Q25 theta, bit for bit
    fc25 = K58.read_json(Q25_FROZEN)
    th7, rec7 = p4.gen_fit_record(mf, Ef7, Xf.n, Xf.y)
    ref7 = np.asarray(fc25["params"]["theta"], np.float64)
    seam_z = {"theta_bitwise": bool(np.array_equal(th7, ref7)), "max_abs_diff": float(np.abs(th7 - ref7).max()),
              "loss": [rec7["loss"], fc25["refit"]["loss"]], "loss_bitwise": bool(rec7["loss"] == fc25["refit"]["loss"]),
              "fit_questions": [int(len(Xf)), int(fc25["refit"]["fit_questions"])]}
    seam_z["ok"] = bool(seam_z["theta_bitwise"] and seam_z["loss_bitwise"]
                        and seam_z["fit_questions"][0] == seam_z["fit_questions"][1])
    log(f"seam Z {seam_z}")
    if not seam_z["ok"]:
        raise SystemExit("seam Z failed: this path does not reproduce the frozen Q25 parameters")
    # ---- the new term on the fit questions (features only), then ONE fit of the 8-term theta
    lg, trec = EV.inner_term(inner, M, par, pr, False)
    seam_t = {"mean_max_q": [trec["mean_max_q_record"], ev["term"]["mean_max_q_record"]],
              "games_in_questions": [trec["games_in_questions"], ev["term"]["games_in_questions"]]}
    seam_t["ok"] = bool(seam_t["mean_max_q"][0] == seam_t["mean_max_q"][1]
                        and seam_t["games_in_questions"][0] == seam_t["games_in_questions"][1])
    if not seam_t["ok"]:
        raise SystemExit(f"seam T failed: the term is not the judged one {seam_t}")
    E_fit = K.question_term(inner.d["inner"]["S_fit"], lg)
    terms8 = terms7 + [EV.TERM]
    th8, rec8 = p4.gen_fit_record(mf, Ef7 + [E_fit], Xf.n, Xf.y)
    params = p4.gen_params(th8, terms8)
    w = params["w"]
    w_lo, w_hi = r.rf.BOUNDS["w"]
    bounds_hit = {t: ("lower" if w[t] <= w_lo + 1e-8 else "upper") for t in terms8
                  if w[t] <= w_lo + 1e-8 or w[t] >= w_hi - 1e-8}
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for name, src in (("prototypes.npz", K.PROTO), ("s0_head.npz", K58.HEAD_NPZ)):
        dst = os.path.join(OUT, name)
        shutil.copyfile(src, dst)
        files[name] = {"source": EV.rel(src), "sha256": K58.sha256_file(dst)}
        if files[name]["sha256"] != K58.sha256_file(src):
            raise SystemExit(f"{name}: copy differs")
    if files["prototypes.npz"]["sha256"] != krec["prototypes_file"]["sha256"] or \
            files["s0_head.npz"]["sha256"] != krec["head"]["sha256"]:
        raise SystemExit("the copied files are not kernel.json's")
    fw = ev["weight_of_new_term"]["by_fold"]
    rep = {"format": "c10-prereg59 rank frozen_config v1", "created": EV.now(), "seconds": round(time.time() - t0, 1),
           "what": "PREREG_59 section 2: frozen Q59 = Q25's seven terms + OPP2 (10 x the mean over the question's games "
                   "of log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9)); one fit of the 8-term theta on all inner fit "
                   "pseudo-questions; no inner score computed here (the judgement is explore/eval.json)",
           "code": cs,
           "adopted_on": {"eval": "results/prereg59/explore/eval.json", "eval_sha256_lf": K58.sha_lf(K.EVAL),
                          "gates": ev["gates"], "inner_diff": ev["candidate"]["vs"]["inner_diff"],
                          "inner_ci95": ev["candidate"]["vs"]["inner_ci95"],
                          "mono_diff": ev["candidate"]["vs"]["mono_diff"],
                          "mono_ci95": ev["candidate"]["vs"]["mono_ci95"]},
           "combo": {"id": "Q59", "terms": terms8, "path": "gen", "pool": "log", "calibrated": True, "opening": True,
                     "base": "Q25 (results/prereg25/rank/frozen_config.json: the seven terms, their models and maps)",
                     "q25_frozen_sha256_lf": K58.sha_lf(Q25_FROZEN)},
           "params": params,
           "term": {"name": EV.TERM, "tau1": par[0], "tau2": par[1], "b": par[2], "scale": K.SCALE, "eps": K.EPS,
                    "kernel": "results/prereg59/explore/kernel.json (M, the grid choice)",
                    "kernel_sha256_lf": K58.sha_lf(K.KERNEL), "files": files,
                    "classes": {"head": krec["classes"]["head"], "prototypes": krec["classes"]["prototypes"]["total"]},
                    "hd_groups": ["HD 0", "HD 1", "HD 2+"], "colour": ["player B", "player W"],
                    "inner_mean_max_q": trec["mean_max_q_record"],
                    "inner_hd_groups_of_question_games": trec["hd_groups_of_question_games"],
                    "inner_player_colour_of_question_games": trec["player_colour_of_question_games"]},
           "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + sum_j w_j E_j + B beta)), E_OPP2 = 10 x (S @ l_g) with S "
                      "the question's mean-pooling matrix over its parsed games; rule B (tools/decide.py)",
           "refit": {"loss": rec8["loss"], "starts": rec8["starts"], "fit_questions": int(len(Xf)),
                     "bounds_hit": bounds_hit, "w_bounds": [w_lo, w_hi],
                     "seven_term_loss_same_questions": rec7["loss"]},
           "record_only": {"oof_weight_of_new_term_by_fold_in_the_judgement": fw,
                           "q25_params": {k: fc25["params"][k] for k in ("a", "gamma", "w")}},
           "seams": {"Q25_oof": seam, "Z_seven_term_refit_equals_frozen_Q25": seam_z, "T_term_is_the_judged_one": seam_t},
           "environment": r.venv_versions(False), "peak_working_set_gb": EV.peak_gb(st)}
    K58.write_json(FROZEN, rep)
    log(f"frozen Q59: w {({k: round(v, 4) for k, v in w.items()})}, a {params['a']:.4f}, gamma {params['gamma']:.4f}; "
        f"bounds {bounds_hit}; loss {rec8['loss']:.6f} (7 terms {rec7['loss']:.6f}); {time.time() - t0:.0f}s, peak "
        f"{EV.peak_gb(st)} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
