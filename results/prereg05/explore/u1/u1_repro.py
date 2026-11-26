"""prereg05 exploration U1, step 1: reproduce PREREG_04 tune's inner OOF of Q00 (R20), Q01 and Q02 and save them.

    .venv/Scripts/python.exe results/prereg05/explore/u1/u1_repro.py
    -> results/prereg05/explore/u1/u1_oof.npz, u1_repro.json

Inner pseudo-questions only (prereg02 fit / eval rounds of split-v1 inner players, PREREG_03 single-colour set);
validation files are never opened (u1_lib guard). Nothing outside results/prereg05/explore/u1 is written.
The OOF numbers must equal results/prereg04/rank/tune_report.json (rank-balanced, per level, per n bin, single colour,
fold thetas) before anything is analysed.
"""

import json
import os
import time

import numpy as np

import u1_lib as U

T0 = time.time()
OUT = U.HERE
TUNE = os.path.join(U.REPO, "results", "prereg04", "rank", "tune_report.json")
RECORDED = {"Q00": 0.5854702521055817, "Q01": 0.6042484782462081, "Q02": 0.6101211982700248}
TOL = 1e-12


def main():
    tr = json.load(open(TUNE, encoding="utf-8"))
    rec_combo = {r["id"]: r for r in tr["combinations"]}
    d = U.load_inputs()
    inner = d["inner"]
    # inputs equal the ones tune hashed
    tune_inputs = tr["inputs_sha256"]
    same_inputs = {k: (tune_inputs.get(k) == v) for k, v in dict(inner["inputs"], **d["sha"]).items()}
    print("inputs equal to tune_report:", same_inputs, flush=True)
    if not all(same_inputs.values()):
        raise SystemExit("inputs differ from the ones PREREG_04 tune used")

    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    out, checks = {}, {}
    runs = {"Q00": lambda: U.run_q00(d),
            "Q01": lambda: U.run_gen(d, ["OP", "O4"]),
            "Q02": lambda: U.run_gen(d, ["OP", "O4", "SWA", "SWB"])}
    for cid, fn in runs.items():
        P, Pm, thetas = fn()
        m, sc = U.metrics(U.rule_b_np(P), Xe.y, Xe.n, Xe.player)
        mm, _ = U.metrics(U.rule_b_np(Pm), Xm.y, Xm.n, Xm.player)
        r = rec_combo[cid]["oof"]
        rmono = rec_combo[cid]["single_colour"]
        chk = {
            "rank_balanced": m["rank_balanced"], "recorded": RECORDED[cid],
            "abs_diff": abs(m["rank_balanced"] - RECORDED[cid]),
            "question_mean_diff": abs(m["question_mean"] - r["question_mean"]),
            "per_level_max_diff": max(abs(m["per_level"][lv] - r["per_level"][lv]) for lv in U.LEVELS),
            "per_n_bin_max_diff": max(abs(m["per_n_bin"][b]["rank_balanced"] - r["per_n_bin"][b]["rank_balanced"])
                                      for b in r["per_n_bin"]),
            "pred_distribution_equal": m["pred_distribution"] == r["pred_distribution"],
            "single_colour_rank_balanced": mm["rank_balanced"],
            "single_colour_recorded": rmono.get("rank_balanced"),
            "single_colour_abs_diff": abs(mm["rank_balanced"] - rmono["rank_balanced"]),
            "fold_theta_max_diff": max(abs(a - b) for th, fr in zip(thetas, rec_combo[cid]["folds"])
                                       for a, b in zip(th, fr["theta"])),
        }
        chk["passed"] = bool(chk["abs_diff"] <= TOL and chk["question_mean_diff"] <= TOL
                             and chk["per_level_max_diff"] <= TOL and chk["per_n_bin_max_diff"] <= TOL
                             and chk["pred_distribution_equal"] and chk["single_colour_abs_diff"] <= TOL
                             and chk["fold_theta_max_diff"] <= TOL)
        checks[cid] = chk
        print(f"{cid}: OOF rank-balanced {m['rank_balanced']!r} (recorded {RECORDED[cid]!r}, diff "
              f"{chk['abs_diff']:.3g}); single-colour {mm['rank_balanced']:.6f}; passed {chk['passed']} "
              f"({time.time() - T0:.0f}s)", flush=True)
        out[cid] = (P, Pm, thetas)
    if not all(c["passed"] for c in checks.values()):
        raise SystemExit("reproduction failed; nothing saved")

    # ---- question-level descriptors
    keys = inner["keys"]
    players = inner["players"]
    colour = np.array(keys["color"])
    is_black = (colour == "B").astype(np.int64)

    def padded(groups, width=20):
        G = np.full((len(groups), width), -1, dtype=np.int32)
        for q, g in enumerate(groups):
            G[q, :len(g)] = g
        return G

    eval_rows = inner["pseudo"]["eval"]                       # (round, player index, positions)
    eval_groups = inner["eval_groups"]
    assert len(eval_groups) == len(Xe) and all(len(g) == n for g, n in zip(eval_groups, Xe.n))
    nb_eval = np.array([is_black[g].sum() for g in eval_groups], dtype=np.int64)
    mono_groups = ms["groups"]
    mono_round = np.array([q[0] for q in ms["questions"]], dtype=np.int64)
    bank_size = np.array([len(p["bank"]) for p in players], dtype=np.int64)
    save = {
        "levels": np.array(U.LEVELS),
        # eval (registered tune evaluation set, 127,200 mixed-colour questions)
        "eval_P_Q00": out["Q00"][0], "eval_P_Q01": out["Q01"][0], "eval_P_Q02": out["Q02"][0],
        "eval_y": Xe.y, "eval_n": Xe.n.astype(np.int64), "eval_player": Xe.player, "eval_fold": Xe.fold,
        "eval_round": np.array([r for r, _, _ in eval_rows], dtype=np.int64),
        "eval_n_black": nb_eval, "eval_games": padded(eval_groups),
        # single-colour inner pseudo-questions (PREREG_03 rev 3; 120,544), OOF with each fold's parameters
        "mono_P_Q00": out["Q00"][1], "mono_P_Q01": out["Q01"][1], "mono_P_Q02": out["Q02"][1],
        "mono_y": Xm.y, "mono_n": Xm.n.astype(np.int64), "mono_player": Xm.player, "mono_fold": Xm.fold,
        "mono_colour": ms["colour"], "mono_round": mono_round, "mono_games": padded(mono_groups),
        # players (pseudo_inner.json order; 'player' arrays index this)
        "player_id": np.array([p["player_id"] for p in players]),
        "player_level": np.array([U.LEVELS.index(p["rank"]) for p in players], dtype=np.int64),
        "player_fold": np.array([p["fold"] for p in players], dtype=np.int64),
        "player_bank_size": bank_size,
        "player_n_inner_games": np.array([p["n_games"] for p in players], dtype=np.int64),
        # inner games (inner_keys.npz order; 'games' arrays index this)
        "game_colour": colour, "game_level": np.asarray(keys["label"], dtype=np.int64),
    }
    path = os.path.join(OUT, "u1_oof.npz")
    np.savez_compressed(path, **save)
    sha = U.p3.sha256_file(path)
    rep = {
        "what": "prereg05 exploration U1 step 1: PREREG_04 tune inner OOF of Q00 / Q01 / Q02 reproduced and saved "
                "(inner pseudo-questions only; design-stage numbers, not validation results)",
        "created": U.p4.now_iso(), "seconds": round(time.time() - T0, 1),
        "checks": checks, "tolerance": TOL,
        "inputs_sha256": dict(inner["inputs"], **d["sha"]),
        "inputs_equal_to_prereg04_tune_report": same_inputs,
        "fold_thetas": {cid: v[2] for cid, v in out.items()},
        "questions": {"eval": int(len(Xe)), "single_colour": int(len(Xm)), "fit": int(len(inner["X_fit"]))},
        "single_colour_sha256": ms["sha256"],
        "output": {"file": "results/prereg05/explore/u1/u1_oof.npz", "sha256": sha, "bytes": os.path.getsize(path)},
        "git_head": U.p3.git_head(),
        "environment": U.p3.environment(),
    }
    with open(os.path.join(OUT, "u1_repro.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB, sha256 {sha[:12]}) in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
