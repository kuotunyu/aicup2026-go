"""Parity checks of hs_lib against KataGo's own python paths (CPU, real checkpoint, fit players only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_parity.py [--skip-c]

(a) issue #1154: empty board, Chinese rules, komi 7.5, the proyear_2020 metadata the issue printed, plain 'model'
    weights -> policy at C16 / D16 / E16, through KataGo's GameState.get_model_outputs and through hs_lib.Scorer.
(b) >= 20 positions from real fit games (data/prereg07/fixtures/fit_sample.json, made by hs_fixtures.py), 4 profiles:
    hs_lib's batched logits vs GameState.get_model_outputs (in process) and vs humanslnet_server.py (KataGo's own
    stdin/stdout server, run unchanged as a subprocess): max abs difference of the probabilities (unmasked and masked).
(c) 2 fit games each from D, A, 1D, 4D, 6D x 11 rank profiles: where the log-likelihood curve peaks per level.
    Criterion written down before the run: peaks non-decreasing in the order D, A, 1D, 4D, 6D and peak(6D) > peak(D).

Writes results/prereg07/explore/pilot_humansl/hs_parity.json (numbers for README.md). The same functions are called
by test_hs_lib.py.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (installs the data/local guard)

FIXTURE = os.path.join(H.REPO, "data", "prereg07", "fixtures", "fit_sample.json")
OUT = os.path.join(HERE, "hs_parity.json")

ISSUE_1154 = {
    "url": "https://github.com/lightvector/KataGo/issues/1154",
    "points": ["C16", "D16", "E16"],
    "tensor_pos": [3 * 19 + 2, 3 * 19 + 3, 3 * 19 + 4],     # moves_and_probs0[59:62] on the empty board, y=3, x=2..4
    "python": [0.036621157079935074, 0.15681298077106476, 5.05671305290889e-05],
    "engine": [0.0366211012, 0.156812504, 5.05667886e-05],
    "metadata": "SGFMetadata(inverseBRank=1, inverseWRank=1, bIsHuman=True, wIsHuman=True, tcIsUnknown=True, "
                "gameDate=2020-06-01, source=5 (GoGoD)); load_model(use_swa='') = plain model weights; "
                "GameState((19,19), RULES_CHINESE) (komi 7.5)",
}
PARITY_PROFILES = ["rank_10k", "rank_1d", "rank_5d", "fox20220601_3d_oppunknown"]
PARITY_LEVELS = ["D", "A", "3D", "6D"]
PARITY_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95]
CURVE_LEVELS = ["D", "A", "1D", "4D", "6D"]
CURVE_PROFILES = ["rank_20k", "rank_15k", "rank_12k", "rank_9k", "rank_6k", "rank_3k", "rank_1k", "rank_2d",
                  "rank_4d", "rank_6d", "rank_9d"]


def load_fixture():
    if not os.path.exists(FIXTURE):
        raise H.HSError(f"{FIXTURE} missing: run hs_fixtures.py first")
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def proyear_2020_meta():
    return H.SGFMetadata(inverseBRank=1, inverseWRank=1, bIsHuman=True, wIsHuman=True, tcIsUnknown=True,
                         gameDate=datetime.date(2020, 6, 1), source=H.SGFMetadata.SOURCE_GOGOD)


# ---------------------------------------------------------------- (a)
def check_issue_1154(model):
    import torch
    meta = proyear_2020_meta()
    p_gs, gs = H.gamestate_policy(model, [], meta, rules=H.GameState.RULES_CHINESE)
    feats = H.Features(model.config, H.POS_LEN)
    bin_nchw, glob = gs.get_input_features(feats)
    row = meta.get_metadata_row(nextPlayer=H.Board.BLACK, boardArea=361)
    with torch.inference_mode():
        out = model(torch.tensor(bin_nchw), torch.tensor(glob), input_meta=torch.tensor(row).reshape(1, -1))
        p_ours = torch.softmax(out[0][0][0, 0, :].float(), dim=0).numpy().astype(np.float64)
    pos = ISSUE_1154["tensor_pos"]
    rec = {"issue": ISSUE_1154, "gamestate": [float(p_gs[i]) for i in pos], "hs_direct": [float(p_ours[i]) for i in pos]}
    rec["max_abs_diff_gamestate_vs_issue_python"] = float(max(abs(a - b) for a, b in zip(rec["gamestate"],
                                                                                          ISSUE_1154["python"])))
    rec["max_abs_diff_gamestate_vs_issue_engine"] = float(max(abs(a - b) for a, b in zip(rec["gamestate"],
                                                                                          ISSUE_1154["engine"])))
    rec["max_rel_diff_gamestate_vs_issue_python"] = float(max(abs(a - b) / b for a, b in zip(rec["gamestate"],
                                                                                              ISSUE_1154["python"])))
    rec["max_abs_diff_hs_vs_gamestate"] = float(np.max(np.abs(p_ours - p_gs)))
    return rec


# ---------------------------------------------------------------- (b)
def parity_positions(fixture):
    """-> list of (game dict, sorted target-move indices) for PARITY_LEVELS, plus up to 2 ko positions."""
    chosen = []
    for lv in PARITY_LEVELS:
        g = next(x for x in fixture["games"] if x["rank"] == lv)
        f = H.featurize_game(g["sgf_content"], g["color"])
        T = len(f["move"])
        idx = sorted({min(T - 1, int(round(fr * (T - 1)))) for fr in PARITY_FRACTIONS})
        ko = [int(i) for i in np.nonzero(f["bin"][:, 6].any(axis=(1, 2)))[0] if int(i) not in idx][:1]
        chosen.append((g, sorted(idx + ko), f))
    return chosen


def check_gamestate(model, fixture, profile_names=PARITY_PROFILES):
    """hs_lib Scorer (batched, CPU fp32) vs GameState.get_model_outputs (batch 1) at the same positions."""
    profiles = H.profiles_from_names(profile_names)
    sc = H.Scorer(model, profiles, device="cpu", precision="fp32", batch=64)
    rows = []
    worst_unmasked, worst_masked, worst_logp = 0.0, 0.0, 0.0
    mask_mismatch = 0
    for g, idx, f in parity_positions(fixture):
        moves = H.parse_moves(g["sgf_content"])
        sub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
        lp_ours = sc.logp_moves(sub)                                           # [P, N], masked
        for pi, t in enumerate(idx):
            turn = int(f["turn"][t])
            pla = int(f["pla"][t])
            for j, prof in enumerate(profiles):
                logits = sc.policy_logits(sub["bin"][pi:pi + 1], sub["glob"][pi:pi + 1], sub["pla"][pi:pi + 1],
                                          [j])[0].astype(np.float64)
                p_un = np.exp(H.log_softmax_np(logits))
                p_gs, gs = H.gamestate_policy(model, moves[:turn], prof.sgfmeta(pla))
                legal = f["legal"][t]
                gs_legal = np.zeros(H.N_POLICY, bool)
                gs_legal[H.PASS_POS] = True
                for y in range(19):
                    for x in range(19):
                        if gs.board.would_be_legal(pla, gs.board.loc(x, y)) and \
                                not gs.board.would_be_suicide(pla, gs.board.loc(x, y)):
                            gs_legal[y * 19 + x] = True
                mask_mismatch += int((gs_legal != legal).sum())
                d_un = float(np.max(np.abs(p_un - p_gs)))
                p_m = np.exp(H.log_softmax_np(np.where(legal, logits, -np.inf)))
                d_m = float(np.max(np.abs(p_m - H.masked_renorm(p_gs, legal))))
                lp_gs = float(np.log(H.masked_renorm(p_gs, legal)[f["move"][t]]))
                d_lp = abs(float(lp_ours[pi, j]) - lp_gs)
                worst_unmasked, worst_masked, worst_logp = max(worst_unmasked, d_un), max(worst_masked, d_m), \
                    max(worst_logp, d_lp)
                rows.append({"level": g["rank"], "game_id": g["game_id"], "colour": g["color"], "turn": turn,
                             "profile": prof.name, "ko_point": bool(f["bin"][t, 6].any()),
                             "max_abs_diff_unmasked": d_un, "max_abs_diff_masked": d_m,
                             "abs_diff_logp_played_move": d_lp, "logp_played_move": float(lp_ours[pi, j])})
    return {"positions": len({(r["game_id"], r["turn"]) for r in rows}), "profiles": profile_names,
            "evaluations": len(rows), "max_abs_diff_unmasked": worst_unmasked, "max_abs_diff_masked": worst_masked,
            "max_abs_diff_logp_played_move": worst_logp, "legal_mask_mismatches": mask_mismatch, "rows": rows}


def check_server(fixture, ckpt=H.CKPT_PATH, profile_names=PARITY_PROFILES, python=None):
    """KataGo's humanslnet_server.py (unchanged, subprocess, CPU) vs hs_lib at the same positions and profiles.
    The server lists only python-legal moves (would_be_legal: single-stone suicide excluded) plus pass, as
    (loc, prob) with prob from the UNMASKED softmax; we compare those entries with our unmasked probabilities."""
    model, _ = H.load_model(ckpt, device="cpu")
    profiles = H.profiles_from_names(profile_names)
    sc = H.Scorer(model, profiles, device="cpu", precision="fp32", batch=64)
    # KataGo's load_model calls torch.load without weights_only; on torch >= 2.6 that means weights_only=True, which
    # refuses this checkpoint's defaultdict entries. The server is run unchanged with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    # only after the file's SHA-256 equals the pinned one and its pickle references only the 7 pinned globals.
    H.check_checkpoint(ckpt, full_hash=True)
    if H.pickle_globals(ckpt) != H.CKPT_GLOBALS:
        raise H.HSError("checkpoint pickle globals differ from the pinned list; not running the server")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    cmd = [python or sys.executable, os.path.join(H.KATAGO_PY, "humanslnet_server.py"), "-checkpoint", ckpt,
           "-device", "cpu"]
    errf = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, cwd=H.KATAGO_PY, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=errf, text=True, env=env, bufsize=1)

    def ask(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            errf.seek(0)
            raise H.HSError("humanslnet_server.py exited: " + errf.read()[-2000:])
        return json.loads(line)["outputs"]

    worst, n_eval, n_pos = 0.0, 0, 0
    rows = []
    try:
        for g, idx, f in parity_positions(fixture):
            moves = H.parse_moves(g["sgf_content"])
            ask({"command": "start", "board_x_size": 19, "board_y_size": 19, "rules": H.RULES})
            b = H.Board(19)
            played = 0
            for t in idx:
                turn = int(f["turn"][t])
                while played < turn:
                    pla, x, y = moves[played]
                    ask({"command": "play", "pla": pla, "loc": b.loc(x, y)})
                    played += 1
                pla = int(f["pla"][t])
                n_pos += 1
                for j, prof in enumerate(profiles):
                    out = ask({"command": "get_model_outputs", "sgfmeta": prof.sgfmeta(pla).to_dict()})
                    logits = sc.policy_logits(f["bin"][t:t + 1], f["glob"][t:t + 1], f["pla"][t:t + 1], [j])[0]
                    p_un = np.exp(H.log_softmax_np(logits))
                    d = 0.0
                    for loc, prob in out["moves_and_probs0"]:
                        pos = H.PASS_POS if loc == H.Board.PASS_LOC else b.loc_y(loc) * 19 + b.loc_x(loc)
                        d = max(d, abs(float(prob) - float(p_un[pos])))
                    worst = max(worst, d)
                    n_eval += 1
                    rows.append({"level": g["rank"], "turn": turn, "profile": prof.name, "listed_moves":
                                 len(out["moves_and_probs0"]), "max_abs_diff_unmasked": d})
    finally:
        proc.stdin.close()
        proc.wait(timeout=120)
        errf.close()
    return {"positions": n_pos, "evaluations": n_eval, "max_abs_diff_unmasked": worst, "rows": rows,
            "server": "third_party/katago/python/humanslnet_server.py (unchanged), -device cpu, plain model; "
                      "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 after the SHA-256 and pickle-globals checks"}


# ---------------------------------------------------------------- (c)
def check_curves(model, fixture, profile_names=CURVE_PROFILES, levels=CURVE_LEVELS, per_level=2, batch=128,
                 device="cpu", precision="fp32"):
    profiles = H.profiles_from_names(profile_names)
    sc = H.Scorer(model, profiles, device=device, precision=precision, batch=batch)
    games, t0, n_eval = [], time.time(), 0
    for lv in levels:
        for g in [x for x in fixture["games"] if x["rank"] == lv][:per_level]:
            r = H.score_game(sc, g["sgf_content"], g["color"])
            n_eval += len(r["side"]) * len(profiles)
            games.append({"level": lv, "game_id": g["game_id"], "colour": g["color"], "target_moves":
                          int(r["counts"][0]), "sum_logp": r["sums"][0].tolist(),
                          "mean_logp": (r["sums"][0] / r["counts"][0]).tolist()})
    secs = time.time() - t0
    per_level_rec = {}
    for lv in levels:
        gs = [g for g in games if g["level"] == lv]
        tot = np.sum([g["sum_logp"] for g in gs], axis=0)
        cnt = sum(g["target_moves"] for g in gs)
        mean = tot / cnt
        w = np.exp(H.log_softmax_np(tot))
        ranks = np.array([H.rank_index(p.rank) for p in profiles], dtype=np.float64)
        per_level_rec[lv] = {"peak_profile": profiles[int(np.argmax(mean))].name,
                             "peak_rank_index": int(ranks[int(np.argmax(mean))]),
                             "mean_logp_per_move": mean.tolist(),
                             "posterior_mean_rank_index_flat_prior": float(w @ ranks),
                             "moves": int(cnt)}
    peaks = [per_level_rec[lv]["peak_rank_index"] for lv in levels]
    ok = all(a <= b for a, b in zip(peaks, peaks[1:])) and peaks[-1] > peaks[0]
    return {"profiles": profile_names, "levels": levels, "games_per_level": per_level, "games": games,
            "per_level": per_level_rec, "peaks_rank_index_in_level_order": peaks,
            "criterion": "peaks non-decreasing in the order D, A, 1D, 4D, 6D and peak(6D) > peak(D) "
                         "(written before the run)", "passed": bool(ok), "evaluations": n_eval,
            "seconds": secs, "evals_per_second": n_eval / secs, "device": device, "precision": precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
    model, state = H.load_model(H.CKPT_PATH, device="cpu")
    fixture = load_fixture()
    rep = {"what": "PREREG_07 pilot: hs_lib parity checks on CPU (fit players only)", "created_utc": H.utc_now(),
           "checkpoint": ck, "checkpoint_keys": sorted(k for k in state.keys() if not k.startswith("_hs")),
           "checkpoint_load": state["_hs_load"], "checkpoint_pickle_globals": H.pickle_globals(H.CKPT_PATH), "model_config_version": state["config"]["version"],
           "has_swa_model": "swa_model" in state, "katago_commit": H.KATAGO_COMMIT, "environment": H.environment(),
           "fixture": {"path": os.path.relpath(FIXTURE, H.REPO).replace("\\", "/"), "seed": fixture["seed"],
                       "games": len(fixture["games"])}}
    t = time.time()
    rep["a_issue_1154"] = check_issue_1154(model)
    rep["a_issue_1154"]["seconds"] = time.time() - t
    print("(a)", json.dumps({k: v for k, v in rep["a_issue_1154"].items() if k != "issue"}), flush=True)
    t = time.time()
    rep["b_gamestate"] = check_gamestate(model, fixture)
    rep["b_gamestate"]["seconds"] = time.time() - t
    print("(b) gamestate", json.dumps({k: v for k, v in rep["b_gamestate"].items() if k != "rows"}), flush=True)
    t = time.time()
    rep["b_server"] = check_server(fixture)
    rep["b_server"]["seconds"] = time.time() - t
    print("(b) server", json.dumps({k: v for k, v in rep["b_server"].items() if k != "rows"}), flush=True)
    if not a.skip_c:
        rep["c_curves"] = check_curves(model, fixture)
        print("(c)", json.dumps({k: v for k, v in rep["c_curves"].items() if k not in ("games",)}), flush=True)
    rep["seconds_total"] = time.time() - t0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
