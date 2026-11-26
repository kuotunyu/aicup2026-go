"""prereg64 step 4: the same-player soft-merge simulation on E n H, copied from Gemini's
data/advice/gemini_1121/03_comp2_and_final_checks.py lines 591-878 (test_c2_3_..., simulation part), with the
PLAN / frozen.json setting (threshold 0.65, (S - 0.65) / 0.35, n_j / 12, beta 1.0, lam 0.3) instead of Gemini's
(null p99.99 threshold, n_j / 15, beta 0.8). No inner, no test file. Called by p64_eval.py mergesim.
"""
import gc
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg39", "explore"),
          os.path.join(REPO, "results", "prereg58", "explore"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p58_lib as K58  # noqa: E402
import p64_lib as K  # noqa: E402

guard = K58.guard
MODES = (("zero_repeat_safety", 0.0), ("moderate_repeat_35pct", 0.35), ("paired_repeat_100pct", 1.0))
POOLS = 20
SEED0 = 9000
RULE = "0% repeats: mean gain 0 with min = max = 0; 35% repeats: mean gain > 0"


def holdout_bit_s25(rank_str, pid_str):       # 03 lines 87-89
    h = hashlib.sha256(f"c10-prereg25|holdout|{rank_str}:{pid_str}".encode("utf-8")).hexdigest()[:8]
    return int(int(h, 16) % 10 == 0)


def run(log=print):
    from collections import defaultdict
    import p09_forward as FW09
    import p10_fuse as F10
    import p19_forward as FW19
    import p25_forward as FW25
    import p39_eval as P39E

    with np.load(guard(K58.FEATS_META), allow_pickle=False) as zm:
        all_pid = np.asarray(zm["player_id"]).astype(str)
        all_col = np.asarray(zm["color"]).astype(str)
        all_rk = np.asarray(zm["rank"]).astype(str)
    with np.load(guard(K.OWN_E), allow_pickle=False) as ze:
        if str(ze["ckpt_sha256"]) != K.C49_CKPT_SHA:
            raise SystemExit("store_E is not c49")
        emb_e = K.l2(ze["emb"])
        cnt_e = np.asarray(ze["count"], np.int64)
        sr_e = np.asarray(ze["store_row"], np.int64)
    by_p = defaultdict(lambda: {"B": [], "W": [], "rk_str": ""})
    for i in range(len(sr_e)):
        if cnt_e[i] > 0:
            p = all_pid[sr_e[i]]
            by_p[p][all_col[sr_e[i]]].append(i)
            by_p[p]["rk_str"] = all_rk[sr_e[i]]

    q09_map = FW09.load_map(guard(os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")))
    s25_models = FW25.load_models(guard(os.path.join(REPO, "data", "prereg25", "models")))
    gE = P39E.read_list("E")
    gE2 = P39E.read_list("E2")
    with open(guard(os.path.join(REPO, "data", "prereg10", "mock", "sets.json")), encoding="utf-8") as f:
        mock_n = len(json.load(f)["games"])
    mock_means = np.zeros((mock_n, 2, 770), np.float32)
    mock_counts = np.zeros((mock_n, 2), np.int64)
    mock_usable = np.zeros(mock_n, bool)
    for mp in F10.chunk_files(guard(os.path.join(REPO, "data", "prereg10", "mock"))):
        with np.load(guard(mp), allow_pickle=False) as z:
            r = np.asarray(z["ref"], np.int64)
            f_ = np.asarray(z["failed"], bool)
            c = np.asarray(z["counts"], np.int64)
            mock_means[r] = np.asarray(z["means"], np.float32)
            mock_counts[r] = c
            mock_usable[r] = (~f_) & (c[:, 0] > 0)

    def load_e_p1(name, games):
        n = len(games)
        means = np.zeros((n, 2, 770), np.float32)
        counts = np.zeros((n, 2), np.int64)
        usable = np.zeros(n, bool)
        for ch in F10.chunk_files(guard(os.path.join(REPO, "data", "prereg39", "p1", name))):
            with np.load(guard(ch), allow_pickle=False) as z:
                r = np.asarray(z["ref"], np.int64)
                f_ = np.asarray(z["failed"], bool)
                c = np.asarray(z["counts"], np.int64)
                means[r] = np.asarray(z["means"], np.float32)
                counts[r] = c
                usable[r] = (~f_) & (c[:, 0] > 0)
        reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
        idx = np.flatnonzero(reuse >= 0)
        if len(idx):
            means[idx] = mock_means[reuse[idx]]
            counts[idx] = mock_counts[reuse[idx]]
            usable[idx] = mock_usable[reuse[idx]]
        return means, counts, usable

    mE1, cE1, uE1 = load_e_p1("E", gE)
    mE2, cE2, uE2 = load_e_p1("E2", gE2)
    del mock_means, mock_counts, mock_usable
    mE = np.concatenate([mE1, mE2], axis=0)
    cE = np.concatenate([cE1, cE2], axis=0)
    uE = np.concatenate([uE1, uE2], axis=0)
    del mE1, cE1, uE1, mE2, cE2, uE2
    gc.collect()
    H_E = [np.zeros((len(uE), FW19.H_DIM), np.float32) for _ in s25_models]
    for s_b in range(0, len(uE), 8192):
        e_b = min(len(uE), s_b + 8192)
        xb = FW09.standardise_x(np.asarray(FW09.x_raw(mE[s_b:e_b], cE[s_b:e_b]), np.float64), q09_map["mu_x"],
                                q09_map["sd_x"])
        for par, Hs in zip(s25_models, H_E):
            Hs[s_b:e_b] = FW19.h_np(par, xb).astype(np.float32)
        del xb
    del mE, cE
    gc.collect()
    row_to_ep1 = {int(g["row"]): i for i, g in enumerate(gE + gE2)}
    sr_to_ce = {int(r): i for i, r in enumerate(sr_e.tolist())}
    eh_players = []
    for p, d in by_p.items():
        if holdout_bit_s25(d["rk_str"], p) == 1 and len(d["B"]) >= 20 and len(d["W"]) >= 20:
            eh_players.append(p)
    log(f"E n H players {len(eh_players)}")

    SM = K58.score_matrix()
    rk_idx = {r: i for i, r in enumerate(K58.RANKS)}
    modes = {}
    for mode_name, rep_frac in MODES:
        gains, base_scs, changed, edges = [], [], [], []
        for sd in range(POOLS):
            rng_s = np.random.default_rng(SEED0 + sd)
            perm_p = rng_s.permutation(eh_players)
            q_ce_idxs, q_p1_idxs, q_cols_s, q_y_s, q_ng_s = [], [], [], [], []
            for p in perm_p:
                y_val = rk_idx[by_p[p]["rk_str"]]
                n_q_for_p = 2 if (rng_s.random() < rep_frac) else 1
                for q_rep in range(n_q_for_p):
                    cch = rng_s.choice(["B", "W"])
                    kv = int(rng_s.choice([5, 10, 15, 20]))
                    avail = [sr_e[i] for i in by_p[p][cch] if int(sr_e[i]) in row_to_ep1 and uE[row_to_ep1[int(sr_e[i])]]]
                    if len(avail) < 2 * kv:
                        continue
                    perm_g = rng_s.permutation(avail)
                    chosen_sr = perm_g[q_rep * kv:(q_rep + 1) * kv]
                    q_ce_idxs.append([sr_to_ce[int(r)] for r in chosen_sr])
                    q_p1_idxs.append([row_to_ep1[int(r)] for r in chosen_sr])
                    q_cols_s.append(str(cch))
                    q_y_s.append(y_val)
                    q_ng_s.append(kv)
            eg_flat, qg_local = [], []
            for grp in q_ce_idxs:
                st_l = len(eg_flat)
                for gi in grp:
                    eg_flat.append(emb_e[gi])
                qg_local.append(list(range(st_l, len(eg_flat))))
            S_sim = K.cc_whiten_sim(np.array(eg_flat), qg_local, q_cols_s)
            _, _, lp_s, _ = FW19.set_term(s25_models, H_E, q_p1_idxs, len(uE), scale=4.0)
            lp_m, W_adj = K.merge(lp_s, S_sim, q_ng_s)
            y_arr = np.array(q_y_s, np.int64)
            d0, d1 = K58.decode(lp_s), K58.decode(lp_m)
            sc0 = K58.rank_balanced(SM[d0, y_arr], y_arr)
            sc1 = K58.rank_balanced(SM[d1, y_arr], y_arr)
            base_scs.append(sc0)
            gains.append(sc1 - sc0)
            changed.append(int((d0 != d1).sum()))
            edges.append(int((np.triu(W_adj, 1) > 0).sum()))
        modes[mode_name] = {"repeat_fraction": rep_frac, "n_unseen_EH_players": len(eh_players),
                            "base_S25_rb": float(np.mean(base_scs)),
                            "merged_minus_base_mean": float(np.mean(gains)),
                            "merged_minus_base_min_max": [float(np.min(gains)), float(np.max(gains))],
                            "merged_minus_base_by_pool": [float(x) for x in gains],
                            "questions_changed_mean": float(np.mean(changed)),
                            "edges_mean": float(np.mean(edges))}
        log(json.dumps({mode_name: {k: v for k, v in modes[mode_name].items() if k != "merged_minus_base_by_pool"}}))
    z = modes["zero_repeat_safety"]
    passed = bool(z["merged_minus_base_mean"] == 0.0 and z["merged_minus_base_min_max"] == [0.0, 0.0]
                  and modes["moderate_repeat_35pct"]["merged_minus_base_mean"] > 0.0)
    return {"what": "Gemini 03 lines 591-878 simulation part (E n H players: S25 holdout bit 1, >= 20 games of each "
                    "colour with c49 and P1; 20 pools per mode, seeds 9000 + sd; base = S25 set term 4 * lp) with the "
                    "PLAN setting threshold 0.65, (S - 0.65) / 0.35, n_j / 12, beta 1.0, lam 0.3",
            "setting": {"lam": K.MERGE_LAM, "threshold": K.MERGE_THR, "n_div": K.MERGE_NDIV, "beta": K.MERGE_BETA},
            "gemini_setting_of_03_json": {"threshold": "null p99.99", "n_div": 15, "beta": 0.8},
            "modes": modes, "rule": RULE, "passed": passed}
