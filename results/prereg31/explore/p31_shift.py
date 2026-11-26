"""prereg31, record-only diagnostic (NOT in PLAN.md; added 2026-11-10 after stage 0a and before any S31 training;
it decides nothing and no gate reads it).

Question: the HD[1] share of the competition-2 test games (22.8%) is higher than the training share (17.1%).
Training has the same number of games at every rank, the test set need not.  Is the difference what the rank mix
of the test questions would produce (harmless: the model input means the same thing), or does HD[1] mean something
else in the test files (a risk for the Private score if S31 were adopted)?

Method (no labels anywhere): for every test question q, the submitted probabilities p_q(r) of model 25 and the
training conditional pi(r, colour) = share of HD[1] among the training games of rank r and that colour give the
expected number of HD[1] games  e_q = sum over games g of sum_r p_q(r) * pi(r, colour_g).  Compare
sum(o_q) / sum(n_q) with sum(e_q) / sum(n_q); bootstrap over questions (HD use is clustered by player, so games
are not the unit).

Limits, to be read with the numbers: p_q is the belief of model 25, not the truth, and it is not perfectly
calibrated; pi is estimated from the training players.  A small difference is weak evidence of no shift; a large
one is a warning.  Reads: hd_stats.json (training shares, test tallies), the official test question files with HD
(paths as arguments, never data/local), results/test25/rank/rank_test_probs.csv, and the HD table of stage 0a.

Two training references.  "game-weighted": every training game of that rank and colour counts once (hd_stats.json).
"player-averaged": the mean over the players with at least 5 games of that colour of the HD[1] share of each
player (from the HD table) - this is the weighting the set model is trained under (p19_lib.sample_question draws
a level, then a player, then games) and the weighting of a test set that has one question per player.
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p31_hd as H                                                    # noqa: E402  (parse_hd, csv_rows, RANKS)

FORMAT = "c10-prereg31 hd shift v1 (record-only)"
POOLS = ("public", "private")
DRAWS = 2000
BOOT_SEED = 31
CONFIDENT = 0.8
MIN_GAMES = 5                            # the smallest question size of the set model (p19_lib.KS)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_probs(path):
    out = {}
    it = H.csv_rows(path)
    header = next(it)
    want = ["question_id", "n_games", "n_parsed"] + ["p_" + r for r in H.RANKS]
    if header != want:
        raise SystemExit("p31_shift: another header in " + path)
    for row in it:
        if row:
            out[row[0]] = [float(x) for x in row[3:]]
    return out


def read_questions(path):
    """question_id -> list of (colour index, 1 if HD[1] else 0); stops on anything unreadable (stage 0a found none)."""
    out = {}
    it = H.csv_rows(path)
    next(it)
    for row in it:
        if not row:
            continue
        games = []
        for i in range(int(row[1])):
            sgf, col = row[2 + i], row[2 + H.N_SLOTS + i]
            hd, _ = H.parse_hd(sgf)
            if hd > 1:
                raise SystemExit("p31_shift: HD >= 2 in a test file (stage 0a saw none)")
            games.append((H.COLOURS.index(col), hd))
        out[row[0]] = games
    return out


def player_average(table_path):
    """pi[r][c]: mean over players with >= MIN_GAMES games (HD <= 1) of that colour of their HD[1] share; players."""
    t = H.load_table(table_path)
    pi = [[None, None] for _ in H.RANKS]
    cnt = [[0, 0] for _ in H.RANKS]
    for r in range(len(H.RANKS)):
        for c in range(2):
            m = (t["rank"] == r) & (t["color"] == c) & (t["hd"] <= 1)
            _u, inv = np.unique(t["player_id"][m], return_inverse=True)
            n = np.bincount(inv)
            o = np.bincount(inv, weights=(t["hd"][m] == 1).astype(np.float64))
            ok = n >= MIN_GAMES
            pi[r][c] = float((o[ok] / n[ok]).mean())
            cnt[r][c] = int(ok.sum())
    return pi, cnt


def share(a, b):
    return a / b if b else None


def summarise(rows, rng):
    """rows: list of (n, o, e_le1, e_all, e_uniform, e_player).  Shares; question bootstrap of observed - expected."""
    n = sum(r[0] for r in rows)
    o = sum(r[1] for r in rows)
    e1 = sum(r[2] for r in rows)
    ea = sum(r[3] for r in rows)
    eu = sum(r[4] for r in rows)
    ep = sum(r[5] for r in rows)
    out = {"questions": len(rows), "games": n, "observed_hd1_share": share(o, n),
           "expected_model25_x_train_hd_le1": share(e1, n), "expected_model25_x_train_all": share(ea, n),
           "expected_uniform_rank_x_train_hd_le1": share(eu, n),
           "expected_model25_x_train_player_avg": share(ep, n),
           "observed_minus_expected_hd_le1": None if not n else o / n - e1 / n,
           "observed_minus_expected_player_avg": None if not n else o / n - ep / n}
    if len(rows) >= 30:
        d, dp = [], []
        k = len(rows)
        for _ in range(DRAWS):
            sn = so = se = sp = 0.0
            for _j in range(k):
                r = rows[rng.randrange(k)]
                sn += r[0]
                so += r[1]
                se += r[2]
                sp += r[5]
            d.append(so / sn - se / sn)
            dp.append(so / sn - sp / sn)
        d.sort()
        dp.sort()
        out["observed_minus_expected_hd_le1_ci95"] = [d[int(0.025 * DRAWS)], d[int(0.975 * DRAWS) - 1]]
        out["observed_minus_expected_player_avg_ci95"] = [dp[int(0.025 * DRAWS)], dp[int(0.975 * DRAWS) - 1]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    st = H.read_json(a.stats)
    if st.get("stage0a") != "PASS" or st.get("encoding") != "Z":
        raise SystemExit("p31_shift: hd_stats.json is not the stage 0a PASS / Z record")
    by = st["train"]["by_rank_colour"]
    pi_le1 = [[by[r][c]["hd1"] / (by[r][c]["hd0"] + by[r][c]["hd1"]) for c in H.COLOURS] for r in H.RANKS]
    pi_all = [[by[r][c]["hd1"] / by[r][c]["n"] for c in H.COLOURS] for r in H.RANKS]
    if sha_file(a.table) != st["table"]["sha256"]:
        raise SystemExit("p31_shift: the HD table is not the one stage 0a read")
    pi_pl, pl_count = player_average(a.table)
    probs = read_probs(a.probs)
    files, per_pool, seam = {}, {}, {}
    for pool in POOLS:
        name = H.RANK_TEST_FILES[pool]
        path = os.path.join(a.test_dir, name)
        files[name] = sha_file(path)
        if files[name] != st["test_files"][name]["sha256"]:
            raise SystemExit("p31_shift: test file differs from the one stage 0a read")
        per_pool[pool] = read_questions(path)
        tally = [[0, 0], [0, 0]]
        for games in per_pool[pool].values():
            for c, hd in games:
                tally[c][hd] += 1
        want = [x[:2] for x in st["test_rank"][pool]["n"]]
        seam[pool] = {"n_by_colour_hd": tally, "hd_stats": want, "ok": tally == want}
        if not seam[pool]["ok"]:
            raise SystemExit("p31_shift: tallies differ from hd_stats.json")
    ids = [q for pool in POOLS for q in per_pool[pool]]
    if sorted(ids) != sorted(probs) or len(set(ids)) != len(ids):
        raise SystemExit("p31_shift: question ids of the test files and the probability file differ")
    max_dev = max(abs(sum(p) - 1.0) for p in probs.values())
    R = len(H.RANKS)
    recs = []
    for pool in POOLS:
        for q, games in per_pool[pool].items():
            p = probs[q]
            row = {"pool": pool, "top": max(range(R), key=lambda r: p[r]), "pmax": max(p), "all": None,
                   "c": [None, None], "p": p}
            for sel in (None, 0, 1):
                gs = [g for g in games if sel is None or g[0] == sel]
                n = len(gs)
                o = sum(hd for _c, hd in gs)
                e1 = sum(p[r] * pi_le1[r][c] for c, _hd in gs for r in range(R))
                ea = sum(p[r] * pi_all[r][c] for c, _hd in gs for r in range(R))
                eu = sum(pi_le1[r][c] for c, _hd in gs for r in range(R)) / R
                ep = sum(p[r] * pi_pl[r][c] for c, _hd in gs for r in range(R))
                if sel is None:
                    row["all"] = (n, o, e1, ea, eu, ep)
                else:
                    row["c"][sel] = (n, o, e1, ea, eu, ep)
            recs.append(row)
    rng = random.Random(BOOT_SEED)
    out = {"format": FORMAT, "created": H.now(),
           "note": "record-only; not in PLAN.md; added after stage 0a (fb84dec) and before any S31 training; "
                   "decides nothing, no gate reads it; no labels used",
           "inputs": {"stats": a.stats.replace(os.sep, "/"), "stats_sha256_lf": H.sha_lf(a.stats),
                      "table": a.table.replace(os.sep, "/"), "table_sha256": st["table"]["sha256"],
                      "probs": a.probs.replace(os.sep, "/"), "probs_sha256": sha_file(a.probs),
                      "test_dir": a.test_dir.replace(os.sep, "/"), "test_files_sha256": files},
           "tool": {"file": "results/prereg31/explore/p31_shift.py", "sha256_lf": H.sha_lf(os.path.abspath(__file__)),
                    "python": sys.version.split()[0], "numpy": np.__version__, "draws": DRAWS,
                    "boot_seed": BOOT_SEED, "min_games": MIN_GAMES},
           "seam_tallies_equal_hd_stats": seam, "probs_max_abs_sum_minus_1": max_dev,
           "train_hd1_share_among_hd_le1": {r: {c: pi_le1[i][j] for j, c in enumerate(H.COLOURS)}
                                             for i, r in enumerate(H.RANKS)},
           "train_hd1_share_player_avg": {r: {c: pi_pl[i][j] for j, c in enumerate(H.COLOURS)}
                                          for i, r in enumerate(H.RANKS)},
           "train_players_with_min_games": {r: {c: pl_count[i][j] for j, c in enumerate(H.COLOURS)}
                                            for i, r in enumerate(H.RANKS)}}
    out["combined"] = summarise([r["all"] for r in recs], rng)
    out["by_pool"] = {pool: summarise([r["all"] for r in recs if r["pool"] == pool], rng) for pool in POOLS}
    out["by_colour"] = {c: summarise([r["c"][j] for r in recs if r["c"][j][0] > 0], rng)
                        for j, c in enumerate(H.COLOURS)}
    out["by_model25_top_rank"] = {r: summarise([x["all"] for x in recs if x["top"] == i], rng)
                                  for i, r in enumerate(H.RANKS)}
    out["confident_questions_pmax_ge_%.1f" % CONFIDENT] = {
        r: summarise([x["all"] for x in recs if x["top"] == i and x["pmax"] >= CONFIDENT], rng)
        for i, r in enumerate(H.RANKS)}
    k = sum(x["all"][0] for x in recs)
    out["model25_mean_rank_probability_game_weighted"] = {
        r: sum(x["p"][i] * x["all"][0] for x in recs) / k for i, r in enumerate(H.RANKS)}
    H.write_json(a.out, out)
    c = out["combined"]
    print("observed %.4f  expected: game-weighted %.4f (diff %+.4f, ci95 %s)  player-averaged %.4f (diff %+.4f, "
          "ci95 %s)  uniform-rank %.4f -> %s"
          % (c["observed_hd1_share"], c["expected_model25_x_train_hd_le1"], c["observed_minus_expected_hd_le1"],
             c.get("observed_minus_expected_hd_le1_ci95"), c["expected_model25_x_train_player_avg"],
             c["observed_minus_expected_player_avg"], c.get("observed_minus_expected_player_avg_ci95"),
             c["expected_uniform_rank_x_train_hd_le1"], a.out))


if __name__ == "__main__":
    main()
