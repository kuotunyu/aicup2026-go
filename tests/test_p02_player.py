"""Tests for the PREREG_02 competition-1 tools (tools/p02_player_open.py, tools/p02_player_fuse.py).
Synthetic data only; nothing here reads a validation file. Run:
    python -m unittest tests.test_p02_player      (or: python -m pytest tests/test_p02_player.py)
"""

import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import p02_moves  # noqa: E402
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402

LETTERS = "abcdefghijklmnopqrs"
NEEDLE = "ans" + "wers"


def random_game(rng, n=40, first="B"):
    """n moves on distinct random points, alternating colours starting with `first`."""
    pts = rng.sample([a + b for a in LETTERS for b in LETTERS], n)
    other = "W" if first == "B" else "B"
    return [((first if i % 2 == 0 else other), p) for i, p in enumerate(pts)]


def sgf_of(mv):
    return "(;SZ[19]" + "".join(";%s[%s]" % (c, p) for c, p in mv) + ")"


def swap_colours(mv):
    return [("W" if c == "B" else "B", p) for c, p in mv]


# ---------------------------------------------------------------- tokens
class Tokens(unittest.TestCase):
    def test_spelling_and_counts(self):
        mv = [("B", "pd"), ("W", "dp"), ("B", "pq"), ("W", "dd")]
        self.assertEqual(po.col_tokens("B", mv),
                         ["B0:pd", "B*:pd", "B1:pq", "B*:pq",
                          "Bq:Bpd>Wdp", "Bq:Wdp>Bpq", "Bq:Bpq>Wdd"])
        self.assertEqual(po.nocol_tokens("W", mv),
                         ["o0:dp", "o1:dd", "o*:dp", "o*:dd", "q:pd>dp", "q:dp>pq", "q:pq>dd"])
        self.assertEqual(po.game_tokens("W", mv, "raw"),
                         po.col_tokens("W", mv) + po.nocol_tokens("W", mv))

    def test_first_ten_own_moves_and_ten_game_moves(self):
        rng = random.Random(1)
        mv = random_game(rng, 40)
        toks = po.game_tokens("W", mv, "raw")
        own = [p for c, p in mv if c == "W"][:10]
        self.assertEqual(sum(t.startswith("W") and t[1].isdigit() for t in toks), 10)
        self.assertIn("W9:%s" % own[9], toks)
        self.assertNotIn("W10:%s" % [p for c, p in mv if c == "W"][10], toks)
        self.assertEqual(sum(t.startswith("q:") for t in toks), 9)       # bigrams of 10 moves
        self.assertEqual(sum(t.startswith("Wq:") for t in toks), 9)

    def test_colour_free_set_invariant_under_colour_swap(self):
        rng = random.Random(2)
        for _ in range(50):
            mv = random_game(rng, rng.randint(1, 40), first=rng.choice("BW"))
            for c in "BW":
                c2 = "W" if c == "B" else "B"
                sw = swap_colours(mv)
                self.assertEqual(po.nocol_tokens(c, mv), po.nocol_tokens(c2, sw))
                if any(x == c for x, _ in mv):
                    self.assertNotEqual(po.col_tokens(c, mv), po.col_tokens(c2, sw))

    def test_canon_tokens_invariant_under_symmetry(self):
        rng = random.Random(3)
        for _ in range(40):
            mv = random_game(rng, 40)
            base = po.game_tokens("B", mv, "canon")
            for s in range(8):
                self.assertEqual(po.game_tokens("B", p02_moves.transform(mv, s), "canon"), base)
            # raw tokens do change under a non-trivial symmetry
            self.assertNotEqual(po.game_tokens("B", p02_moves.transform(mv, 1), "raw"),
                                po.game_tokens("B", mv, "raw"))

    def test_moves_come_from_the_shared_parser(self):
        mv = random_game(random.Random(4), 70)
        self.assertEqual(po.moves_of(sgf_of(mv)), mv[:60])

    def test_bad_colour_rejected(self):
        with self.assertRaises(ValueError):
            po.game_tokens("", [("B", "pd")], "raw")
        with self.assertRaises(ValueError):
            po.orient_moves([("B", "pd")], "mirror")


# ---------------------------------------------------------------- TF-IDF
class OpenCos(unittest.TestCase):
    def test_hand_computed_tfidf(self):
        # vocabulary a, b, c (+ d only in the query). candidates: c0 = {a:2, b:1}, c1 = {b:1, c:1}
        import scipy.sparse as sp
        G = sp.csr_matrix(np.array([[2, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], float))
        cos, has = po.open_cos(G, [[0], [1]], G, [[2]])
        P = 2
        idf = np.log((1 + P) / (1 + np.array([1, 2, 1, 0]))) + 1
        c0 = np.array([1 + math.log(2), 1, 0, 0]) * idf
        c1 = np.array([0, 1, 1, 0]) * idf
        q = np.array([1, 0, 0, 1]) * idf
        n = lambda v: v / (np.linalg.norm(v) + 1e-12)  # noqa: E731
        np.testing.assert_allclose(cos[0], [n(q) @ n(c0), n(q) @ n(c1)], rtol=1e-12)
        self.assertTrue(has[0])

    def test_query_without_tokens(self):
        import scipy.sparse as sp
        G = sp.csr_matrix(np.array([[1, 0], [0, 1], [0, 0]], float))
        cos, has = po.open_cos(G, [[0], [1]], G, [[2]])
        self.assertFalse(has[0])
        self.assertTrue(np.all(cos == 0))

    def test_token_matrices_share_columns(self):
        rng = random.Random(5)
        a = [("B", random_game(rng, 20)) for _ in range(3)]
        b = [("W", random_game(rng, 20)) for _ in range(2)]
        ga, gb = po.token_matrices([a, b], "raw")
        self.assertEqual(ga.shape[1], gb.shape[1])
        self.assertEqual(ga.shape[0], 3)
        self.assertEqual(int(ga[0].sum()), len(po.game_tokens("B", a[0][1], "raw")))


# ---------------------------------------------------------------- self-retrieval split
class SelfRetrievalSplit(unittest.TestCase):
    def setUp(self):
        self.games = {"p%02d" % i: ["g%02d_%03d" % (i, j) for j in range(100)] for i in range(7)}

    def test_parts_disjoint_query_not_in_references_80_references(self):
        rounds = po.sr_rounds(self.games)
        self.assertEqual(len(rounds), 10)
        for r, items in enumerate(rounds):
            self.assertEqual([p for p, _q, _r in items], sorted(self.games))
            for p, q, refs in items:
                parts = po.sr_parts(p, self.games[p])
                self.assertEqual([len(x) for x in parts], [20] * 5)
                self.assertEqual(len(refs), 80)
                self.assertEqual(len(set(refs)), 80)
                self.assertFalse(set(q) & set(refs))
                self.assertTrue(set(q) <= set(parts[r % 5]))
                self.assertTrue(5 <= len(q) <= 20)
                self.assertEqual(set(refs) | set(parts[r % 5]), set(self.games[p]))

    def test_deterministic_and_order_free(self):
        a = po.sr_rounds(self.games)
        shuffled = dict(reversed(list(self.games.items())))
        shuffled = {p: list(reversed(g)) for p, g in shuffled.items()}
        self.assertEqual(a, po.sr_rounds(shuffled))
        self.assertNotEqual([q for _p, q, _r in a[0]], [q for _p, q, _r in a[5]])  # same part, new draw

    def test_k_draws_follow_the_registered_seed(self):
        rng = random.Random(int(__import__("hashlib").sha256(b"c10-prereg02|sr-k|3").hexdigest()[:16], 16))
        p0 = sorted(self.games)[0]
        k = rng.randint(5, 20)
        part = po.sr_parts(p0, self.games[p0])[3]
        self.assertEqual(po.sr_rounds(self.games)[3][0][1], rng.sample(part, k))

    def test_strict_needs_100_games(self):
        g = dict(self.games)
        g["p00"] = g["p00"][:99]
        with self.assertRaises(ValueError):
            po.sr_rounds(g)
        rounds = po.sr_rounds(g, strict=False)
        self.assertEqual(len(rounds), 10)
        with self.assertRaises(ValueError):
            po.sr_rounds({"x": ["a", "a"] + ["g%d" % i for i in range(98)]})


# ---------------------------------------------------------------- fusion, ranking, selection
class Fusion(unittest.TestCase):
    def test_zscore_edge_cases(self):
        M = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.0, 1.0 + 1e-14, 1.0], [7.0, 8.0, 9.0]])
        Z = pf.zscore_rows(M, [True, True, True, False])
        np.testing.assert_allclose(Z[0], (M[0] - 2.0) / np.std(M[0]))
        self.assertTrue(np.all(Z[1] == 0))       # constant row
        self.assertTrue(np.all(Z[2] == 0))       # std < 1e-12
        self.assertTrue(np.all(Z[3] == 0))       # missing term
        self.assertEqual(pf.zscore_rows(np.zeros((0, 3)), []).shape, (0, 3))

    def test_fuse_formula_and_missing_terms(self):
        E = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        O = np.array([[3.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        terms = {"E0": (E, np.array([True, False])), "raw": (O, np.array([True, False]))}
        c = pf.COMBO_BY_ID["Q04"]                      # E0 + O_raw, w = 1
        s = pf.fuse(c, terms)
        np.testing.assert_allclose(s[0], pf.zscore_rows(E[:1], [True])[0] + pf.zscore_rows(O[:1], [True])[0])
        self.assertTrue(np.all(s[1] == 0))             # both terms missing
        self.assertEqual(pf.all_missing(c, terms).tolist(), [False, True])
        s2 = pf.fuse(pf.COMBO_BY_ID["Q06"], terms)     # w = 4
        np.testing.assert_allclose(s2[0], pf.zscore_rows(E[:1], [True])[0] + 4 * pf.zscore_rows(O[:1], [True])[0])
        s3 = pf.fuse(pf.COMBO_BY_ID["Q22"], terms)     # opening only, weight 1
        np.testing.assert_allclose(s3[0], pf.zscore_rows(O[:1], [True])[0])

    def test_all_missing_gives_official_fallback(self):
        ids = ["c%d" % i for i in range(8)]
        S = np.zeros((1, 8))
        self.assertEqual(pf.top_ids(S, ids), [ids[:5]])

    def test_ties_by_candidate_id_and_ranks_agree(self):
        ids = ["a", "b", "c", "d", "e", "f", "g"]
        S = np.array([[1.0, 3.0, 3.0, 0.0, 3.0, -1.0, 2.0]])
        self.assertEqual(pf.top_ids(S, ids)[0], ["b", "c", "e", "g", "a"])
        order = pf.top_ids(S, ids, n=7)[0]
        for t in range(7):
            self.assertEqual(int(po.true_ranks(S, [t])[0]), order.index(ids[t]) + 1)

    def test_points(self):
        np.testing.assert_allclose(po.points([1, 2, 5, 6, 400]), [1, math.exp(-1), math.exp(-4), 0, 0])

    def test_combination_list(self):
        self.assertEqual([c["id"] for c in pf.COMBOS], ["Q%02d" % i for i in range(24)])
        self.assertEqual(pf.COMBO_BY_ID["Q02"], {"id": "Q02", "e": "E0", "o": "raw", "w": 0.25})
        self.assertEqual(pf.COMBO_BY_ID["Q11"], {"id": "Q11", "e": "E1", "o": "raw", "w": 4.0})
        self.assertEqual(pf.COMBO_BY_ID["Q14"], {"id": "Q14", "e": "E0", "o": "canon", "w": 1.0})
        self.assertEqual(pf.COMBO_BY_ID["Q21"], {"id": "Q21", "e": "E1", "o": "canon", "w": 4.0})
        self.assertEqual((pf.COMBO_BY_ID["Q23"]["e"], pf.COMBO_BY_ID["Q23"]["o"]), (None, "canon"))

    def test_choose_tolerance_and_order(self):
        order = ["Q00", "Q01", "Q02", "Q03"]
        self.assertEqual(pf.choose({"Q00": 0.5, "Q01": 0.6, "Q02": 0.6025, "Q03": 0.599}, order),
                         ("Q02", "Q01"))
        self.assertEqual(pf.choose({"Q00": 0.5, "Q01": 0.6, "Q02": 0.604, "Q03": 0.6}, order),
                         ("Q02", "Q02"))
        self.assertEqual(pf.choose({"Q00": 0.7, "Q01": 0.7, "Q02": 0.1, "Q03": 0.1}, order),
                         ("Q00", "Q00"))

    def test_e_vectors(self):
        emb = np.array([[2.0, 0.0], [0.0, 3.0], [9.0, 9.0]])
        nw = np.array([2.0, 1.0, 0.0])
        ok = np.array([True, True, False])
        V, has = pf.e_vectors("E0", emb, nw, ok, [[0, 1], [2]])
        np.testing.assert_allclose(V[0], [2 / 3, 1.0])
        self.assertEqual(has.tolist(), [True, False])
        V1, has1 = pf.e_vectors("E1", emb, nw, ok, [[0, 1]])
        np.testing.assert_allclose(V1[0], np.array([1.0, 1.0]) / math.sqrt(2))
        M, qh, nmiss = pf.e_scores("E0", (emb, nw, ok), [[0], [2]], (emb, nw, ok), [[1]])
        self.assertEqual(nmiss, 1)
        self.assertEqual(M[0, 1], M[0, 0])             # candidate without windows gets the worst score


# ---------------------------------------------------------------- guard
class Guard(unittest.TestCase):
    TH = po.guard_thresholds(1.0, 0.8, 0.5)          # theta1 0.9, theta2 0.65

    def test_thresholds(self):
        self.assertAlmostEqual(self.TH["theta1"], 0.9)
        self.assertAlmostEqual(self.TH["theta2"], 0.65)
        self.assertAlmostEqual(self.TH["theta_RX"], 0.75)
        self.assertTrue(self.TH["theta1_enabled"] and self.TH["theta2_enabled"])
        th = po.guard_thresholds(1.0, 0.97, 0.5)
        self.assertFalse(th["theta1_enabled"])
        self.assertTrue(th["theta2_enabled"])

    def test_levels(self):
        lv = lambda rho, bc=0.5, bq=0.5, th=None: po.guard_level(rho, th or self.TH, bc, bq)[0]  # noqa: E731
        self.assertEqual(lv(0.95), 1)
        self.assertEqual(lv(0.9), 1)
        self.assertEqual(lv(0.8), 2)
        self.assertEqual(lv(0.65), 2)
        self.assertEqual(lv(0.6), 3)
        self.assertEqual(lv(0.95, 0.9, 0.3), 3)        # colour shares differ by 0.6
        self.assertEqual(lv(0.95, 0.8, 0.3), 1)        # exactly 0.5 is allowed
        no1 = po.guard_thresholds(1.0, 0.97, 0.5)
        self.assertEqual(lv(0.9, th=no1), 1)           # theta1 disabled: no level 2
        self.assertEqual(lv(0.7, th=no1), 3)
        no2 = po.guard_thresholds(1.0, 0.8, 0.78)      # theta2 disabled: theta_RX = 0.89
        self.assertEqual(lv(0.88, th=no2), 3)
        self.assertEqual(lv(0.89, th=no2), 1)
        self.assertEqual(lv(float("nan")), 1)
        self.assertEqual(lv(None, 1.0, 0.0), 3)

    def test_decisions(self):
        g = {"thresholds": {"raw": self.TH, "canon": self.TH}}
        C = pf.COMBO_BY_ID
        dec = lambda cid, rho, cnn=True, bc=0.5, bq=0.5: pf.guard_decision(  # noqa: E731
            C[cid], {"raw": rho, "canon": rho}, g, bc, bq, cnn)[:2]
        self.assertEqual(dec("Q00", 0.1), (C["Q00"], 0))           # CNN only: exempt
        self.assertEqual(dec("Q01", 0.1, bc=1.0, bq=0.0), (C["Q01"], 0))
        self.assertEqual(dec("Q06", 0.95), (C["Q06"], 1))
        self.assertEqual(dec("Q06", 0.8), (C["Q04"], 2))            # w 4 -> 1
        self.assertEqual(dec("Q03", 0.8), (C["Q03"], 2))            # w 0.5 stays
        self.assertEqual(dec("Q20", 0.8), (C["Q19"], 2))            # E1 + canon, w 2 -> 1
        self.assertEqual(dec("Q22", 0.8), (C["Q04"], 2))
        self.assertEqual(dec("Q23", 0.8), (C["Q14"], 2))
        self.assertEqual(dec("Q09", 0.5), (C["Q01"], 3))            # E1 part only
        self.assertEqual(dec("Q16", 0.95, bc=0.9, bq=0.1), (C["Q00"], 3))
        self.assertEqual(dec("Q22", 0.5), (C["Q00"], 3))
        self.assertEqual(dec("Q23", 0.5, cnn=False), (C["Q23"], 3))  # fallback list: kept
        self.assertEqual(pf.guard_decision(C["Q22"], {"raw": 0.1}, None, 0.5, 0.5, True)[:2], (C["Q22"], 0))

    def test_colour_share(self):
        self.assertEqual(po.colour_share("BBWW"), 0.5)
        self.assertIsNone(po.colour_share([]))

    def test_calibration_on_a_synthetic_fit_set(self):
        rng = random.Random(9)
        levels = po.LEVELS
        by_player, level = {}, {}
        for li, lv in enumerate(levels):
            for j in range(3):
                p = "fit_%s_%d" % (lv, j)
                n = 110 + j
                games = []
                for i in range(n):
                    c = "B" if i < n // 2 else "W"          # file order: black games first
                    games.append(("g_%07d" % (li * 1000 + j * 200 + i), c,
                                  sgf_of(random_game(rng, 20, first="B"))))
                by_player[p], level[p] = games, lv
        by_player["fit_small"], level["fit_small"] = by_player["fit_D_0"][:50], "D"   # not eligible
        pool, cand, query = po.sim_pool(by_player, {lv: [p for p in by_player if level[p] == lv
                                                         and len(by_player[p]) >= 105] for lv in levels},
                                        0, "colour_separated", per_level=2)
        self.assertEqual(len(pool), 20)
        for p in pool:
            self.assertEqual(len(cand[p]), 100)
            self.assertEqual(query[p], [g for g, _c, _s in by_player[p]][-len(query[p]):])
            self.assertFalse(set(query[p]) & set(cand[p]))
        cal = po.calibrate(by_player, level, per_level=2, seeds=(0,), log=lambda s: None)
        self.assertEqual([x["split"] for x in cal["pools"]], list(po.CAL_SPLITS))
        self.assertEqual(cal["eligible_players_by_level"]["D"], 3)
        for o in po.ORIENTS:
            th = cal["guard"][o]
            self.assertAlmostEqual(th["theta1"], (th["rho_R"] + th["rho_F"]) / 2)
            self.assertAlmostEqual(th["theta2"], (th["rho_F"] + th["rho_X"]) / 2)
        x = [p for p in cal["pools"] if p["split"] == "colour_separated"][0]
        self.assertLess(x["b_q"], 0.1)                  # query = last games = white


# ---------------------------------------------------------------- end to end on a synthetic pool
def set_notebook_smoke(p02_dir, flag):
    with open(os.path.join(p02_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"smoke": flag}, f)


def make_pool(root, n_players=12, n_games=100, seed=0):
    """Candidate file, question file, dumps and prereg01-style files for a synthetic pool."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    pts = [a + b for a in LETTERS for b in LETTERS]
    players = ["syn_p_%02d" % i for i in range(n_players)]
    style = {p: rng.sample(pts, 12) for p in players}
    centre = {p: nrng.normal(size=16) for p in players}

    def game(p):
        colour = rng.choice("BW")
        opening = [x for x in style[p] if rng.random() < 0.7]
        rest = [x for x in rng.sample(pts, 60) if x not in opening]
        seq = (opening + rest)[:50]
        mv = [("B" if i % 2 == 0 else "W", x) for i, x in enumerate(seq)]
        if colour == "W":           # the player's style goes on its own moves
            mv = [("B", rest[-1])] + [("W" if c == "B" else "B", x) for c, x in mv]
        return colour, sgf_of(mv)

    def emb(p):
        nw = int(nrng.integers(20, 60))
        return (centre[p] + nrng.normal(scale=2.0, size=16)).astype(np.float32) * nw, nw

    cand, cg = [], {k: [] for k in ("game_id", "player_id", "color", "emb_sum", "n_windows")}
    gi = 0
    for p in players:
        for _ in range(n_games):
            gi += 1
            c, s = game(p)
            cand.append(["syn_g_%05d" % gi, p, s, c])
            e, n = emb(p)
            for k, v in zip(cg, ("syn_g_%05d" % gi, p, c, e, n)):
                cg[k].append(v)
    rng.shuffle(cand)
    os.makedirs(os.path.join(root, "p02", "dumps"))
    os.makedirs(os.path.join(root, "p01"))
    set_notebook_smoke(os.path.join(root, "p02"), True)     # the notebook's own flag: SMOKE-like dumps
    with open(os.path.join(root, "cands.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "player_id", "sgf_content", "color"])
        w.writerows(cand)
    np.savez(os.path.join(root, "p02", "dumps", "cand_games.npz"),
             **{k: (np.stack(v) if k == "emb_sum" else np.array(v)) for k, v in cg.items()})

    qs, qg = [], {k: [] for k in ("question_id", "slot", "color", "emb_sum", "n_windows", "parsed")}
    for i, p in enumerate(players):
        qid = "syn_q_%02d" % i
        n = rng.randint(5, 20)
        games = [game(p) for _ in range(n)]
        qs.append([qid, n] + [s for _c, s in games] + [""] * (20 - n) + [c for c, _s in games] + [""] * (20 - n))
        for slot, (c, _s) in enumerate(games, 1):
            e, nw = emb(p)
            parsed = i != 2                         # question 3 has no parsed game
            for k, v in zip(qg, (qid, slot, c, e if parsed else np.full(16, np.nan, np.float32),
                                 nw if parsed else 0, parsed)):
                qg[k].append(v)
    with open(os.path.join(root, "questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] +
                   ["color_%d" % i for i in range(1, 21)])
        w.writerows(qs)
    np.savez(os.path.join(root, "p02", "dumps", "query_games.npz"),
             **{k: (np.stack(v) if k == "emb_sum" else np.array(v)) for k, v in qg.items()})

    # prereg01-style reference outputs computed independently (float32, L2, argsort)
    sums, cnt = {}, {}
    for p, e, n in zip(cg["player_id"], cg["emb_sum"], cg["n_windows"]):
        sums[p] = sums.get(p, 0) + e
        cnt[p] = cnt.get(p, 0) + n
    names = sorted(sums)
    C = np.stack([sums[n] / cnt[n] for n in names]).astype(np.float32)
    qsum, qcnt = {}, {}
    for q, e, n, ok in zip(qg["question_id"], qg["emb_sum"], qg["n_windows"], qg["parsed"]):
        if ok:
            qsum[q] = qsum.get(q, 0) + e
            qcnt[q] = qcnt.get(q, 0) + n
    with open(os.path.join(root, "p01", "player_val_top5.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id"] + pf.TOP_COLS)
        for r in qs:
            q = r[0]
            if q not in qsum:
                w.writerow([q] + names[:5])
            else:
                d = np.linalg.norm(C - qsum[q] / qcnt[q], axis=1)
                w.writerow([q] + [names[i] for i in np.argsort(d)[:5]])
    qids = [r[0] for r in qs if r[0] in qsum]
    np.savez(os.path.join(root, "p01", "player_val_embeddings.npz"), candidate_ids=np.array(names),
             centroids=C, question_ids=np.array(qids),
             queries=np.stack([qsum[q] / qcnt[q] for q in qids]).astype(np.float32))
    return root


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = make_pool(os.path.join(cls.tmp, "pool"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_tool(self, *argv):
        with redirect_stdout(StringIO()):
            pf.main(list(argv))

    def args(self, out, *extra):
        r = self.root
        return ["--p02-dir", os.path.join(r, "p02"), "--candidates", os.path.join(r, "cands.csv"),
                "--out-dir", out, "--smoke"] + list(extra)

    def full_run(self, out, guard=None):
        q = ["--questions", os.path.join(self.root, "questions.csv")]
        self.run_tool("repro", "--p01-dir", os.path.join(self.root, "p01"), *self.args(out, *q))
        if guard:
            with open(os.path.join(out, "guard_calibration.json"), "w", encoding="utf-8") as f:
                json.dump(guard, f)
        self.run_tool("select", *self.args(out))
        self.run_tool("apply", *self.args(out, *q))

    def read(self, out, name):
        with open(os.path.join(out, name), "rb") as f:
            return f.read()

    def test_repro_select_apply_deterministic(self):
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        self.full_run(a)
        self.full_run(b)
        rep = json.loads(self.read(a, "repro_local.json"))
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["n_top5_identical"], 12)
        self.assertLess(rep["max_abs_centroid_diff"], 1e-4)
        self.assertEqual(self.read(a, "frozen_config.json"), self.read(b, "frozen_config.json"))
        self.assertEqual(self.read(a, "player_val_top5.csv"), self.read(b, "player_val_top5.csv"))
        fz = json.loads(self.read(a, "frozen_config.json"))
        self.assertEqual(fz["combination_list"], ["Q%02d" % i for i in range(24)])
        self.assertEqual(set(fz["self_retrieval_mean"]), set(fz["combination_list"]))
        sr = json.loads(self.read(a, "sr_report.json"))
        self.assertEqual(sr["pseudo_queries_per_round"], [12] * 10)
        with open(os.path.join(a, "player_val_top5.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["question_id"] + pf.TOP_COLS)
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(len(set(r[1:])) == 5 for r in rows[1:]))
        app = json.loads(self.read(a, "apply_report.json"))
        self.assertEqual(app["guard_level"], 0 if fz["selected"]["o"] is None else 0)  # no calibration
        # the openings carry the identity in this synthetic pool
        self.assertGreater(fz["self_retrieval_mean"]["Q22"], fz["self_retrieval_mean"]["Q00"])

    def test_guard_applied_from_calibration(self):
        out = os.path.join(self.tmp, "g")
        th = po.guard_thresholds(3.0, 1.5, 0.01)       # theta1 2.25, theta2 0.755: rho ~1 -> level 2
        self.full_run(out, guard={"smoke": True, "guard": {"raw": th, "canon": th}})
        fz = json.loads(self.read(out, "frozen_config.json"))
        app = json.loads(self.read(out, "apply_report.json"))
        sel = fz["selected"]
        if sel["o"] is None:
            self.assertEqual(app["guard_level"], 0)
        else:
            self.assertEqual(app["guard_level"], 2)
            want = pf.guard_decision(sel, app["rho"], fz["guard"], app["b_c_candidate_black_share"],
                                     app["b_q_question_black_share"], True)[0]
            self.assertEqual(app["applied"], want)

    def test_no_cnn_fallback_list(self):
        out = os.path.join(self.tmp, "n")
        with self.assertRaises(SystemExit):
            self.run_tool("select", *self.args(out, "--no-cnn"))            # --reason required
        self.run_tool("select", *self.args(out, "--no-cnn", "--reason", "test"))
        fz = json.loads(self.read(out, "frozen_config.json"))
        self.assertEqual(fz["combination_list"], ["Q22", "Q23"])
        # PREREG_02 rev 3, 7.7: no Q00, so the gate is "selected self-retrieval mean >= 0.50"
        self.assertEqual(fz["gate"]["threshold"], pf.FALLBACK_GATE_MIN)
        self.assertIsNone(fz["gate"]["q00_mean"])
        self.assertEqual(fz["gate"]["passed"], fz["gate"]["selected_mean"] >= 0.50)
        self.run_tool("apply", "--p02-dir", os.path.join(self.tmp, "nowhere"), "--candidates",
                      os.path.join(self.root, "cands.csv"), "--questions",
                      os.path.join(self.root, "questions.csv"), "--out-dir", out, "--smoke")
        app = json.loads(self.read(out, "apply_report.json"))
        self.assertIn(app["applied"]["id"], ("Q22", "Q23"))
        self.assertEqual(app["score_allowed"], fz["gate"]["passed"] is True and app["guard_level"] != 3)

    def test_fallback_level3_blocks_scoring(self):
        # the fallback list has no CNN term: level 3 keeps the selection but scoring is blocked (rev 3, 7.7)
        th = {"theta1": 0.9, "theta2": 0.5, "theta_RX": 0.7, "theta1_enabled": True, "theta2_enabled": True}
        sel = pf.COMBO_BY_ID["Q22"]
        applied, level, _ = pf.guard_decision(sel, {sel["o"]: 0.1}, {"thresholds": {sel["o"]: th}},
                                              0.5, 0.5, False)
        self.assertEqual((applied["id"], level), ("Q22", 3))

    def test_repro_needs_a_clean_notebook_run_without_smoke(self):
        p02 = os.path.join(self.tmp, "p02_copy")
        shutil.copytree(os.path.join(self.root, "p02"), p02)
        set_notebook_smoke(p02, False)                  # a real run refuses SMOKE dumps
        out = os.path.join(self.tmp, "nb")
        real = ["repro", "--p01-dir", os.path.join(self.root, "p01"), "--p02-dir", p02,
                "--candidates", os.path.join(self.root, "cands.csv"),
                "--questions", os.path.join(self.root, "questions.csv"), "--out-dir", out]
        with self.assertRaises(SystemExit):              # no repro_check.json / run_log.json
            self.run_tool(*real)
        with open(os.path.join(p02, "repro_check.json"), "w", encoding="utf-8") as f:
            json.dump({"passed": True}, f)
        with open(os.path.join(p02, "run_log.json"), "w", encoding="utf-8") as f:
            json.dump({"runs": [{"run": 1, "status": "final_check_failed", "p01_unchanged": False}]}, f)
        with self.assertRaises(SystemExit):              # prereg01 changed during the notebook run
            self.run_tool(*real)
        rep = json.loads(self.read(out, "repro_local.json"))
        self.assertFalse(rep["passed"])
        self.assertEqual(rep["notebook_run"]["last_run_status"], "final_check_failed")
        with open(os.path.join(p02, "run_log.json"), "w", encoding="utf-8") as f:
            json.dump({"runs": [{"run": 1, "status": "final_check_failed"},
                                {"run": 2, "status": "completed", "p01_unchanged": True}]}, f)
        self.run_tool(*real)                             # the last run is the one that counts
        rep = json.loads(self.read(out, "repro_local.json"))
        self.assertTrue(rep["passed"] and rep["notebook_run_ok"] and rep["notebook_run_required"])

    def test_select_needs_passed_repro(self):
        out = os.path.join(self.tmp, "r")
        with self.assertRaises(SystemExit):
            self.run_tool("select", *self.args(out))

    def test_smoke_refused_inside_repo_and_apply_needs_commit(self):
        inside = os.path.join(po.REPO, "results", "prereg02", "player")
        with self.assertRaises(SystemExit):
            po.check_out_dir(inside, True)
        po.check_out_dir(self.tmp, True)
        po.check_out_dir(inside, False)
        # prereg01's results are never an output dir, with or without --smoke; nor is --p01-dir
        for smoke in (False, True):
            with self.assertRaises(SystemExit):
                po.check_out_dir(os.path.join(po.REPO, "results", "prereg01", "player"), smoke)
        with self.assertRaises(SystemExit):
            po.check_out_dir(os.path.join(self.root, "p01", "sub"), True, extra_readonly=(os.path.join(self.root, "p01"),))
        with self.assertRaises(SystemExit):     # repro refuses --out-dir inside --p01-dir
            self.run_tool("repro", "--p01-dir", os.path.join(self.root, "p01"),
                          *self.args(os.path.join(self.root, "p01"), "--questions",
                                     os.path.join(self.root, "questions.csv")))
        out = os.path.join(self.tmp, "a2")
        self.full_run(out)
        q = ["--questions", os.path.join(self.root, "questions.csv")]
        r = self.root
        with self.assertRaises(SystemExit):     # not --smoke: frozen_config must be committed
            self.run_tool("apply", "--p02-dir", os.path.join(r, "p02"), "--candidates",
                          os.path.join(r, "cands.csv"), "--out-dir", out, *q)

    def test_smoke_never_on_real_inputs_or_real_dumps(self):
        r = self.root
        out = os.path.join(self.tmp, "sm")
        real_cands = os.path.join(po.REPO, "data", "local", "player_val_candidates.csv")
        for extra in ([], ["--no-cnn", "--reason", "test"]):     # the real candidate file (inside the repo)
            with self.assertRaises(SystemExit):
                self.run_tool("select", "--p02-dir", os.path.join(r, "p02"), "--candidates", real_cands,
                              "--out-dir", out, "--smoke", *extra)
        self.assertFalse(os.path.exists(os.path.join(out, "frozen_config.json")))
        # a copy of a data/local file outside the repository is refused too (REPO patched to a fake repo)
        fake = os.path.join(self.tmp, "fake_repo")
        os.makedirs(os.path.join(fake, "data", "local"))
        shutil.copyfile(os.path.join(r, "cands.csv"), os.path.join(fake, "data", "local", "player_val_candidates.csv"))
        old = pf.REPO
        pf.REPO = fake
        try:
            with self.assertRaises(SystemExit):
                pf.check_smoke_inputs(True, os.path.join(r, "cands.csv"))
        finally:
            pf.REPO = old
        pf.check_smoke_inputs(True, os.path.join(r, "cands.csv"))
        pf.check_smoke_inputs(False, real_cands)             # real mode: the real files are the inputs
        # CNN dumps: --smoke needs the notebook's SMOKE flag, a real run refuses it
        p02 = os.path.join(self.tmp, "p02_real")
        shutil.copytree(os.path.join(r, "p02"), p02)
        set_notebook_smoke(p02, False)
        q = ["--questions", os.path.join(r, "questions.csv")]
        with self.assertRaises(SystemExit):
            self.run_tool("repro", "--p01-dir", os.path.join(r, "p01"), "--p02-dir", p02, "--candidates",
                          os.path.join(r, "cands.csv"), "--out-dir", out, "--smoke", *q)
        os.remove(os.path.join(p02, "config.json"))
        with self.assertRaises(SystemExit):
            pf.check_dump_kind(p02, True)
        with self.assertRaises(SystemExit):
            pf.check_dump_kind(os.path.join(r, "p02"), False)
        self.assertTrue(pf.check_dump_kind(os.path.join(r, "p02"), True))

    def test_no_cnn_needs_a_registered_basis_in_real_mode(self):
        from types import SimpleNamespace
        r = self.root
        out = os.path.join(self.tmp, "nc")
        os.makedirs(out)
        args = SimpleNamespace(p02_dir=os.path.join(r, "p02"), out_dir=out, smoke=False)
        with self.assertRaises(SystemExit):              # dumps there, no failed repro
            pf.no_cnn_basis(args)
        with open(os.path.join(out, "repro_local.json"), "w", encoding="utf-8") as f:
            json.dump({"passed": True, "created": "2026-10-01T10:00:00+08:00"}, f)
        with self.assertRaises(SystemExit):              # repro passed
            pf.no_cnn_basis(args)
        with open(os.path.join(out, "repro_local.json"), "w", encoding="utf-8") as f:
            json.dump({"passed": False, "created": "2026-10-01T10:00:00+08:00"}, f)
        b = pf.no_cnn_basis(args)
        self.assertTrue(b["repro_failed"] and not b["dumps_missing"])
        self.assertEqual(b["repro_local_json"]["created"], "2026-10-01T10:00:00+08:00")
        b = pf.no_cnn_basis(SimpleNamespace(p02_dir=os.path.join(self.tmp, "nowhere"), out_dir=self.tmp, smoke=False))
        self.assertTrue(b["dumps_missing"])
        self.assertTrue(pf.no_cnn_basis(SimpleNamespace(p02_dir=os.path.join(r, "p02"), out_dir=self.tmp,
                                                        smoke=True))["cand_games_npz_exists"])
        # select writes the basis into frozen_config.json
        sel = os.path.join(self.tmp, "nc2")
        self.run_tool("select", *self.args(sel, "--no-cnn", "--reason", "test"))
        fz = json.loads(self.read(sel, "frozen_config.json"))
        self.assertIn("rule", fz["no_cnn_basis"])

    def test_self_retrieval_references_and_queries_are_disjoint(self):
        """Every self-retrieval call: the candidate (reference) rows and the pseudo-query rows do not
        overlap, each candidate has exactly 80 references, and IDF / E see the references only."""
        d = pf.Data(os.path.join(self.root, "cands.csv"), os.path.join(self.root, "p02"), True, True)
        seen = {"open": 0, "e": 0}
        orig_open, orig_e = po.open_cos, pf.e_scores

        def check(c_groups, q_groups):
            refs = {i for g in c_groups for i in g}
            qs = {i for g in q_groups for i in g}
            self.assertFalse(refs & qs)
            self.assertTrue(all(len(g) == 80 for g in c_groups))
            self.assertTrue(all(5 <= len(g) <= 20 for g in q_groups))

        def open_spy(Gc, c_groups, Gq, q_groups):
            self.assertIs(Gc, Gq)
            check(c_groups, q_groups)
            seen["open"] += 1
            return orig_open(Gc, c_groups, Gq, q_groups)

        def e_spy(kind, c_arr, c_groups, q_arr, q_groups):
            self.assertIs(c_arr, q_arr)
            check(c_groups, q_groups)
            seen["e"] += 1
            return orig_e(kind, c_arr, c_groups, q_arr, q_groups)
        po.open_cos, pf.e_scores = open_spy, e_spy
        try:
            pf.self_retrieval(d, pf.COMBOS, strict=True, log=lambda s: None)
        finally:
            po.open_cos, pf.e_scores = orig_open, orig_e
        self.assertEqual(seen, {"open": 2 * po.SR_ROUNDS, "e": 2 * po.SR_ROUNDS})


class CommittedFrozenConfig(unittest.TestCase):
    """apply's git check, on a throw-away repository (REPO patched)."""

    def setUp(self):
        import subprocess
        self.sub = subprocess
        self.tmp = tempfile.mkdtemp()
        self.old = (po.REPO, pf.REPO)
        po.REPO = pf.REPO = self.tmp
        g = ["git", "-C", self.tmp]
        subprocess.run(g + ["init", "-q"], check=True)
        self.path = os.path.join(self.tmp, "frozen_config.json")
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write('{\n "selected": "Q04"\n}\n')
        subprocess.run(g + ["add", "frozen_config.json"], check=True)
        subprocess.run(g + ["-c", "user.name=kuotunyu", "-c",
                            "user.email=61350295+kuotunyu@users.noreply.github.com",
                            "-c", "commit.gpgsign=false", "commit", "-q", "-m", "t"], check=True)

    def tearDown(self):
        po.REPO, pf.REPO = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_committed_passes_changed_fails(self):
        self.assertTrue(pf.check_committed(self.path))
        with open(self.path, "wb") as f:                     # same content, CRLF line ends
            f.write(b'{\r\n "selected": "Q04"\r\n}\r\n')
        st = self.sub.run(["git", "-C", self.tmp, "status", "--porcelain"], capture_output=True, text=True)
        if not st.stdout.strip():                             # git itself sees no change
            self.assertTrue(pf.check_committed(self.path))
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write('{\n "selected": "Q06"\n}\n')
        with self.assertRaises(SystemExit):
            pf.check_committed(self.path)
        other = os.path.join(self.tmp, "sub")
        os.makedirs(other)
        with open(os.path.join(other, "frozen_config.json"), "w", encoding="utf-8") as f:
            f.write("{}\n")
        with self.assertRaises(SystemExit):                  # never committed
            pf.check_committed(os.path.join(other, "frozen_config.json"))
        outside = tempfile.mkdtemp()
        try:
            with self.assertRaises(SystemExit):              # outside the repository
                pf.check_committed(os.path.join(outside, "frozen_config.json"))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_notebook_results_must_be_committed(self):
        p02 = os.path.join(self.tmp, "results", "prereg02", "player")
        os.makedirs(os.path.join(p02, "dumps"))
        for rel in ("dumps/cand_games.npz", "dumps/query_games.npz", "repro_check.json", "run_log.json"):
            with open(os.path.join(p02, *rel.split("/")), "w", encoding="utf-8") as f:
                f.write("{}\n")
        with self.assertRaises(SystemExit):                  # not committed
            pf.require_committed_results(p02, need_dumps=True)
        g = ["git", "-C", self.tmp]
        self.sub.run(g + ["add", "-A"], check=True)
        self.sub.run(g + ["-c", "user.name=kuotunyu", "-c", "user.email=61350295+kuotunyu@users.noreply.github.com",
                          "-c", "commit.gpgsign=false", "commit", "-q", "-m", "r"], check=True)
        self.assertEqual(pf.require_committed_results(p02, need_dumps=True)["files"], 4)
        with open(os.path.join(p02, "run_log.json"), "w", encoding="utf-8") as f:
            f.write('{"runs": []}\n')                        # changed after the commit
        with self.assertRaises(SystemExit):
            pf.require_committed_results(p02, need_dumps=True)
        empty = os.path.join(self.tmp, "results", "prereg02", "none")
        os.makedirs(empty)
        with self.assertRaises(SystemExit):                  # CNN needs the dumps
            pf.require_committed_results(empty, need_dumps=True)
        self.assertEqual(pf.require_committed_results(empty, need_dumps=False)["files"], 0)


class NoAnswerFiles(unittest.TestCase):
    def test_sources_do_not_contain_the_word(self):
        for name in ("p02_player_open.py", "p02_player_fuse.py", "p02_moves.py"):
            with open(os.path.join(TOOLS, name), encoding="utf-8") as f:
                self.assertNotIn(NEEDLE, f.read().lower(), name)
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            self.assertNotIn(NEEDLE, f.read().lower())

    def test_forbidden_paths_refused(self):
        with self.assertRaises(SystemExit):
            po.refuse_forbidden(os.path.join("data", "local", "player_val_" + NEEDLE + ".csv"))
        with self.assertRaises(SystemExit):
            pf.read_candidates(os.path.join("x", "Player_" + NEEDLE.upper() + ".csv"))
        self.assertEqual(po.refuse_forbidden("player_val_questions.csv"), "player_val_questions.csv")


if __name__ == "__main__":
    unittest.main()
