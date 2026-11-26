"""Tests for tools/p04_opp.py (PREREG_04 1.2: O4, the opponent-move count table; plan.md 5).
Run: .venv/Scripts/python.exe -m pytest tests/test_p04_opp.py   (or -m unittest tests.test_p04_opp)

Synthetic data only (random legal games made with sgfmill, random counts), plus the committed design-phase scripts
in results/prereg04/design/evidence/B_proxy/ (imported without writing bytecode) as the reference the tool is
ported from. No real game, no question file, no label file, no GPU. The O4 of validation questions is never
computed here: the 'val' gate is tested in a scratch git repository and by checking that cmd_val calls it first.
With C10_SLOW=1 the real selftest also runs (loads the fit games through c10data.load_rows, about 2 minutes).
"""

import ast
import contextlib
import csv
import glob
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import p04_opp as P  # noqa: E402

DESIGN = os.path.join(REPO, "results", "prereg04", "design", "evidence", "B_proxy")
SLOW = os.environ.get("C10_SLOW", "")


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def design_module(name):
    """Import a committed design script (feat / count / score_inner / prep) without writing bytecode into the
    committed folder; its own 'import feat' resolves to the design feat.py."""
    key = f"_p04design_{name}"
    if key in sys.modules:
        return sys.modules[key]
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, DESIGN)
    try:
        if "feat" in sys.modules and not os.path.dirname(sys.modules["feat"].__file__) == DESIGN:
            raise RuntimeError("another module named feat is loaded")
        spec = importlib.util.spec_from_file_location(key, os.path.join(DESIGN, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(DESIGN)
        sys.dont_write_bytecode = old
    return mod


def funcs(path_or_src, strip_doc=True):
    src = path_or_src if "\n" in path_or_src else open(path_or_src, encoding="utf-8").read()
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            if strip_doc and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:]
            out[n.name] = n
    return out


class _Unfeat(ast.NodeTransformer):
    """feat.X -> X (the port lives in the same module as the feat functions); count.py's environment globals
    STRIDE / LIMIT are the shard's arguments stride / limit in the port."""
    RENAME = {"STRIDE": "stride", "LIMIT": "limit"}

    def visit_Name(self, node):
        if node.id in self.RENAME:
            return ast.copy_location(ast.Name(id=self.RENAME[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "feat":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def random_game(rng, n_moves, corner=0.7):
    """A legal game without suicide (sgfmill allows self-capture; the replay does not model it), dense in one corner
    so that captures, ataris and liberty sharing happen often."""
    from sgfmill import boards
    B = boards.Board(19)
    codes, colour = [], "b"
    for _ in range(n_moves):
        for _try in range(30):
            if rng.random() < corner:
                r, c = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            else:
                r, c = int(rng.integers(0, 19)), int(rng.integers(0, 19))
            if B.get(r, c) is not None:
                continue
            trial = B.copy()
            trial.play(r, c, colour)
            if trial.get(r, c) is None:          # self-capture
                continue
            B = trial
            break
        else:
            break
        codes.append((512 if colour == "w" else 0) + (18 - r) * 19 + c)
        colour = "w" if colour == "b" else "b"
    return np.array(codes, dtype=np.uint16)


def synthetic_moves(games, labels, colours):
    lens = np.array([len(g) for g in games], dtype=np.int64)
    off = np.zeros(len(games) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    return {"moves": np.concatenate(games).astype(np.uint16), "off": off, "label": np.array(labels, dtype=np.int8),
            "color": np.array(colours), "rank": np.array([P.LEVELS[x] for x in labels]),
            "game_id": np.array([f"g_{i:07d}" for i in range(len(games))]),
            "player_id": np.array([f"p_{i % 3}" for i in range(len(games))])}


def random_counts(rng):
    C = rng.integers(0, 4, size=(10, P.N_CELLS)).astype(np.int64)
    A = C + rng.integers(0, 40, size=(10, P.N_CELLS)).astype(np.int64)
    C2 = rng.integers(0, 4, size=(10, P.N_CELLS)).astype(np.int64)
    A2 = C2 + rng.integers(0, 40, size=(10, P.N_CELLS)).astype(np.int64)
    return {"C_t": C, "A_t": A, "C_o": C2, "A_o": A2}


GAMES = None


def games():
    global GAMES
    if GAMES is None:
        rng = np.random.default_rng(7)
        GAMES = [random_game(rng, int(n)) for n in (5, 40, 90, 140, 160, 33, 120, 75, 1, 150, 110, 60)]
    return GAMES


# ================================================================ the port is the design code
class TestPort(unittest.TestCase):
    def test_feat_functions_are_verbatim(self):
        d, m = funcs(os.path.join(DESIGN, "feat.py")), funcs(os.path.join(TOOLS, "p04_opp.py"))
        self.assertEqual(sorted(d), ["_canon_table", "_dist_tables", "_enc", "_grid", "_libs4", "_neighbours",
                                     "cells_of", "phase_of", "replay"])
        for name, node in d.items():
            self.assertEqual(ast.dump(node), ast.dump(m[name]), name)

    def test_feat_constants_equal(self):
        f = design_module("feat")
        for k in ("SIZE", "NPT", "OFFS", "N_TAC", "N_DOPP", "N_DOWN", "N_PHASE", "NEI", "N_PAT", "N_CELLS", "W", "PAD"):
            self.assertEqual(getattr(f, k), getattr(P, k), k)
        for k in ("CANON", "CANON32", "DOPP", "DOWN", "STRUCT", "NB4", "NB8", "INNER"):
            a, b = getattr(f, k), getattr(P, k)
            self.assertEqual(a.dtype, b.dtype, k)
            self.assertTrue(np.array_equal(a, b), k)
        self.assertEqual((P.N_PAT, P.N_CELLS), (1107, 557928))

    def test_encode_is_prep_verbatim(self):
        d, m = funcs(os.path.join(DESIGN, "prep.py")), funcs(os.path.join(TOOLS, "p04_opp.py"))
        self.assertEqual(ast.dump(d["encode"]), ast.dump(m["encode"]))

    def test_count_shard_loop_is_verbatim(self):
        d = _Unfeat().visit(funcs(os.path.join(DESIGN, "count.py"))["_shard"])
        m = funcs(os.path.join(TOOLS, "p04_opp.py"))["_count_shard"]
        loops = lambda f: [ast.dump(n) for n in f.body if isinstance(n, ast.For)]  # noqa: E731
        self.assertEqual(len(loops(d)), 1)
        self.assertEqual(loops(d), loops(m))
        acc = lambda f: [ast.dump(n) for n in f.body if isinstance(n, ast.Assign)  # noqa: E731
                         and any(isinstance(t, ast.Name) and t.id in ("acc", "games", "G") for t in n.targets)]
        self.assertEqual(len(acc(d)), 3)
        self.assertEqual(acc(d), acc(m))

    def test_build_tables_slices_equal_design(self):
        rng = np.random.default_rng(3)
        counts = random_counts(rng)
        si = design_module("score_inner")
        dt, dinfo = si.build_tables(counts)
        mt, minfo = P.build_tables(counts)
        self.assertTrue(np.array_equal(mt[0], dt[1]))      # t_choice_a20
        self.assertTrue(np.array_equal(mt[1], dt[4]))      # o_choice_a20
        self.assertEqual(minfo, dinfo)
        self.assertEqual(si.VARIANTS[1], "t_choice_a20")
        self.assertEqual(si.VARIANTS[4], "o_choice_a20")
        self.__class__.tables = (mt, dt)

    def test_score_game_equals_design_game(self):
        rng = np.random.default_rng(4)
        counts = random_counts(rng)
        si = design_module("score_inner")
        dt, _ = si.build_tables(counts)
        mt, _ = P.build_tables(counts)
        gs = games()
        mv = synthetic_moves(gs, [i % 10 for i in range(len(gs))], ["B" if i % 2 else "W" for i in range(len(gs))])
        tcol = np.where(mv["color"] == "B", 1, 2).astype(np.int64)
        si._G.update({"mv": mv["moves"], "tab": dt, "off": mv["off"], "tcol": tcol})
        for g in range(len(gs)):
            _, ll, nm, an, _ = si._game(g)
            m = mv["moves"][mv["off"][g]:mv["off"][g + 1]]
            mll, mnm, man = P.score_game(m, int(tcol[g]), mt)
            self.assertTrue(np.array_equal(mll[0], ll[1]), g)
            self.assertTrue(np.array_equal(mll[1], ll[4]), g)
            self.assertTrue(np.array_equal(mnm, nm), g)
            self.assertEqual(man, an)
        # the phase sum as fuse.py: llv[:, j].sum(1)
        L = np.stack([P.score_game(mv["moves"][mv["off"][g]:mv["off"][g + 1]], int(tcol[g]), mt)[0]
                      for g in range(len(gs))])
        t4, o4 = P.side_vectors(L)
        self.assertTrue(np.array_equal(t4, L[:, 0].sum(1)))
        self.assertTrue(np.array_equal(o4, L[:, 1].sum(1)))


# ================================================================ replay and features
class TestReplayAndFeatures(unittest.TestCase):
    def test_capture_logic_equals_sgfmill(self):
        caps = 0
        for m in games():
            bad, an = P.capture_mismatches(m)
            self.assertEqual((bad, an), (0, 0))
            snaps, _, _ = P.replay(m, lambda i, c: True)
            stones = [sum(1 for v in s if v) for s in snaps]
            caps += sum(1 for i in range(1, len(stones)) if stones[i] < stones[i - 1] + 1)
        self.assertGreater(caps, 5, "the synthetic games must contain captures")

    def test_features_equal_slow_reference(self):
        npos = 0
        for i, m in enumerate(games()):
            a, b = P.feature_mismatches(m, 1 if i % 2 else 2, step=3)
            npos += a
            self.assertEqual(b, 0)
        self.assertGreaterEqual(npos, P.SELFTEST_MIN_POSITIONS)

    def test_tactical_codes_seen(self):
        tacs = set()
        for m in games():
            snaps, meta, _ = P.replay(m, lambda i, c: True)
            if not meta:
                continue
            cell, te, played, ph = P.cells_of(snaps, meta)
            tacs |= set(((cell // (P.N_DOPP * P.N_DOWN * P.N_PHASE)) % P.N_TAC).tolist())
        self.assertTrue({0, 1, 2, 4} <= tacs, tacs)

    def test_anomaly_stops_the_replay(self):
        m = np.array([0 * 19 + 3, 512 + 4, 0 * 19 + 3, 512 + 5], dtype=np.uint16)   # B d1, W e1, B d1 again
        snaps, meta, an = P.replay(m, lambda i, c: True)
        self.assertEqual(an, 1)
        self.assertEqual(len(meta), 2)

    def test_simple_capture_by_hand(self):
        # black stone at b2 (point 1*19+1) surrounded by white a2, c2, b1, b3 -> captured by the last white move
        B, Wc = 0, 512
        seq = [B + 1 * 19 + 1, Wc + 1 * 19 + 0, B + 10 * 19 + 10, Wc + 1 * 19 + 2, B + 12 * 19 + 12, Wc + 0 * 19 + 1,
               B + 14 * 19 + 14, Wc + 2 * 19 + 1, B + 16 * 19 + 16]
        snaps, meta, an = P.replay(np.array(seq, dtype=np.uint16), lambda i, c: True)
        self.assertEqual(an, 0)
        self.assertEqual(snaps[7][1 * 19 + 1], 1)          # before white's capturing move
        self.assertEqual(snaps[8][1 * 19 + 1], 0)          # captured
        # distance buckets: meta holds (i, colour, point, last opponent move, own previous move)
        self.assertEqual(meta[0][3:], (P.NPT, P.NPT))
        self.assertEqual(meta[2][3:], (1 * 19 + 0, 1 * 19 + 1))


class TestEncode(unittest.TestCase):
    def test_encode_codes_and_passes(self):
        sgf = "(;GM[1]SZ[19]AB[dd];B[pd];W[dp]C[x];B[];W[tt];B[aa];W[sa])"
        m = P.encode_one(sgf)
        self.assertEqual(m.dtype, np.uint16)
        # 'pd' = column p (15), row d (3) -> 3 * 19 + 15; W adds 512; passes and setup stones are skipped
        self.assertEqual(m.tolist(), [3 * 19 + 15, 512 + 15 * 19 + 3, 0, 512 + 0 * 19 + 18])

    def test_target_colour_value(self):
        self.assertEqual((P.target_colour_value("B"), P.target_colour_value("W")), (1, 2))
        with self.assertRaises(P.P04Error):
            P.target_colour_value("b")


# ================================================================ count and score against brute force
def brute_counts(mv, keep):
    out = {k: np.zeros((10, P.N_CELLS), dtype=np.int64) for k in ("C_t", "A_t", "C_o", "A_o")}
    stats = {"games": 0, "target_moves": 0, "opponent_moves": 0}
    for g in keep:
        m = mv["moves"][mv["off"][g]:mv["off"][g + 1]]
        tc = 1 if mv["color"][g] == "B" else 2
        lv = int(mv["label"][g])
        snaps, meta, _ = P.replay(m, lambda i, c: True)
        for s, x in zip(snaps, meta):
            side = "t" if x[1] == tc else "o"
            ref = P.ref_cell(s, x)
            for cell in ref.values():
                out["A_" + side][lv, cell] += 1
            out["C_" + side][lv, ref[x[2]]] += 1
            stats["target_moves" if side == "t" else "opponent_moves"] += 1
        stats["games"] += 1
    return out, stats


class TestCountAndScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs = games()
        cls.mv = synthetic_moves(gs, [(3 * i) % 10 for i in range(len(gs))],
                                 ["B" if i % 3 else "W" for i in range(len(gs))])

    def test_count_equals_brute_force_with_stride_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stride, limit, workers in ((1, 0, 2), (3, 0, 2), (2, 9, 3)):
                tot, stats = quiet(P.count_tables, self.mv, workers=workers, stride=stride, limit=limit,
                                   tmp_root=tmp)
                G = len(self.mv["off"]) - 1
                keep = [g for g in range(min(limit, G) if limit else G) if g % stride == 0]
                ref, rstats = brute_counts(self.mv, keep)
                for k in ref:
                    self.assertTrue(np.array_equal(tot[k], ref[k]), (k, stride, limit))
                summ = P.count_summary(tot, stats, stride, limit, 0.0)
                for k, v in rstats.items():
                    self.assertEqual(summ[k], v, (k, stride, limit))
                self.assertEqual(summ["anomalies"], 0)
                self.assertEqual(os.listdir(tmp), [], "temporary shard folder not removed")

    def test_int32_guard_in_merge(self):
        z = lambda: np.zeros(10 * P.N_CELLS, dtype=np.int32)  # noqa: E731
        ok_stats = {"target_moves": 10, "opponent_moves": 12}
        with tempfile.TemporaryDirectory() as tmp:
            def shard(name, arrays, stats):
                path = os.path.join(tmp, name)
                np.savez(path, **arrays)
                return path, stats
            a = {k: z() for k in ("C_t", "A_t", "C_o", "A_o")}
            a["A_o"][7] = 2 ** 31 - 1
            b = {k: z() for k in ("C_t", "A_t", "C_o", "A_o")}
            b["A_o"][7] = 5
            tot = P.merge_shards([shard("s0.npz", a, ok_stats), shard("s1.npz", b, ok_stats)])
            self.assertEqual(int(tot["A_o"][7]), 2 ** 31 + 4)              # int64 total above 2^31 is fine
            self.assertEqual(os.listdir(tmp), [])
            wrapped = {k: z() for k in ("C_t", "A_t", "C_o", "A_o")}
            wrapped["A_t"][3] = np.int32(-5)                                 # a partial that passed 2^31 - 1
            with self.assertRaisesRegex(P.P04Error, "overflowed"):
                P.merge_shards([shard("s2.npz", wrapped, ok_stats)])
            limit = (2 ** 32) // 361                                          # 361 x moves must stay below 2^32
            with self.assertRaisesRegex(P.P04Error, "2\\^32"):
                P.merge_shards([shard("s3.npz", b, {"target_moves": limit + 1, "opponent_moves": 0})])
            self.assertLess(P.int32_bound({"target_moves": limit, "opponent_moves": 3}), 2 ** 32)
            # the registered count: 8 workers, the largest side 79,985,811 moves -> about 10.0 M moves per shard
            self.assertLess(361 * (79985811 // 8 + 1), 2 ** 32)
            wide = {k: z().astype(np.int64) for k in ("C_t", "A_t", "C_o", "A_o")}
            with self.assertRaisesRegex(P.P04Error, "int64"):
                P.merge_shards([shard("s4.npz", wide, ok_stats)])

    def test_int32_preflight_before_the_workers(self):
        # shard_moves gives each worker exactly the moves _count_shard counts (no anomaly in these games)
        with tempfile.TemporaryDirectory() as tmp:
            for stride, limit, workers in ((1, 0, 3), (2, 9, 2)):
                _, stats = quiet(P.count_tables, self.mv, workers=workers, stride=stride, limit=limit, tmp_root=tmp)
                self.assertEqual(P.shard_moves(self.mv, workers, stride, limit),
                                 [(s["target_moves"], s["opponent_moves"]) for s in stats], (stride, limit, workers))
                pre = P.int32_preflight(self.mv, workers, stride, limit)
                self.assertEqual(pre["max_shard_bound"], max(P.int32_bound(s) for s in stats))
            # a bound at 2^32 stops before any temporary folder or worker exists
            saved = P.INT32_WRAP
            P.INT32_WRAP = P.int32_preflight(self.mv, 2)["max_shard_bound"]
            try:
                with self.assertRaisesRegex(P.P04Error, "use more workers"):
                    P.count_tables(self.mv, workers=2, tmp_root=tmp)
                self.assertEqual(os.listdir(tmp), [])
            finally:
                P.INT32_WRAP = saved

    def test_saved_counts_are_deterministic_and_reload(self):
        tot, _ = quiet(P.count_tables, self.mv, workers=1, stride=1, limit=4)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
            P.save_counts(a, tot)
            P.save_counts(b, {k: tot[k].copy() for k in ("A_o", "C_o", "A_t", "C_t")})
            self.assertEqual(P.sha256_file(a), P.sha256_file(b))
            back = P.load_counts(a)
            for k in tot:
                self.assertTrue(np.array_equal(back[k], tot[k]))
            with np.load(a) as z:
                self.assertEqual(z.files, ["C_t", "A_t", "C_o", "A_o"])

    def test_tables_record(self):
        import hashlib
        tabs, _ = P.build_tables(random_counts(np.random.default_rng(21)))
        rec = P.tables_record(tabs)
        self.assertEqual(rec["tables_sha256"], hashlib.sha256(tabs.tobytes()).hexdigest())
        self.assertEqual(rec["tables_array"]["shape"], [2, P.N_CELLS, 10])
        self.assertEqual(rec["tables_array"]["dtype"], "float64")
        other = tabs.copy()
        other[1, 5, 3] = np.nextafter(other[1, 5, 3], 1.0)
        self.assertNotEqual(P.tables_record(other)["tables_sha256"], rec["tables_sha256"])
        # a non-contiguous view of the same values hashes the same values (C order)
        self.assertEqual(P.tables_record(np.asfortranarray(tabs))["tables_sha256"], rec["tables_sha256"])

    def test_score_game_by_hand(self):
        rng = np.random.default_rng(11)
        tabs = rng.uniform(0.001, 0.02, size=(2, P.N_CELLS, 10))
        for g, m in enumerate(games()[:6]):
            tc = 1 if g % 2 else 2
            ll, nm, an = P.score_game(m, tc, tabs)
            want = np.zeros((2, 3, 10))
            wnm = np.zeros((2, 3), dtype=np.int64)
            snaps, meta, _ = P.replay(m, lambda i, c: True)
            for s, x in zip(snaps, meta):
                si = 0 if x[1] == tc else 1                 # 0 = the target's own move, 1 = the opponent's
                ref = P.ref_cell(s, x)
                T = tabs[si]
                den = np.sum([T[c] for c in ref.values()], axis=0)
                want[si, P.phase_of(x[0])] += np.log(T[ref[x[2]]]) - np.log(den)
                wnm[si, P.phase_of(x[0])] += 1
            self.assertLess(np.max(np.abs(ll - want)), 1e-9, g)
            self.assertTrue(np.array_equal(nm, wnm))

    def test_sides_follow_the_target_colour(self):
        rng = np.random.default_rng(12)
        tabs = rng.uniform(0.001, 0.02, size=(2, P.N_CELLS, 10))
        m = games()[4]
        a, _, _ = P.score_game(m, 1, tabs)
        b, _, _ = P.score_game(m, 2, tabs[::-1].copy())
        self.assertTrue(np.array_equal(a[0], b[1]))
        self.assertTrue(np.array_equal(a[1], b[0]))

    def test_score_games_pool_equals_direct(self):
        rng = np.random.default_rng(13)
        tabs = rng.uniform(0.001, 0.02, size=(2, P.N_CELLS, 10))
        gs = games()
        items = [(i, gs[i], 1 + i % 2) for i in range(0, len(gs), 2)]
        with tempfile.TemporaryDirectory() as tmp:
            ll, nm, an = quiet(P.score_games, items, tabs, workers=2, tmp_root=tmp, chunk=2)
            self.assertEqual(os.listdir(tmp), [])
        for i, m, tc in items:
            d, dn, da = P.score_game(m, tc, tabs)
            self.assertTrue(np.array_equal(ll[i], d))
            self.assertTrue(np.array_equal(nm[i], dn))
        self.assertTrue(np.isnan(ll[1]).all())             # a row not given stays NaN


# ================================================================ question file, val, dupcheck
def write_questions(path, qs):
    """qs: [(question_id, [(colour, sgf), ...])]"""
    header = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for q, gl in qs:
            row = [q, len(gl)] + [s for _, s in gl] + [""] * (20 - len(gl)) + [c for c, _ in gl] + [""] * (20 - len(gl))
            w.writerow(row)


def to_sgf(m):
    let = "abcdefghijklmnopqrs"
    out = []
    for code in m.tolist():
        c = "W" if code >= 512 else "B"
        y, x = divmod(code & 511, 19)
        out.append(f";{c}[{let[x]}{let[y]}]")
    return "(;SZ[19]" + "".join(out) + ")"


class TestQuestionsValDup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        gs = games()
        self.qs = [("val_q_0001", [("B", to_sgf(gs[1])), ("W", to_sgf(gs[2])), ("W", "(;SZ[19])")]),
                   ("val_q_0002", [("W", to_sgf(gs[3])), ("B", to_sgf(np.array([3, 3], dtype=np.uint16)))])]
        self.qpath = os.path.join(self.tmp, "rank_val_questions.csv")
        write_questions(self.qpath, self.qs)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_slots_and_sgf_round_trip(self):
        slots = P.read_question_slots(self.qpath)
        self.assertEqual([(q, s, c) for q, s, c, _ in slots],
                         [("val_q_0001", 1, "B"), ("val_q_0001", 2, "W"), ("val_q_0001", 3, "W"),
                          ("val_q_0002", 1, "W"), ("val_q_0002", 2, "B")])
        self.assertTrue(np.array_equal(P.encode_one(slots[0][3]), games()[1]))

    def test_slots_must_match_val_keys(self):
        slots = P.read_question_slots(self.qpath)
        vk = {"n": 5, "question_id": [s[0] for s in slots], "slot": [s[1] for s in slots],
              "color": [s[2] for s in slots]}
        P.slots_match_keys(slots, vk)
        vk["color"] = list(vk["color"])
        vk["color"][1] = "B"
        with self.assertRaises(P.P04Error):
            P.slots_match_keys(slots, vk)

    def test_val_scores_and_failures(self):
        slots = P.read_question_slots(self.qpath)
        tabs = np.random.default_rng(5).uniform(0.001, 0.02, size=(2, P.N_CELLS, 10))
        res = quiet(P.val_scores, slots, tabs, workers=1, tmp_root=self.tmp)
        self.assertEqual(res["replay_failed"].tolist(), [False, False, True, False, True])  # empty; anomaly
        self.assertEqual(res["moves_parsed"].tolist(), [len(games()[1]), len(games()[2]), 0, len(games()[3]), 2])
        ll, _, _ = P.score_game(games()[2], 2, tabs)
        self.assertTrue(np.array_equal(res["o4"][1], ll[1].sum(0)))
        self.assertTrue(np.array_equal(res["t4"][1], ll[0].sum(0)))

    def test_dupcheck_core(self):
        # dupcheck.py's semantics: the same move sequence; 'same' = a fit row with the slot's target colour, 'other'
        # (colour swapped) = a fit row with the other target colour (the same game from the other player's side)
        gs = games()
        fit = synthetic_moves(gs[:6], [0] * 6, ["B", "W", "B", "B", "W", "B"])
        fit["moves"] = np.concatenate([gs[i] for i in (0, 1, 2, 3, 4, 2)]).astype(np.uint16)   # game 2 twice: B and B
        fit["off"] = np.r_[0, np.cumsum([len(gs[i]) for i in (0, 1, 2, 3, 4, 2)])].astype(np.int64)
        fit["color"] = np.array(["B", "W", "B", "B", "W", "W"])                                 # ... now B and W
        h = P.fit_hashes(fit)
        self.assertEqual(h[P.seq_hash(gs[2])], 3)
        self.assertEqual(h[P.seq_hash(gs[1])], 2)
        flipped = (gs[1] ^ np.uint16(512)).astype(np.uint16)       # colour bits flipped: not the same game
        seqs = [gs[1], gs[1], gs[2], gs[7], np.zeros(0, dtype=np.uint16), gs[0], flipped]
        cols = ["W", "B", "B", "B", "B", "W", "W"]
        same, other, empty = P.match_games(seqs, cols, h)
        self.assertEqual(same.tolist(), [True, False, True, False, False, False, False])
        self.assertEqual(other.tolist(), [False, True, True, False, False, True, False])
        self.assertEqual(empty.tolist(), [False, False, False, False, True, False, False])
        slots = [(f"q{i}", 1, c, "") for i, c in enumerate(cols)]
        rec = P.dup_record(slots, same, other, empty)
        self.assertEqual((rec["slots_in_fit_any"], rec["slots_in_fit_same_target_colour"],
                          rec["slots_in_fit_other_target_colour"], rec["stop"]), (4, 2, 3, True))
        self.assertEqual(rec["matched_slots"], [["q0", 1, "same"], ["q1", 1, "other"], ["q2", 1, "same+other"],
                                                ["q5", 1, "other"]])
        with self.assertRaises(P.P04Error):
            P.match_games(seqs, cols[:-1], h)
        with self.assertRaises(P.P04Error):
            P.match_games(seqs[:1], ["b"], h)

    def test_dupcheck_matches_design_script(self):
        # the design's dupcheck.py loop (hash -> fit rows; hit, 'any fit row colour != this colour'), same data
        import hashlib
        gs = games()
        fit = synthetic_moves([gs[i] for i in (0, 1, 2, 2, 3)], [0] * 5, ["B", "W", "B", "W", "W"])
        inner = synthetic_moves([gs[i] for i in (1, 2, 3, 4, 0)], [0] * 5, ["W", "W", "B", "B", "W"])
        dh = {}
        for g in range(5):
            k = hashlib.sha1(np.asarray(fit["moves"][fit["off"][g]:fit["off"][g + 1]]).tobytes()).digest()
            dh.setdefault(k, []).append(g)
        want_hit, want_other = [], []
        for g in range(5):
            k = hashlib.sha1(np.asarray(inner["moves"][inner["off"][g]:inner["off"][g + 1]]).tobytes()).digest()
            hit = dh.get(k, [])
            want_hit.append(bool(hit))
            want_other.append(any(fit["color"][j] != inner["color"][g] for j in hit))
        seqs = [inner["moves"][inner["off"][g]:inner["off"][g + 1]] for g in range(5)]
        same, other, _ = P.match_games(seqs, inner["color"].tolist(), P.fit_hashes(fit))
        self.assertEqual((same | other).tolist(), want_hit)
        self.assertEqual(other.tolist(), want_other)
        self.assertEqual(want_hit, [True, True, True, False, True])
        self.assertEqual(want_other, [False, True, True, False, True])
        big = [(f"q{i}", 1, "B", "") for i in range(100)]
        z = np.zeros(100, bool)
        one = z.copy()
        one[5] = True
        self.assertTrue(P.dup_record(big, one, z, z)["stop"])               # exactly 1 % stops
        self.assertFalse(P.dup_record(big + [("x", 1, "B", "")], np.r_[one, False], np.r_[z, False],
                                      np.r_[z, False])["stop"])            # 1 of 101 does not
        self.assertEqual(P.dup_record(big, one, z, z)["matched_slots"], [["q5", 1, "same"]])

    def test_answer_paths_refused(self):
        for p in (os.path.join(REPO, "data", "local", "rank_val_answers.csv"), "C:/x/Answers/q.csv",
                  os.path.join(self.tmp, "my_answer.csv")):
            with self.assertRaises(P.P04Error):
                P.refuse_answer_path(p)
        bad = os.path.join(self.tmp, "answers_q.csv")
        write_questions(bad, self.qs)
        with self.assertRaises(P.P04Error):
            P.read_question_slots(bad)
        with self.assertRaises(P.P04Error):
            P.write_json(os.path.join(self.tmp, "answer.json"), {})
        with self.assertRaises(P.P04Error):
            P.load_counts(os.path.join(self.tmp, "answers.npz"))

    def test_no_label_file_name_in_tool(self):
        src = open(os.path.join(TOOLS, "p04_opp.py"), encoding="utf-8").read()
        self.assertNotIn("answers.csv", src)
        self.assertNotIn("score.py", src.replace("no score.py", ""))


# ================================================================ gates and output folders
def git(repo, *args):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false"] + list(args),
                          capture_output=True, text=True)


class TestGates(unittest.TestCase):
    def test_val_gate_in_a_scratch_repo(self):
        with tempfile.TemporaryDirectory() as repo:
            self.assertEqual(git(repo, "init", "-q").returncode, 0)
            fz = os.path.join(repo, "results", "prereg04", "rank", "frozen_config.json")
            with self.assertRaisesRegex(P.P04Error, "missing"):
                P.require_frozen_config(fz, repo)
            os.makedirs(os.path.dirname(fz))
            good = {"format": P.FROZEN_CONFIG_FORMAT, "smoke": False, "preview": False, "design_input": False,
                    "combo": {"id": "Q01", "sources": ["C5", "OP", "O4"]}}

            def put(obj, commit=True):
                with open(fz, "w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(obj) + "\n")
                if commit:
                    git(repo, "add", "-A")
                    self.assertEqual(git(repo, "commit", "-q", "--allow-empty", "-m", "x").returncode, 0)
            put(good, commit=False)
            with self.assertRaisesRegex(P.P04Error, "not committed"):
                P.require_frozen_config(fz, repo)
            git(repo, "add", "-A")                                   # staged only: still not committed
            with self.assertRaisesRegex(P.P04Error, "changed"):
                P.require_frozen_config(fz, repo)
            self.assertEqual(git(repo, "commit", "-q", "-m", "x").returncode, 0)
            rec = P.require_frozen_config(fz, repo)
            self.assertEqual(rec["state"], "committed and unchanged")
            self.assertEqual(rec["combo"], "Q01")
            with open(fz, "a") as f:
                f.write(" ")
            with self.assertRaisesRegex(P.P04Error, "changed"):
                P.require_frozen_config(fz, repo)
            # committed but not a real frozen config with O4 -> refused (content, not only git state)
            for bad, msg in (({"selected": "Q01"}, "not a"),
                             (dict(good, format="c10-prereg04 rank tune_report v1"), "not a"),
                             (dict(good, preview=True), "not a real frozen config"),
                             (dict(good, design_input=True), "not a real frozen config"),
                             (dict(good, smoke=True), "not a real frozen config"),
                             ({k: v for k, v in good.items() if k != "preview"}, "not a real frozen config"),
                             (dict(good, combo={"id": "Q00", "sources": ["C5", "OP"]}), "does not contain O4")):
                put(bad)
                with self.assertRaisesRegex(P.P04Error, msg):
                    P.require_frozen_config(fz, repo)
            put(dict(good, combo={"id": "Q02", "sources": ["C5", "OP", "O4", "SWA", "SWB"]}))
            self.assertEqual(P.require_frozen_config(fz, repo)["combo"], "Q02")

    def test_val_calls_the_gate_before_anything(self):
        calls = []

        def gate(*a, **k):
            calls.append("gate")
            raise P.P04Error("gate")

        spy = lambda name: (lambda *a, **k: calls.append(name))  # noqa: E731
        old = {n: getattr(P, n) for n in ("require_frozen_config", "registered_counts", "read_question_slots",
                                           "load_counts", "val_scores", "resolve_out")}
        try:
            P.require_frozen_config = gate
            for n in ("registered_counts", "read_question_slots", "load_counts", "val_scores"):
                setattr(P, n, spy(n))
            P.resolve_out = lambda a, d, cmd: types.SimpleNamespace(out_dir=tempfile.gettempdir(), real=False,
                                                                     preview=True, subset=False, cmd=cmd, tools={})
            rc = quiet(P.main, ["val", "--out-dir", tempfile.gettempdir()])
        finally:
            for n, f in old.items():
                setattr(P, n, f)
        self.assertEqual(rc, 2)
        self.assertEqual(calls, ["gate"])

    def test_val_has_no_bypass_option(self):
        ap = P.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["val", "--smoke"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["val", "--counts", "x.npz"])

    def test_out_dir_rules(self):
        ns = lambda **k: types.SimpleNamespace(**{"out_dir": None, "stride": 1, "limit": 0, "workers": 1, **k})  # noqa
        with self.assertRaisesRegex(P.P04Error, "inside the repository must be"):
            P.resolve_out(ns(out_dir=os.path.join(REPO, "results", "prereg04", "design")), P.OPP_OUT, "count")
        with self.assertRaisesRegex(P.P04Error, "inside the repository must be"):
            P.resolve_out(ns(out_dir=os.path.join(REPO, "data", "prereg04")), P.OPP_OUT, "inner")
        with self.assertRaisesRegex(P.P04Error, "preview"):
            P.resolve_out(ns(stride=2), P.OPP_OUT, "count")
        with self.assertRaisesRegex(P.P04Error, "preview"):
            P.resolve_out(ns(limit=5), P.OPP_OUT, "inner")
        with tempfile.TemporaryDirectory() as tmp:
            c = P.resolve_out(ns(out_dir=os.path.join(tmp, "o"), stride=3), P.OPP_OUT, "count")
            self.assertTrue(c.preview and not c.real and c.subset)
            self.assertIn("p04_opp.py", c.tools["files_sha256"])
            self.assertIn("p03_rank_fuse.py", c.tools["files_sha256"])
            self.assertTrue(os.path.isdir(os.path.join(tmp, "o")))

    def test_tool_state_covers_every_p02_tool(self):
        # the same frozen set as p04_rank_fuse.frozen_files(): every tools/p02_*.py, p03_rank_fuse, c10data, decide
        want = {os.path.basename(p) for p in glob.glob(os.path.join(TOOLS, "p02_*.py"))} | \
            {"p03_rank_fuse.py", "c10data.py", "decide.py", "p04_opp.py"}
        self.assertTrue({"p02_player_fuse.py", "p02_player_open.py"} <= want)
        self.assertEqual(set(P.tool_state()["files_sha256"]), want)

    def test_real_mode_needs_committed_tools(self):
        for tools in ({"git_head": "x", "dirty_or_untracked": ["tools/p04_opp.py"]},
                      {"git_head": None, "dirty_or_untracked": []}, {"git_head": "x", "dirty_or_untracked": None}):
            with self.assertRaises(P.P04Error):
                P.require_own_committed(tools)
        P.require_own_committed({"git_head": "x", "dirty_or_untracked": []})

    def test_registered_constants(self):
        self.assertEqual(P.PINNED_COUNT, {"games": 720307, "target_moves": 79985811, "opponent_moves": 79985352,
                                          "anomalies": 0, "n_cells": 557928})
        self.assertEqual((P.INNER_GAMES, P.VAL_SLOTS, P.VAL_QUESTIONS), (73556, 19156, 1589))
        self.assertEqual((P.ALPHA, P.TOL, P.DUP_STOP_SHARE), (20.0, 1e-12, 0.01))
        self.assertEqual(P.manifest_sha("counts_fit.npz"),
                         "f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5")
        self.assertEqual(P.manifest_sha("proxy_inner_all.npz"),
                         "3e6e3efd90816fb8770691fd1786840e834a66cbccebcd4c2afa9da8ce163b34")

    def test_inner_rows_alignment(self):
        inner = {"rank": np.array(["D", "C", "D"]), "game_id": np.array(["g1", "g2", "g3"]),
                 "player_id": np.array(["p1", "p2", "p3"]), "color": np.array(["B", "W", "B"])}
        keys = {"rank": ["C", "D", "D"], "game_id": ["g2", "g3", "g1"], "player_id": ["p2", "p3", "p1"],
                "color": ["W", "B", "B"]}
        self.assertEqual(P.inner_rows(inner, keys).tolist(), [1, 2, 0])
        keys["player_id"] = ["p2", "p1", "p1"]
        with self.assertRaises(P.P04Error):
            P.inner_rows(inner, keys)


@unittest.skipUnless(SLOW, "set C10_SLOW=1 for the real selftest (fit games through load_rows, about 2 min)")
class TestRealSelftest(unittest.TestCase):
    def test_selftest_on_fit(self):
        rec = quiet(P.run_selftest, quiet(P.load_part, "fit"))
        self.assertTrue(rec["passed"], rec["problems"])
        self.assertEqual(rec["capture"]["games_with_mismatch"], 0)
        self.assertGreaterEqual(rec["features"]["positions"], 89)
        self.assertEqual(rec["features"]["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
