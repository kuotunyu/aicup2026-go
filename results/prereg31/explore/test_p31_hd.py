"""Unit tests of p31_hd.py (prereg31 explore, stage 0a). Synthetic data only: no real game, no data folder.

    PY -m pytest -q -p no:cacheprovider results/prereg31/explore/test_p31_hd.py
"""

import ast
import hashlib
import json
import os
import sys
import tarfile
from fractions import Fraction

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p31_hd as H  # noqa: E402

PLAIN = "(;SZ[19];B[qd];W[dq];B[aa];W[bb])"
STAR_B = "(;SZ[19];B[pd];W[dq];B[aa];W[bb])"
STAR_W = "(;SZ[19];B[qd];W[dp];B[aa];W[bb])"


def with_hd(sgf, n):
    assert sgf.startswith("(;SZ[19]")
    return "(;SZ[19]HD[%d]" % n + sgf[len("(;SZ[19]"):]


# ================================================================ one game
class TestParseHd:
    @pytest.mark.parametrize("n", [0, 1, 2, 9])
    def test_value_and_removal(self, n):
        assert H.parse_hd(with_hd(PLAIN, n)) == (n, PLAIN)

    def test_hd_before_sz(self):
        assert H.parse_hd("(;HD[3]SZ[19];B[qd])") == (3, "(;SZ[19];B[qd])")

    def test_root_only_game(self):
        assert H.parse_hd("(;SZ[19]HD[1])") == (1, "(;SZ[19])")

    @pytest.mark.parametrize("sgf,why", [
        (PLAIN, "hd_missing"),
        ("(;SZ[19];B[qd]HD[1];W[dq])", "hd_missing"),          # not in the root node
        ("(;SZ[19]HD[0]HD[1];B[qd])", "hd_multiple"),
        ("(;SZ[19]HD[];B[qd])", "hd_value"),
        ("(;SZ[19]HD[x];B[qd])", "hd_value"),
        ("(;SZ[19]HD[-1];B[qd])", "hd_value"),
        ("(;SZ[19]HD[ 1];B[qd])", "hd_value"),
        ("(;SZ[19]HD[10];B[qd])", "hd_range"),
        ("(;SZ[19]XHD[1];B[qd])", "hd_identifier"),
        ("SZ[19]HD[1]", "no_root"),
        ("", "no_root"),
    ])
    def test_malformed(self, sgf, why):
        with pytest.raises(H.HdError) as e:
            H.parse_hd(sgf)
        assert str(e.value) == why

    def test_hd_group(self):
        assert [H.hd_group(n) for n in (0, 1, 2, 5, 9)] == [0, 1, 2, 2, 2]


class TestFingerprint:
    def test_no_star_move(self):
        assert H.fingerprint(PLAIN) is True
        assert H.fingerprint(with_hd(PLAIN, 2)) is True

    @pytest.mark.parametrize("sgf", [STAR_B, STAR_W, "(;SZ[19];B[qd];W[pd])", "(;SZ[19];B[dp];W[qq])",
                                     "(;SZ[19];B[qd];W[aa];B[bb];W[cc];B[dp])"])
    def test_star_move_of_either_colour_anywhere(self, sgf):
        assert H.fingerprint(sgf) is False

    def test_other_points_do_not_count(self):
        assert H.fingerprint("(;SZ[19];B[dd];W[pp];B[dq];W[pe];B[qd];W[cp])") is True

    def test_unusual_spacing(self):
        assert H.fingerprint("(;SZ[19]; B [pd];W[qq])") is False
        assert H.fingerprint("(;SZ[19];C[pd];W[qq])") is True       # not a move


# ================================================================ seam A
def row(pid="p1", gid="g_0000001", rank="D", col="B", sgf=PLAIN):
    return [pid, gid, rank, col, sgf]


class TestCompareRow:
    def test_equal(self):
        probs, hd, f, c = H.compare_row(row(sgf=STAR_B), row(sgf=with_hd(STAR_B, 1)), "D")
        assert (probs, hd, f, c) == ([], 1, False, 0)
        probs, hd, f, c = H.compare_row(row(col="W"), row(col="W", sgf=with_hd(PLAIN, 4)), "D")
        assert (probs, hd, f, c) == ([], 4, True, 1)

    @pytest.mark.parametrize("r11,why", [
        (row(pid="p2", sgf=with_hd(PLAIN, 0)), "player_id"),
        (row(gid="g_0000002", sgf=with_hd(PLAIN, 0)), "game_id"),
        (row(rank="C", sgf=with_hd(PLAIN, 0)), "rank"),
        (row(col="W", sgf=with_hd(PLAIN, 0)), "color"),
        (row(sgf=PLAIN), "hd_parse"),
        (row(sgf=with_hd(PLAIN, 0)[:-1] + ";B[cc])"), "sgf"),
        (row(sgf=with_hd(PLAIN, 0) + " "), "sgf"),
        (row()[:4], "width"),
    ])
    def test_each_problem(self, r11, why):
        probs, _hd, _f, _c = H.compare_row(row(), r11, "D")
        assert probs == [why]

    def test_rank_of_the_file(self):
        probs, _hd, _f, _c = H.compare_row(row(rank="C"), row(rank="C", sgf=with_hd(PLAIN, 0)), "D")
        assert probs == ["rank"]

    def test_colour_must_be_b_or_w(self):
        probs, _hd, _f, _c = H.compare_row(row(col="X"), row(col="X", sgf=with_hd(PLAIN, 0)), "D")
        assert probs == ["color"]


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write(",".join(header) + "\r\n")
        for r in rows:
            f.write(",".join(r) + "\r\n")


def game(i, star, extra=0):
    """A distinct synthetic game; star=True puts a move on pd (no fingerprint)."""
    a, b = "abcdefghijklmno"[i % 15], "abcdefghijklmno"[(i // 15) % 15]
    first = "pd" if star else "qd"
    return "(;SZ[19];B[%s];W[%s%s];B[%s%s]%s)" % (first, a, b, b, a, ";W[ss]" * extra)


def rank_rows(rank, spec):
    """spec: per colour a list of (hd, F). Returns (v1.0 rows, v1.1 rows); game ids in file order."""
    r10, r11, i = [], [], 0
    for col in H.COLOURS:
        for hd, f in spec[col]:
            i += 1
            sgf = game(i, star=not f)
            base = ["%s_p%d" % (rank, i % 3), "g_%07d" % i, rank, col]
            r10.append(base + [sgf])
            r11.append(base + [with_hd(sgf, hd)])
    return r10, r11


SPEC = [(0, True)] + [(0, False)] * 5 + [(1, True)] + [(1, False)] * 2 + [(3, True)]      # 10 games per colour


def make_training(tmp, spec=None, mutate=None):
    """v1.0 folder, v1.1 folder and a tar.xz of the v1.1 files (plus a macOS decoy). mutate(rank, r10, r11) may edit
    the row lists in place."""
    spec = spec or {c: SPEC for c in H.COLOURS}
    v10, v11 = os.path.join(tmp, "v10"), os.path.join(tmp, "v11", "training")
    for rank in H.RANKS:
        r10, r11 = rank_rows(rank, spec)
        if mutate:
            mutate(rank, r10, r11)
        write_csv(os.path.join(v10, "train_%s.csv" % rank), H.TRAIN_HEADER, r10)
        write_csv(os.path.join(v11, "train_%s.csv" % rank), H.TRAIN_HEADER, r11)
    decoy = os.path.join(tmp, "decoy", "._train_D.csv")
    os.makedirs(os.path.dirname(decoy))
    with open(decoy, "wb") as f:
        f.write(b"\x00not a csv")
    tarball = os.path.join(tmp, "v11.tar.xz")
    with tarfile.open(tarball, "w:xz") as tf:
        tf.add(decoy, arcname="training/._train_D.csv")
        for rank in H.RANKS:
            tf.add(os.path.join(v11, "train_%s.csv" % rank), arcname="training/train_%s.csv" % rank)
    return v10, v11, tarball


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def reg(monkeypatch):
    """Registers a synthetic tarball as the expected one."""
    def go(tarball):
        monkeypatch.setattr(H, "TARBALL_SHA256", sha(tarball))
        monkeypatch.setattr(H, "TARBALL_BYTES", os.path.getsize(tarball))
    return go


def quiet(*_a, **_k):
    pass


class TestBuild:
    def test_pass_writes_table(self, tmp_path, reg):
        v10, v11, tb = make_training(str(tmp_path))
        reg(tb)
        out = str(tmp_path / "o" / "hd_table.npz")
        rec = H.build(v10, v11, tb, out, log=quiet)
        assert rec["status"] == "PASS" and rec["stop_reasons"] == [] and rec["seam_a"]["ok"] is True
        assert rec["totals"]["rows_v10"] == rec["totals"]["rows_v11"] == rec["totals"]["rows_equal"] == 200
        assert rec["totals"]["plain_form_v10"] == rec["totals"]["plain_form_v11"] == 200
        assert rec["table"]["rows"] == 200 and rec["table"]["sha256"] == sha(out)
        t = H.load_table(out)
        assert t["hd"].dtype == np.int8 and t["F"].dtype == np.bool_ and t["game_id"].dtype.kind == "S"
        assert len(set(zip(t["rank"].tolist(), t["game_id"].tolist()))) == 200
        d = t["rank"] == 0
        assert t["game_id"][d][0] == b"g_0000001" and t["player_id"][d][0] == b"D_p1"
        assert t["hd"][d].tolist() == [h for h, _f in SPEC] * 2
        assert t["F"][d].tolist() == [f for _h, f in SPEC] * 2
        assert t["color"][d].tolist() == [0] * 10 + [1] * 10
        with np.load(out, allow_pickle=False) as z:           # no pickled object inside
            assert sorted(z.files) == ["F", "color", "format", "game_id", "hd", "player_id", "rank", "ranks"]

    def test_table_bytes_are_deterministic(self, tmp_path, reg):
        v10, v11, tb = make_training(str(tmp_path))
        reg(tb)
        a, b = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
        H.build(v10, v11, tb, a, log=quiet)
        H.build(v10, v11, tb, b, log=quiet)
        assert sha(a) == sha(b)

    def stopped(self, tmp_path, reg, mutate, register=True):
        v10, v11, tb = make_training(str(tmp_path), mutate=mutate)
        if register:
            reg(tb)
        out = str(tmp_path / "hd_table.npz")
        rec = H.build(v10, v11, tb, out, log=quiet)
        assert rec["status"] == "STOP" and rec["table"] is None and not os.path.exists(out)
        assert rec["seam_a"]["ok"] is False
        return rec

    def test_sgf_mismatch_stops(self, tmp_path, reg):
        def mutate(rank, r10, r11):
            if rank == "B":
                r11[3][4] = r11[3][4].replace(";B[pd]", ";B[pe]")
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["rows_all_equal"]
        assert rec["files"]["B"]["problems"]["sgf"] == 1 and rec["files"]["B"]["examples"][0]["row"] == 4
        assert rec["totals"]["rows_not_equal"] == 1 and rec["totals"]["rows_equal"] == 199

    def test_missing_hd_stops(self, tmp_path, reg):
        def mutate(rank, r10, r11):
            if rank == "6D":
                r11[0][4] = r10[0][4]
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["rows_all_equal"] and rec["totals"]["problems"]["hd_parse"] == 1

    @pytest.mark.parametrize("col,name", [(0, "player_id"), (3, "color")])
    def test_other_columns_stop(self, tmp_path, reg, col, name):
        def mutate(rank, r10, r11):
            if rank == "A":
                r11[5][col] = "W" if name == "color" else "someone"
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["rows_all_equal"] and rec["totals"]["problems"][name] == 1

    def test_row_count_stops(self, tmp_path, reg):
        def mutate(rank, r10, r11):
            if rank == "1D":
                del r11[-1]
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["row_counts_equal"]
        assert (rec["files"]["1D"]["rows_v10"], rec["files"]["1D"]["rows_v11"]) == (20, 19)

    def test_extra_row_in_v11_stops(self, tmp_path, reg):
        def mutate(rank, r10, r11):
            if rank == "D":
                r11.append(["x", "g_9999999", "D", "B", with_hd(PLAIN, 0)])
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["row_counts_equal"]

    def test_duplicate_key_stops(self, tmp_path, reg):
        def mutate(rank, r10, r11):
            if rank == "C":
                r10[7][1] = r11[7][1] = r10[2][1]
        rec = self.stopped(tmp_path, reg, mutate)
        assert rec["stop_reasons"] == ["keys_unique"] and rec["files"]["C"]["duplicate_game_ids"] == 1

    def test_tarball_sha_stops(self, tmp_path, reg):
        rec = self.stopped(tmp_path, reg, None, register=False)
        assert rec["stop_reasons"] == ["tarball_sha"] and rec["totals"]["rows_not_equal"] == 0

    def test_extracted_file_not_from_tarball_stops(self, tmp_path, reg):
        v10, v11, tb = make_training(str(tmp_path))
        reg(tb)
        p = os.path.join(v11, "train_2D.csv")                 # same games, another HD: seam A rows still equal
        with open(p, "rb") as f:
            raw = f.read()
        with open(p, "wb") as f:
            f.write(raw.replace(b"HD[3]", b"HD[4]"))
        rec = H.build(v10, v11, tb, str(tmp_path / "t.npz"), log=quiet)
        assert rec["status"] == "STOP" and rec["stop_reasons"] == ["extracted_matches_tarball"]
        assert rec["totals"]["rows_not_equal"] == 0 and rec["table"] is None

    def test_cli_runs_once(self, tmp_path, reg):
        v10, v11, tb = make_training(str(tmp_path))
        reg(tb)
        out, rec = str(tmp_path / "t.npz"), str(tmp_path / "b.json")
        argv = ["build", "--v10", v10, "--v11", v11, "--tarball", tb, "--out", out, "--record", rec]
        assert H.main(argv) == 0
        assert json.load(open(rec, encoding="utf-8"))["status"] == "PASS"
        with pytest.raises(SystemExit) as e:
            H.main(argv)
        assert "runs once" in str(e.value)

    def test_cli_stop_exit_code(self, tmp_path):
        v10, v11, tb = make_training(str(tmp_path))            # tarball not registered
        out, rec = str(tmp_path / "t.npz"), str(tmp_path / "b.json")
        assert H.main(["build", "--v10", v10, "--v11", v11, "--tarball", tb, "--out", out, "--record", rec]) == 3
        assert json.load(open(rec, encoding="utf-8"))["status"] == "STOP" and not os.path.exists(out)


# ================================================================ 1.3: the Z / C1 rule
class TestFingerprintDecision:
    TRAIN = ([800, 150, 50], [80, 30, 50])       # f = .1 .2 1; pi = .8 .15 .05; s1 = .05 * .8 / .2 = .2; s1/2 = .1

    def test_numbers(self):
        d = H.fingerprint_decision(*self.TRAIN, [100, 100], [10, 30])
        assert (d["f0"], d["f1"], d["f2"]) == (0.1, 0.2, 1.0)
        assert (d["pi0"], d["pi1"], d["pi2"]) == (0.8, 0.15, 0.05)
        assert (d["g0"], d["g1"]) == (0.1, 0.3)
        assert d["s1"] == pytest.approx(0.2) and d["half_s1"] == pytest.approx(0.1) and d["D"] == pytest.approx(0.1)
        assert d["record_only"]["d1"] == pytest.approx(0.1) and d["record_only"]["d0"] == pytest.approx(0.0)
        assert d["record_only"]["s0"] == pytest.approx(0.05 * 0.9 / 0.85)
        assert d["sanity"]["ok"] is True

    def test_boundary_is_c1(self):
        assert H.fingerprint_decision(*self.TRAIN, [100, 100], [10, 30])["decision"] == "C1"     # D == s1 / 2 exactly

    def test_just_below_is_z(self):
        assert H.fingerprint_decision(*self.TRAIN, [100, 100], [10, 29])["decision"] == "Z"
        assert H.fingerprint_decision(*self.TRAIN, [100000, 100000], [10000, 29999])["decision"] == "Z"

    def test_above_is_c1_and_far_below_is_z(self):
        assert H.fingerprint_decision(*self.TRAIN, [100, 100], [10, 90])["decision"] == "C1"
        assert H.fingerprint_decision(*self.TRAIN, [100, 100], [10, 20])["decision"] == "Z"
        assert H.fingerprint_decision(*self.TRAIN, [100, 100], [30, 10])["decision"] == "Z"      # D < 0

    @pytest.mark.parametrize("n,nf", [
        ([800, 150, 50], [80, 30, 44]),          # f2 = .88 < .90
        ([800, 150, 50], [401, 30, 50]),         # f0 > .50
        ([800, 150, 50], [80, 76, 50]),          # f1 > .50
        ([800, 150, 0], [80, 30, 0]),            # no HD >= 2 game: f2 undefined
    ])
    def test_sanity_stop(self, n, nf):
        assert H.fingerprint_decision(n, nf, [100, 100], [10, 30])["sanity"]["ok"] is False

    @pytest.mark.parametrize("n,nf", [([800, 150, 50], [80, 30, 45]), ([800, 150, 50], [400, 75, 50])])
    def test_sanity_boundaries_pass(self, n, nf):                # f2 == .90; f0 == f1 == .50
        assert H.fingerprint_decision(n, nf, [100, 100], [10, 30])["sanity"]["ok"] is True

    def test_empty_test_group_gives_no_decision(self):
        d = H.fingerprint_decision(*self.TRAIN, [100, 0], [10, 0])
        assert d["decision"] is None and d["g1"] is None


class TestEncoding:
    def test_flags(self):
        assert [H.flag_of(h, "Z") for h in (0, 1, 2, 9)] == [0, 1, 0, 0]
        assert [H.flag_of(h, "C1") for h in (0, 1, 2, 9)] == [0, 1, 1, 1]
        assert H.flag_count([70, 20, 10], "Z") == 20 and H.flag_count([70, 20, 10], "C1") == 30
        with pytest.raises(ValueError):
            H.flag_of(1, "X")
        with pytest.raises(ValueError):
            H.flag_count([1, 1, 1], "X")


# ================================================================ section 2 item 5: the guard
def train_n(lo_rank=None, hi_rank=None):
    """N[rank][colour][group]: flag share .20 (Z) / .30 (C1) everywhere, except optional other ranks."""
    N = np.zeros((10, 2, 3), dtype=np.int64)
    N[:, :] = [70, 20, 10]
    if lo_rank is not None:
        N[lo_rank, :] = [80, 10, 10]             # Z .10, C1 .20
    if hi_rank is not None:
        N[hi_rank, :] = [50, 40, 10]             # Z .40, C1 .50
    return N


def qn(b1, w1, n=100):
    return [[n - b1, b1, 0], [n - w1, w1, 0]]


class TestGuard:
    @pytest.mark.parametrize("b1,w1,ok", [(20, 20, True), (17, 23, True), (16, 20, False), (20, 24, False),
                                          (23, 17, True), (24, 20, False), (20, 16, False)])
    def test_z_margins(self, b1, w1, ok):                        # p = .20 everywhere: [.17, .23] inclusive
        g, _rec = H.guard(train_n(), qn(b1, w1), "Z")
        assert g["ok"] is ok
        assert g["bounds"]["B"]["ok"] is (17 <= b1 <= 23) and g["bounds"]["W"]["ok"] is (17 <= w1 <= 23)

    def test_range_over_ranks(self):
        N = train_n(lo_rank=0, hi_rank=9)                        # Z: p in [.10, .40] -> [.07, .43]
        assert H.guard(N, qn(7, 43), "Z")[0]["ok"] is True
        assert H.guard(N, qn(6, 43), "Z")[0]["ok"] is False
        assert H.guard(N, qn(7, 44), "Z")[0]["ok"] is False
        g, _rec = H.guard(N, qn(7, 43), "Z")
        assert g["bounds"]["B"] == {"min_p": 0.1, "max_p": 0.4, "lo": float(Fraction(7, 100)),
                                    "hi": float(Fraction(43, 100)), "ok": True}
        assert g["p"]["D"] == {"B": 0.1, "W": 0.1} and g["p"]["6D"] == {"B": 0.4, "W": 0.4}
        assert g["t"] == {"B": 0.07, "W": 0.43}

    def test_encoding_changes_p(self):
        N = train_n()                                            # C1: p = .30 -> [.27, .33]
        assert H.guard(N, qn(20, 20), "C1")[0]["ok"] is False
        assert H.guard(N, qn(27, 33), "C1")[0]["ok"] is True
        assert H.guard(N, qn(27, 33), "Z")[0]["ok"] is False

    def test_colours_are_separate(self):
        N = train_n()
        N[:, 1] = [50, 40, 10]                                   # white: Z .40
        assert H.guard(N, qn(20, 40), "Z")[0]["ok"] is True
        assert H.guard(N, qn(40, 20), "Z")[0]["ok"] is False

    def test_t_counts_every_game_of_the_colour(self):
        t = [[70, 20, 10], [80, 20, 0]]                          # black: HD[1] share 20 / 100, not 20 / 90
        g, _rec = H.guard(train_n(), t, "Z")
        assert g["t"] == {"B": 0.2, "W": 0.2}

    def test_record_only(self):
        _g, rec = H.guard(train_n(lo_rank=0, hi_rank=9), qn(20, 20), "Z")
        assert rec["mean_p"]["B"] == pytest.approx((0.1 + 0.4 + 8 * 0.2) / 10)
        assert rec["adjacent_pairs"][0] == "C-D" and rec["adjacent_pairs"][-1] == "6D-5D"
        assert rec["adjacent_diff"]["W"] == pytest.approx([0.1] + [0.0] * 7 + [0.2])

    def test_empty_cell_fails(self):
        N = train_n()
        N[3, 0] = 0
        assert H.guard(N, qn(20, 20), "Z")[0]["ok"] is False
        assert H.guard(train_n(), [[0, 0, 0], [80, 20, 0]], "Z")[0]["ok"] is False


# ================================================================ the test files and stats end to end
Q_HEADER = ["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] + ["color_%d" % i for i in range(1, 21)]
C_HEADER = ["game_id", "player_id", "sgf_content", "color"]


def question_rows(prefix, colour_games):
    """colour_games: {colour: [(hd, F)]}; questions of 5 games, one colour each."""
    rows, q, i = [], 0, 0
    for col, games in colour_games.items():
        for s in range(0, len(games), 5):
            part = games[s:s + 5]
            q += 1
            sg = []
            for hd, f in part:
                i += 1
                sg.append(with_hd(game(i, star=not f), hd))
            rows.append(["%s_q_%04d" % (prefix, q), str(len(part))] + sg + [""] * (20 - len(part))
                        + [col] * len(part) + [""] * (20 - len(part)))
    return rows


def mix(n0, f0, n1, f1):
    return [(0, True)] * f0 + [(0, False)] * (n0 - f0) + [(1, True)] * f1 + [(1, False)] * (n1 - f1)


def make_test_dir(tmp, per_colour, break_sha=None):
    d = os.path.join(tmp, "test_original")
    files = {}
    for pool, pre in (("public", "pub"), ("private", "priv")):
        write_csv(os.path.join(d, H.RANK_TEST_FILES[pool]), Q_HEADER,
                  question_rows(pre, {"B": per_colour, "W": per_colour}))
        write_csv(os.path.join(d, H.PLAYER_TEST_FILES[pool]), Q_HEADER, question_rows(pre, {"B": mix(4, 1, 1, 0)}))
        write_csv(os.path.join(d, H.PLAYER_CAND_FILES[pool]), C_HEADER,
                  [["%s_g_%07d" % (pre, k + 1), "%s_p_0001" % pre, with_hd(game(k, star=True), int(k == 0)), "B"]
                   for k in range(4)])
    for names in (H.RANK_TEST_FILES, H.PLAYER_TEST_FILES, H.PLAYER_CAND_FILES):
        for name in names.values():
            files[name] = {"sha256": sha(os.path.join(d, name))}
    if break_sha:
        files[break_sha]["sha256"] = "0" * 64
    with open(os.path.join(d, "SOURCE.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "c10-test02 source v1", "files": files}, f)
    return d


class TestQuestionFiles:
    def test_only_the_first_num_games_slots(self, tmp_path):
        p = str(tmp_path / "q.csv")
        r = ["q1", "2", with_hd(PLAIN, 1), with_hd(STAR_B, 0), with_hd(PLAIN, 0)] + [""] * 17 + ["W", "W", "W"] \
            + [""] * 17
        write_csv(p, Q_HEADER, [r])
        assert list(H.question_games(p)) == [("W", with_hd(PLAIN, 1)), ("W", with_hd(STAR_B, 0))]
        t = H.tally_games(H.question_games(p))
        assert t["games"] == 2 and t["unreadable"] == 0 and t["n"] == [[0, 0, 0], [1, 1, 0]]
        assert t["nF"] == [[0, 0, 0], [0, 1, 0]] and t["hd_values"] == {"0": 1, "1": 1}

    def test_unreadable_games_are_counted(self, tmp_path):
        p = str(tmp_path / "q.csv")
        rows = [["q1", "2", with_hd(PLAIN, 1), ""] + [""] * 18 + ["B", "B"] + [""] * 18,           # empty SGF
                ["q2", "1", PLAIN] + [""] * 19 + ["B"] + [""] * 19,                                # no HD
                ["q3", "1", with_hd(PLAIN, 0)] + [""] * 19 + ["X"] + [""] * 19,                    # colour
                ["q4", "0"] + [""] * 40]                                                           # num_games
        write_csv(p, Q_HEADER, rows)
        t = H.tally_games(H.question_games(p))
        assert t["games"] == 5 and t["unreadable"] == 4 and t["n"] == [[0, 1, 0], [0, 0, 0]]

    def test_other_header_refused(self, tmp_path):
        p = str(tmp_path / "q.csv")
        write_csv(p, Q_HEADER[:-1], [])
        with pytest.raises(SystemExit):
            list(H.question_games(p))
        write_csv(p, C_HEADER[::-1], [])
        with pytest.raises(SystemExit):
            list(H.candidate_games(p))

    def test_candidates(self, tmp_path):
        p = str(tmp_path / "c.csv")
        write_csv(p, C_HEADER, [["g1", "p1", with_hd(PLAIN, 1), "B"], ["g2", "p1", with_hd(STAR_W, 0), "B"]])
        t = H.tally_games(H.candidate_games(p))
        assert t["n"] == [[1, 1, 0], [0, 0, 0]] and t["nF"] == [[0, 1, 0], [0, 0, 0]]


class TestStats:
    """Training: per rank and colour 6 HD[0] (1 F), 3 HD[1] (1 F), 1 HD[3] (F): f = 1/6, 1/3, 1; pi = .6 .3 .1;
    s1 = .1 * (2/3) / .4 = 1/6; s1 / 2 = 1/12; p = .30 (Z) / .40 (C1) in every cell."""

    def run(self, tmp_path, reg, per_colour, spec=None, break_sha=None):
        v10, v11, tb = make_training(str(tmp_path), spec=spec)
        reg(tb)
        table = str(tmp_path / "hd_table.npz")
        brec = H.build(v10, v11, tb, table, log=quiet)
        assert brec["status"] == "PASS"
        td = make_test_dir(str(tmp_path), per_colour, break_sha=break_sha)
        return H.stats(table, brec, td, log=quiet), table, brec, td

    def test_z_pass(self, tmp_path, reg):
        out, table, _b, _td = self.run(tmp_path, reg, mix(14, 2, 6, 2))      # g0 = 1/7, g1 = 1/3; D = 1/42
        assert out["stage0a"] == "PASS" and out["stop_reasons"] == []
        assert out["encoding"] == "Z" and out["sanity_ok"] is True and out["guard_ok"] is True
        fp = out["fingerprint"]
        assert fp["train_games"] == [120, 60, 20] and fp["train_F"] == [20, 20, 20]
        assert fp["test_games"] == [56, 24] and fp["test_F"] == [8, 8] and fp["test_hd_ge2_games"] == 0
        assert fp["D"] == pytest.approx(1 / 42) and fp["s1"] == pytest.approx(1 / 6)
        assert out["seam_a"]["ok"] is True and out["seam_a"]["rows_by_file"]["D"] == [20, 20]
        assert out["table"]["sha256"] == sha(table) and out["table"]["unique_keys"] == 200
        tr = out["train"]
        assert tr["by_rank_colour"]["4D"]["W"] == {
            "hd0": 6, "hd1": 3, "hd_ge2": 1, "n": 10, "share_hd0": 0.6, "share_hd1": 0.3, "share_hd_ge2": 0.1,
            "flag_share_Z": 0.3, "flag_share_C1": 0.4}
        assert tr["overall"]["n"] == 200 and tr["hd_values"] == {"0": 120, "1": 60, "3": 20}
        assert tr["fingerprint_by_rank"]["A"] == {"f0": pytest.approx(1 / 6), "f1": pytest.approx(1 / 3)}
        assert out["test_rank"]["public"]["t_B"] == 0.3 and out["test_rank"]["combined"]["t_W"] == 0.3
        assert out["test_rank"]["combined"]["games"] == 80
        assert out["guard"]["encoding"] == "Z" and out["guard"]["t"] == {"B": 0.3, "W": 0.3}
        assert out["record_only"]["mean_p"]["B"] == pytest.approx(0.3)
        assert out["test_player_record_only"]["player_candidates_public"]["t_B"] == 0.25
        assert all(f["ok"] for f in out["test_files"].values()) and len(out["test_files"]) == 6

    def test_c1_pass(self, tmp_path, reg):
        out, *_ = self.run(tmp_path, reg, mix(12, 2, 8, 6))                  # g0 = 1/6, g1 = 3/4; D = 5/12
        assert out["stage0a"] == "PASS" and out["encoding"] == "C1"
        assert out["guard"]["encoding"] == "C1" and out["guard"]["p"]["D"] == {"B": 0.4, "W": 0.4}
        assert out["guard"]["t"] == {"B": 0.4, "W": 0.4}

    def test_guard_stop(self, tmp_path, reg):
        out, *_ = self.run(tmp_path, reg, mix(10, 1, 10, 3))                 # Z; t = .50 > .33
        assert out["encoding"] == "Z" and out["guard_ok"] is False and out["sanity_ok"] is True
        assert out["stage0a"] == "STOP" and out["stop_reasons"] == ["distribution_guard"]

    def test_sanity_stop(self, tmp_path, reg):
        spec = {c: SPEC[:-1] + [(3, False)] for c in H.COLOURS}              # f2 = 0
        out, *_ = self.run(tmp_path, reg, mix(14, 2, 6, 2), spec=spec)
        assert out["sanity_ok"] is False and out["stage0a"] == "STOP"
        assert "fingerprint_sanity" in out["stop_reasons"]

    def test_test_file_sha_stop(self, tmp_path, reg):
        out, *_ = self.run(tmp_path, reg, mix(14, 2, 6, 2), break_sha="rank_prediction_test_private.csv")
        assert out["stage0a"] == "STOP" and out["stop_reasons"] == ["test_sha"]
        assert out["test_files"]["rank_prediction_test_private.csv"]["ok"] is False

    def test_table_changed_after_build_stops(self, tmp_path, reg):
        out, table, brec, td = self.run(tmp_path, reg, mix(14, 2, 6, 2))
        brec = json.loads(json.dumps(brec))
        brec["table"]["sha256"] = "0" * 64
        out = H.stats(table, brec, td, log=quiet)
        assert out["stage0a"] == "STOP" and out["stop_reasons"] == ["table_sha"]

    def test_stopped_build_gives_stop(self, tmp_path):
        v10, v11, tb = make_training(str(tmp_path))                          # tarball not registered
        brec = H.build(v10, v11, tb, str(tmp_path / "t.npz"), log=quiet)
        out = H.stats(str(tmp_path / "t.npz"), brec, str(tmp_path / "nowhere"), log=quiet)
        assert out["stage0a"] == "STOP" and out["stop_reasons"] == ["seam_a"] and out["encoding"] is None
        assert out["build"]["stop_reasons"] == ["tarball_sha"]

    def test_cli_runs_once_and_writes_json(self, tmp_path, reg):
        v10, v11, tb = make_training(str(tmp_path))
        reg(tb)
        table, brec = str(tmp_path / "t.npz"), str(tmp_path / "b.json")
        assert H.main(["build", "--v10", v10, "--v11", v11, "--tarball", tb, "--out", table, "--record", brec]) == 0
        td = make_test_dir(str(tmp_path), mix(10, 1, 10, 3))
        out = str(tmp_path / "s" / "hd_stats.json")
        argv = ["stats", "--table", table, "--build-record", brec, "--test-dir", td, "--out", out]
        assert H.main(argv) == 3                                             # the guard stop
        d = json.load(open(out, encoding="utf-8"))
        assert d["stage0a"] == "STOP" and d["format"] == H.STATS_FORMAT and d["build"]["status"] == "PASS"
        with pytest.raises(SystemExit) as e:
            H.main(argv)
        assert "runs once" in str(e.value)


# ================================================================ isolation
class TestIsolation:
    def test_only_stdlib_and_numpy(self):
        with open(os.path.join(HERE, "p31_hd.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                mods.add(node.module.split(".")[0])
        assert mods <= set(sys.stdlib_module_names) | {"numpy"}
        assert mods - set(sys.stdlib_module_names) == {"numpy"}

    def test_no_data_path_inside(self):
        with open(os.path.join(HERE, "p31_hd.py"), encoding="utf-8") as f:
            code = f.read().split('"""', 2)[2]                               # after the module docstring
        assert "data/" not in code and "data\\" not in code

    def test_registered_constants(self):
        assert H.RANKS == ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
        assert H.TARBALL_SHA256 == "e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e"
        assert H.TARBALL_BYTES == 220710452
        assert (H.SANITY_F2_MIN, H.SANITY_F01_MAX, H.GUARD_MARGIN) == (Fraction(9, 10), Fraction(1, 2),
                                                                       Fraction(3, 100))
