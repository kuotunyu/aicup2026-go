"""Unit tests of p31_lib.py, p31_train.py, p31_eval.py and ram_watch.py (prereg31 explore, stage 0b and the eval).

Synthetic data only: no real game, no real feature, no real model, no inner score of anything real. The only repo
files read are committed records (hd_stats.json, the prereg19 / 25 / 29 reports and train records) and git.

    PY -m pytest -q -p no:cacheprovider results/prereg31/explore/test_p31.py
"""

import json
import math
import os
import sys
import types

import numpy as np
import pytest
from scipy import sparse

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p31_hd as HD  # noqa: E402
import p31_lib as L  # noqa: E402
import p31_train as PT  # noqa: E402
import p31_eval as E  # noqa: E402
import ram_watch as RW  # noqa: E402

NAMES = E.K19.PARAM_NAMES


def table_parts(rows):
    """rows: [(rank index, game_id, player_id, colour index, hd)] -> the parts of p31_hd.save_table."""
    parts = []
    for r in sorted({x[0] for x in rows}):
        sel = [x for x in rows if x[0] == r]
        parts.append((r, {"game_id": np.array([x[1].encode("ascii") for x in sel], dtype="S"),
                          "player_id": np.array([x[2].encode("ascii") for x in sel], dtype="S"),
                          "color": np.array([x[3] for x in sel], dtype=np.uint8),
                          "hd": np.array([x[4] for x in sel], dtype=np.int8),
                          "F": np.zeros(len(sel), dtype=bool)}))
    return parts


def make_table(path, rows):
    HD.save_table(str(path), table_parts(rows))
    return str(path)


def write_json(path, obj):
    with open(str(path), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)
    return str(path)


def small_params(rng, width, hidden=16, rho=8):
    """A synthetic set model with the shapes of p19_lib (first layer `width` wide, h 64 wide, 10 levels)."""
    return {"W1": rng.normal(0.0, 0.05, (hidden, width)), "b1": rng.normal(0.0, 0.1, hidden),
            "W2": rng.normal(0.0, 0.2, (E.K19.H_DIM, hidden)), "b2": rng.normal(0.0, 0.1, E.K19.H_DIM),
            "R1": rng.normal(0.0, 0.2, (rho, E.K19.H_DIM)), "c1": rng.normal(0.0, 0.1, rho),
            "R2": rng.normal(0.0, 0.5, (1, rho)), "c2": rng.normal(0.0, 0.1, 1),
            "t0": np.array([-2.0]), "dt": rng.normal(0.0, 0.3, E.K19.L - 2)}


# ---------------------------------------------------------------- p31_lib: names and constants
class TestNames:
    def test_model_name(self):
        assert L.model_name(3) == "s31_seed3.npz"
        assert L.model_name(0, hd_off_retrain=True) == "s31_seed0_hdoff_retrain.npz"

    def test_subset_seeds(self):
        assert L.subset_seeds(3) == (0, 1, 2)
        assert L.subset_seeds(5) == (0, 1, 2, 3, 4)
        assert L.subset_seeds(10) == L.SEEDS
        with pytest.raises(ValueError):
            L.subset_seeds(4)

    def test_constants_are_the_registration(self):
        assert L.SEEDS == tuple(range(10)) == tuple(L.L29.S25_SEEDS) + tuple(L.L29.NEW_SEEDS)
        assert L.SEAM_R_SEED == 0 and L.SUBSETS == (3, 5, 10)
        assert L.GO_MEAN == -0.010 and L.GO_PAIRS == 8
        assert L.D_IN_HD == E.FW.X_DIM + L.N_HD == 1542
        assert L.N_PROBE == 270000 and L.N_V == 74946
        assert L.ENCODINGS == ("Z", "C1")

    def test_rank_and_colour_order_of_the_repo(self):
        assert HD.RANKS == tuple(E.r.LEVELS)
        assert HD.COLOURS == tuple(E.K19.COLOURS)
        assert len(HD.RANKS) == E.K19.L

    def test_plan_references_are_the_committed_reports(self):
        got = {**E.E23.refs(), **L.L29.refs25(), **L.refs29()}
        assert set(L.REF_PLAN) == set(got)
        for k, v in L.REF_PLAN.items():
            assert got[k] == v, k


class TestGuard:
    def test_forbidden_folder_is_refused(self, tmp_path):
        bad = os.path.join(str(tmp_path), "data", "lo" + "cal", "x.json")
        with pytest.raises(SystemExit):
            L.guard(bad)
        with pytest.raises(SystemExit):
            E.guard_path(bad)

    def test_other_paths_pass(self, tmp_path):
        p = os.path.join(str(tmp_path), "x.json")
        assert L.guard(p) == p


# ---------------------------------------------------------------- p31_lib: input width
class Untouchable:
    def __getattr__(self, name):
        raise AssertionError("read " + name)

    def __setattr__(self, name, value):
        raise AssertionError("set " + name)


class TestInputWidth:
    def test_sets_and_restores(self):
        k = types.SimpleNamespace(D_IN=1540)
        with L.input_width(k, 1542):
            assert k.D_IN == 1542
        assert k.D_IN == 1540

    def test_restores_after_an_exception(self):
        k = types.SimpleNamespace(D_IN=1540)
        with pytest.raises(RuntimeError):
            with L.input_width(k, 1542):
                raise RuntimeError("x")
        assert k.D_IN == 1540

    def test_hd_off_touches_nothing(self):
        with L.input_width(Untouchable(), None):
            pass

    def test_the_real_module(self):
        before = E.K19.D_IN
        assert before == E.FW.X_DIM
        with L.input_width(E.K19, L.D_IN_HD):
            assert E.K19.D_IN == 1542
        assert E.K19.D_IN == before


# ---------------------------------------------------------------- p31_lib: the 2 HD inputs
class TestHdColumns:
    HD_ = np.array([0, 1, 2, 1, 0, 9])
    CI = np.array([0, 0, 1, 1, 1, 0])

    def test_encoding_z(self):
        c = L.hd_columns(self.HD_, self.CI, "Z")
        assert c.dtype == np.float64 and c.shape == (6, 2)
        assert c[:, 0].tolist() == [0, 1, 0, 0, 0, 0]
        assert c[:, 1].tolist() == [0, 0, 0, 1, 0, 0]

    def test_encoding_c1(self):
        c = L.hd_columns(self.HD_, self.CI, "C1")
        assert c[:, 0].tolist() == [0, 1, 0, 0, 0, 1]
        assert c[:, 1].tolist() == [0, 0, 1, 1, 0, 0]

    def test_never_both_columns(self):
        for enc in L.ENCODINGS:
            c = L.hd_columns(self.HD_, self.CI, enc)
            assert (c.sum(axis=1) <= 1).all()

    def test_flag_is_the_one_of_stage_0a(self):
        hd = np.arange(10)
        for enc in L.ENCODINGS:
            assert L.flag_array(hd, enc).astype(int).tolist() == [HD.flag_of(int(h), enc) for h in hd]
        with pytest.raises(ValueError):
            L.flag_array(hd, "X")

    def test_refusals(self):
        with pytest.raises(L.P31Error):
            L.hd_columns(np.array([0, 1]), np.array([0, 1, 0]), "Z")
        with pytest.raises(L.P31Error):
            L.hd_columns(np.array([0, -1]), np.array([0, 1]), "Z")
        with pytest.raises(L.P31Error):
            L.hd_columns(np.array([0, 1]), np.array([0, 2]), "Z")
        with pytest.raises(ValueError):
            L.hd_columns(np.array([0, 1]), np.array([0, 1]), "other")


class TestMoments:
    def cols(self):
        return L.hd_columns(np.array([1, 0, 1, 0, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0, 1, 0, 1]), "Z")

    def test_role_a_rows_only(self):
        cols = self.cols()
        rows_a = np.array([0, 1, 2, 3])
        mu, sd = L.hd_moments(cols, rows_a)
        assert mu.tolist() == [0.25, 0.25]
        assert np.allclose(sd, [math.sqrt(0.1875)] * 2, rtol=0, atol=1e-15)
        other = cols.copy()
        other[4:] = 7.0
        mu2, sd2 = L.hd_moments(other, rows_a)
        assert np.array_equal(mu, mu2) and np.array_equal(sd, sd2)

    def test_sigma_zero_is_refused(self):
        cols = self.cols()
        with pytest.raises(L.P31Error):
            L.hd_moments(cols, np.array([1, 3]))            # no flag at all among these rows
        with pytest.raises(L.P31Error):
            L.hd_moments(cols, np.array([0, 4]))            # x_W constant
        with pytest.raises(L.P31Error):
            L.hd_moments(cols, np.array([], dtype=np.int64))
        with pytest.raises(L.P31Error):
            L.hd_moments(np.zeros((4, 3)), np.array([0, 1]))

    def test_standardise_and_even_game(self):
        cols = self.cols()
        mu, sd = L.hd_moments(cols, np.arange(8))
        z = L.hd_standardise(cols, mu, sd)
        assert np.array_equal(z, (cols - mu) / sd)
        assert np.allclose(z.mean(axis=0), 0.0, atol=1e-15) and np.allclose(z.std(axis=0), 1.0, atol=1e-15)
        ev = L.even_value(mu, sd)
        assert np.array_equal(ev, (0.0 - mu) / sd)
        assert np.array_equal(L.hd_standardise(np.zeros((1, 2)), mu, sd)[0], ev)
        assert np.array_equal(z[1], ev)                     # an even game of the data

    def test_bad_moments_are_refused(self):
        for mu, sd in (([0.1, 0.2], [0.0, 1.0]), ([0.1], [1.0]), ([0.1, float("nan")], [1.0, 1.0])):
            with pytest.raises(L.P31Error):
                L.hd_standardise(np.zeros((2, 2)), mu, sd)
            with pytest.raises(L.P31Error):
                L.even_value(mu, sd)


# ---------------------------------------------------------------- p31_lib: the HD table and the join
ROWS = [(0, "g1", "pa", 0, 0), (0, "g2", "pa", 1, 1), (3, "g1", "pb", 0, 2), (9, "g7", "pc", 1, 1)]


class TestHdTable:
    def table(self, tmp_path):
        return L.HdTable(make_table(tmp_path / "t.npz", ROWS))

    def test_join_all_found(self, tmp_path):
        t = self.table(tmp_path)
        assert t.n == 4 and t.sha256 == L.sha_file(str(tmp_path / "t.npz"))
        hd, rec = t.join(["6D", "D", "A", "D"], ["g7", "g2", "g1", "g1"], ["pc", "pa", "pb", "pa"],
                         ["W", "W", "B", "B"])
        assert hd.dtype == np.int8 and hd.tolist() == [1, 1, 2, 0]
        assert rec == {"rows": 4, "joined": 4, "missing": 0, "player_mismatch": 0, "colour_mismatch": 0,
                       "hd_groups": [1, 2, 1], "ok": True}

    def test_same_game_id_in_two_ranks(self, tmp_path):
        hd, rec = self.table(tmp_path).join(["D", "A"], ["g1", "g1"], ["pa", "pb"], ["B", "B"])
        assert hd.tolist() == [0, 2] and rec["ok"]

    def test_missing_game(self, tmp_path):
        hd, rec = self.table(tmp_path).join(["D", "C", "D"], ["g1", "g1", "zz"], ["pa", "pa", "pa"],
                                            ["B", "B", "B"])
        assert hd.tolist() == [0, -1, -1]
        assert rec["joined"] == 1 and rec["missing"] == 2 and rec["ok"] is False
        assert rec["hd_groups"] == [1, 0, 0]

    def test_player_mismatch(self, tmp_path):
        hd, rec = self.table(tmp_path).join(["D", "A"], ["g1", "g1"], ["pa", "px"], ["B", "B"])
        assert hd.tolist() == [0, 2]
        assert rec["player_mismatch"] == 1 and rec["colour_mismatch"] == 0 and rec["ok"] is False

    def test_colour_mismatch(self, tmp_path):
        hd, rec = self.table(tmp_path).join(["D", "A"], ["g1", "g1"], ["pa", "pb"], ["W", "B"])
        assert rec["colour_mismatch"] == 1 and rec["player_mismatch"] == 0 and rec["ok"] is False

    def test_refusals(self, tmp_path):
        t = self.table(tmp_path)
        with pytest.raises(L.P31Error):
            t.join(["D", "A"], ["g1"], ["pa", "pb"], ["B", "B"])
        with pytest.raises(L.P31Error):
            t.join(["7D"], ["g1"], ["pa"], ["B"])
        with pytest.raises(L.P31Error):
            t.join(["D"], ["g1"], ["pa"], ["X"])

    def test_duplicate_key_is_refused(self, tmp_path):
        p = make_table(tmp_path / "dup.npz", ROWS + [(3, "g1", "pz", 1, 0)])
        with pytest.raises(L.P31Error):
            L.HdTable(p)

    def test_rank_and_colour_index(self):
        assert L.rank_index(list(HD.RANKS)).tolist() == list(range(10))
        assert L.colour_index(["W", "B", "B"]).tolist() == [1, 0, 0]


class TestWalks:
    GA = [{"rank": "D", "game_id": "a0", "player_id": "p1", "color": "B"},
          {"rank": "D", "game_id": "a1", "player_id": "p1", "color": "W"}]
    GB = [{"rank": "3D", "game_id": "b0", "player_id": "p2", "color": "W"}]

    def sets(self, tmp_path):
        return {"a": write_json(tmp_path / "a.json", {"probe": {"games": self.GA}}),
                "b": write_json(tmp_path / "b.json", {"probe": {"games": self.GB}})}

    def test_probe_games(self, tmp_path):
        g = L.probe_games(["a", "b", "a"], [1, 0, 0], ["D:p1", "3D:p2", "D:p1"], self.sets(tmp_path))
        assert g == {"rank": ["D", "3D", "D"], "game_id": ["a1", "b0", "a0"], "player_id": ["p1", "p2", "p1"],
                     "color": ["W", "W", "B"]}

    def test_probe_games_other_player(self, tmp_path):
        with pytest.raises(L.P31Error):
            L.probe_games(["a", "b"], [1, 0], ["D:p1", "3D:p9"], self.sets(tmp_path))

    def v_files(self, tmp_path, colour):
        sets = write_json(tmp_path / "sets.json", {"probe": {"games": self.GA + self.GB}})
        meta = str(tmp_path / "v_meta.npz")
        np.savez(meta, ref=np.array([2, 0]), players=np.array(["3D:p2", "D:p1"]), color=np.array(colour))
        return meta, sets

    def test_v_games(self, tmp_path):
        meta, sets = self.v_files(tmp_path, ["W", "B"])
        g = L.v_games(meta, sets, L.sha_file(sets))
        assert g == {"rank": ["3D", "D"], "game_id": ["b0", "a0"], "player_id": ["p2", "p1"], "color": ["W", "B"]}
        with pytest.raises(L.P31Error):
            L.v_games(meta, sets, "0" * 64)

    def test_v_games_other_colour(self, tmp_path):
        meta, sets = self.v_files(tmp_path, ["W", "W"])
        with pytest.raises(L.P31Error):
            L.v_games(meta, sets, L.sha_file(sets))

    def test_concat_and_rows_agree(self):
        a = {"rank": ["D"], "game_id": ["a0"], "player_id": ["p1"], "color": ["B"]}
        b = {"rank": ["3D", "A"], "game_id": ["b0", "c0"], "player_id": ["p2", "p3"], "color": ["W", "B"]}
        g = L.concat_games(a, b)
        assert g["game_id"] == ["a0", "b0", "c0"] and a["game_id"] == ["a0"]
        keys, levels, colours = np.array(["D:p1", "3D:p2", "A:p3"]), np.array([0, 6, 3]), np.array(["B", "W", "B"])
        assert L.rows_agree(g, keys, levels, colours) == {"rows": 3, "key_mismatch": 0, "level_mismatch": 0,
                                                          "colour_mismatch": 0, "ok": True}
        r = L.rows_agree(g, np.array(["D:p1", "3D:p2", "A:p9"]), levels, colours)
        assert r["key_mismatch"] == 1 and r["ok"] is False
        r = L.rows_agree(g, keys, np.array([0, 5, 3]), colours)
        assert r["level_mismatch"] == 1 and r["ok"] is False
        r = L.rows_agree(g, keys, levels, np.array(["B", "B", "B"]))
        assert r["colour_mismatch"] == 1 and r["ok"] is False
        assert L.rows_agree(g, keys[:2], levels, colours) == {"rows": 3, "length_mismatch": True, "ok": False}


# ---------------------------------------------------------------- p31_lib: the S31 model file
class TestModelFile:
    def test_round_trip(self, tmp_path):
        par = small_params(np.random.default_rng(1), L.D_IN_HD)
        path = os.path.join(str(tmp_path), "m", L.model_name(4))
        sha = L.save_model(path, par, 4, 1200, False, [0.1, 0.2], [0.3, 0.4], "Z", NAMES)
        assert sha == L.sha_file(path) and os.listdir(os.path.dirname(path)) == [L.model_name(4)]
        got, meta = L.load_model(path, NAMES)
        assert set(got) == set(NAMES) and all(np.array_equal(got[k], par[k]) for k in NAMES)
        assert meta["seed"] == 4 and meta["best_step"] == 1200 and meta["smoke"] is False
        assert meta["encoding"] == "Z"
        assert meta["hd_mu"].tolist() == [0.1, 0.2] and meta["hd_sd"].tolist() == [0.3, 0.4]

    def test_a_1540_wide_model_is_not_an_s31_file(self, tmp_path):
        par = small_params(np.random.default_rng(1), E.FW.X_DIM)
        path = os.path.join(str(tmp_path), "m", "x.npz")
        L.save_model(path, par, 0, 1, True, [0.1, 0.2], [0.3, 0.4], "C1", NAMES)
        with pytest.raises(SystemExit):
            L.load_model(path, NAMES)

    def test_another_format_is_refused(self, tmp_path):
        par = small_params(np.random.default_rng(1), L.D_IN_HD)
        path = os.path.join(str(tmp_path), "x.npz")
        np.savez(path, format=np.array(L.L29.S25_FORMAT), seed=np.array(0), best_step=np.array(1),
                 smoke=np.array(False), **{"p_" + k: par[k] for k in NAMES})
        with pytest.raises(SystemExit):
            L.load_model(path, NAMES)

    def test_save_refusals(self, tmp_path):
        par = small_params(np.random.default_rng(1), L.D_IN_HD)
        path = os.path.join(str(tmp_path), "m", "x.npz")
        with pytest.raises(ValueError):
            L.save_model(path, par, 0, 1, False, [0.1, 0.2], [0.3, 0.4], "Y", NAMES)
        with pytest.raises(L.P31Error):
            L.save_model(path, par, 0, 1, False, [0.1, 0.2], [0.3, 0.0], "Z", NAMES)
        assert not os.path.exists(path)


# ---------------------------------------------------------------- p31_lib: stage 0b
def nll(value):
    return {s: value for s in L.SEEDS}


class TestStage0b:
    def test_go(self):
        d = L.stage0b_decision(nll(1.98), nll(2.0))
        assert d["decision"] == "GO" and d["condition_mean"] and d["condition_pairs"]
        assert d["pairs_s31_lower"] == 10 and d["seeds"] == list(L.SEEDS)
        assert d["threshold_mean_diff"] == -0.010 and d["threshold_pairs"] == 8
        assert abs(d["mean_diff_s31_minus_control"] + 0.02) < 1e-12
        assert d["mean_s31"] == 1.98 and d["mean_control"] == 2.0
        assert len(d["pair_diff_s31_minus_control"]) == 10

    def test_mean_just_inside_and_just_outside(self):
        assert L.stage0b_decision(nll(2.0 - 0.010 - 1e-9), nll(2.0))["decision"] == "GO"
        d = L.stage0b_decision(nll(2.0 - 0.010 + 1e-9), nll(2.0))
        assert d["decision"] == "NO-GO" and d["condition_mean"] is False and d["condition_pairs"] is True

    def test_the_threshold_is_inclusive_and_has_no_tolerance(self, monkeypatch):
        monkeypatch.setattr(L, "GO_MEAN", -0.125)           # a number with an exact difference in binary
        assert L.stage0b_decision(nll(0.875), nll(1.0))["decision"] == "GO"
        d = L.stage0b_decision(nll(0.875 + 2.0 ** -40), nll(1.0))
        assert d["decision"] == "NO-GO" and d["mean_diff_s31_minus_control"] == -0.125 + 2.0 ** -40

    def test_pairs(self):
        c = nll(2.0)
        s = {i: (1.0 if i < 7 else 2.5) for i in L.SEEDS}
        d = L.stage0b_decision(s, c)
        assert d["condition_mean"] is True and d["pairs_s31_lower"] == 7 and d["decision"] == "NO-GO"
        s[7] = 1.0
        d = L.stage0b_decision(s, c)
        assert d["pairs_s31_lower"] == 8 and d["decision"] == "GO"

    def test_an_equal_pair_is_not_lower(self):
        c = nll(2.0)
        s = {i: (1.0 if i < 7 else 2.0) for i in L.SEEDS}
        d = L.stage0b_decision(s, c)
        assert d["pairs_s31_lower"] == 7 and d["condition_mean"] is True and d["decision"] == "NO-GO"

    def test_ten_small_wins_are_not_enough(self):
        d = L.stage0b_decision(nll(1.995), nll(2.0))
        assert d["pairs_s31_lower"] == 10 and d["condition_mean"] is False and d["decision"] == "NO-GO"

    def test_refusals(self):
        nine = {s: 1.9 for s in range(9)}
        with pytest.raises(L.P31Error):
            L.stage0b_decision(nine, nll(2.0))
        with pytest.raises(L.P31Error):
            L.stage0b_decision(nll(1.9), nine)
        for bad in (float("nan"), float("inf")):
            s = nll(1.9)
            s[3] = bad
            with pytest.raises(L.P31Error):
                L.stage0b_decision(s, nll(2.0))
            with pytest.raises(L.P31Error):
                L.stage0b_decision(nll(1.9), s)

    def test_control_is_the_committed_records(self):
        c, src = L.control_nll()
        assert tuple(sorted(c)) == L.SEEDS and all(math.isfinite(v) for v in c.values())
        assert src["prereg25_seeds"] == [0, 1, 2] and src["prereg29_seeds"] == [3, 4, 5, 6, 7, 8, 9]
        assert src["holdout_questions_prereg25"] == src["holdout_questions_prereg29"]
        t25, t29 = L.read_json(L.L29.P25_TRAIN), L.read_json(L.P29_TRAIN)
        assert c[2] == t25["seeds"]["2"]["best_holdout_nll"] and c[3] == t29["seeds"]["3"]["best_holdout_nll"]


def good_train_record():
    return {"format": L.TRAIN_FORMAT, "smoke": False, "code": {"head": "h", "dirty": ""}, "stage0a": {"ok": True},
            "seam_a": {"ok": True}, "seam_r": {"ok": True, "applicable": True},
            "seeds": {str(s): {"port_check": {"ok": True}, "best_holdout_nll": 2.0, "best_step": 100 + s}
                      for s in L.SEEDS},
            "players": {"same_as_prereg25": True}}


class TestTrainRecordChecks:
    KEYS = {"format", "not_smoke", "code_clean", "stage0a_ok", "seam_a_ok", "seam_r_ok", "seeds_0_to_9",
            "port_checks_ok", "same_players_as_prereg25"}

    def test_good(self):
        c = L.train_record_checks(good_train_record())
        assert set(c) == self.KEYS and all(v is True for v in c.values())
        assert PT.train_record_checks is L.train_record_checks

    def test_empty(self):
        c = L.train_record_checks({})
        assert set(c) == self.KEYS and not any(c.values())

    @pytest.mark.parametrize("name,change", [
        ("format", lambda t: t.update(format="c10-prereg29 explore train v1")),
        ("not_smoke", lambda t: t.update(smoke=True)),
        ("not_smoke", lambda t: t.pop("smoke")),
        ("code_clean", lambda t: t["code"].update(dirty=" M results/prereg31/explore/p31_lib.py")),
        ("code_clean", lambda t: t["code"].pop("dirty")),
        ("stage0a_ok", lambda t: t["stage0a"].update(ok=False)),
        ("seam_a_ok", lambda t: t["seam_a"].update(ok=None)),
        ("seam_r_ok", lambda t: t["seam_r"].update(ok=False)),
        ("seam_r_ok", lambda t: t["seam_r"].update(applicable=False)),
        ("seeds_0_to_9", lambda t: t["seeds"].pop("9")),
        ("seeds_0_to_9", lambda t: t["seeds"].update({"10": {"port_check": {"ok": True}}})),
        ("port_checks_ok", lambda t: t["seeds"]["4"]["port_check"].update(ok=False)),
        ("port_checks_ok", lambda t: t["seeds"]["4"].pop("port_check")),
        ("same_players_as_prereg25", lambda t: t["players"].update(same_as_prereg25=None)),
    ])
    def test_one_thing_wrong(self, name, change):
        t = good_train_record()
        change(t)
        c = L.train_record_checks(t)
        assert c[name] is False
        assert all(v is True for k, v in c.items() if k != name)


# ---------------------------------------------------------------- p31_lib: HD-only naive Bayes (record-only)
class TestNaiveBayes:
    def test_table(self):
        flag = np.array([1, 0, 1, 1, 0, 1])
        level = np.array([0, 0, 1, 1, 1, 2])
        col = np.array([0, 0, 1, 1, 1, 0])
        P, n, n1 = L.nb_table(flag, level, col, np.arange(5), n_levels=3)      # row 5 is not role A
        assert n.tolist() == [[2, 0], [0, 3], [0, 0]] and n1.tolist() == [[1, 0], [0, 2], [0, 0]]
        assert P.tolist() == [[0.5, 0.5], [0.5, 0.6], [0.5, 0.5]]
        assert L.nb_table(flag, level, col, np.arange(5))[0].shape == (10, 2)

    def test_no_signal_is_the_uniform_nll(self):
        P = np.full((10, 2), 0.5)
        hold = {5: (np.arange(20).reshape(4, 5), np.array([0, 3, 9, 5]))}
        rec = L.nb_nll(P, hold, np.arange(20) % 3 == 0, np.arange(20) % 2)
        assert rec["questions"] == 4 and rec["uniform_nll"] == math.log(10)
        assert abs(rec["mean_nll_minus_uniform"]) < 1e-12 and abs(rec["mean_nll"] - math.log(10)) < 1e-12

    def test_hand_computed(self):
        P = np.array([[0.9, 0.2], [0.1, 0.2]])              # the flag says something for Black only
        flag = np.array([1, 1, 1, 0, 1, 0])
        col = np.array([0, 0, 0, 0, 1, 1])
        hold = {2: (np.array([[0, 1], [2, 3], [4, 5]]), np.array([0, 1, 1])),
                3: (np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64))}
        rec = L.nb_nll(P, hold, flag, col)
        q1 = -math.log(0.81 / 0.82)                         # two Black flags, true level 0
        q2 = math.log(2.0)                                  # one flag and one not: a tie
        q3 = math.log(2.0)                                  # White games: no information
        assert rec["questions"] == 3 and set(rec["per_n"]) == {"2"}
        assert abs(rec["mean_nll"] - (q1 + q2 + q3) / 3) < 1e-12
        assert rec["uniform_nll"] == math.log(2.0)
        assert abs(rec["mean_nll_minus_uniform"] - (rec["mean_nll"] - math.log(2.0))) < 1e-15
        assert rec["per_n"]["2"]["questions"] == 3

    def test_no_question(self):
        P = np.full((10, 2), 0.5)
        with pytest.raises(L.P31Error):
            L.nb_nll(P, {}, np.zeros(4), np.zeros(4, dtype=np.int64))
        with pytest.raises(L.P31Error):
            L.nb_nll(P, {5: (np.zeros((0, 5), dtype=np.int64), np.zeros(0, dtype=np.int64))}, np.zeros(4),
                     np.zeros(4, dtype=np.int64))


# ---------------------------------------------------------------- p31_train / p31_eval: the two files agree
class TestConsistency:
    def test_shared_names(self):
        assert PT.STOP == L.STOP and PT.T.STOP == L.STOP
        assert PT.FORMAT == L.TRAIN_FORMAT and PT.RECORD == L.TRAIN_RECORD
        assert E.S0B_FORMAT == PT.S0B_FORMAT
        assert E.TOL == 1e-12 and E.N_TRAIN_ROWS == 1000000

    def test_forward_files_are_in_both_code_states(self):
        ce, ct = E.code_state(True), PT.code_state()
        assert set(E.FORWARD_FILES) <= set(ce["sha256_lf"])
        assert set(E.FORWARD_FILES) <= set(ct["sha256_lf"])
        for f in E.FORWARD_FILES:
            assert ce["sha256_lf"][f] == ct["sha256_lf"][f]
        assert "results/prereg31/explore/test_p31.py" in ce["sha256_lf"]
        assert "results/prereg31/explore/PLAN.md" in ce["sha256_lf"] and "results/prereg31/explore/PLAN.md" in \
            ct["sha256_lf"]
        assert not any(k.endswith(("train_record.json", "stage0b.json")) and "prereg31" in k
                       for k in ce["sha256_lf"])

    def test_same_players(self):
        a = {"A_players": 6619, "A_games": 300000, "H_players": 748, "H_games": 44946, "holdout_questions": 3740,
             "same_as_prereg25": None}
        b = {k: v for k, v in a.items() if k != "same_as_prereg25"}
        assert PT.same_players(a, b) is True
        for k in b:
            assert PT.same_players(a, {**b, k: b[k] + 1}) is False

    def test_hd_record(self):
        rec = PT.hd_record(np.array([0.25, 0.125]), np.array([0.5, 0.75]), "Z")
        assert rec == {"encoding": "Z", "hd_mu": [0.25, 0.125], "hd_sd": [0.5, 0.75]}
        assert all(type(x) is float for x in rec["hd_mu"] + rec["hd_sd"])
        json.dumps(rec)

    def test_launchers(self):
        for name in ("run_train.sh", "run_eval.sh"):
            with open(os.path.join(HERE, name), "rb") as f:
                raw = f.read()
            assert raw.count(b"\r") == 0, name
            assert b"unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS" in raw
            assert b"export PYTHONDONTWRITEBYTECODE=1" in raw
        with open(os.path.join(HERE, "run_train.sh"), encoding="utf-8") as f:
            sh = f.read()
        assert 'echo "end rc=$? $(date -u +%FT%TZ) run $RUN" >> "$LOG"' in sh
        assert '--run-id "$RUN"' in sh and "--stop data/prereg31/STOP" in sh
        assert 'p31_train.py train "$@"' in sh


# ---------------------------------------------------------------- p31_eval: gates
def cmp_rec(inner_diff, lo, mono_lo=0.001, b=0.01, w=0.01):
    mono_hi = None if mono_lo is None else mono_lo + 0.02
    return {"inner_diff": inner_diff, "inner_ci95": [lo, lo + 0.02], "mono_diff": 0.01,
            "mono_ci95": [mono_lo, mono_hi], "mono_by_colour": {"B": b, "W": w}}


class TestGates:
    def test_all_four(self):
        c29, c25 = cmp_rec(0.006, 0.001), cmp_rec(0.004, 0.0005)
        g = E.gates(c29, c25)
        assert g["passed"] is True and g["G1_G2_G4_against"] == "Q29"
        assert g["G5"] == {"against": "Q25", "delta": 0.004, "ci95": [0.0005, 0.0205], "passed": True}
        k = E.K23.gates(c29["inner_diff"], c29["inner_ci95"], c29["mono_diff"], c29["mono_ci95"],
                        c29["mono_by_colour"])
        assert (g["G1"], g["G2"], g["G4"]) == (k["G1"], k["G2"], k["G4"])
        assert g["G1"]["threshold"] == E.K23.GATE == 0.005

    def test_g5_is_strict(self):
        for lo in (0.0, -0.0001):
            g = E.gates(cmp_rec(0.006, 0.001), cmp_rec(0.004, lo))
            assert g["G5"]["passed"] is False and g["passed"] is False
            assert g["G1"]["passed"] and g["G2"]["passed"] and g["G4"]["passed"]

    def test_g5_has_no_threshold_on_the_delta(self):
        assert E.gates(cmp_rec(0.006, 0.001), cmp_rec(0.0001, 0.00001))["passed"] is True

    def test_g1(self):
        assert E.gates(cmp_rec(E.K23.GATE, 0.001), cmp_rec(0.004, 0.0005))["passed"] is True
        g = E.gates(cmp_rec(0.0049, 0.001), cmp_rec(0.004, 0.0005))
        assert g["G1"]["passed"] is False and g["G5"]["passed"] is True and g["passed"] is False

    def test_g2(self):
        g = E.gates(cmp_rec(0.006, 0.0), cmp_rec(0.004, 0.0005))
        assert g["G2"]["passed"] is False and g["passed"] is False

    def test_g4(self):
        ok25 = cmp_rec(0.004, 0.0005)
        assert E.gates(cmp_rec(0.006, 0.001, mono_lo=0.0), ok25)["G4"]["passed"] is False
        assert E.gates(cmp_rec(0.006, 0.001, mono_lo=None), ok25)["G4"]["passed"] is False
        assert E.gates(cmp_rec(0.006, 0.001, b=-0.0001), ok25)["passed"] is False
        assert E.gates(cmp_rec(0.006, 0.001, w=None), ok25)["passed"] is False
        assert E.gates(cmp_rec(0.006, 0.001, w=-1e-13), ok25)["passed"] is True
        # the single-colour numbers of Q31 - Q25 are not part of G5
        assert E.gates(cmp_rec(0.006, 0.001), cmp_rec(0.004, 0.0005, mono_lo=-1.0, b=-1.0))["passed"] is True


# ---------------------------------------------------------------- p31_eval: seam records
SHA = "ab" * 32


def seam_a_args():
    st = {"build": {"tarball": {"sha256": HD.TARBALL_SHA256, "ok": True}},
          "seam_a": {"ok": True, "checks": {"same_rows": True, "plain_form": True},
                     "totals": {"rows_v10": 1000000, "rows_v11": 1000000, "rows_equal": 1000000,
                                "rows_not_equal": 0, "duplicate_game_ids": 0}},
          "table": {"sha256": SHA}}
    tr = {"seam_a": {"ok": True, "probe_rows": 270000, "v_rows": 74946, "table_sha256": SHA,
                     "join": {"ok": True, "rows": 344946, "missing": 0, "player_mismatch": 0,
                              "colour_mismatch": 0}},
          "hd": {"table_sha256": SHA}}
    jrec = {"ok": True, "rows": 1600, "joined": 1600, "missing": 0, "player_mismatch": 0, "colour_mismatch": 0}
    return {"st": st, "s0a": {"ok": True, "table_sha256_hd_stats": SHA}, "table_sha": SHA, "table_rows": 1000000,
            "tr": tr, "jrec": jrec, "n_inner": 1600}


class TestSeamA:
    FLAGS = ("tarball_is_plan", "rows_all_equal", "stage0a_ok", "table_is_hd_stats", "table_is_train_record",
             "train_rows_joined", "inner_rows_joined")

    def test_good(self):
        rec = E.seam_a_record(**seam_a_args())
        assert rec["ok"] is True and all(rec[k] is True for k in self.FLAGS)
        json.dumps(rec)

    @pytest.mark.parametrize("flags,change", [
        (("tarball_is_plan",), lambda a: a["st"]["build"]["tarball"].update(sha256="0" * 64)),
        (("tarball_is_plan",), lambda a: a["st"]["build"]["tarball"].update(ok=False)),
        (("tarball_is_plan",), lambda a: a["st"].pop("build")),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"].update(ok=False)),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"]["checks"].update(plain_form=False)),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"].update(checks={})),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"]["totals"].update(rows_equal=999999)),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"]["totals"].update(rows_v11=1000001)),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"]["totals"].update(rows_not_equal=1)),
        (("rows_all_equal",), lambda a: a["st"]["seam_a"]["totals"].update(duplicate_game_ids=1)),
        (("rows_all_equal",), lambda a: a.update(table_rows=999999)),
        (("stage0a_ok",), lambda a: a["s0a"].update(ok=False)),
        (("table_is_hd_stats", "table_is_train_record"), lambda a: a.update(table_sha="cd" * 32)),
        (("table_is_hd_stats", "table_is_train_record"), lambda a: a.update(table_sha=None)),
        (("table_is_hd_stats",), lambda a: a["st"]["table"].update(sha256="cd" * 32)),
        (("table_is_hd_stats",), lambda a: a["s0a"].update(table_sha256_hd_stats="cd" * 32)),
        (("table_is_train_record",), lambda a: a["tr"]["seam_a"].update(table_sha256="cd" * 32)),
        (("table_is_train_record",), lambda a: a["tr"]["hd"].update(table_sha256="cd" * 32)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"].update(ok=False)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"].update(probe_rows=269999)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"].update(v_rows=74947)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"]["join"].update(rows=344945)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"]["join"].update(missing=1)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"]["join"].update(player_mismatch=1)),
        (("train_rows_joined",), lambda a: a["tr"]["seam_a"]["join"].update(colour_mismatch=1)),
        (("inner_rows_joined",), lambda a: a["jrec"].update(ok=False)),
        (("inner_rows_joined",), lambda a: a["jrec"].update(joined=1599)),
        (("inner_rows_joined",), lambda a: a["jrec"].update(missing=1)),
        (("inner_rows_joined",), lambda a: a["jrec"].update(colour_mismatch=2)),
        (("inner_rows_joined",), lambda a: a.update(n_inner=1601)),
    ])
    def test_one_thing_wrong(self, flags, change):
        a = seam_a_args()
        change(a)
        rec = E.seam_a_record(**a)
        assert rec["ok"] is False
        assert tuple(k for k in self.FLAGS if rec[k] is False) == flags

    def test_the_committed_stage_0a_record(self):
        st = L.read_json(L.HD_STATS)
        sha, rows = st["table"]["sha256"], st["table"]["rows"]
        a = seam_a_args()
        a.update(st=st, s0a={"ok": True, "table_sha256_hd_stats": sha}, table_sha=sha, table_rows=rows)
        a["tr"]["seam_a"]["table_sha256"] = a["tr"]["hd"]["table_sha256"] = sha
        rec = E.seam_a_record(**a)
        assert rec["tarball_is_plan"] is True and rec["rows_all_equal"] is True and rec["ok"] is True
        assert rec["rows_v10"] == rec["rows_v11"] == rec["rows_equal"] == E.N_TRAIN_ROWS


class TestSeam1:
    SAME = {"fit": True, "eval": True, "mono": True}

    def test_good(self):
        seam = {k: (v, v) for k, v in L.REF_PLAN.items()}
        rec = E.seam1_record(seam, dict(L.REF_PLAN), dict(self.SAME))
        assert rec == {"within_tol": True, "bitwise": True, "refs_are_plan": True,
                       "hd_off_same_as_s19_term": self.SAME, "ok": True}

    def test_close_is_not_enough(self):
        seam = {k: (v, v) for k, v in L.REF_PLAN.items()}
        seam["Q25"] = (L.REF_PLAN["Q25"] + 1e-13, L.REF_PLAN["Q25"])
        rec = E.seam1_record(seam, dict(L.REF_PLAN), dict(self.SAME))
        assert rec["within_tol"] is True and rec["bitwise"] is False and rec["ok"] is False
        seam["Q25"] = (L.REF_PLAN["Q25"] + 1e-6, L.REF_PLAN["Q25"])
        rec = E.seam1_record(seam, dict(L.REF_PLAN), dict(self.SAME))
        assert rec["within_tol"] is False and rec["ok"] is False

    def test_references_must_be_the_plan(self):
        seam = {k: (v, v) for k, v in L.REF_PLAN.items()}
        ref = dict(L.REF_PLAN)
        ref["Q29"] = ref["Q29"] + 1e-15
        assert ref["Q29"] != L.REF_PLAN["Q29"]
        rec = E.seam1_record(seam, ref, dict(self.SAME))
        assert rec["refs_are_plan"] is False and rec["ok"] is False
        ref = dict(L.REF_PLAN)
        ref.pop("Q09_mono")
        assert E.seam1_record(seam, ref, dict(self.SAME))["ok"] is False

    def test_hd_off_arrays_must_be_the_same(self):
        seam = {k: (v, v) for k, v in L.REF_PLAN.items()}
        assert E.seam1_record(seam, dict(L.REF_PLAN), {**self.SAME, "mono": False})["ok"] is False
        assert E.seam1_record(seam, dict(L.REF_PLAN), {})["ok"] is False


class TestStage0bRecord:
    def files(self, tmp_path, delta=-0.02, **over):
        control, _src = L.control_nll()
        tr = good_train_record()
        for s in L.SEEDS:
            tr["seeds"][str(s)]["best_holdout_nll"] = control[s] + delta
        tpath = write_json(tmp_path / "train_record.json", tr)
        s0b = {"format": E.S0B_FORMAT, "decision": "GO", "mean_diff_s31_minus_control": delta,
               "pairs_s31_lower": 10, "train_record_sha256_lf": E.E23.sha_lf(tpath)}
        s0b.update(over)
        return tpath, write_json(tmp_path / "stage0b.json", s0b)

    def test_go(self, tmp_path):
        rec = E.stage0b_record(*self.files(tmp_path))
        assert rec["exists"] is True and rec["go"] is True and rec["decision_recomputed"] == "GO"
        assert rec["format_ok"] and rec["is_this_train_record"]
        assert abs(rec["mean_diff_recomputed"] + 0.02) < 1e-12

    def test_no_file(self, tmp_path):
        tpath, _spath = self.files(tmp_path)
        assert E.stage0b_record(tpath, str(tmp_path / "none.json")) == {"exists": False, "go": False}

    def test_another_train_record(self, tmp_path):
        rec = E.stage0b_record(*self.files(tmp_path, train_record_sha256_lf="0" * 64))
        assert rec["is_this_train_record"] is False and rec["go"] is False

    def test_no_go_in_the_file(self, tmp_path):
        rec = E.stage0b_record(*self.files(tmp_path, decision="NO-GO"))
        assert rec["decision_recomputed"] == "GO" and rec["go"] is False

    def test_a_go_the_rule_does_not_give(self, tmp_path):
        rec = E.stage0b_record(*self.files(tmp_path, delta=-0.005))
        assert rec["decision"] == "GO" and rec["decision_recomputed"] == "NO-GO" and rec["go"] is False

    def test_another_format(self, tmp_path):
        rec = E.stage0b_record(*self.files(tmp_path, format="other"))
        assert rec["format_ok"] is False and rec["go"] is False


class TestOnceGuard:
    def paths(self, tmp_path, monkeypatch):
        started, out = str(tmp_path / "eval_started.json"), str(tmp_path / "eval_report.json")
        monkeypatch.setattr(E, "STARTED", started)
        return started, out

    def args(self, rerun=False, reason=None):
        return types.SimpleNamespace(rerun=rerun, reason=reason)

    def test_first_run(self, tmp_path, monkeypatch):
        _started, out = self.paths(tmp_path, monkeypatch)
        assert E.once_guard(self.args(), {"head": "aaa"}, out) is None

    def test_second_run_is_refused(self, tmp_path, monkeypatch):
        started, out = self.paths(tmp_path, monkeypatch)
        write_json(started, {"code": {"head": "aaa"}})
        with pytest.raises(SystemExit):
            E.once_guard(self.args(), {"head": "bbb"}, out)
        with pytest.raises(SystemExit):
            E.once_guard(self.args(rerun=True), {"head": "bbb"}, out)
        with pytest.raises(SystemExit):
            E.once_guard(self.args(reason="x"), {"head": "bbb"}, out)

    def test_rerun_needs_a_newer_commit(self, tmp_path, monkeypatch):
        started, out = self.paths(tmp_path, monkeypatch)
        write_json(started, {"code": {"head": "aaa"}})
        with pytest.raises(SystemExit):
            E.once_guard(self.args(rerun=True, reason="bug in x"), {"head": "aaa"}, out)
        assert E.once_guard(self.args(rerun=True, reason="bug in x"), {"head": "bbb"}, out) == ["aaa"]

    def test_a_report_alone_also_counts(self, tmp_path, monkeypatch):
        started, out = self.paths(tmp_path, monkeypatch)
        write_json(out, {"code": {"head": "ccc"}})
        with pytest.raises(SystemExit):
            E.once_guard(self.args(), {"head": "ddd"}, out)
        write_json(started, {"code": {"head": "aaa"}})
        assert E.once_guard(self.args(rerun=True, reason="bug in x"), {"head": "ddd"}, out) == ["aaa", "ccc"]
        with pytest.raises(SystemExit):
            E.once_guard(self.args(rerun=True, reason="bug in x"), {"head": "ccc"}, out)


class TestTrainSummary:
    def test_seed_order(self):
        tr = {"seeds": {"10": {"best_step": 3, "best_holdout_nll": 1.5, "curve": [1]},
                        "9": {"best_step": 2, "best_holdout_nll": 1.6}, "2": {"best_step": 1}}}
        s = E.train_summary(tr)
        assert list(s) == ["2", "9", "10"]
        assert s["10"] == {"best_step": 3, "best_holdout_nll": 1.5}
        assert s["2"] == {"best_step": 1, "best_holdout_nll": None}
        assert E.train_summary({}) == {}


# ---------------------------------------------------------------- ram_watch
GB = 2 ** 30


class TestRamWatch:
    def test_limit(self):
        assert RW.LIMIT_GB == 4.0
        assert RW.must_stop(4 * GB) is False and RW.must_stop(4 * GB - 1) is True
        assert RW.must_stop(10 * GB, limit_gb=12) is True

    def test_run_ended(self, tmp_path):
        log = str(tmp_path / "train_log.txt")
        assert RW.run_ended(log, "A1") is False            # no log yet
        with open(log, "w", encoding="utf-8") as f:
            f.write("start 2026-11-10T00:00:00Z head abc run A1 args \n")
            f.write("seed 0 step 100 the end of run A1\n")
        assert RW.run_ended(log, "A1") is False
        with open(log, "a", encoding="utf-8") as f:
            f.write("end rc=0 2026-11-10T09:00:00Z run XA1\n")
        assert RW.run_ended(log, "A1") is False
        with open(log, "a", encoding="utf-8") as f:
            f.write("end rc=0 2026-11-10T09:00:00Z run A1\n")
        assert RW.run_ended(log, "A1") is True and RW.run_ended(log, "A2") is False

    def test_watch_writes_stop_once_and_ends_with_the_run(self, tmp_path):
        log, stop, out = (str(tmp_path / n) for n in ("log.txt", os.path.join("d", "STOP"), "watch.jsonl"))
        avail = iter([8 * GB, 3 * GB, 2 * GB, 8 * GB])
        naps = []

        def nap(seconds):
            naps.append(seconds)
            if len(naps) == 3:
                with open(log, "w", encoding="utf-8") as f:
                    f.write("end rc=3 2026-11-10T09:00:00Z run R1\n")

        res = RW.watch(log, stop, out, "R1", every=7, avail=lambda: next(avail), sleep=nap)
        assert res == {"samples": 4, "min_avail_gb": 2.0, "stop_written": True}
        assert naps == [7, 7, 7]
        with open(stop, encoding="utf-8") as f:
            text = f.read()
        assert "3.00 GB < 4.0 GB" in text and "run R1" in text      # written at the first low sample only
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f]
        assert [x["stop_written"] for x in rows] == [False, True, True, True]
        assert [x["avail_gb"] for x in rows] == [8.0, 3.0, 2.0, 8.0] and all(x["run"] == "R1" for x in rows)

    def test_watch_without_pressure(self, tmp_path):
        log, stop, out = (str(tmp_path / n) for n in ("log.txt", "STOP", "watch.jsonl"))
        with open(log, "w", encoding="utf-8") as f:
            f.write("end rc=0 2026-11-10T09:00:00Z run R2\n")
        res = RW.watch(log, stop, out, "R2", avail=lambda: 9 * GB, sleep=lambda s: None)
        assert res == {"samples": 1, "min_avail_gb": 9.0, "stop_written": False}
        assert not os.path.exists(stop)

    def test_watch_gives_up_after_max_hours(self, tmp_path):
        log, stop, out = (str(tmp_path / n) for n in ("log.txt", "STOP", "watch.jsonl"))
        res = RW.watch(log, stop, out, "R3", max_hours=-1, avail=lambda: 9 * GB, sleep=lambda s: None)
        assert res["samples"] == 1


# ---------------------------------------------------------------- a synthetic inner world (no real data)
RF, P3 = E.E23.rf, E.E23.p3
N_PLAYERS, PER = 40, 40                # 4 players per level; 40 games each: 20 as Black, then 20 as White
N_INNER = N_PLAYERS * PER


def pool(rows_list, n_cols):
    """Averaging matrix [Q, n_cols] (csr): row q has 1 / n at the games of question q."""
    indptr = np.cumsum([0] + [len(x) for x in rows_list])
    data = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in rows_list])
    return sparse.csr_matrix((data, np.concatenate(rows_list), indptr), shape=(len(rows_list), n_cols))


class World:
    def __init__(self, seed=0, all_even=False):
        rng = np.random.default_rng(seed)
        level = np.repeat(np.arange(10), N_PLAYERS // 10)                  # player -> level
        row_player = np.repeat(np.arange(N_PLAYERS), PER)
        colour = np.tile(np.array(["B"] * (PER // 2) + ["W"] * (PER // 2)), N_PLAYERS)
        keys = {"rank": [HD.RANKS[level[p]] for p in row_player], "game_id": ["g%05d" % i for i in range(N_INNER)],
                "player_id": ["p%03d" % p for p in row_player], "color": colour.tolist(), "n": N_INNER}
        self.covered = np.ones(N_INNER, dtype=bool)
        self.covered[rng.choice(N_INNER, 30, replace=False)] = False       # games without P1 features
        ref = rng.permutation(np.flatnonzero(self.covered)).astype(np.int64)
        self.prep = types.SimpleNamespace(
            inner_ref=ref, inner_X=rng.normal(0.0, 1.0, (len(ref), E.FW.X_DIM)).astype(np.float32))
        self.mp = {"mu_x": rng.normal(0.0, 0.1, E.FW.X_DIM), "sd_x": rng.uniform(0.5, 1.5, E.FW.X_DIM)}
        hd = rng.choice(np.array([0, 1, 2, 5]), size=N_INNER, p=[0.70, 0.25, 0.03, 0.02])
        hd[:4 * PER] = 0                                                   # players 0-3: even games only
        self.hd = np.zeros(N_INNER, dtype=np.int8) if all_even else hd.astype(np.int8)
        fit, ev, mono = [], [], []
        ey, en, ep, my, mk, mpl, mc = [], [], [], [], [], [], []
        for p in range(N_PLAYERS):
            base = p * PER
            for _ in range(2):
                fit.append(base + rng.choice(PER, 10, replace=False))
            for n in E.K19.KS:
                ev.append(base + rng.choice(PER, n, replace=False))
                ey.append(level[p]), en.append(n), ep.append(p)
            for ci, c in enumerate(HD.COLOURS):
                for k in E.K19.KS:
                    mono.append(base + ci * (PER // 2) + rng.choice(PER // 2, k, replace=False))
                    my.append(level[p]), mk.append(k), mpl.append(p), mc.append(c)
        i64 = lambda x: np.array(x, dtype=np.int64)  # noqa: E731
        self.Xe = types.SimpleNamespace(y=i64(ey), n=i64(en), player=i64(ep))
        self.Xm = types.SimpleNamespace(y=i64(my), n=i64(mk), player=i64(mpl))
        self.ms = {"y": i64(my), "player": i64(mpl), "k": i64(mk), "colour": np.array(mc), "n_players": N_PLAYERS}
        self.d = {"inner": {"S_fit": pool(fit, N_INNER), "S_eval": pool(ev, N_INNER), "S_mono": pool(mono, N_INNER),
                            "keys": keys,
                            "players": [{"rank": HD.RANKS[level[p]], "player_id": "p%03d" % p}
                                        for p in range(N_PLAYERS)]}}
        self.base = {}
        for which, y in (("fit", np.repeat(level, 2)), ("eval", self.Xe.y), ("mono", self.Xm.y)):
            z = rng.normal(0.0, 1.0, (len(y), 10))
            z[np.arange(len(y)), y] += 1.5
            self.base[which] = z
        self.E_q09 = {w: rng.normal(0.0, 0.3, z.shape) for w, z in self.base.items()}
        self.pars25 = [small_params(rng, E.FW.X_DIM) for _ in range(3)]
        self.pars29 = self.pars25 + [small_params(rng, E.FW.X_DIM) for _ in range(7)]
        self.pars31 = [small_params(rng, L.D_IN_HD) for _ in range(10)]
        for par in self.pars31:
            par["W1"][:, E.FW.X_DIM:] = rng.normal(0.0, 1.0, (par["W1"].shape[0], L.N_HD))
        self.hdmeta = {"encoding": "Z", "hd_mu": np.array([0.12, 0.11]), "hd_sd": np.array([0.33, 0.31])}


class FakeInner(E.r.Inner):
    """The frozen Inner without its data and without the fusion fit: run() adds the terms with weight 1 to fixed
    base logits. Scoring, bootstrap, compare and short are the frozen code."""

    def __init__(self, w):                                                 # the frozen __init__ reads inner data
        self.d, self.n_inner, self.covered = w.d, N_INNER, w.covered
        self.Xe, self.Xm, self.ms, self.base = w.Xe, w.Xm, w.ms, w.base
        self.boot = RF.PlayerBootstrap(w.Xe.player, w.Xe.y, N_PLAYERS)
        self.mboot = RF.PlayerBootstrap(w.ms["player"], w.ms["y"], N_PLAYERS)
        self.E_hs = {k: np.zeros_like(v) for k, v in w.base.items()}

    def run(self, terms, Emap):
        z = {k: self.base[k] + sum(Emap[t][k] for t in terms if t in Emap) for k in ("eval", "mono")}
        P, Pm = RF.softmax(z["eval"]), RF.softmax(z["mono"])
        m, sc = RF.metrics(RF.rule_b_np(P), self.Xe.y, self.Xe.n, self.Xe.player)
        mrec, msc = P3.mono_score_record(RF.rule_b_np(Pm), self.Xm, self.ms, self.mboot)
        folds = [{"fold": 0, "params": {"w": {t: 1.0 for t in terms}}}]
        return {"P": P, "Pm": Pm, "folds": folds, "terms": list(terms), "m": m, "sc": sc, "mrec": mrec, "msc": msc,
                "aux": {}, "rb": m["rank_balanced"], "mono_rb": mrec["rank_balanced"]}


@pytest.fixture(scope="module")
def world():
    w = World()
    return w, FakeInner(w)


def same_term(a, b):
    return all(np.array_equal(x[w], y[w]) for x, y in zip(a, b) for w in E.WHICH)


class TestSetTerm:
    def test_hd_off_is_s19_term(self, world):
        w, inner = world
        a = E.set_term(w.prep, inner, w.mp, w.pars29)
        b = E.E23.s19_term(w.prep, inner, w.mp, w.pars29)
        assert same_term(a, b)
        assert a[0]["eval"].shape == (len(w.Xe.y), 10) and a[2]["eval"].all() and np.abs(a[0]["eval"]).max() > 0

    def test_hd_on_by_hand(self, world):
        w, inner = world
        hd_std = E.hd_inputs(inner, w.hd, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)
        got = E.set_term(w.prep, inner, w.mp, w.pars31, hd_std)
        X = E.FW.standardise_x(w.prep.inner_X.astype(np.float64), w.mp["mu_x"], w.mp["sd_x"])
        X = np.concatenate([X, hd_std[w.prep.inner_ref]], axis=1)
        H = []
        for par in w.pars31:
            h = np.zeros((N_INNER, E.K19.H_DIM))
            h[w.prep.inner_ref] = E.K19.h_np(par, X)
            H.append(h)
        for which in E.WHICH:
            groups = E.K19.usable_groups(w.d["inner"]["S_" + which], w.covered)
            e, _per, lp, has = E.K19.set_term(w.pars31, H, groups, N_INNER)
            assert np.allclose(got[0][which], e, rtol=1e-9, atol=1e-12)
            assert np.allclose(got[1][which], lp, rtol=1e-9, atol=1e-12)
            assert np.array_equal(got[2][which], has)

    def test_hd_columns_without_weight_change_nothing(self, world):
        w, inner = world
        hd_std = E.hd_inputs(inner, w.hd, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)
        wide, narrow = [], []
        for par in w.pars31[:3]:
            a = dict(par)
            a["W1"] = par["W1"].copy()
            a["W1"][:, E.FW.X_DIM:] = 0.0
            wide.append(a)
            narrow.append({**par, "W1": par["W1"][:, :E.FW.X_DIM].copy()})
        a = E.set_term(w.prep, inner, w.mp, wide, hd_std)
        b = E.set_term(w.prep, inner, w.mp, narrow)
        for which in E.WHICH:
            assert np.allclose(a[0][which], b[0][which], rtol=1e-9, atol=1e-12)
            assert np.array_equal(a[2][which], b[2][which])
        c = E.set_term(w.prep, inner, w.mp, w.pars31[:3], hd_std)
        assert not np.allclose(c[0]["eval"], b[0]["eval"], rtol=1e-9, atol=1e-12)

    def test_hd_inputs_are_taken_by_inner_row(self, world):
        w, inner = world
        S = w.d["inner"]["S_eval"]
        r0 = int(next(c for c in S.indices[S.indptr[5]:S.indptr[6]] if w.covered[c]))
        zero = np.zeros((N_INNER, L.N_HD))
        one = zero.copy()
        one[r0] = [3.0, -2.0]
        a = E.set_term(w.prep, inner, w.mp, w.pars31[:2], zero)
        b = E.set_term(w.prep, inner, w.mp, w.pars31[:2], one)
        for which in ("eval", "mono"):
            holds = np.asarray((w.d["inner"]["S_" + which][:, r0] != 0).todense()).ravel()
            changed = (a[0][which] != b[0][which]).any(axis=1)
            assert holds.any() and np.array_equal(changed, holds)

    def test_smoke_noise_does_not_read_the_features(self, world):
        w, inner = world
        other = types.SimpleNamespace(inner_ref=w.prep.inner_ref, inner_X=np.zeros_like(w.prep.inner_X))
        a = E.set_term(w.prep, inner, w.mp, w.pars29[:2], noise=True)
        b = E.set_term(other, inner, w.mp, w.pars29[:2], noise=True)
        assert same_term(a, b)
        assert not np.array_equal(a[0]["eval"], E.set_term(w.prep, inner, w.mp, w.pars29[:2])[0]["eval"])


class TestHdInputs:
    def test_registered_columns(self, world):
        w, inner = world
        mu, sd = w.hdmeta["hd_mu"], w.hdmeta["hd_sd"]
        ci = np.array([0 if c == "B" else 1 for c in w.d["inner"]["keys"]["color"]])
        for enc in L.ENCODINGS:
            cols = L.hd_columns(w.hd, ci, enc)
            got = E.hd_inputs(inner, w.hd, mu, sd, enc, False)
            assert got.shape == (N_INNER, 2) and np.array_equal(got, (cols - mu) / sd)
            perm = np.random.default_rng(E.E23.NOISE_SEED).permutation(N_INNER)
            assert np.array_equal(E.hd_inputs(inner, w.hd, mu, sd, enc, True), ((cols - mu) / sd)[perm])
        z = E.hd_inputs(inner, w.hd, mu, sd, "Z", False)
        black_hd1 = (w.hd == 1) & (ci == 0)
        assert black_hd1.any() and np.array_equal(z[black_hd1][:, 0], np.full(black_hd1.sum(), (1 - mu[0]) / sd[0]))
        assert np.array_equal(z[w.hd != 1], np.tile(L.even_value(mu, sd), (int((w.hd != 1).sum()), 1)))

    def test_one_row_per_inner_row(self, world):
        w, inner = world
        with pytest.raises(L.P31Error):
            E.hd_inputs(inner, w.hd[:-1], w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)
        missing = w.hd.copy()
        missing[7] = -1
        with pytest.raises(L.P31Error):
            E.hd_inputs(inner, missing, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)


class TestQuestionSubsets:
    def test_counts(self):
        S = pool([np.array([0, 1, 2]), np.array([3, 4]), np.array([5])], 6)
        n1, n2 = E.question_hd_counts(S, np.array([1, 0, 2, 1, 1, 5]))
        assert n1.tolist() == [1, 2, 0] and n2.tolist() == [1, 0, 1]
        assert n1.dtype == np.int64 and n2.dtype == np.int64

    def test_empty_subset(self):
        y = np.array([0, 1, 2])
        rec = E.subset_record(np.ones(3), np.zeros(3), y, np.array([0, 1, 2]), 3, np.zeros(3, dtype=bool))
        assert rec == {"n_questions": 0, "note": "no question in this subset"}

    def test_subset_is_p23_subset_diff(self):
        y, player = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 2, 3, 4, 5])
        a, b = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        mask = np.array([True, True, True, True, False, False])
        rec = E.subset_record(a, b, y, player, 6, mask)
        want = E.E23.subset_diff(a, b, y, player, 6, mask)
        assert {k: rec[k] for k in want} == want and rec["levels_present"] == 2
        assert rec["n_questions"] == 4 and rec["diff"] == 0.5

    def test_world_subsets(self, world):
        w, inner = world
        q29 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars29)[0])
        q25 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars25)[0])
        rec = E.hd_subsets(inner, q25, q29, w.hd)
        for name, total in (("eval", len(w.Xe.y)), ("mono", len(w.ms["y"]))):
            r_ = rec[name]
            assert set(r_) == {"with_hd1_game", "without_hd1_game", "without_hd_ge2_game", "with_hd_ge2_game"}
            assert r_["with_hd1_game"]["n_questions"] + r_["without_hd1_game"]["n_questions"] == total
            assert r_["with_hd_ge2_game"]["n_questions"] + r_["without_hd_ge2_game"]["n_questions"] == total
            assert r_["without_hd1_game"]["n_questions"] >= 4 * (total // N_PLAYERS)    # players 0-3
        json.dumps(rec)
        wrong = types.SimpleNamespace(d={"inner": {**w.d["inner"], "S_eval": w.d["inner"]["S_mono"]}}, ms=w.ms,
                                      Xe=w.Xe)
        with pytest.raises(L.P31Error):
            E.hd_subsets(wrong, q25, q29, w.hd)


# ---------------------------------------------------------------- the judged part, end to end on the synthetic world
def run_judge(w, inner, path, smoke=False):
    q25 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars25)[0])
    q29 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars29)[0])
    tr = {"seeds": {str(s): {"best_step": 100 * s, "best_holdout_nll": 2.0 - 0.01 * s} for s in L.SEEDS}}
    base = {"format": E.FORMAT, "smoke": smoke, "code": {"head": "test"}}
    out = E.judge(w.prep, inner, w.mp, w.E_q09, q25, q29, w.pars31, w.hd, w.hdmeta, smoke, path, base, tr)
    return out, q25, q29


@pytest.fixture(scope="module")
def judged(world, tmp_path_factory):
    w, inner = world
    path = str(tmp_path_factory.mktemp("judge") / "eval_judged.json")
    out, q25, q29 = run_judge(w, inner, path)
    return out, q25, q29, path


class TestJudge:
    def test_shape_of_the_result(self, judged):
        out = judged[0]
        assert set(out) == {"q31", "gates", "cmp29", "cmp25", "record_only"}
        assert set(out["record_only"]) == {
            "own_check", "seed_count_curve", "s31_alone_rule_b", "by_n_games", "by_level", "by_colour",
            "by_hd_in_question", "hd_inputs_set_to_even_game", "weights_by_fold", "training"}

    def test_gates_are_the_two_comparisons(self, world, judged):
        _w, inner = world
        out, q25, q29, _path = judged
        assert out["cmp29"] == inner.compare(out["q31"], q29) and out["cmp25"] == inner.compare(out["q31"], q25)
        assert out["gates"] == E.gates(out["cmp29"], out["cmp25"])
        assert out["cmp29"]["inner_diff"] == out["q31"]["rb"] - q29["rb"]
        assert out["gates"]["G5"]["delta"] == out["q31"]["rb"] - q25["rb"]

    def test_the_candidate_is_the_fusion_of_the_hd_term(self, world, judged):
        w, inner = world
        hd_std = E.hd_inputs(inner, w.hd, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)
        q31 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars31, hd_std)[0])
        assert q31["rb"] == judged[0]["q31"]["rb"] and q31["mono_rb"] == judged[0]["q31"]["mono_rb"]
        assert np.array_equal(q31["sc"], judged[0]["q31"]["sc"])

    def test_judged_file_is_written_first(self, judged):
        out, q25, q29, path = judged
        j = L.read_json(path)
        assert j["format"] == E.FORMAT and j["smoke"] is False and j["code"] == {"head": "test"}
        assert j["gates"] == json.loads(json.dumps(out["gates"]))
        assert j["Q31"] == [out["q31"]["rb"], out["q31"]["mono_rb"]]
        assert j["Q29"] == [q29["rb"], q29["mono_rb"]] and j["Q25"] == [q25["rb"], q25["mono_rb"]]

    def test_record_only_part(self, world, judged):
        w, inner = world
        out, q25, q29, _path = judged
        ro = out["record_only"]
        assert ro["own_check"]["ok"] is True
        curve = ro["seed_count_curve"]
        assert list(curve) == ["3", "5", "10"]
        assert [c["seeds"] for c in curve.values()] == [[0, 1, 2], [0, 1, 2, 3, 4], list(range(10))]
        assert curve["10"]["inner_rank_balanced"] == out["q31"]["rb"]
        assert curve["10"]["vs_q29"] == out["cmp29"] and curve["10"]["vs_q25"] == out["cmp25"]
        hd_std = E.hd_inputs(inner, w.hd, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", False)
        q3 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars31[:3], hd_std)[0])
        assert curve["3"]["inner_rank_balanced"] == q3["rb"] and curve["3"]["vs_q25"] == inner.compare(q3, q25)
        assert set(ro["by_n_games"]) == {"mono_by_k", "eval_by_n_exact", "eval_by_n_bin"}
        assert [x["level"] for x in ro["by_level"]["eval"]] == list(HD.RANKS)
        assert set(ro["by_colour"]) == {"q31_single_colour", "q31_minus_q29", "q31_minus_q25"}
        assert set(ro["by_colour"]["q31_single_colour"]) == {"B", "W"}
        assert set(ro["by_hd_in_question"]) == {"eval", "mono"}
        assert set(ro["weights_by_fold"]) == {"Q31", "Q29", "Q25"}
        assert ro["weights_by_fold"]["Q31"] == [{"fold": 0, "w_P1TO": 1.0, "P1TO_at_bound": False, "w_SET": 1.0,
                                                 "SET_at_bound": False}]
        assert list(ro["training"]) == [str(s) for s in L.SEEDS]
        ev = ro["hd_inputs_set_to_even_game"]
        assert ev["even_game_value"] == [float(x) for x in L.even_value(w.hdmeta["hd_mu"], w.hdmeta["hd_sd"])]
        even = np.tile(L.even_value(w.hdmeta["hd_mu"], w.hdmeta["hd_sd"]), (N_INNER, 1))
        qe = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars31, even)[0])
        assert ev["inner_rank_balanced"] == qe["rb"] and ev["q31_minus_even"] == inner.compare(out["q31"], qe)
        assert ev["even_vs_q29"] == inner.compare(qe, q29)

    def test_everything_reported_can_be_written(self, world, judged, tmp_path):
        _w, inner = world
        out, _q25, q29, _path = judged
        rep = {"gates": out["gates"], "cmp29": out["cmp29"], "cmp25": out["cmp25"],
               "candidate": inner.short(out["q31"], q29), "record_only": out["record_only"]}
        p = str(tmp_path / "eval_report.json")
        E.E23.write_json(p, rep)
        back = L.read_json(p)
        assert back["gates"]["passed"] == out["gates"]["passed"]
        assert back["candidate"]["inner_rank_balanced"] == out["q31"]["rb"]

    def test_without_any_hd_game_the_even_game_run_is_the_candidate(self, tmp_path):
        w = World(seed=1, all_even=True)
        inner = FakeInner(w)
        out, _q25, _q29 = run_judge(w, inner, str(tmp_path / "j.json"))
        ev = out["record_only"]["hd_inputs_set_to_even_game"]
        assert ev["inner_rank_balanced"] == out["q31"]["rb"]
        assert ev["q31_minus_even"]["inner_diff"] == 0.0 and ev["q31_minus_even"]["inner_ci95"] == [0.0, 0.0]
        sub = out["record_only"]["by_hd_in_question"]["eval"]
        assert sub["with_hd1_game"] == {"n_questions": 0, "note": "no question in this subset"}
        assert sub["without_hd1_game"]["n_questions"] == len(w.Xe.y)

    def test_smoke_mode_runs(self, world, tmp_path):
        w, inner = world
        path = str(tmp_path / "j.json")
        out, _q25, _q29 = run_judge(w, inner, path, smoke=True)
        assert L.read_json(path)["smoke"] is True and out["record_only"]["own_check"]["ok"] is True
        perm = E.hd_inputs(inner, w.hd, w.hdmeta["hd_mu"], w.hdmeta["hd_sd"], "Z", True)
        q31 = E.fuse(inner, w.E_q09, E.set_term(w.prep, inner, w.mp, w.pars31, perm, noise=True)[0])
        assert q31["rb"] == out["q31"]["rb"]
