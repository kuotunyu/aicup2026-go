"""Unit tests of p34_ogs.py (prereg34 explore, stage 1a). Synthetic JSON lines only: no real game, no data folder.

    PY -m pytest -q -p no:cacheprovider results/prereg34/explore/test_p34_ogs.py
"""

import copy
import csv
import gzip
import io
import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p34_ogs as P  # noqa: E402

T_OK = 1600000000            # 2020-09-13


def moves(n, start=0):
    out = []
    for i in range(n):
        k = start + i
        out.append([k % 19, (k // 19) % 19, 1000])
    return out


def game(**kw):
    g = {"game_id": 123, "width": 19, "height": 19, "handicap": 0, "ranked": True, "phase": "finished",
         "outcome": "Resignation", "initial_state": {"black": "", "white": ""}, "initial_player": "black",
         "free_handicap_placement": False, "time_control": {"speed": "live"}, "start_time": T_OK,
         "players": {"black": {"id": 11, "username": "alice", "rank": 20.5, "professional": False},
                     "white": {"id": 22, "username": "carol", "rank": 21.25, "professional": False}},
         "moves": moves(120)}
    g.update(kw)
    return g


def reason(g):
    try:
        P.check_game(g)
    except P.Reject as e:
        return e.reason
    return None


def line(g):
    return json.dumps(g, separators=(",", ":")).encode()


# ---------------------------------------------------------------- filters

def test_base_game_accepted():
    xy, speed = P.check_game(game())
    assert len(xy) == 120 and speed == "live"


@pytest.mark.parametrize("kw,why", [
    ({"width": 13, "height": 13}, "size"), ({"height": 13}, "size"), ({"width": "19"}, "size"),
    ({"handicap": 2}, "handicap"), ({"handicap": False}, "handicap"), ({"handicap": None}, "handicap"),
    ({"ranked": False}, "ranked"), ({"ranked": 0}, "ranked"), ({"ranked": "true"}, "ranked"),
    ({"ranked": None}, "ranked"),
    ({"phase": "play"}, "phase"),
    ({"outcome": "Cancellation"}, "outcome_cancellation"),
    ({"initial_state": {"black": "dd", "white": ""}}, "initial_state"),
    ({"initial_state": {"black": "", "white": "pp"}}, "initial_state"), ({"initial_state": None}, "initial_state"),
    ({"initial_player": "white"}, "initial_player"), ({"initial_player": None}, "initial_player"),
    ({"free_handicap_placement": True}, "free_handicap_placement"),
    ({"free_handicap_placement": None}, "free_handicap_placement"),
    ({"time_control": {"speed": "correspondence"}}, "speed_correspondence"),
    ({"start_time": 1546271999}, "start_time"),         # 2018-12-31 23:59:59 Taipei
    ({"start_time": 1748707200}, "start_time"),         # 2025-06-01 00:00:00 Taipei
    ({"start_time": None}, "start_time"), ({"start_time": 1.6e9}, "start_time"),
    ({"game_id": None}, "game_id"), ({"game_id": "123"}, "game_id"),
])
def test_filter_rejects(kw, why):
    assert reason(game(**kw)) == why


def test_ranked_int_1_accepted():
    assert reason(game(ranked=1)) is None


def test_missing_free_handicap_rejected():
    g = game()
    del g["free_handicap_placement"]
    assert reason(g) == "free_handicap_placement"


def test_speed_missing_or_other_accepted():
    assert reason(game(time_control=None)) is None
    assert reason(game(time_control={"speed": "blitz"})) is None


def test_date_window_edges_inclusive():
    assert P.T_MIN == 1546272000 and P.T_MAX == 1748707199
    assert reason(game(start_time=P.T_MIN)) is None
    assert reason(game(start_time=P.T_MAX)) is None


def _players(side, **kw):
    g = game()
    g["players"][side].update(kw)
    return g


@pytest.mark.parametrize("side", ["black", "white"])
def test_player_filters(side):
    assert reason(_players(side, professional=True)) == "professional"
    assert reason(_players(side, professional=None)) is None
    assert reason(_players(side, rank=None)) == "rank_not_numeric"
    assert reason(_players(side, rank="20")) == "rank_not_numeric"
    assert reason(_players(side, rank=True)) == "rank_not_numeric"
    assert reason(_players(side, rank=float("nan"))) == "rank_not_numeric"
    assert reason(_players(side, rank=20)) is None
    assert reason(_players(side, username="GnuGoBot")) == "bot_username"
    assert reason(_players(side, username="roBOTnik")) == "bot_username"
    assert reason(_players(side, username="bo_t")) is None
    assert reason(_players(side, id=None)) == "player_ids"


def test_same_player_both_sides_rejected():
    g = game()
    g["players"]["white"]["id"] = 11
    assert reason(g) == "player_ids"


# ---------------------------------------------------------------- moves

def test_trailing_passes_removed():
    g = game(moves=moves(100) + [[-1, -1, 5], [-1, -1, 5]])
    xy, _ = P.check_game(g)
    assert len(xy) == 100 and None not in xy


def test_trailing_passes_do_not_count_towards_100():
    assert reason(game(moves=moves(99) + [[-1, -1, 5], [-1, -1, 5]])) == "too_few_moves"
    assert reason(game(moves=moves(99))) == "too_few_moves"
    assert reason(game(moves=moves(100))) is None


def test_middle_pass_rejected():
    m = moves(60) + [[-1, -1, 5]] + moves(60, 60)
    assert reason(game(moves=m)) == "middle_pass"
    m2 = moves(60) + [[-1, -1, 5]] + moves(60, 60) + [[-1, -1, 5]]
    assert reason(game(moves=m2)) == "middle_pass"


@pytest.mark.parametrize("bad", [[19, 3, 1], [3, 19, 1], [-1, 3, 1], [3, -2, 1], [3.0, 3, 1], [3], "dd", None])
def test_bad_coordinates_rejected(bad):
    assert reason(game(moves=moves(110) + [bad] + moves(10, 200))) == "bad_coord"


def test_extra_move_fields_ok():
    m = [mv + [{"blur": 1}] for mv in moves(110)]
    assert reason(game(moves=m)) is None


# ---------------------------------------------------------------- SGF and rows

def test_sgf_format_and_coordinates():
    assert P.to_sgf([(0, 0), (18, 0), (0, 18), (3, 15)]) == "(;SZ[19];B[aa];W[sa];B[as];W[dp])"
    assert P.to_sgf([(16, 3)]) == "(;SZ[19];B[qd])"     # x = column 16 -> q, y = row 3 -> d


def test_two_rows_per_game():
    g = game(moves=moves(100) + [[-1, -1, 0]])
    xy, sp = P.check_game(g)
    rows = P.game_rows(g, xy, sp)
    assert [r["color"] for r in rows] == ["B", "W"]
    assert [r["player_id"] for r in rows] == ["ogs_11", "ogs_22"]
    assert {r["game_id"] for r in rows} == {"ogs_123"}
    assert [r["ogs_rank"] for r in rows] == [20.5, 21.25]
    assert rows[0]["sgf"] == rows[1]["sgf"] and rows[0]["sgf"].startswith("(;SZ[19];B[aa];W[ba];")
    assert rows[0]["sgf"].count(";") == 101 and rows[0]["n_moves"] == 100


def test_prefilter():
    assert P.prefilter(line(game()))
    assert not P.prefilter(line(game(width=9, height=9)))
    assert not P.prefilter(line(game(handicap=3)))
    assert P.prefilter(json.dumps(game()).encode())            # spaced variant
    assert not P.prefilter(line(game(width=190)))


# ---------------------------------------------------------------- parse end to end

def _write_src(path, games, extra_lines=()):
    with gzip.open(path, "wb") as f:
        for g in games:
            f.write(line(g) + b"\n")
        for x in extra_lines:
            f.write(x + b"\n")


def test_parse_end_to_end(tmp_path):
    src = tmp_path / "src.json.gz"
    games = [game(game_id=1), game(game_id=2, phase="play"), game(game_id=3, width=9, height=9),
             game(game_id=4, moves=moves(150) + [[-1, -1, 0]])]
    _write_src(src, games, [b'{"width":19,"handicap":0, broken'])
    out = tmp_path / "out"
    assert P.main(["parse", "--src", str(src), "--out-dir", str(out)]) == 0
    rec = json.load(open(out / "parse_record.json"))
    assert rec["lines"] == 5 and rec["accepted_games"] == 2 and rec["rows_written"] == 4
    assert rec["rejected_first_reason"]["phase"] == 1 and rec["rejected_first_reason"]["prefilter"] == 1
    assert rec["json_errors"] == 1 and rec["players_seen"] == 2
    assert rec["accepted_notes"]["trailing_pass_games"] == 1
    with gzip.open(out / "rows.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert tuple(rows[0]) == P.COLUMNS and len(rows) == 5
    assert rows[1][:4] == ["ogs_11", "ogs_1", "20.5", "B"] and rows[2][:4] == ["ogs_22", "ogs_1", "21.25", "W"]
    assert rows[3][6] == "150"
    idx = np.fromfile(out / "rows_index.bin", dtype=P.IDX_DTYPE)
    assert idx["row"].tolist() == [0, 1, 2, 3] and idx["gid"].tolist() == [1, 1, 4, 4]
    assert P.main(["parse", "--src", str(src), "--out-dir", str(out)]) == 1        # no overwrite


# ---------------------------------------------------------------- window rule

def test_window_short_player():
    assert P.window([20.0] * 19) == (0, False, None)


def test_window_exactly_20():
    k, ok, r = P.window([20.0] * 20)
    assert (k, ok, r) == (20, True, 20.0)


def test_window_cap_40():
    k, ok, r = P.window([25.0] * 30 + [20.0] * 45)
    assert (k, ok, r) == (40, True, 20.0)


def test_window_tolerance_inclusive():
    # newest 21 rows alternate 20 and 22: median of the run stays within [20, 22], max distance <= 1 only when
    # the median is 21. A run [20, 22] has median 21 -> ok; [22] alone ok; [20,22,20] median 20 -> 22 is 2 away.
    assert P.window([20.0, 21.0] * 15)[1] is True               # every distance <= 1.0
    k, ok, _ = P.window([18.0] * 10 + [21.0] * 20)               # 18 is 3 away -> stops at 20 rows
    assert (k, ok) == (20, True)
    k, ok, _ = P.window([19.0] * 10 + [20.0] * 20)               # 19 exactly 1.0 from the median 20 -> extends
    assert k == 30 and ok


def test_window_stops_at_first_break():
    # newest 15 rows at 20, then an outlier, then 20 more at 20: greedy stops at 15 -> rejected
    ranks = [20.0] * 20 + [30.0] + [20.0] * 15
    assert P.window(ranks)[:2] == (15, False)


def test_window_median_of_kept_rows():
    ranks = [20.0] * 10 + [20.2] * 10 + [20.9] * 11
    k, ok, r = P.window(ranks)
    assert ok and k == 31 and r == 20.2


def test_window_drift_rejects_old_rows():
    ranks = [10.0 + 0.2 * i for i in range(60)]        # steady improvement (0.2 per game)
    k, ok, r = P.window(ranks)
    assert ok and k == 21                               # 21 rows span 4.0 -> median +-2.0 (PLAN revision 1)
    assert r == pytest.approx(ranks[-11])


# ---------------------------------------------------------------- bins and sampling

def test_bin_edges():
    assert P.bin_of(15.0) == 15 and P.bin_of(15.999) == 15 and P.bin_of(16.0) == 16
    assert P.bin_of(29.99) == 29 and P.bin_of(30.0) == 30 and P.bin_of(36.5) == 36
    assert P.bin_of(3.0) == 15 and P.bin_of(40.0) == 36             # clipped
    assert P.BIN_HI - P.BIN_LO + 1 == 22


def _idx(players):
    """players: {pid: [ranks old -> new]} -> structured index (time = position, gid unique)."""
    rows = []
    gid = 1
    for pid, ranks in players.items():
        for i, r in enumerate(ranks):
            rows.append((pid, gid, r, 1600000000 + i * 60, len(rows), 0, 0, 120))
            gid += 1
    return np.array(rows, dtype=P.IDX_DTYPE)


def test_select_players_and_kept_rows():
    idx = _idx({5: [20.0] * 30, 7: [20.0] * 10, 9: [25.0] * 50})
    acc, n = P.select_players(idx)
    assert n == 3 and [p["pid"] for p in acc] == [5, 9]
    p9 = acc[1]
    assert len(p9["kept"]) == 40 and idx["t"][p9["kept"]].tolist() == sorted(idx["t"][p9["kept"]].tolist())
    assert idx["t"][p9["kept"][-1]] == idx["t"][idx["pid"] == 9].max()


def test_select_players_sorts_by_time_not_file_order():
    idx = _idx({5: [20.0] * 25})
    idx["t"] = idx["t"][::-1].copy()                  # file order newest first
    idx["rank"][0] = 30.0                              # the newest row (first in the file) is an outlier
    acc, _ = P.select_players(idx)
    assert acc == []                                   # run starts at the outlier: only 1 row


def test_draw_deterministic_and_capped():
    players = {pid: [15.0 + (pid % 22) + 0.5] * 30 for pid in range(1, 22 * 160 + 1)}
    acc, _ = P.select_players(_idx(players))
    c1, bins = P.draw(acc)
    c2, _ = P.draw(list(reversed(acc)))                # input order does not matter
    assert [(c["bin"], c["pid"]) for c in c1] == [(c["bin"], c["pid"]) for c in c2]
    assert all(v == 160 for v in bins.values())
    per = {}
    for c in c1:
        per[c["bin"]] = per.get(c["bin"], 0) + 1
        assert len(c["rows"]) == 25
    assert all(v == 150 for v in per.values()) and len(per) == 22
    assert [c["bin"] for c in c1] == sorted(c["bin"] for c in c1)


def test_draw_uses_one_generator_in_bin_order():
    players = {pid: [15.5 if pid <= 3 else 16.5] * 25 for pid in range(1, 7)}
    acc, _ = P.select_players(_idx(players))
    chosen, _ = P.draw(acc)
    rng = np.random.default_rng(34)
    p1 = rng.permutation(3)
    p2 = rng.permutation(3)
    assert [c["pid"] for c in chosen] == [[1, 2, 3][i] for i in p1] + [[4, 5, 6][i] for i in p2]


def test_sample_end_to_end(tmp_path):
    src = tmp_path / "src.json.gz"
    games = []
    gid = 1
    for pair in range(30):                                # 60 players, 25 games each
        for k in range(25):
            g = game(game_id=gid, start_time=T_OK + gid * 60)
            g["players"] = {"black": {"id": 1000 + 2 * pair, "username": "a", "rank": 18.0 + (pair % 15),
                                      "professional": False},
                            "white": {"id": 1001 + 2 * pair, "username": "b", "rank": 18.2 + (pair % 15),
                                      "professional": False}}
            games.append(g)
            gid += 1
    games.append(copy.deepcopy(games[0]))                # duplicate game in the dump
    _write_src(src, games)
    out = tmp_path / "parsed"
    assert P.main(["parse", "--src", str(src), "--out-dir", str(out)]) == 0
    s = tmp_path / "s"
    assert P.main(["sample", "--rows", str(out), "--out-dir", str(s)]) == 3     # < 1500 players: registered stop
    rec = json.load(open(s / "sample_record.json"))
    assert rec["duplicate_rows_dropped"] == 2 and rec["players_accepted_window"] == 60
    assert rec["rows_sampled"] == 60 * 25 and rec["stop"] is True and len(rec["stop_reasons"]) == 2
    assert rec["black_share"] == 0.5
    with gzip.open(s / "sample_games.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 60 * 25 and rows[1][7].startswith("(;SZ[19];B[")
    s2 = tmp_path / "s2"
    P.main(["sample", "--rows", str(out), "--out-dir", str(s2)])
    assert open(s / "sample_games.csv.gz", "rb").read() == open(s2 / "sample_games.csv.gz", "rb").read()
