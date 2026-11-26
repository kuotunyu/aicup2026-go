"""prereg06 exploration c2_headroom: shared helpers (inner pseudo-questions only; no validation data).

Reads results/prereg05/explore/u1/u1_oof.npz through u1_lib (which installs the data/local guard on import).
Nothing here writes a file. Scoring is the official formula (1 exact, 1/e adjacent, 0 otherwise).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
U1 = os.path.join(REPO, "results", "prereg05", "explore", "u1")
if U1 not in sys.path:
    sys.path.insert(0, U1)

import u1_lib as U  # noqa: E402  (installs the data/local guard)

LEVELS = U.LEVELS
E1 = U.E1
KS = (5, 10, 15, 20)
# players per level in the whole training set (docs/DATA.md, "一覽")
TRAIN_PLAYERS = np.array([890, 817, 1048, 1582, 433, 610, 618, 640, 658, 711], dtype=np.float64)


def load_mono(combo="Q02"):
    z = U.load_saved()
    return {"P": z[f"mono_P_{combo}"], "y": z["mono_y"], "n": z["mono_n"], "player": z["mono_player"],
            "fold": z["mono_fold"], "colour": z["mono_colour"], "round": z["mono_round"],
            "games": z["mono_games"], "player_level": z["player_level"], "n_players": len(z["player_level"]),
            "P_Q00": z["mono_P_Q00"], "player_n_inner_games": z["player_n_inner_games"],
            "game_colour": z["game_colour"], "game_level": z["game_level"]}


def expected_scores(P):
    return U.expected_scores(P)


def rule_b(P):
    return np.argmax(expected_scores(P), axis=1)


def score(pred, y):
    return U.score(pred, y)


def rank_balanced(sc, y):
    return U.rank_balanced(sc, y)


def pct(a, qs=(2.5, 5, 50, 95, 97.5)):
    a = np.asarray(a, dtype=np.float64)
    return {str(q): float(np.percentile(a, q)) for q in qs}


def summary(a):
    a = np.asarray(a, dtype=np.float64)
    out = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "min": float(a.min()),
           "max": float(a.max()), "n": int(len(a))}
    out.update({"p" + k: v for k, v in pct(a).items()})
    return out


class PlayerBoot:
    """PlayerBootstrap as PREREG_04 G2 (players resampled within each level, 1000 draws, default_rng(0)),
    for the rank-balanced mean of any per-question score vector over a subset of questions."""

    def __init__(self, player, y, n_players, n_boot=1000, seed=0):
        self.b = U.PlayerBootstrap(player, y, n_players)
        assert len(self.b.draws[0]) == n_boot or n_boot == 1000

    def ci(self, sc, sc_ref=None):
        st = self.b.stats(sc)
        if sc_ref is not None:
            st = st - self.b.stats(sc_ref)
        return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], float(np.std(st, ddof=1))


def dump(obj, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    return path
