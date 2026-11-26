"""prereg58: S0 (prereg43) per-game vectors of the OPPONENT's side (colour flipped). Arguments = p43_run.py embed.

    GPUPY results/prereg58/explore/p58_opp.py embed --ckpt data/prereg43/S0/ckpt/step_320000.pt --source ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results", "prereg43", "explore"))
import p43_run as R  # noqa: E402
import p36_embed  # noqa: E402

_pool, _store = p36_embed.pool_games, p36_embed.store_games


def pool_games(src, pool):
    codes, start, end, is_white = _pool(src, pool)
    return codes, start, end, ~is_white


def store_games(players, cache):
    rows, codes, start, end, is_white = _store(players, cache)
    return rows, codes, start, end, ~is_white


p36_embed.pool_games, p36_embed.store_games = pool_games, store_games

if __name__ == "__main__":
    sys.exit(R.main(sys.argv[1:]))
