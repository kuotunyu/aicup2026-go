"""prereg56 probe, opponent side: the same c49 per-game vectors but for the OTHER player of every game (colour
flipped). Arguments = p49_run.py embed arguments.

    GPUPY results/prereg56/explore/p56_opp.py embed --ckpt ... --source pool|store ... --emb-root data/prereg56/emb_opp_*
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "results", "prereg49", "explore"))
import p49_run as R  # noqa: E402  (sets up sys.path for the older modules)
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
