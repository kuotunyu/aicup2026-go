"""Null control: a proxy file with the same keys whose per-game 'log-likelihoods' are pure noise (seed 0).
Same shape and roughly the same spread as a real per-game sum over ~110 moves (sd 3 nats per level)."""
import sys
sys.dont_write_bytecode = True
import numpy as np
z = np.load("moves_inner.npz")
G = len(z["off"]) - 1
rng = np.random.default_rng(0)
ll = rng.normal(0.0, 1.0, size=(G, 6, 3, 10)) * np.sqrt(3.0)
np.savez("proxy_null.npz", rank=z["rank"], game_id=z["game_id"], player_id=z["player_id"], color=z["color"],
         label=z["label"], ll=ll, n_moves=np.full((G, 2, 3), 36), variants=np.array(
             ["t_choice_a5", "t_choice_a20", "t_choice_a100", "t_nb", "o_choice_a20", "o_nb"]))
print("ok", G)
