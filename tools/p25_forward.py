"""PREREG_25 competition 2: the S25 model files (the question-level set model S19's architecture retrained with the
validation players, docs/PREREG_25_SET_MORE_PLAYERS.md section 1.2).

Only the loader is new: the registered SHA-256s, format, seeds and steps of the three S25 files. Every computation is
tools/p19_forward.py (imported, never edited): the architecture, the standardisation with Q09's p1e_map.npz, the
pooling over the usable games, the 3-seed ensemble and the scale 4 are the same as 19's.

Pure numpy / scipy: this module never imports torch, u1_lib, hs_lib, pilot_* or p1_* (tests/test_p25_hygiene.py).
"""

import os

import numpy as np

import p19_forward as SF

L = SF.L
SCALE = SF.SCALE
H_DIM = SF.H_DIM
SEEDS = SF.SEEDS
PARAM_NAMES = SF.PARAM_NAMES
SHAPES = SF.SHAPES
MODEL_FORMAT = "c10-prereg25 s25 model v1"
# PREREG_25 1.2: the three model files, byte for byte the exploration's (results/prereg25/explore/train_record.json)
MODEL_SHA256 = {0: "4a01d6a60355cd514e9cc4988c08424a7952fc33302b47a63542a8caf67163bb",
                1: "ab53d69b2762a7c01bed354ac6c218cf167fda44bbd3d88eed06a6f4adf3cba4",
                2: "20e190f56866c7a28b4eeef0c0d4d656867b79a908e066ba91a47570601c4fbb"}
BEST_STEP = {0: 11000, 1: 7500, 2: 10500}

# the computation (p19_forward, unchanged)
game_h, set_term, question_set_term = SF.game_h, SF.set_term, SF.question_set_term
single_game_logp, usable_groups, mean_matrix = SF.single_game_logp, SF.usable_groups, SF.mean_matrix
sha256_file = SF.sha256_file
SetForwardError = SF.SetForwardError


def model_name(seed):
    return f"s25_seed{int(seed)}.npz"


def load_model(path, seed):
    """One S25 model file (the registered SHA-256, format, seed, step, not smoke) -> float64 parameters."""
    seed = int(seed)
    if sha256_file(path) != MODEL_SHA256.get(seed):
        raise SetForwardError(f"{path}: SHA-256 is not the registered S25 seed {seed} model (PREREG_25 1.2)")
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != MODEL_FORMAT or int(z["seed"]) != seed or int(z["best_step"]) != BEST_STEP[seed] \
                or bool(z["smoke"]):
            raise SetForwardError(f"{path}: not the S25 model of seed {seed}")
        par = {k: np.asarray(z[f"p_{k}"]).astype(np.float64) for k in PARAM_NAMES}
    bad = [k for k in PARAM_NAMES if par[k].shape != SHAPES[k]]
    if bad:
        raise SetForwardError(f"{path}: parameter shapes {bad} differ")
    return par


def load_models(folder):
    """The three seeds in order -> [params]."""
    return [load_model(os.path.join(folder, model_name(s)), s) for s in SEEDS]
