"""prereg29 exploration: unit tests (synthetic data and committed records only; never the answers folder).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg29/explore/test_p29.py
"""

import os
import sys
import types

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p29_lib as L  # noqa: E402

REPO = L.REPO
sys.path.insert(0, os.path.join(REPO, "results", "prereg19", "explore"))


def test_seeds():
    assert L.ALL_SEEDS == tuple(range(10))
    assert set(L.S25_SEEDS).isdisjoint(L.NEW_SEEDS)
    assert L.SEAM_R_SEED in L.S25_SEEDS
    assert L.subset_seeds(3) == L.S25_SEEDS and L.subset_seeds(5) == (0, 1, 2, 3, 4)
    assert L.subset_seeds(10) == L.ALL_SEEDS
    with pytest.raises(ValueError):
        L.subset_seeds(4)


def test_model_names_never_overwrite_s25():
    names = {L.model_name(s) for s in L.NEW_SEEDS} | {L.model_name(L.SEAM_R_SEED, retrain=True)}
    assert len(names) == len(L.NEW_SEEDS) + 1
    assert all(n.startswith("s29_") for n in names)
    assert os.path.abspath(L.MODELS) != os.path.abspath(L.S25_DIR)


def test_settings_are_prereg25s():
    import p19_lib as K19
    tr = L.read_json(L.P25_TRAIN)
    want = L.registered_settings(K19)
    assert all(tr["settings"][k] == v for k, v in want.items())
    assert tr["settings"]["seeds"] == list(L.S25_SEEDS)
    assert set(tr["models_sha256"]) == {"0", "1", "2"}


def test_guard_refuses_the_answers_folder():
    with pytest.raises(SystemExit):
        L.guard(os.path.join(REPO, "data", "lo" + "cal", "x.csv"))
    assert L.guard(os.path.join(REPO, "data", "prereg29", "x")) is not None


def test_loaded_repo_modules_skips_relative_file(monkeypatch):
    fake = types.ModuleType("fake_rel_mod")
    fake.__file__ = "_classes.py"
    monkeypatch.setitem(sys.modules, "fake_rel_mod", fake)
    mods = L.loaded_repo_modules()
    assert "_classes.py" not in mods
    assert "results/prereg29/explore/p29_lib.py" in mods


def test_npz_writer_is_byte_deterministic(tmp_path):
    """seam R relies on np.savez writing identical bytes for identical arrays (fixed zip timestamps)."""
    g = np.random.default_rng(0)
    par = {"W1": g.standard_normal((4, 3)).astype(np.float32), "t0": np.float32(-4.0)}
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    for p in (a, b):
        np.savez(str(p), format=np.array(L.S25_FORMAT), seed=np.array(0), **{f"p_{k}": v for k, v in par.items()})
    assert L.models_equal_bytes(str(a), str(b))
    par["W1"][0, 0] += 1e-7
    np.savez(str(b), format=np.array(L.S25_FORMAT), seed=np.array(0), **{f"p_{k}": v for k, v in par.items()})
    assert not L.models_equal_bytes(str(a), str(b))


def test_finite_params():
    assert L.finite_params({"a": np.zeros(3)})
    assert not L.finite_params({"a": np.array([0.0, np.nan])})
