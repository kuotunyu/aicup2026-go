"""Unit tests of p34_extract.py and p34_colab_nb.py (prereg34 explore, PLAN.md 3). Synthetic data only: no real game,
no GPU, no data folder.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg34/explore/test_p34_extract.py
"""

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
import p34_extract as X  # noqa: E402
import p34_colab_nb as NBB  # noqa: E402

SGF = "(;SZ[19];B[pd];W[dp];B[pp];W[dd])"


def write_gz(path, header, rows):
    with open(path, "wb") as fo_raw:
        fo = io.TextIOWrapper(gzip.GzipFile(fileobj=fo_raw, mode="wb", mtime=0), encoding="utf-8", newline="")
        wr = csv.writer(fo, lineterminator="\n")
        wr.writerow(header)
        for r in rows:
            wr.writerow(r)
        fo.close()
    return str(path)


def ogs_rows():
    # (player_id, game_id, ogs_rank, color, start_time, speed, n_moves, sgf, bin, player_ogs_median)
    return [["ogs_11", "ogs_100", "20.5", "B", "1600000000", "live", "120", SGF + "a", "20", "20.5"],
            ["ogs_22", "ogs_100", "21.25", "W", "1600000000", "live", "120", SGF + "a", "21", "21.0"],
            ["ogs_11", "ogs_101", "20.75", "W", "1600000100", "blitz", "130", SGF + "b", "20", "20.5"]]


@pytest.fixture
def ogs(tmp_path):
    p = write_gz(tmp_path / "sample_games.csv.gz", X.OGS_COLUMNS + ("bin", "player_ogs_median"), ogs_rows())
    return p, X.build_ogs_sets(p, p)


# ---------------------------------------------------------------- routing
def test_wrapper_installed():
    assert X.E.set_games is X.set_games


def test_other_sources_go_to_the_original(monkeypatch):
    seen = []

    def orig(sets, name):
        seen.append((sets[name]["source"], name))
        return ["original"]

    monkeypatch.setattr(X, "_ORIGINAL", orig)
    for src in ("fit", "inner", "val"):
        assert X.set_games({"probe": {"source": src}}, "probe") == ["original"]
    assert seen == [("fit", "probe"), ("inner", "probe"), ("val", "probe")]


def test_ogs_source_reads_the_file_never_the_original(monkeypatch, ogs):
    monkeypatch.setattr(X, "_ORIGINAL", lambda *a: pytest.fail("original called for an ogs probe"))
    _p, sets = ogs
    got = X.set_games(sets, "probe")
    assert [g["ref"] for g in got] == [0, 1, 2]
    assert [(g["player_id"], g["game_id"], g["color"]) for g in got] == [
        ("ogs_11", "ogs_100", "B"), ("ogs_22", "ogs_100", "W"), ("ogs_11", "ogs_101", "W")]
    assert [g["sgf_content"] for g in got] == [SGF + "a", SGF + "a", SGF + "b"]
    assert all(g["rank"] == "ogs" for g in got)


def test_player_mismatch_refused(ogs):
    _p, sets = ogs
    sets["probe"]["games"][1]["player_id"] = "ogs_11"
    with pytest.raises(X.E.P.P1Error, match="player or colour"):
        X.set_games(sets, "probe")


def test_colour_mismatch_refused(ogs):
    _p, sets = ogs
    sets["probe"]["games"][2]["color"] = "B"          # ogs_101 has no black row of ogs_11 in the file
    with pytest.raises(X.E.P.P1Error, match="not in"):
        X.set_games(sets, "probe")
    _p, sets2 = ogs
    sets2 = json.loads(json.dumps(sets2))
    sets2["probe"]["games"][0]["color"], sets2["probe"]["games"][1]["color"] = "W", "B"   # swapped colours
    with pytest.raises(X.E.P.P1Error, match="player or colour"):
        X.set_games(sets2, "probe")


def test_games_file_hash_checked(ogs):
    _p, sets = ogs
    sets["probe"]["games_file_sha256"] = "0" * 64
    with pytest.raises(X.E.P.P1Error, match="SHA-256"):
        X.set_games(sets, "probe")


def test_seam_copy_source_checks_rank(tmp_path):
    p = write_gz(tmp_path / "seam_games.csv.gz", X.SEAM_COLUMNS,
                 [["pa", "g_1", "D", "B", SGF], ["pb", "g_2", "5D", "W", SGF + "x"]])
    sets = {"probe": {"source": "p3_seam", "games_file": p, "games_file_sha256": X.sha_raw(p),
                      "games": [{"rank": "D", "game_id": "g_1", "player_id": "pa", "color": "B"},
                                {"rank": "5D", "game_id": "g_2", "player_id": "pb", "color": "W"}]}}
    got = X.set_games(sets, "probe")
    assert [(g["rank"], g["sgf_content"], g["ref"]) for g in got] == [("D", SGF, 0), ("5D", SGF + "x", 1)]
    sets["probe"]["games"][1]["rank"] = "4D"
    with pytest.raises(X.E.P.P1Error):
        X.set_games(sets, "probe")


# ---------------------------------------------------------------- sets format
def test_sets_format_and_load_sets(tmp_path, ogs):
    p, sets = ogs
    assert sets["format"] == X.E.SETS_FORMAT == "c10-prereg08 p1 sets v1"
    pr = sets["probe"]
    assert pr["source"] == "ogs" and pr["games_file_sha256"] == X.sha_raw(p)
    assert pr["games"][0] == {"game_id": "ogs_100", "player_id": "ogs_11", "color": "B", "ogs_rank": "20.5"}
    assert sets["counts"] == {"probe": 3, "probe_players": 2, "bank": 0, "mono": 0}
    out = tmp_path / "sets.json"
    X.write_json(str(out), sets)
    loaded = X.E.load_sets(str(out))
    assert len(X.set_games(loaded, "probe")) == 3


def test_sets_rejects_duplicates_and_foreign_files(tmp_path):
    rows = ogs_rows()
    p = write_gz(tmp_path / "dup.csv.gz", X.OGS_COLUMNS + ("bin", "player_ogs_median"), rows + [rows[0]])
    with pytest.raises(SystemExit, match="twice"):
        X.build_ogs_sets(p, p)
    q = write_gz(tmp_path / "official.csv.gz", X.SEAM_COLUMNS, [["pa", "g_1", "D", "B", SGF]])
    with pytest.raises(SystemExit, match="not a p34_ogs"):
        X.build_ogs_sets(q, q)


def test_sets_subcommand_refuses_overwrite_and_outside_repo(tmp_path, ogs):
    p, _s = ogs
    with pytest.raises(SystemExit, match="not inside the repository"):
        X.main(["sets", "--games", p, "--out", str(tmp_path / "s.json")])
    (tmp_path / "s2.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        X.main(["sets", "--games", p, "--out", str(tmp_path / "s2.json")])


def test_guard_refuses_local():
    with pytest.raises(SystemExit):
        X.guard(os.path.join(X.REPO, "data", "local", "x"))


# ---------------------------------------------------------------- seam comparison (Colab check)
def synth_chunks(rng, n=X.SEAM_GAMES):
    means = rng.normal(size=(n, 2, 770)).astype(np.float32)
    counts = rng.integers(0, 120, size=(n, 2)).astype(np.int32)
    counts[:, 0] = np.maximum(counts[:, 0], 1)
    counts[5, 1] = 0
    failed = np.zeros(n, dtype=bool)
    failed[7] = True
    return {"ref": np.arange(n, dtype=np.int64), "failed": failed, "counts": counts,
            "n_moves": counts.sum(1).astype(np.int16), "means": means}


def split2(c):
    return [{k: v[:200] for k, v in c.items()}, {k: v[200:] for k, v in c.items()}]


def test_std_x_matches_p09_forward():
    import p09_forward as FW
    rng = np.random.default_rng(1)
    c = synth_chunks(rng, 50)
    mu, sd = rng.normal(size=1540), rng.uniform(0.5, 2, size=1540)
    sd[3] = 0.0
    a = X.std_x(c["means"], c["counts"], mu, sd)
    b = FW.standardise_x(FW.x_raw32(c["means"], c["counts"]), mu, sd)
    assert np.array_equal(a, b)


def test_seam_compare_pass_and_fail():
    rng = np.random.default_rng(2)
    ref = synth_chunks(rng)
    ref.update(mu_x=np.zeros(1540), sd_x=np.ones(1540))
    new = split2({k: ref[k].copy() for k in ("ref", "failed", "counts", "n_moves", "means")})
    new[1]["means"][3, 0, 10] += 0.04
    r = X.seam_compare(new, ref)
    assert r["passed"] and not r["bitwise_equal_means"] and abs(r["max_abs_std_feature"] - 0.04) < 1e-5
    new[1]["means"][3, 0, 10] += 0.02
    assert not X.seam_compare(new, ref)["passed"]
    new2 = split2({k: ref[k].copy() for k in ("ref", "failed", "counts", "n_moves", "means")})
    new2[0]["counts"][0, 0] += 1
    r2 = X.seam_compare(new2, ref)
    assert not r2["passed"] and r2["equal"]["counts"] is False


def test_seam_colab_command(tmp_path):
    rng = np.random.default_rng(3)
    ref = synth_chunks(rng)
    np.savez(tmp_path / "ref.npz", **ref, mu_x=np.zeros(1540), sd_x=np.ones(1540))
    d = tmp_path / "probe"
    d.mkdir()
    for i, part in enumerate(split2(ref)):
        np.savez(d / f"chunk_{i:05d}.npz", precision=np.array("fp16"), batch=np.array(1024),
                 torch=np.array("2.11.0+cu128"), device=np.array("cuda"), **part)
    out = tmp_path / "seam_colab.json"
    assert X.main(["seam-colab", "--new-dir", str(d), "--ref", str(tmp_path / "ref.npz"), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["passed"] and rec["bitwise_equal_means"] and rec["new_run"]["torch"] == "2.11.0+cu128"


def test_p1_code_is_the_kit_code():
    """The Colab path runs the kit's P1 code: it must be the repository's (colab-bundle refuses otherwise)."""
    with open(X.KIT_RECORD, encoding="utf-8") as f:
        kit = json.load(f)
    assert X.E.P.code_shas() == kit["manifest"]["constants"]["p1_code_record"]


# ---------------------------------------------------------------- notebook
KIT = {"kit_id": "g11-kit-v1-test", "zip": {"file": "C10_gpu_kit_v1.zip", "sha256": "a" * 64}}
BUNDLE = {"zip": {"file": "C10_p34_bundle.zip", "sha256": "b" * 64},
          "manifest": {"kit_id": "g11-kit-v1-test", "kit_zip_sha256": "a" * 64}}


def test_notebook_build():
    b = NBB.build(KIT, BUNDLE)
    assert b == NBB.build(KIT, BUNDLE)                    # deterministic
    nb = json.loads(b.decode("utf-8"))
    assert nb["nbformat"] == 4
    ids = [c["id"] for c in nb["cells"]]
    assert ids == ["p34-intro", "p34-setup", "p34-env", "p34-unpack", "p34-seam", "p34-run", "p34-pack"]
    src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    for cid, s in src.items():
        if cid != "p34-intro":
            compile(s, cid, "exec")
    assert "pip list 2>/dev/null | grep -iE '^torch |nvidia-cudnn'" in src["p34-env"]
    assert "TORCH_TAG = '+cu128'" in src["p34-setup"]
    assert "ENV['torch'].endswith(TORCH_TAG)" in src["p34-env"] and "stop(" in src["p34-env"]
    assert "2026.07" in src["p34-env"] and "2026.07" in src["p34-intro"]
    assert "seam_sets.json" in src["p34-seam"] and "seam-colab" in src["p34-seam"]
    assert "data/prereg34/p1/sets.json" in src["p34-run"]
    assert "'/content/drive/MyDrive/C10/gpu_fallback/C10_gpu_kit_v1.zip'" in src["p34-setup"]
    assert "'b" + "b" * 63 + "'" in src["p34-setup"]
    assert "gpu_fallback_core.py" in src["p34-unpack"] and "'verify'" in src["p34-unpack"]


def test_notebook_refuses_bundle_of_another_kit():
    other = json.loads(json.dumps(BUNDLE))
    other["manifest"]["kit_zip_sha256"] = "c" * 64
    with pytest.raises(SystemExit, match="another kit"):
        NBB.build(KIT, other)


def test_notebook_against_the_committed_kit_record():
    with open(NBB.KIT_RECORD, encoding="utf-8") as f:
        kit = json.load(f)
    b = dict(BUNDLE, manifest={"kit_id": kit["kit_id"], "kit_zip_sha256": kit["zip"]["sha256"]})
    src = json.loads(NBB.build(kit, b).decode("utf-8"))
    assert kit["zip"]["sha256"] in "".join(src["cells"][1]["source"])
