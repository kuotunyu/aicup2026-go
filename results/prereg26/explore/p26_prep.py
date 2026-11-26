"""prereg26 (CPU, .venv): the raw EX of the V games (PLAN.md 1, seam V). No model, no score.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe
    $S/p26_prep.py vex   -> data/prereg26/prep/v_ex.npy + results/prereg26/explore/v_ex_record.json

The 375 V chunks of prereg25 (data/prereg25/v/probe, p1_extract unchanged) are re-read with tools/p09_map.read_chunk_rows
(unchanged; the function 09's prep used), which returns X and EX of the usable games. Seam V: the re-read X, ref, y and
player keys equal prereg25's v_X.npy / v_meta.npz bit for bit and the chunk SHA list equals prereg25's prep_record.json;
then the EX rows (float32, raw) are written in the same order. Reads no inner data, never data/local.
"""

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p26_lib as L  # noqa: E402
import numpy as np  # noqa: E402

import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402

CHUNK = 200
FORMAT = "c10-prereg26 explore v_ex v1"


def log(*x):
    print(*x, flush=True)


def cmd_vex(_a):
    t0 = time.time()
    if os.path.exists(L.V_EX) or os.path.exists(L.V_EX_RECORD):
        raise SystemExit("v_ex.npy / v_ex_record.json exist")
    rec25 = L.read_json(os.path.join(L.V_PREP, "prep_record.json"))
    sets = L.read_json(os.path.join(L.V_DIR, "sets.json"))
    games = sets["probe"]["games"]
    n_ch = (len(games) + CHUNK - 1) // CHUNK
    if n_ch != rec25["chunks"]:
        raise SystemExit(f"{n_ch} chunks, prereg25 prep had {rec25['chunks']}")
    vX = np.load(L.guard(os.path.join(L.V_PREP, "v_X.npy")), mmap_mode="r")
    with np.load(L.guard(os.path.join(L.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vref, vy, vk = np.asarray(z["ref"]), np.asarray(z["y"]), np.asarray(z["players"]).astype(str)
    n = len(vref)
    out_dir = os.path.dirname(L.V_EX)
    os.makedirs(out_dir, exist_ok=True)
    tmp = L.V_EX + ".tmp"
    EX = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, FW.NEX))
    shas, i = [], 0
    eq = {"X": True, "ref": True, "y": True, "players": True}
    n_nan_rows = 0
    n_inf = 0
    for k in range(n_ch):
        p = os.path.join(L.V_DIR, "probe", f"chunk_{k:05d}.npz")
        c = M.read_chunk_rows(L.guard(p), None, (), games)
        g = len(c["ref"])
        eq["X"] = eq["X"] and bool(np.array_equal(c["X"], np.asarray(vX[i:i + g]), equal_nan=True))
        eq["ref"] = eq["ref"] and bool(np.array_equal(c["ref"], vref[i:i + g]))
        eq["y"] = eq["y"] and bool(np.array_equal(c["y"], vy[i:i + g]))
        eq["players"] = eq["players"] and bool(np.array_equal(c["players"].astype(str), vk[i:i + g]))
        n_inf += int(np.isinf(c["EX"]).sum())
        n_nan_rows += int(np.isnan(c["EX"]).any(axis=1).sum())
        EX[i:i + g] = c["EX"]
        shas.append(c["sha"])
        i += g
        if k % 100 == 0:
            log(f"chunk {k}/{n_ch}, {i} rows, {time.time() - t0:.0f}s")
    seam = {"rows_equal": bool(i == n), **{f"{k}_equal_prereg25": v for k, v in eq.items()},
            "chunk_sha256_equal_prereg25": bool(shas == rec25["chunk_sha256"]), "ex_inf_values": n_inf}
    seam["passed"] = bool(all(v for k, v in seam.items() if k != "ex_inf_values") and n_inf == 0)
    EX.flush()
    del EX
    if not seam["passed"]:
        os.remove(tmp)
        log(json.dumps(seam))
        raise SystemExit(f"seam V failed: {seam}")
    os.replace(tmp, L.V_EX)
    rec = {"format": FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "what": "PLAN.md 1 / 5.4: raw EX (44, float32) of the V games, re-read with tools/p09_map.read_chunk_rows "
                   "(unchanged) from prereg25's V chunks, in v_meta.npz order",
           "rows": int(n), "chunks": n_ch, "seam_v": seam, "rows_with_some_nan_ex_record_only": n_nan_rows,
           "v_ex_sha256": L.sha_file(L.V_EX), "v_X_sha256": L.sha_file(os.path.join(L.V_PREP, "v_X.npy")),
           "v_meta_sha256": L.sha_file(os.path.join(L.V_PREP, "v_meta.npz")),
           "prereg25_prep_record_sha256_lf": L.sha_lf(os.path.join(L.V_PREP, "prep_record.json")),
           "seconds": round(time.time() - t0, 1)}
    L.write_json(L.V_EX_RECORD, rec)
    log(json.dumps(rec, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("vex")
    a = ap.parse_args(argv)
    return {"vex": cmd_vex}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
