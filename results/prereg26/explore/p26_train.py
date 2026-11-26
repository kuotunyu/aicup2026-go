"""prereg26 exploration (competition 2): train P1TO_E26 (PLAN.md 3) = 09's six ens3fam members retrained on probe + V
(CPU, .venv torch, 8 threads). Reads no inner data, never data/local; computes no candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg26/explore
    $S/p26_train.py train            -> data/prereg26/map/p1e26_map.npz (+ members/) + $S/train_record.json (run once;
                                        the code must be committed and clean)
    $S/p26_train.py train --smoke    -> data/prereg26/smoke/ (first 6,000 probe + 2,000 V rows, features replaced by
                                        default_rng(12345) normal noise, 2-epoch schedules: pipeline and timing only)
    --resume keeps a finished member after an interruption (same commit and code as train_started.json).

Everything that defines a member is imported unchanged from tools/p09_map.py and tools/p09_forward.py (PREREG_09 1.3):
fit_member (player-grouped 5-fold CV, folds by player_folds seed 0 over the actual training players, e* by the mean OOF
log-loss over the 24-epoch schedule, refit on every row stopped at e*; lowrank fixed 30), Rows (float32 input),
model_params, set_threads, save_map, torch_numpy_max_diff; the moments are p09_rank_fuse.Prep.moments. Only the rows
differ: 09's probe rows (data/prereg17/prep = 09's prep) followed by the V rows (prereg25 v_X / v_meta, this
exploration's v_ex).
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
import p26_lib as L  # noqa: E402  (tools/ on sys.path)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402  (pins the p08 tools before importing them)
import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402

REPO = L.REPO
DATA = L.DATA
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p26_lib.py", "p26_prep.py", "p26_train.py", "v_ex_record.json")
IMPORTED = ("tools/p09_map.py", "tools/p09_forward.py", "tools/p09_rank_fuse.py", "results/prereg25/explore/p25_lib.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/check_report.json",
            "results/prereg09/rank/train_report.json", "results/prereg09/rank/frozen_config.json")
FORMAT = "c10-prereg26 explore train v1"
MEMBER_FORMAT = "c10-prereg26 member v1"
MAP_NAME = "p1e26_map.npz"


def log(*x):
    print(*x, flush=True)


def git(*args):
    import subprocess
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    rel = sorted(set(rel) | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: L.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def settings(smoke):
    return {"members": [dict(m) for m in FW.MEMBERS], "configs": json.loads(json.dumps(FW.CONFIGS)),
            "batch": M.BATCH, "predict_batch": M.PREDICT_BATCH, "folds": M.FOLDS, "fold_seed": 0,
            "threads": r.VENV["threads"], "smoke_epochs": L.SMOKE_EPOCHS if smoke else None,
            "smoke_rows": list(L.SMOKE_ROWS) if smoke else None, "port_rows_each": L.PORT_ROWS}


def seam_p(v_rec):
    want = (L.read_json(L.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {f: L.sha_file(os.path.join(L.PROBE_PREP, f)) for f in L.PROBE_FILES}
    rec25 = L.read_json(os.path.join(L.V_PREP, "prep_record.json"))
    v = {"v_X.npy": L.sha_file(os.path.join(L.V_PREP, "v_X.npy")),
         "v_meta.npz": L.sha_file(os.path.join(L.V_PREP, "v_meta.npz")), "v_ex.npy": L.sha_file(L.V_EX)}
    out = {"probe_want": {f: want.get(f) for f in got}, "probe_got": got, "v": v,
           "v_X_meta_are_prereg25": bool(v["v_X.npy"] == rec25["files_sha256"]["v_X.npy"]
                                         and v["v_meta.npz"] == rec25["files_sha256"]["v_meta.npz"]),
           "v_ex_is_record": bool(v["v_ex.npy"] == v_rec["v_ex_sha256"]),
           "seam_v_passed": bool(v_rec["seam_v"]["passed"])}
    out["ok"] = bool(all(want.get(f) == s for f, s in got.items()) and out["v_X_meta_are_prereg25"]
                     and out["v_ex_is_record"] and out["seam_v_passed"])
    return out


def seam_t(probe):
    """PLAN.md 5.5: on the probe rows alone, the same path gives Q09's moments bit for bit and 09's fold sizes."""
    mp = FW.load_map(L.Q09_MAP)
    t = time.time()
    mom = L.moments(r, probe["X"], probe["EX"], np.arange(probe["n"]))
    eq = L.moments_equal(mom, mp)
    fold = FW.player_folds(probe["players"])
    sizes = {str(f): int((fold == f).sum()) for f in range(M.FOLDS)}
    want = L.read_json(L.Q09_TRAIN)["training"]["folds"]
    out = {"moments_equal_q09_map": eq, "fold_sizes": sizes, "fold_sizes_09": want,
           "rows": int(mom["rows"]), "seconds": round(time.time() - t, 1)}
    out["ok"] = bool(all(eq.values()) and sizes == want and mom["rows"] == 270000)
    return out, mp


def save_member(path, spec, params, cv, e_star, binding):
    arrays = {"format": np.array(MEMBER_FORMAT), "name": np.array(spec["name"]), "kind": np.array(spec["kind"]),
              "seed": np.array(spec["seed"]), "e_star": np.array(e_star), "cv": np.array(json.dumps(cv)),
              "binding": np.array(json.dumps(binding, sort_keys=True))}
    arrays.update({f"p_{k}": v for k, v in params.items()})
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return L.sha_file(path)


def load_member(path, spec, binding):
    with np.load(L.guard(path), allow_pickle=False) as z:
        if str(z["format"]) != MEMBER_FORMAT or (str(z["name"]), str(z["kind"]), int(z["seed"])) != \
                (spec["name"], spec["kind"], spec["seed"]):
            raise SystemExit(f"{path}: not the member {spec}")
        if json.loads(str(z["binding"])) != json.loads(json.dumps(binding, sort_keys=True)):
            raise SystemExit(f"{path}: trained with another binding (code / rows / moments)")
        return {k: np.asarray(z[f"p_{k}"]) for k in FW.PARAMS[spec["kind"]]}, int(z["e_star"]), \
            json.loads(str(z["cv"]))


def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_dir = SMOKE_DIR if smoke else DATA
    mem_dir = os.path.join(out_dir, "members")
    map_path = os.path.join(out_dir, "map", MAP_NAME)
    rec_path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else RECORD
    venv = r.venv_check(with_torch=True, preview=smoke)
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg26 files or the imported tools are not committed and clean: {cs['dirty']}")
        prior = [p for p in (rec_path, STARTED) if os.path.exists(p)]
        if prior and not a.resume and not (a.rerun and a.reason):
            raise SystemExit(f"{prior} exist: run once (--resume after an interruption; --rerun --reason only after "
                             f"a commit that names a concrete program error)")
        if a.resume and os.path.exists(rec_path):
            raise SystemExit(f"{rec_path} exists: training finished, --resume is only for an interrupted run")
        if a.rerun and prior:
            heads = {(L.read_json(p).get("code") or {}).get("head") for p in prior}
            if cs["head"] in heads:
                raise SystemExit("--rerun needs a newer commit than the earlier run's")
        if a.resume and os.path.exists(STARTED):
            stc = L.read_json(STARTED).get("code") or {}
            if stc.get("sha256_lf") != cs["sha256_lf"] or stc.get("head") != cs["head"]:
                raise SystemExit("--resume needs the same commit and code as train_started.json")
        else:
            L.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                   "rerun_reason": a.reason if a.rerun else None})
    map_sha09 = L.sha_file(L.Q09_MAP)
    if map_sha09 != L.read_json(L.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    aud = L.read_json(os.path.join(L.P25, "audit.json"))
    pre_d = {"V_players_in_inner": aud["val"]["players_in_inner"],
             "V_games_vs_inner_8_symmetries": aud["duplicates"]["val_vs_inner_8_symmetries"]}
    if any(pre_d.values()):
        raise SystemExit(f"prereg25 audit.json: V overlaps inner {pre_d}")
    v_rec = L.read_json(L.V_EX_RECORD)
    sp = seam_p(v_rec)
    if not sp["ok"]:
        raise SystemExit(f"seam P failed: {sp}")
    probe, v = L.load_probe(r), L.load_v(r)
    overlap = {"probe_and_V_keys": len(set(probe["players"].tolist()) & set(v["players"].tolist())),
               "probe_and_V_player_id": len(L.player_ids(probe["players"].tolist())
                                            & L.player_ids(v["players"].tolist()))}
    if overlap["probe_and_V_keys"] or overlap["probe_and_V_player_id"]:
        raise SystemExit(f"probe and V share players {overlap}")
    st, mp09 = seam_t(probe)
    log(f"seam P ok; seam T {json.dumps(st)}")
    if not st["ok"]:
        raise SystemExit(f"seam T failed: {st}")
    # ---- the training rows: probe (09's order) then V
    t = time.time()
    if smoke:
        n_p, n_v = L.SMOKE_ROWS
        g = np.random.default_rng(L.NOISE_SEED)
        X = g.standard_normal((n_p + n_v, FW.X_DIM)).astype(np.float32)
        EX = g.standard_normal((n_p + n_v, FW.NEX)).astype(np.float32)
    else:
        n_p, n_v = probe["n"], v["n"]
        X = np.concatenate([np.asarray(probe["X"]), np.asarray(v["X"])])
        EX = np.concatenate([np.asarray(probe["EX"]), np.asarray(v["EX"])])
    y = np.concatenate([probe["y"][:n_p], v["y"][:n_v]])
    players = np.concatenate([probe["players"][:n_p], v["players"][:n_v]])
    n = n_p + n_v
    mom = L.moments(r, X, EX, np.arange(n))
    fold = FW.player_folds(players)
    rows_rec = {"probe_rows": int(n_p), "v_rows": int(n_v), "rows": int(n),
                "players": int(len(set(players.tolist()))), "probe_players": int(len(set(players[:n_p].tolist()))),
                "v_players": int(len(set(players[n_p:].tolist()))),
                "folds": {str(f): int((fold == f).sum()) for f in range(M.FOLDS)},
                "v_rows_per_fold": {str(f): int((fold[n_p:] == f).sum()) for f in range(M.FOLDS)},
                "level_counts": np.bincount(y, minlength=FW.L).tolist(), "probe_V_overlap": overlap,
                "seconds_load_and_moments": round(time.time() - t, 1)}
    mom_diff = {k: float(np.nanmax(np.abs(np.asarray(mom[k]) - np.asarray(mp09[k])))) for k in
                ("mu_x", "sd_x", "mu_ex", "sd_ex")}
    log(f"rows {json.dumps({k: rows_rec[k] for k in ('rows', 'players', 'folds')})}; moments vs Q09 max |diff| "
        f"{mom_diff} ({time.time() - t:.0f}s)")
    used_threads = M.set_threads(r.VENV["threads"])
    import torch
    binding = {"head": cs["head"], "code_sha256_lf": {f: cs["sha256_lf"].get(f) for f in
                                                       ("tools/p09_map.py", "tools/p09_forward.py",
                                                        "results/prereg26/explore/p26_train.py")},
               "rows": int(n), "v_ex_sha256": sp["v"]["v_ex.npy"], "smoke": smoke, "threads": used_threads,
               "torch": torch.__version__, "moments_mu_x_sum": float(np.sum(mom["mu_x"]))}
    os.makedirs(mem_dir, exist_ok=True)
    saved_epochs = {k: FW.CONFIGS[k]["epochs"] for k in FW.CONFIGS}
    if smoke:
        for k in FW.CONFIGS:
            FW.CONFIGS[k]["epochs"] = L.SMOKE_EPOCHS
    members, recs = [], {}
    try:
        for spec in FW.MEMBERS:
            path = os.path.join(mem_dir, f"member_{spec['name']}.npz")
            jpath = os.path.join(mem_dir, f"member_{spec['name']}.json")
            if a.resume and os.path.exists(path) and os.path.exists(jpath):
                params, e_star, cv = load_member(path, spec, binding)
                recs[spec["name"]] = {**L.read_json(jpath), "kept_on_resume": True}
                if recs[spec["name"]]["sha256"] != L.sha_file(path):
                    raise SystemExit(f"{path}: SHA differs from its record")
                log(f"{spec['name']}: kept (resume)")
            else:
                get = M.Rows(X, EX, mom, FW.uses_ex(spec["kind"]))
                tm = time.time()
                model, cv = M.fit_member(spec, get, get.d, y, fold, log=log)
                params = M.model_params(model, spec["kind"])
                e_star = int(cv["epochs_chosen"])
                sha = save_member(path, spec, params, cv, e_star, binding)
                recs[spec["name"]] = {"e_star": e_star, "cv": cv, "seconds": round(time.time() - tm, 1),
                                      "sha256": sha, "file": os.path.relpath(path, REPO).replace("\\", "/")}
                L.write_json(jpath, recs[spec["name"]])
                del model
            members.append({"name": spec["name"], "kind": spec["kind"], "seed": spec["seed"], "e_star": e_star,
                            "params": params})
    finally:
        for k, e in saved_epochs.items():
            FW.CONFIGS[k]["epochs"] = e
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    map_rec = {"what": "prereg26 P1TO_E26: 09's ens3fam retrained on probe (P1+P2+P3) + V (validation players)",
               "probe_used": "P1+P2+P3+V", "games": int(n), "players": rows_rec["players"], "smoke": smoke,
               "cv": {k: v["cv"] for k, v in recs.items()}, "e_star": {k: v["e_star"] for k, v in recs.items()}}
    FW.save_map(map_path, members, mom["mu_x"], mom["sd_x"], mom["mu_ex"], mom["sd_ex"], map_rec)
    map_sha = L.sha_file(map_path)
    # ---- port check (PREREG_09 rev 2 item 1: float64 torch vs numpy, ensemble lp <= 1e-4) on training rows
    tp = time.time()
    k = L.PORT_ROWS
    pr = np.concatenate([np.arange(min(k, n_p)), n_p + np.arange(min(k, n_v))])
    pc = M.torch_numpy_max_diff(map_path, X[pr], EX[pr])
    pc["ok"] = bool(pc["max_abs_lp_diff"] <= 1e-4)
    pc["seconds"] = round(time.time() - tp, 1)
    log(f"port check {pc['max_abs_lp_diff']:.3g} ok {pc['ok']}")
    q09_tr = L.read_json(L.Q09_TRAIN)
    rec = {"format": FORMAT, "created": r.now_iso(), "smoke": smoke,
           "smoke_note": ("first 6,000 probe + 2,000 V rows, features replaced by default_rng(12345) normal noise, "
                          "2-epoch schedules: never a result") if smoke else None,
           "seconds": round(time.time() - t_all, 1), "code": cs,
           "what": "PLAN.md 3: P1TO_E26 = 09's six ens3fam members (tools/p09_map.fit_member unchanged) on probe + V; "
                   "no inner data read",
           "settings": settings(smoke), "venv": venv, "environment": {"torch_threads": used_threads},
           "q09_map_sha256": map_sha09, "pre_seam_d_prereg25_audit": pre_d, "seam_p": sp, "seam_t": st, "rows": rows_rec,
           "moments_vs_q09_max_abs_diff_record_only": mom_diff, "members": recs,
           "q09_members_record_only": {k: {"cv_logloss": v["cv"]["logloss"], "top1": v["cv"]["top1"],
                                           "e_star": v["e_star"], "seconds": v.get("seconds")}
                                       for k, v in q09_tr["members"].items()},
           "port_check": pc, "map_file": os.path.relpath(map_path, REPO).replace("\\", "/"), "map_sha256": map_sha}
    L.write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0 if pc["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
