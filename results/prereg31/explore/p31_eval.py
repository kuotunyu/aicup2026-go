"""prereg31 exploration (competition 2): inner-only judgement of PLAN.md 4 (one candidate Q31 = Q29 with the SET
term's ten no-HD models replaced by the ten S31 models, which read 2 extra per-game HD inputs), judged once. CPU
only (.venv; numpy forward, no torch).

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg31/explore
    $S/p31_eval.py eval            -> $S/eval_report.json (judged once; refuses unless the code, train_record.json
                                      and stage0b.json are committed and clean and stage0b.json says GO)
    $S/p31_eval.py eval --smoke    -> data/prereg31/smoke/eval_report.json (the smoke models of p31_train.py; the S31
                                      term is computed on noise features with the HD columns permuted across the inner
                                      rows: plumbing and timing only, never a result). NOTE: building the frozen
                                      Inner object and seam 1 recompute the already known inner numbers of the old
                                      models (Q02H ... Q29) also in a smoke run.

The Q09 / Q19 rebuild, the scores, the gates and the record-only helpers are imported unchanged from
results/prereg23/explore/p23_eval.py and p23_lib.py; the fusion is tools/p09_rank_fuse.Inner (unchanged). The SET term
is this file's set_term: p23_eval.s19_term line for line, with one optional step (the 2 standardised HD columns are
appended after the standardisation and before h_np). With HD off it must reproduce Q25 and Q29 bit for bit (seam 1).
Never the answers folder, never validation features, never tools/score.py, never data/training_v1.1 (the HD table is
read through p31_lib.HdTable; seam A of the build is taken from the committed hd_stats.json).
"""

import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
for p in (os.path.join(REPO, "tools"), P23, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

import p23_eval as E23  # noqa: E402  (Q19 rebuild, s19_term, scores; puts prereg19 / prereg12 folders on sys.path)
import p23_lib as K23  # noqa: E402  (gates with the +0.005 threshold)
import p19_lib as K19  # noqa: E402
import p31_lib as L  # noqa: E402  (puts prereg25 / prereg29 explore on sys.path)
import p29_lib as L29  # noqa: E402

r, ft_eval, FW, rf = E23.r, E23.ft_eval, E23.FW, E23.rf
guard_path = E23.guard_path
TOL = 1e-12
N_TRAIN_ROWS = 1000000                                      # PLAN.md 4.2 seam A: rows compared v1.0 against v1.1
STARTED = os.path.join(L.DATA, "eval_started.json")
SEAM_FAIL = os.path.join(L.DATA, "eval_seam_fail.json")     # a seam stop is written here (no candidate number exists)
JUDGED = os.path.join(L.DATA, "eval_judged.json")           # the gates, written as soon as they exist
REPORT = os.path.join(HERE, "eval_report.json")
SMOKE_ONLY_ABSENT = ("train_record.json", "stage0b.json")   # committed records a smoke run does not need
OWN_FILES = ("PLAN.md", "p31_hd.py", "p31_lib.py", "p31_train.py", "p31_eval.py", "test_p31.py", "test_p31_hd.py",
             "hd_stats.json", "train_record.json", "stage0b.json", "run_train.sh", "run_eval.sh", "ram_watch.py")
IMPORTED = tuple(E23.IMPORTED) + (
    "results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
    "results/prereg19/explore/p19_train.py", "results/prereg25/explore/p25_lib.py",
    "results/prereg25/explore/p25_train.py", "results/prereg25/explore/eval_report.json",
    "results/prereg25/explore/train_record.json", "results/prereg25/explore/audit.json",
    "results/prereg25/rank/s25_seed0.npz", "results/prereg25/rank/s25_seed1.npz",
    "results/prereg25/rank/s25_seed2.npz", "results/prereg29/explore/p29_lib.py",
    "results/prereg29/explore/p29_train.py", "results/prereg29/explore/train_record.json",
    "results/prereg29/explore/eval_report.json")
# the forward code of PLAN.md 4.2 seam M: LF SHA at the eval = LF SHA in train_record.json
FORWARD_FILES = ("results/prereg19/explore/p19_lib.py", "results/prereg19/explore/p19_train.py",
                 "results/prereg25/explore/p25_lib.py", "results/prereg25/explore/p25_train.py",
                 "tools/p09_forward.py", "results/prereg31/explore/p31_lib.py",
                 "results/prereg31/explore/p31_train.py")
FORMAT = "c10-prereg31 explore eval v1"
S0B_FORMAT = "c10-prereg31 explore stage0b v1"              # = p31_train.S0B_FORMAT (test_p31.py compares them)
TERMS_SET = ("P1TO", "SET")
WHICH = ("fit", "eval", "mono")


def log(*x):
    print(*x, flush=True)


# ---------------------------------------------------------------- code state and the once guard (as prereg29)
def code_state(smoke):
    own = [f for f in OWN_FILES if not (smoke and f in SMOKE_ONLY_ABSENT)]
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in own] + list(IMPORTED)
    rel = sorted(set(rel) | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = E23.git("status", "--porcelain", "--", *rel)
    return {"head": E23.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: E23.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = {(r.read_json(p).get("code") or {}).get("head") for p in prior}
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the one of the earlier run (the fix of the named error)")
    return sorted(h for h in heads if h)


def stage0b_record(train_path, stage0b_path=None):
    """The committed stage 0b decision (PLAN.md 3 / 6): go only if it is a GO made from this train_record.json and
    the registered rule, recomputed here from the committed records, gives GO again."""
    path = stage0b_path or L.STAGE0B
    if not os.path.exists(path):
        return {"exists": False, "go": False}
    s = L.read_json(path)
    tr = L.read_json(train_path)
    control, _src = L.control_nll()
    again = L.stage0b_decision({int(k): float(v["best_holdout_nll"]) for k, v in (tr.get("seeds") or {}).items()},
                               control)
    rec = {"exists": True, "format_ok": bool(s.get("format") == S0B_FORMAT), "decision": s.get("decision"),
           "mean_diff_s31_minus_control": s.get("mean_diff_s31_minus_control"),
           "pairs_s31_lower": s.get("pairs_s31_lower"),
           "train_record_sha256_lf": s.get("train_record_sha256_lf"),
           "is_this_train_record": bool(s.get("train_record_sha256_lf") == E23.sha_lf(train_path)),
           "decision_recomputed": again["decision"],
           "mean_diff_recomputed": again["mean_diff_s31_minus_control"]}
    rec["go"] = bool(rec["format_ok"] and rec["decision"] == "GO" and rec["is_this_train_record"]
                     and rec["decision_recomputed"] == "GO")
    return rec


# ---------------------------------------------------------------- the SET term (PLAN.md 4.1)
def set_term(prep, inner, mp, pars, hd_std=None, noise=False):
    """p23_eval.s19_term with one optional step: hd_std ([n_inner, 2] float64, indexed by inner row) is appended
    to the standardised features before h_np. hd_std None = HD off = s19_term line for line (seam 1).
    noise (smoke only): the 1,540 features are default_rng([12345, 1, block start]) normal noise."""
    H = [np.zeros((inner.n_inner, K19.H_DIM)) for _ in pars]
    n = len(prep.inner_ref)
    for s in range(0, n, E23.BLOCK):
        e = min(s + E23.BLOCK, n)
        if noise:
            X = np.random.default_rng([E23.NOISE_SEED, 1, s]).standard_normal(
                (e - s, FW.X_DIM)).astype(np.float32).astype(np.float64)
        else:
            X = FW.standardise_x(np.asarray(prep.inner_X[s:e], dtype=np.float64), mp["mu_x"], mp["sd_x"])
        rows = prep.inner_ref[s:e]
        if hd_std is not None:
            X = np.concatenate([X, hd_std[rows]], axis=1)
        for par, Hs in zip(pars, H):
            Hs[rows] = K19.h_np(par, X)
    di = inner.d["inner"]
    E, lp, has = {}, {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        groups = K19.usable_groups(S, inner.covered)
        E[which], _per, lp[which], has[which] = K19.set_term(pars, H, groups, inner.n_inner)
    return E, lp, has


def hd_inputs(inner, hd, mu, sd, encoding, smoke):
    """[n_inner, 2] standardised HD inputs of every inner row: the decided encoding, the colour the target plays
    (inner keys), the mu / sigma stored at training (PLAN.md 4.1). smoke: the rows are permuted (no real HD
    signal)."""
    keys = inner.d["inner"]["keys"]
    cols = L.hd_columns(hd, L.colour_index(keys["color"]), encoding)
    if len(cols) != inner.n_inner:
        raise L.P31Error("HD inputs: not one row per inner row")
    if smoke:
        cols = cols[np.random.default_rng(E23.NOISE_SEED).permutation(len(cols))]
    return L.hd_standardise(cols, mu, sd)


def fuse(inner, E_q09, E_set):
    return inner.run(list(r.Q08_TERMS) + ["SET"], {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set})


# ---------------------------------------------------------------- seams (PLAN.md 4.2)
def seam_m(tr, smoke, cs, s0a, s0b):
    """The ten S31 models = the train record (SHA, seed, best step, smoke flag, hd_mu / hd_sd / encoding);
    settings registered; training code clean; forward code unchanged; port checks; stage 0a holds, stage 0b GO.
    -> (ten float64 parameter dicts in seed order, {encoding, hd_mu, hd_sd}, record)."""
    tr25 = L.read_json(L29.P25_TRAIN)
    reg = L.registered_settings(K19)
    want = {**reg, "seeds": list(L.SEEDS), "seam_r_seed": L.SEAM_R_SEED, "model_format": L.MODEL_FORMAT,
            "model_format_hd_off": L29.S25_FORMAT, "input_width": L.D_IN_HD, "input_width_hd_off": FW.X_DIM}
    if smoke:
        want.update(steps=K19.SMOKE["steps"], eval_every=K19.SMOKE["eval_every"], hold_per=K19.SMOKE["hold_per"])
    s = tr.get("settings") or {}
    hdr = tr.get("hd") or {}
    out = {"models": {},
           "format": bool(tr.get("format") == L.TRAIN_FORMAT),
           "train_settings_registered": bool(s == want),
           "prereg25_settings_same": bool(all(tr25["settings"].get(k) == v for k, v in reg.items())),
           "settings_not_in_prereg25": sorted(k for k in s if k not in tr25["settings"]),
           "train_code_clean": bool((tr.get("code") or {}).get("dirty") == ""),
           "train_record_checks": L.train_record_checks(tr),
           "stage0a_ok": bool(s0a.get("ok") is True),
           "stage0a_same_as_train_record": bool((tr.get("stage0a") or {}).get("hd_stats_sha256_lf")
                                                == s0a.get("hd_stats_sha256_lf")),
           "encoding": s0a.get("encoding"),
           "stage0b_go": None if smoke else bool(s0b is not None and s0b.get("go") is True)}
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    out["forward_code_same_as_train"] = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f))
                                         for f in FORWARD_FILES}
    ok = bool(out["format"] and out["train_settings_registered"] and out["prereg25_settings_same"]
              and all(out["forward_code_same_as_train"].values()) and out["stage0a_ok"]
              and out["stage0a_same_as_train_record"]
              and (smoke or (out["train_code_clean"] and all(out["train_record_checks"].values())
                             and out["stage0b_go"])))
    model_dir = os.path.join(L.SMOKE_DIR, "models") if smoke else L.MODELS
    pars, metas = [], []
    for sd in L.SEEDS:
        rec = (tr.get("seeds") or {})[str(sd)]
        mpath = os.path.join(REPO, *rec["model_file"].split("/"))
        at_place = bool(os.path.normcase(os.path.abspath(mpath))
                        == os.path.normcase(os.path.abspath(os.path.join(model_dir, L.model_name(sd)))))
        sha = E23.sha_file(mpath)
        par, meta = L.load_model(mpath, K19.PARAM_NAMES)
        rhd = rec.get("hd") or {}
        hd_same = bool(meta["encoding"] == hdr.get("encoding") == rhd.get("encoding") == s0a.get("encoding")
                       and meta["hd_mu"].tolist() == hdr.get("hd_mu") == rhd.get("hd_mu")
                       and meta["hd_sd"].tolist() == hdr.get("hd_sd") == rhd.get("hd_sd"))
        good = bool(at_place and sha == tr["models_sha256"][str(sd)] and sha == rec["model_sha256"]
                    and meta["seed"] == sd and rec.get("seed") == sd and meta["best_step"] == rec["best_step"]
                    and meta["smoke"] == smoke and hd_same and rec.get("input_width") == L.D_IN_HD
                    and (rec.get("port_check") or {}).get("ok") is True and L.finite_params(par))
        out["models"][str(sd)] = {"sha256": sha, "best_step": meta["best_step"], "at_registered_place": at_place,
                                  "hd_same_as_record": hd_same, "ok": good}
        ok = ok and good
        pars.append(K19.p64(par))
        metas.append(meta)
    shas = [m["sha256"] for m in out["models"].values()]
    out["all_ten_distinct"] = bool(len(set(shas)) == len(shas))
    out["ok"] = bool(ok and out["all_ten_distinct"])
    hdmeta = {"encoding": metas[0]["encoding"], "hd_mu": metas[0]["hd_mu"], "hd_sd": metas[0]["hd_sd"]}
    return pars, hdmeta, out


def seam_r_record(tr, smoke):
    """Seam R was judged by p31_train.py before any S31 seed was trained; here its record is re-checked: the HD-off
    retrain of seed 0 on disk = train_record.json = the frozen S25 seed 0 = prereg25 train_record.json."""
    tr25 = L.read_json(L29.P25_TRAIN)
    sr = tr.get("seam_r") or {}
    rpath = os.path.join(REPO, *sr["retrain_file"].split("/")) if sr.get("retrain_file") else None
    rsha = E23.sha_file(rpath) if rpath and os.path.exists(rpath) else None
    frozen0 = E23.sha_file(os.path.join(L29.S25_DIR, "s25_seed0.npz"))
    out = {"train_record_ok": bool(sr.get("ok") is True), "applicable": sr.get("applicable"),
           "retrain_sha256_now": rsha, "s25_seed0_frozen_sha256": frozen0,
           "retrain_is_train_record": bool(rsha is not None and rsha == sr.get("retrain_sha256")),
           "retrain_is_s25_seed0": bool(rsha is not None and rsha == frozen0 == tr25["models_sha256"]["0"])}
    out["ok"] = bool(smoke or (out["train_record_ok"] and out["applicable"] is True
                               and out["retrain_is_train_record"] and out["retrain_is_s25_seed0"]))
    return out


def seam_p(tr, smoke):
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": E23.sha_file(os.path.join(E23.PROBE_PREP, "probe_X.npy")),
           "inner_X.npy": E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))}
    vrec = r.read_json(os.path.join(L29.V_PREP, "prep_record.json"))
    v_sha = E23.sha_file(os.path.join(L29.V_PREP, "v_X.npy"))
    m_sha = E23.sha_file(os.path.join(L29.V_PREP, "v_meta.npz"))
    tsp = tr.get("seam_p") or {}
    out = {"want": {k: want.get(k) for k in got}, "got": got, "v_X.npy": v_sha, "v_meta.npz": m_sha,
           "v_meta_is_prep_record": bool(m_sha == vrec["files_sha256"]["v_meta.npz"]),
           "v_X_is_prep_record": bool(v_sha == vrec["files_sha256"]["v_X.npy"]),
           "v_X_is_train_record": bool(v_sha == tsp.get("v_X_sha256")),
           "v_meta_is_train_record": bool(m_sha == tsp.get("v_meta_sha256")),
           "probe_X_is_train_record": bool(got["probe_X.npy"] == tsp.get("got"))}
    out["ok"] = bool(all(want.get(k) == v for k, v in got.items()) and out["v_X_is_prep_record"]
                     and out["v_meta_is_prep_record"]
                     and (smoke or (out["v_X_is_train_record"] and out["v_meta_is_train_record"]
                                    and out["probe_X_is_train_record"])))
    return out


def seam_d(inner, tr, smoke):
    with np.load(guard_path(os.path.join(E23.PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = set(np.asarray(z["players"]).astype(str).tolist())
    with np.load(guard_path(os.path.join(L29.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vkeys = set(np.asarray(z["players"]).astype(str).tolist())
    allk = pkeys | vkeys
    a_keys = {k for k in allk if L.L25.holdout_bit(k) == 0}
    h_keys = {k for k in allk if L.L25.holdout_bit(k) == 1}
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    i_ids = {p["player_id"] for p in players}
    pid = lambda ks: {k.split(":", 1)[1] for k in ks}  # noqa: E731
    aud = L.read_json(L29.P25_AUDIT)
    out = {"A_players": len(a_keys), "H_players": len(h_keys), "I_players": len(i_keys), "V_players": len(vkeys),
           "A_and_H": len(a_keys & h_keys), "train_and_I": len(allk & i_keys),
           "train_and_I_player_id": len(pid(allk) & i_ids),
           "probe_and_V": len(pkeys & vkeys), "probe_and_V_player_id": len(pid(pkeys) & pid(vkeys)),
           "V_games_vs_inner_8_symmetries_audit": aud["duplicates"]["val_vs_inner_8_symmetries"],
           "train_record_players_match": bool(tr["players"]["A_players"] == len(a_keys)
                                              and tr["players"]["H_players"] == len(h_keys))}
    out["ok"] = bool(out["A_and_H"] == 0 and out["train_and_I"] == 0 and out["train_and_I_player_id"] == 0
                     and out["probe_and_V"] == 0 and out["probe_and_V_player_id"] == 0
                     and out["V_games_vs_inner_8_symmetries_audit"] == 0 and out["train_record_players_match"])
    return out


def seam_a_record(st, s0a, table_sha, table_rows, tr, jrec, n_inner):
    """Seam A of PLAN.md 4.2 as booleans. st: the committed hd_stats.json (the tarball SHA and the row comparison
    of the build); s0a: p31_lib.stage0a(); tr: train_record.json; jrec: the join record of every inner row."""
    tb = (st.get("build") or {}).get("tarball") or {}
    sa = st.get("seam_a") or {}
    tot = sa.get("totals") or {}
    checks = sa.get("checks") or {}
    tsa = tr.get("seam_a") or {}
    tj = tsa.get("join") or {}
    out = {"tarball_sha256": tb.get("sha256"),
           "tarball_is_plan": bool(tb.get("sha256") == L.HD.TARBALL_SHA256 and tb.get("ok") is True),
           "build_checks": checks,
           "rows_v10": tot.get("rows_v10"), "rows_v11": tot.get("rows_v11"), "rows_equal": tot.get("rows_equal"),
           "rows_not_equal": tot.get("rows_not_equal"),
           "rows_all_equal": bool(sa.get("ok") is True and bool(checks) and all(v is True for v in checks.values())
                                  and tot.get("rows_v10") == tot.get("rows_v11") == tot.get("rows_equal")
                                  == N_TRAIN_ROWS == table_rows and tot.get("rows_not_equal") == 0
                                  and tot.get("duplicate_game_ids") == 0),
           "stage0a_ok": bool(s0a.get("ok") is True),
           "table_sha256": table_sha, "table_rows": table_rows,
           "table_is_hd_stats": bool(table_sha is not None and table_sha == s0a.get("table_sha256_hd_stats")
                                     == (st.get("table") or {}).get("sha256")),
           "table_is_train_record": bool(table_sha is not None and table_sha == tsa.get("table_sha256")
                                         == (tr.get("hd") or {}).get("table_sha256")),
           "train_rows_joined": bool(tsa.get("ok") is True and tsa.get("probe_rows") == L.N_PROBE
                                     and tsa.get("v_rows") == L.N_V and tj.get("ok") is True
                                     and tj.get("rows") == L.N_PROBE + L.N_V and tj.get("missing") == 0
                                     and tj.get("player_mismatch") == 0 and tj.get("colour_mismatch") == 0),
           "train_join": tj, "inner_join": jrec, "inner_rows": int(n_inner),
           "inner_rows_joined": bool(jrec.get("ok") is True and jrec.get("rows") == n_inner
                                     and jrec.get("joined") == n_inner and jrec.get("missing") == 0
                                     and jrec.get("player_mismatch") == 0 and jrec.get("colour_mismatch") == 0)}
    out["ok"] = bool(out["tarball_is_plan"] and out["rows_all_equal"] and out["stage0a_ok"]
                     and out["table_is_hd_stats"] and out["table_is_train_record"] and out["train_rows_joined"]
                     and out["inner_rows_joined"])
    return out


def seam_a(inner, tr, s0a):
    """-> (HD of every inner row (int8, by inner row), record)."""
    tab = L.HdTable()
    keys = inner.d["inner"]["keys"]
    hd, jrec = tab.join(keys["rank"], keys["game_id"], keys["player_id"], keys["color"])
    return hd, seam_a_record(L.read_json(L.HD_STATS), s0a, tab.sha256, tab.n, tr, jrec, inner.n_inner)


def seam1_record(seam, ref, hd_off_same):
    """Seam 1: every value equals its reference bit for bit; the references read from the old reports are the
    numbers written in PLAN.md 4.2; set_term of this file with HD off gives the arrays of p23_eval.s19_term."""
    out = {"within_tol": bool(all(abs(x - y) <= TOL for x, y in seam.values())),
           "bitwise": bool(all(x == y for x, y in seam.values())),
           "refs_are_plan": bool(all(ref.get(k) == v for k, v in L.REF_PLAN.items())),
           "hd_off_same_as_s19_term": hd_off_same}
    out["ok"] = bool(out["within_tol"] and out["bitwise"] and out["refs_are_plan"]
                     and bool(hd_off_same) and all(hd_off_same.values()))
    return out


# ---------------------------------------------------------------- gates (PLAN.md 4.3)
def gates(cmp29, cmp25):
    """G1 / G2 / G4 = p23_lib.gates on Q31 - Q29; G5 = lower bound of the paired PlayerBootstrap 95% interval of
    Q31 - Q25 > 0. passed: all four."""
    g = K23.gates(cmp29["inner_diff"], cmp29["inner_ci95"], cmp29["mono_diff"], cmp29["mono_ci95"],
                  cmp29["mono_by_colour"])
    g5 = bool(cmp25["inner_ci95"][0] > 0)
    return {"G1": g["G1"], "G2": g["G2"], "G4": g["G4"], "G1_G2_G4_against": "Q29",
            "G5": {"against": "Q25", "delta": cmp25["inner_diff"], "ci95": cmp25["inner_ci95"], "passed": g5},
            "passed": bool(g["passed"] and g5)}


# ---------------------------------------------------------------- record-only helpers (PLAN.md 4.3)
def question_hd_counts(S, hd):
    """Per question (row of the pooling matrix): how many of its games have HD == 1, and how many HD >= 2.
    Every game the question lists counts (not only the P1-usable ones)."""
    B = (sparse.csr_matrix(S) != 0).astype(np.float64)
    hd = np.asarray(hd)
    n1 = np.asarray(B @ (hd == 1).astype(np.float64)).ravel()
    n2 = np.asarray(B @ (hd >= 2).astype(np.float64)).ravel()
    return np.rint(n1).astype(np.int64), np.rint(n2).astype(np.int64)


def subset_record(sc_a, sc_b, y, player, n_players, mask):
    """p23_eval.subset_diff, or a note when the subset has no question."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"n_questions": 0, "note": "no question in this subset"}
    out = E23.subset_diff(sc_a, sc_b, y, player, n_players, mask)
    out["levels_present"] = int(len(set(np.asarray(y)[mask].tolist())))
    return out


def hd_subsets(inner, cand, base, hd):
    """Q31 - Q29 on the questions with / without an HD[1] game and on the questions without an HD >= 2 game."""
    di, ms = inner.d["inner"], inner.ms
    n_players = int(ms["n_players"])
    out = {}
    for name, S, sa, sb, y, player in (
            ("eval", di["S_eval"], cand["sc"], base["sc"], np.asarray(inner.Xe.y), np.asarray(inner.Xe.player)),
            ("mono", di["S_mono"], cand["msc"], base["msc"], np.asarray(ms["y"]), np.asarray(ms["player"]))):
        n1, n2 = question_hd_counts(S, hd)
        if len(n1) != len(y):
            raise L.P31Error("hd_subsets: pooling matrix and questions of different lengths")
        out[name] = {"with_hd1_game": subset_record(sa, sb, y, player, n_players, n1 > 0),
                     "without_hd1_game": subset_record(sa, sb, y, player, n_players, n1 == 0),
                     "without_hd_ge2_game": subset_record(sa, sb, y, player, n_players, n2 == 0),
                     "with_hd_ge2_game": subset_record(sa, sb, y, player, n_players, n2 > 0)}
    return out


def train_summary(tr):
    return {str(s): {"best_step": v.get("best_step"), "best_holdout_nll": v.get("best_holdout_nll")}
            for s, v in sorted((tr.get("seeds") or {}).items(), key=lambda kv: int(kv[0]))}


def judge(prep, inner, mp, E_q09, q25, q29, pars31, hd, hdmeta, smoke, judged_path, base, tr):
    """Everything after the seams: the Q31 term and fusion, the gates (written to judged_path as soon as they
    exist), then the record-only part. -> dict for the report."""
    enc, mu, sd = hdmeta["encoding"], hdmeta["hd_mu"], hdmeta["hd_sd"]
    hd_std = hd_inputs(inner, hd, mu, sd, enc, smoke)
    E_s31, lp, has = set_term(prep, inner, mp, pars31, hd_std, noise=smoke)
    q31 = fuse(inner, E_q09, E_s31)
    cmp29, cmp25 = inner.compare(q31, q29), inner.compare(q31, q25)
    g = gates(cmp29, cmp25)
    log(f"gates {g}")
    E23.write_json(judged_path, {**base, "created": r.now_iso(), "gates": g, "Q31": [q31["rb"], q31["mono_rb"]],
                                 "Q29": [q29["rb"], q29["mono_rb"]], "Q25": [q25["rb"], q25["mono_rb"]],
                                 "note": "written before the record-only part"})
    # ---- record-only (cannot rescue the candidate)
    chk = {"eval_rb_own": K19.rank_balanced(q31["sc"], inner.Xe.y),
           "mono_rb_own": K19.rank_balanced(q31["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - q31["rb"]) <= TOL and abs(chk["mono_rb_own"] - q31["mono_rb"]) <= TOL)
    curve = {}
    for k in L.SUBSETS:
        if k == len(pars31):
            res = q31
        else:
            E_k, _lp, _has = set_term(prep, inner, mp, pars31[:k], hd_std, noise=smoke)
            res = fuse(inner, E_q09, E_k)
        curve[str(k)] = {"seeds": list(L.subset_seeds(k)), "inner_rank_balanced": res["rb"],
                         "single_colour_rank_balanced": res["mono_rb"], "vs_q29": inner.compare(res, q29),
                         "vs_q25": inner.compare(res, q25), "weights_by_fold": E23.w_record(res, list(TERMS_SET))}
    even = np.tile(L.even_value(mu, sd), (inner.n_inner, 1))
    E_ev, lp_ev, has_ev = set_term(prep, inner, mp, pars31, even, noise=smoke)
    q31e = fuse(inner, E_q09, E_ev)
    rec_only = {
        "own_check": chk,
        "seed_count_curve": curve,
        "s31_alone_rule_b": E23.alone(inner, lp, has),
        "by_n_games": E23.by_count(inner, q31, q29),
        "by_level": E23.by_level(inner, q31, q29),
        "by_colour": {"q31_single_colour": inner.short(q31)["single_colour_by_colour"],
                      "q31_minus_q29": cmp29["mono_by_colour"], "q31_minus_q25": cmp25["mono_by_colour"]},
        "by_hd_in_question": hd_subsets(inner, q31, q29, hd),
        "hd_inputs_set_to_even_game": {
            "even_game_value": [float(x) for x in L.even_value(mu, sd)],
            "inner_rank_balanced": q31e["rb"], "single_colour_rank_balanced": q31e["mono_rb"],
            "s31_alone_rule_b": E23.alone(inner, lp_ev, has_ev),
            "q31_minus_even": inner.compare(q31, q31e), "even_vs_q29": inner.compare(q31e, q29),
            "even_vs_q25": inner.compare(q31e, q25), "weights_by_fold": E23.w_record(q31e, list(TERMS_SET))},
        "weights_by_fold": {"Q31": E23.w_record(q31, list(TERMS_SET)), "Q29": E23.w_record(q29, list(TERMS_SET)),
                            "Q25": E23.w_record(q25, list(TERMS_SET))},
        "training": train_summary(tr)}
    return {"q31": q31, "gates": g, "cmp29": cmp29, "cmp25": cmp25, "record_only": rec_only}


# ---------------------------------------------------------------- eval
def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(L.SMOKE_DIR, "eval_report.json") if smoke else REPORT
    train_path = os.path.join(L.SMOKE_DIR, "train_record.json") if smoke else L.TRAIN_RECORD
    fail_path = os.path.join(L.SMOKE_DIR, "eval_seam_fail.json") if smoke else SEAM_FAIL
    judged_path = os.path.join(L.SMOKE_DIR, "eval_judged.json") if smoke else JUDGED
    cs = code_state(smoke)
    earlier, s0b = None, None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg31 files or the imported tools are not committed and clean: {cs['dirty']}")
        s0b = stage0b_record(train_path)
        if not s0b["go"]:
            raise SystemExit(f"stage 0b is not a GO from this train_record.json: no inner evaluation (PLAN.md 3 / 6): "
                             f"{s0b}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit(f"{train_path}: smoke flag is {tr.get('smoke')!r}, this run is smoke={smoke}")
    s0a = L.stage0a()
    pars31, hdmeta, sm = seam_m(tr, smoke, cs, s0a, s0b)
    srr = seam_r_record(tr, smoke)
    log(f"seam M ok={sm['ok']} seam R record ok={srr['ok']}")
    s19_pars, sf19 = E23.s19_models()
    s29_pars, sf29 = L.s29_models(K19, E23.load_npz_model)       # seed order; the first three are the frozen S25
    s25_pars = s29_pars[:len(L29.S25_SEEDS)]
    map_sha = r.sha_raw(E23.Q09_MAP)
    sf = {"S19": sf19, "S25": sf29["s25"], "S29_seeds_3_to_9": sf29["models"], "q09_map_sha256": map_sha,
          "q09_map_is_frozen": bool(map_sha == r.read_json(E23.Q09_FROZEN)["map"]["sha256"])}
    sf["ok"] = bool(sf19["ok"] and sf29["ok"] and sf["q09_map_is_frozen"] and len(s29_pars) == len(L.SEEDS))
    prep = ft_eval.InnerPrep(guard_path(E23.INNER_PREP))
    inner = r.Inner(prep, log=log)
    ref = {**E23.refs(), **L29.refs25(), **L.refs29()}
    mp = FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    terms = terms9 + ["SET"]
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    E_s19, _lp, _h = E23.s19_term(prep, inner, mp, s19_pars)
    q19 = fuse(inner, E_q09, E_s19)
    # Q25 and Q29 through the SET-term code of this file with HD off (PLAN.md 4.2 seam 1)
    E_s25_ref, _lp, _h = E23.s19_term(prep, inner, mp, s25_pars)
    E_s25, _lp, _h = set_term(prep, inner, mp, s25_pars)
    hd_off_same = {w: bool(np.array_equal(E_s25[w], E_s25_ref[w])) for w in WHICH}
    del E_s25_ref
    q25 = fuse(inner, E_q09, E_s25)
    E_s29, _lp, _h = set_term(prep, inner, mp, s29_pars)
    q29 = fuse(inner, E_q09, E_s29)
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]],
            "Q09": [q09["rb"], ref["Q09"]], "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]],
            "Q19": [q19["rb"], ref["Q19"]], "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]],
            "Q25": [q25["rb"], ref["Q25"]], "Q25_mono": [q25["mono_rb"], ref["Q25_mono"]],
            "Q29": [q29["rb"], ref["Q29"]], "Q29_mono": [q29["mono_rb"], ref["Q29_mono"]]}
    s1 = seam1_record(seam, ref, hd_off_same)
    log(f"seam 1 {seam} ok={s1['ok']}")
    sp = seam_p(tr, smoke)
    sd = seam_d(inner, tr, smoke)
    hd, sa = seam_a(inner, tr, s0a)
    log(f"seam F ok={sf['ok']} P ok={sp['ok']} D ok={sd['ok']} A ok={sa['ok']}")
    seams = {"seam": seam, "seam1": s1, "seam_f": sf, "seam_p": sp, "seam_d": sd, "seam_r": srr, "seam_a": sa,
             "seam_m": sm}
    if not (s1["ok"] and sf["ok"] and sp["ok"] and sd["ok"] and srr["ok"] and sa["ok"] and sm["ok"]):
        E23.write_json(fail_path, {**base, "created": r.now_iso(), "status": "stop: seam failed", **seams})
        log(f"STOP: seam failed -> {fail_path}")
        return 1
    if not smoke:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    j = judge(prep, inner, mp, E_q09, q25, q29, pars31, hd, hdmeta, smoke, judged_path, base, tr)
    q31, g = j["q31"], j["gates"]
    status = "gates passed (worth a PREREG_31; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": "smoke models (noise features, HD columns permuted, 300 steps); the S31 term computed on "
                         "noise features with the HD columns permuted across the inner rows: never a result"
                         if smoke else None,
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md 4: Q31 = Inner.run(Q08_TERMS + [SET]) with SET = 4 * lse mean of the ten S31 models "
                   "(2 standardised HD inputs per game appended after the standardisation); G1 / G2 / G4 against "
                   "Q29 (same ten seeds, no HD), G5 against the incumbent Q25; judged once",
           "status": status, **seams,
           "train_record_sha256_lf": E23.sha_lf(train_path), "stage0b": s0b,
           "hd": {"encoding": hdmeta["encoding"], "hd_mu": [float(x) for x in hdmeta["hd_mu"]],
                  "hd_sd": [float(x) for x in hdmeta["hd_sd"]], "inner_hd_groups": sa["inner_join"]["hd_groups"],
                  "smoke_columns_permuted": smoke},
           "candidate": {"name": "Q31", "terms": terms, "seeds": list(L.SEEDS), **inner.short(q31, q29),
                         "vs_incumbent_q25": j["cmp25"]},
           "control": {"name": "Q29", "terms": terms, "seeds": list(L.SEEDS), **inner.short(q29, q25)},
           "incumbent": {"name": "Q25", "terms": terms, "seeds": list(L29.S25_SEEDS), **inner.short(q25, q19)},
           "gates": g, "record_only": j["record_only"], "q09_map_sha256": map_sha,
           "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"{status} -> {out_path} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--smoke", action="store_true")
    e.add_argument("--rerun", action="store_true")
    e.add_argument("--reason")
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
