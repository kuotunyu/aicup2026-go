"""POST-HOC diagnostic of the PILOT (NOT in pilot_variants.json, does not change any criterion): why does
self-identification drop on the single-colour questions? Checks that this pipeline reproduces D2's self-identification
on D2's mixed-colour eval questions (so the drop is the question format, not a bug), and splits the single-colour
self-identification by colour.

    .venv/Scripts/python.exe results/prereg05/explore/pilot_D2/pilot_diag_mixed.py
    -> results/prereg05/explore/pilot_D2/pilot_diag_mixed.json

Needs _tmp/work_raw.npz and _tmp/common.npz, which pilot_probe.py writes (stage 1, about 1 minute) and which are
deleted after the run; re-run pilot_probe.py first to recreate them. Raw orientation only; 100 inner players with
>= 5 non-bank games sampled with numpy default_rng(7); references = the PILOT's 6,403 profiles (other inner players
full), own column replaced by (A) D2's own profile (non-bank games) or (B) the PILOT's (all inner games minus the
question's games). Inner pseudo-questions only; nothing scored; data/local never opened (u1_lib guard).
"""

import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "u1"))

import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

import u1_lib as U  # noqa: E402

TMP = os.path.join(HERE, "_tmp")
NFIT = 5767
KS = (5, 10, 15, 20)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    z = U.load_saved()
    a = np.load(os.path.join(HERE, "pilot_arrays.npz"))
    w = np.load(os.path.join(TMP, "work_raw.npz"))
    P = sparse.csc_matrix((w["P_data"], w["P_indices"], w["P_indptr"]), shape=tuple(w["P_shape"]))
    G = sparse.csr_matrix((w["G_data"], w["G_indices"], w["G_indptr"]), shape=tuple(w["G_shape"]))
    idf = w["idf"]
    c = np.load(os.path.join(TMP, "common.npz"))
    pg_ptr, pg_rows = c["pg_ptr"], c["pg_rows"]
    eg, ep, en = z["eval_games"], z["eval_player"], z["eval_n"]
    mrank = a["own_rank_raw"]
    res = {"A_eval_mixed_own_nonbank_as_D2": {k: [] for k in KS}, "B_eval_mixed_own_all_minus_question": {k: [] for k in KS},
           "C_mono_same_players_own_all_minus_question": {k: [] for k in KS}}
    rng = np.random.default_rng(7)
    cands = []
    for pi in range(636):
        ga = pg_rows[pg_ptr[pi]:pg_ptr[pi + 1]]
        qe = np.nonzero(ep == pi)[0]
        bank = np.unique(eg[qe][eg[qe] >= 0])
        if len(np.setdiff1d(ga, bank)) >= 5 and len(qe):
            cands.append(pi)
    pick = rng.choice(cands, size=100, replace=False)
    for pi in pick:
        ga = pg_rows[pg_ptr[pi]:pg_ptr[pi + 1]]
        qe = np.nonzero(ep == pi)[0]
        bank = np.unique(eg[qe][eg[qe] >= 0])
        nb = np.setdiff1d(ga, bank)
        cols = np.unique(G[ga].indices)
        idc = idf[cols]
        full = np.asarray(G[ga][:, cols].sum(0)).ravel()
        nbc = np.asarray(G[nb][:, cols].sum(0)).ravel()
        own_nb = np.where(nbc > 0, 1 + np.log(np.maximum(nbc, 1)), 0) * idc
        own_nb /= np.sqrt((own_nb ** 2).sum())
        Cq = np.stack([np.asarray(G[eg[q][eg[q] >= 0]][:, cols].sum(0)).ravel() for q in qe])
        Wq = np.where(Cq > 0, 1 + np.log(np.maximum(Cq, 1)), 0) * idc
        Wq /= np.sqrt((Wq ** 2).sum(1))[:, None]
        Own = full[None, :] - Cq
        Wo = np.where(Own > 0, 1 + np.log(np.maximum(Own, 1)), 0) * idc
        Wo /= np.sqrt((Wo ** 2).sum(1))[:, None]
        qm = (Cq > 0).any(0)
        Sm = np.asarray(P[:, cols[qm]] @ Wq[:, qm].astype(np.float32).T).T.astype(np.float64)
        own = NFIT + pi
        for name, oc in (("A_eval_mixed_own_nonbank_as_D2", Wq @ own_nb),
                         ("B_eval_mixed_own_all_minus_question", (Wq * Wo).sum(1))):
            S = Sm.copy()
            S[:, own] = oc
            rk = (S > oc[:, None]).sum(1)
            for k in KS:
                m = en[qe] == k
                res[name][k].extend((rk[m] == 0).tolist())
        mq = np.nonzero(z["mono_player"] == pi)[0]
        for k in KS:
            m = (z["mono_n"][mq] == k) & (mrank[mq] >= 0)
            res["C_mono_same_players_own_all_minus_question"][k].extend((mrank[mq][m] == 0).tolist())
    out = {"what": "POST-HOC diagnostic, not in pilot_variants.json, no criterion changed: own-profile-is-top-1 rate "
                   "(raw orientation, no threshold) on 100 sampled inner players",
           "players": [int(p) for p in pick],
           "self_top1_by_k": {name: {str(k): {"rate": float(np.mean(v)), "questions": len(v)} for k, v in d.items()}
                              for name, d in res.items()},
           "self_top1_all": {name: float(np.mean(sum(d.values(), []))) for name, d in res.items()},
           "d2_reference": "d2_report.json lookup_diagnostics pi1 main top1_is_self_by_n: 5 0.290, 10 0.525, 15 0.619, "
                           "20 0.671 (both orientations); own_profile_rank_pi1 raw own_is_top1_main 0.546"}
    col, n = z["mono_colour"], z["mono_n"]
    out["mono_all_players_self_top1_by_colour_and_k"] = {}
    for d in ("raw", "canon"):
        rk = a["own_rank_" + d]
        has = rk >= 0
        out["mono_all_players_self_top1_by_colour_and_k"][d] = {
            cc: {str(k): float((rk[has & (col == cc) & (n == k)] == 0).mean()) for k in KS} for cc in ("B", "W")}
    t = a["trig_pi1"]
    out["mono_pi1_trigger_rate_by_colour_and_k"] = {cc: {str(k): float(t[(col == cc) & (n == k)].mean()) for k in KS}
                                                    for cc in ("B", "W")}
    out["inputs_sha256"] = {"pilot_arrays.npz": sha(os.path.join(HERE, "pilot_arrays.npz")),
                            "pilot_variants.json": sha(os.path.join(HERE, "pilot_variants.json"))}
    with open(os.path.join(HERE, "pilot_diag_mixed.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["self_top1_all"]), json.dumps(out["mono_pi1_trigger_rate_by_colour_and_k"]))


if __name__ == "__main__":
    main()
