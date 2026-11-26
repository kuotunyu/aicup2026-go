"""prereg35 explore (competition 1, v2): apply the chosen combination to the two official new pools and assemble the
1,600-row submission (PLAN.md 4).

    PY results/prereg35/explore/p35_apply.py apply --src data/test_v2_download/x/tests --pool public|private
         [--eval results/prereg35/explore/eval.json] -> results/prereg35/apply/<pool>/top5.csv + apply_report.json
    PY results/prereg35/explore/p35_apply.py assemble --v1 results/test06/player/on/submission_player.csv
         --src data/test_v2_download/x/tests -> results/prereg35/apply/submission_player.csv + assemble_report.json

The per-game features are computed by tools/p06_player._feat_chunk (the function that built the fit feature
store); the score terms by tools/p06_sim.pool_terms on a duck-typed store of the pool's games, so the official pool
goes through exactly the code of the simulation. LDA: one fit per colour on the whole 06 LDA training set (the
official players are not in it).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import p06_player as pp  # noqa: E402
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
import p35_sim as ps  # noqa: E402,F401  (HERE is on sys.path as the script dir)

FEATS = os.path.join(REPO, "data", "prereg06", "feats")
OUT = os.path.join(REPO, "results", "prereg35", "apply")
PREFIX = {"public": "pub2_", "private": "priv2_"}
csv.field_size_limit(10 ** 9)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PoolStore:
    """The pool's games in the shape p06_sim.pool_terms reads: color, alt, csr(oraw/jo/resp), dense(s2/s34);
    resp columns aligned to the fit store's resp vocabulary (tokens outside it dropped)."""

    def __init__(self, items, train_resp_vocab):
        S2, S34, alt, nm, blocks = pp._feat_chunk(items)
        self.color = np.array([c for _s, c in items])
        self.alt = alt
        self.nmov = nm
        self.n = len(items)
        self.dense_m = {"s2": S2, "s34": S34}
        self.csr_m = {}
        for name in ("oraw", "jo"):
            M, vocab = pp.merge_blocks([blocks[name]])
            self.csr_m[name] = M
        voc, dat, ind, ptr = blocks["resp"]
        pos = {t: i for i, t in enumerate(train_resp_vocab)}
        m = np.array([pos.get(t, -1) for t in voc], np.int64)
        ind2 = m[ind] if len(ind) else ind
        keep = ind2 >= 0
        rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
        self.csr_m["resp"] = sp.csr_matrix((dat[keep], (rows[keep], ind2[keep])),
                                           shape=(len(ptr) - 1, len(train_resp_vocab))).tocsr()

    def csr(self, name):
        return self.csr_m[name]

    def dense(self, name):
        return self.dense_m[name]


class PoolLda:
    def __init__(self, store, cols):
        self.store, self.cols = store, cols

    def project(self, rows, model):
        r = np.asarray(rows, np.int64)
        X = st.ldain_matrix(np.asarray(self.store.dense("s2")[r]), self.store.csr("resp")[r], self.cols)
        return st.lda_project(np.asarray(X, np.float64), model)


def read_pool(src, pool):
    cands = list(csv.DictReader(open(os.path.join(src, "player_identification_candidates2_%s.csv" % pool),
                                     encoding="utf-8", newline="")))
    qs = list(csv.DictReader(open(os.path.join(src, "player_identification_test2_%s.csv" % pool),
                                  encoding="utf-8", newline="")))
    return cands, qs


def cmd_apply(a):
    t0 = time.time()
    with open(a.eval, encoding="utf-8") as f:
        ev = json.load(f)
    w, k = tuple(ev["chosen_weights"]), ev["chosen_csls_k"]
    cands, qs = read_pool(a.src, a.pool)
    pre = PREFIX[a.pool]
    cand_ids = sorted({r["player_id"] for r in cands})
    if not all(c.startswith(pre + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    items, cand_rows = [], {c: [] for c in cand_ids}
    for r in cands:
        cand_rows[r["player_id"]].append(len(items))
        items.append((r["sgf_content"], r["color"]))
    q_rows, qids = [], []
    for q in qs:
        n = int(q["num_games"])
        g = []
        for i in range(1, n + 1):
            g.append(len(items))
            items.append((q["sgf_%d" % i], q["color_%d" % i]))
        q_rows.append(g)
        qids.append(q["question_id"])
    fit = sim.Store(FEATS, load=())
    train_resp = fit.csr("resp")
    store = PoolStore(items, train_resp.vocab)
    ldas, models = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(FEATS, c, fit)
        models[c] = ld.fit_excluding(fit, set())
        ldas[c] = PoolLda(store, ld.cols)
    rec = {"cand_rows": [cand_rows[c] for c in cand_ids], "q_rows": q_rows}
    T = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
    F = st.fuse([T[n] for n in ps.TERMS], list(w))
    S = st.csls(F, k)
    tops = st.top5(S, cand_ids)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, "top5.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
        for q, t in zip(qids, tops):
            wr.writerow([q] + t)
    rep = {"format": "c10-prereg35 apply v1", "pool": a.pool, "weights_O_L_JO_S2_PA": list(w), "csls_k": k,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "not_alternating": int((~store.alt).sum()), "no_move": int((store.nmov == 0).sum()),
           "terms_rows_without_signal": {n: int((~np.asarray(T[n][1])).sum()) for n in ps.TERMS},
           "lda_train": {c: [models[c]["n_players"], models[c]["n_games"]] for c in models},
           "eval_sha256": sha256_file(a.eval), "top5_sha256": sha256_file(path),
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(od, "apply_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    rows = list(csv.reader(open(a.v1, encoding="utf-8", newline="")))
    head, v1 = rows[0], rows[1:]
    out = [r for r in v1 if r[0].startswith(("pub_", "priv_"))]
    for pool in ("public", "private"):
        out += list(csv.reader(open(os.path.join(a.out, pool, "top5.csv"), encoding="utf-8", newline="")))[1:]
    # checks (PLAN.md 4)
    problems = []
    want = {}
    for name, pre in (("test_public", "pub_"), ("test_private", "priv_"), ("test2_public", "pub2_"),
                      ("test2_private", "priv2_")):
        q = [r["question_id"] for r in csv.DictReader(open(os.path.join(a.src, "player_identification_%s.csv" % name),
                                                             encoding="utf-8", newline=""))]
        cf = name.replace("test", "candidates")
        c = {r["player_id"] for r in csv.DictReader(open(os.path.join(a.src, "player_identification_%s.csv" % cf),
                                                          encoding="utf-8", newline=""))}
        for x in q:
            want[x] = (pre, c)
    seen = set()
    for r in out:
        if len(r) != 6:
            problems.append("row width %s" % r[:1])
            continue
        q, t = r[0], r[1:]
        if q in seen:
            problems.append("duplicate %s" % q)
        seen.add(q)
        if q not in want:
            problems.append("unknown %s" % q)
            continue
        pre, c = want[q]
        if len(set(t)) != 5 or any(not x.startswith(pre + "p_") or x not in c for x in t):
            problems.append("bad top5 %s" % q)
    missing = set(want) - seen
    if missing:
        problems.append("missing %d" % len(missing))
    if head != ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]:
        problems.append("header")
    path = os.path.join(a.out, "submission_player.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(head)
        wr.writerows(out)
    rep = {"rows": len(out), "problems": problems, "ok": not problems, "v1_file": a.v1,
           "v1_sha256": sha256_file(a.v1), "v1_rows_kept": sum(1 for r in out if r[0].startswith(("pub_", "priv_"))),
           "sha256": sha256_file(path), "by_prefix": {p: sum(1 for r in out if r[0].startswith(p))
                                                      for p in ("pub_", "priv_", "pub2_", "priv2_")}}
    with open(os.path.join(a.out, "assemble_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    p.add_argument("--out", default=OUT)
    s = sub.add_parser("assemble")
    s.add_argument("--v1", required=True)
    s.add_argument("--src", required=True)
    s.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    return {"apply": cmd_apply, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
