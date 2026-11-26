"""PREREG_02 competition 2: level-conditional opening n-gram (PREREG_02 1.2, plan.md 4.2).

    python tools/p02_rank_opening.py build [--out-dir results/prereg02/rank]
                                           [--questions data/local/rank_val_questions.csv] [--smoke]

Counts come from load_rows(part="fit") ONLY (tools/c10data.py, leak-checked).
The model scores
    every inner game of load_rows(part="inner")      -> <out-dir>/opening_inner.npz
    every game slot of the rank question file         -> <out-dir>/opening_val.npz
No label file is read; the question file has no labels.

Model (fixed in PREREG_02; M and alpha were chosen on inner in results/prereg02/explore/)
  * a game = its first M = 12 moves (tools/p02_moves.parse_moves), in the canonical
    orientation canon(mv, 12) (the D4 transform making the 12 points smallest);
  * context = (target player's colour, canonical prefix); every move of both colours
    is predicted: for L = 0..len-1, with pre = the first L points and mv the next one,
        ll_k += log((N_k(c,pre,mv) + alpha*pg) / (N_k(c,pre) + alpha)),
        pg    = (N(c,pre,mv) + 1) / (N(c,pre) + 361),   alpha = 20,
    N_k = number of fit games of level k with that prefix (N_k(c,pre) counts the fit
    games that have the prefix and at least one more move), N = sum over the levels;
  * stop at the first context never seen in fit (no more terms for that game);
  * o = log_softmax(ll) over the 10 levels; a game with no term has has=False
    (o is stored as 0 and must not be used).
Only the (colour, prefix) keys that occur in the scored games are counted; their
counts are the same integers as in the full fit table (keys are prefix-closed), as in
results/prereg02/explore/opening_rank_ngram.py.

Arrays in opening_inner.npz: rank, game_id, player_id, color (str), ll, o (float64
[N,10]), has (bool), n_terms, n_moves (int16), meta (JSON string).
opening_val.npz: question_id, slot (1-based), color, ll, o, has, n_terms, n_moves, meta.
meta holds M, alpha, the input SHA-256s and a build_id shared by the two files.

--smoke: the output directory must be outside the repository (the model and data are
the same as the real run; smoke only keeps the outputs out of results/).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c10data import EXPECTED, LEVELS, load_rows  # noqa: E402
from p02_moves import canon, parse_moves, points_string  # noqa: E402
from p02_rank_pseudo import (DEFAULTS, P02Error, check_out_dir, load_npz, now_iso,  # noqa: E402
                             refuse_label_file, sha256_file, write_json)

M = 12
ALPHA = 20.0
N_POINTS = 361
LI = {lv: i for i, lv in enumerate(LEVELS)}
QUESTIONS_DEFAULT = os.path.join(os.path.dirname(HERE), "data", "local", "rank_val_questions.csv")
MAX_SLOTS = 20


def game_string(sgf, m=M):
    """Canonical points of the first m moves as one string ('pddp...', 2 letters per move)."""
    mv = parse_moves(sgf, limit=m)
    return points_string(canon(mv, m))


def norm_colour(c):
    if c not in ("B", "W"):
        raise P02Error(f"colour must be 'B' or 'W', got {c!r}")
    return c


class OpeningModel:
    """Counts of the fit games on the keys the scored games can ask for."""

    def __init__(self, scored, m=M):
        """scored: iterable of (colour, canonical string) of the games that will be scored."""
        self.m = m
        self.ctx_index, self.cnt_index = {}, {}
        for c, s in scored:
            for L in range(len(s) // 2):
                self.ctx_index.setdefault(c + s[:2 * L], len(self.ctx_index))
                self.cnt_index.setdefault(c + s[:2 * L + 2], len(self.cnt_index))
        self.n_fit = 0
        self.fit_by_level = [0] * 10

    def count(self, fit_games):
        """fit_games: iterable of (colour, canonical string, level index)."""
        ctx_hits, cnt_hits = [], []
        ci, mi = self.ctx_index, self.cnt_index
        for c, s, lv in fit_games:
            self.n_fit += 1
            self.fit_by_level[lv] += 1
            for L in range(len(s) // 2):
                i = ci.get(c + s[:2 * L])
                if i is None:
                    break  # keys are prefix-closed: no longer prefix can be a key either
                ctx_hits.append(i * 10 + lv)
                j = mi.get(c + s[:2 * L + 2])
                if j is not None:
                    cnt_hits.append(j * 10 + lv)
        self.ctx = np.bincount(np.asarray(ctx_hits, dtype=np.int64),
                               minlength=10 * len(ci)).reshape(len(ci), 10).astype(np.float64)
        self.cnt = np.bincount(np.asarray(cnt_hits, dtype=np.int64),
                               minlength=10 * len(mi)).reshape(len(mi), 10).astype(np.float64)
        self.ctx_tot = self.ctx.sum(1)
        self.cnt_tot = self.cnt.sum(1)

    def score(self, games, alpha=ALPHA):
        """games: list of (colour, canonical string) -> ll [G,10], n_terms [G]."""
        G = len(games)
        ll = np.zeros((G, 10))
        n_terms = np.zeros(G, dtype=np.int64)
        if not self.ctx_index:
            return ll, n_terms
        ctx_idx = np.full((G, self.m), -1, dtype=np.int64)
        cnt_idx = np.full((G, self.m), -1, dtype=np.int64)
        for g, (c, s) in enumerate(games):
            for L in range(len(s) // 2):
                ctx_idx[g, L] = self.ctx_index[c + s[:2 * L]]
                cnt_idx[g, L] = self.cnt_index[c + s[:2 * L + 2]]
        alive = np.ones(G, dtype=bool)
        for L in range(self.m):
            i, j = ctx_idx[:, L], cnt_idx[:, L]
            alive &= i >= 0
            alive &= np.where(i >= 0, self.ctx_tot[np.maximum(i, 0)] > 0, False)  # unseen context: stop
            if not alive.any():
                break
            rows = np.nonzero(alive)[0]
            cx = self.ctx[i[rows]]
            cm = self.cnt[j[rows]]
            pg = (self.cnt_tot[j[rows]] + 1.0) / (self.ctx_tot[i[rows]] + float(N_POINTS))
            ll[rows] += np.log((cm + alpha * pg[:, None]) / (cx + alpha))
            n_terms[rows] += 1
        return ll, n_terms


def log_softmax(x):
    mx = x.max(axis=1, keepdims=True)
    z = x - mx
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def read_questions(path):
    """[(question_id, slot, colour, sgf)] in file order, slots 1..num_games."""
    refuse_label_file(path)
    csv.field_size_limit(10**8)
    out = []
    qids = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qid = r["question_id"]
            if qid in qids:
                raise P02Error(f"{path}: duplicate question_id {qid}")
            qids.add(qid)
            n = int(r["num_games"])
            if not 1 <= n <= MAX_SLOTS:
                raise P02Error(f"{path}: {qid} num_games {n}")
            for i in range(1, n + 1):
                sgf = r[f"sgf_{i}"]
                if not sgf.strip():
                    raise P02Error(f"{path}: {qid} slot {i} is empty but num_games is {n}")
                out.append((qid, i, norm_colour(r[f"color_{i}"].strip()), sgf))
    return out


def build(train_dir, split, local_dir, questions_path, log=print):
    t0 = time.time()
    rows = load_rows(train_dir, split, part="inner", local_dir=local_dir)
    inner = [(r["rank"], r["game_id"], r["player_id"], norm_colour(r["color"]),
              game_string(r["sgf_content"])) for r in rows]
    del rows
    log(f"inner: {len(inner)} games ({time.time() - t0:.0f}s)")
    val = [(q, s, c, game_string(sgf)) for q, s, c, sgf in read_questions(questions_path)]
    log(f"val slots: {len(val)} ({time.time() - t0:.0f}s)")

    model = OpeningModel([(g[3], g[4]) for g in inner] + [(v[2], v[3]) for v in val])
    rows = load_rows(train_dir, split, part="fit", local_dir=local_dir)
    fit = [(norm_colour(r["color"]), game_string(r["sgf_content"]), LI[r["rank"]]) for r in rows]
    del rows
    log(f"fit: {len(fit)} games parsed ({time.time() - t0:.0f}s)")
    model.count(fit)
    fit_moves = Counter(len(s) // 2 for _, s, _ in fit)
    del fit
    log(f"counted: {len(model.ctx_index)} contexts, {len(model.cnt_index)} entries "
        f"({time.time() - t0:.0f}s)")

    ll_i, nt_i = model.score([(g[3], g[4]) for g in inner])
    ll_v, nt_v = model.score([(v[2], v[3]) for v in val])
    return inner, val, model, (ll_i, nt_i), (ll_v, nt_v), fit_moves, time.time() - t0


def _save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_outputs(out_dir, inner, val, model, scored_i, scored_v, fit_moves, inputs, seconds):
    ll_i, nt_i = scored_i
    ll_v, nt_v = scored_v
    has_i, has_v = nt_i > 0, nt_v > 0
    o_i = np.where(has_i[:, None], log_softmax(ll_i), 0.0)
    o_v = np.where(has_v[:, None], log_softmax(ll_v), 0.0)
    params = {"M": M, "alpha": ALPHA, "n_points": N_POINTS, "levels": LEVELS,
              "canonical": "p02_moves.canon(mv, 12): D4 transform making the first 12 points smallest, "
                           "ties -> smallest symmetry index",
              "counted_on": "load_rows(part='fit') only"}
    # one id for the pair of files: apply checks opening_val.npz came from the same build
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(ll_i.tobytes())
    h.update(ll_v.tobytes())
    build_id = h.hexdigest()
    meta = {"what": "PREREG_02 competition 2 opening n-gram (tools/p02_rank_opening.py)",
            "params": params, "build_id": build_id, "created": now_iso(),
            "fit_games": model.n_fit, "fit_games_by_level": dict(zip(LEVELS, model.fit_by_level)),
            "inputs": inputs,
            "tool_sha256": {n: sha256_file(os.path.join(HERE, n))
                            for n in ("p02_rank_opening.py", "p02_moves.py", "c10data.py")}}
    _save_npz(os.path.join(out_dir, "opening_inner.npz"),
              rank=np.array([g[0] for g in inner]), game_id=np.array([g[1] for g in inner]),
              player_id=np.array([g[2] for g in inner]), color=np.array([g[3] for g in inner]),
              ll=ll_i, o=o_i, has=has_i, n_terms=nt_i.astype(np.int16),
              n_moves=np.array([len(g[4]) // 2 for g in inner], dtype=np.int16),
              meta=np.array(json.dumps(dict(meta, part="inner"), ensure_ascii=False)))
    _save_npz(os.path.join(out_dir, "opening_val.npz"),
              question_id=np.array([v[0] for v in val]), slot=np.array([v[1] for v in val], dtype=np.int16),
              color=np.array([v[2] for v in val]),
              ll=ll_v, o=o_v, has=has_v, n_terms=nt_v.astype(np.int16),
              n_moves=np.array([len(v[3]) // 2 for v in val], dtype=np.int16),
              meta=np.array(json.dumps(dict(meta, part="val"), ensure_ascii=False)))

    # summary (inner labels are training data; per-game accuracy of the opening alone)
    lab = np.array([LI[g[0]] for g in inner])
    acc = float((o_i[has_i].argmax(1) == lab[has_i]).mean()) if has_i.any() else None
    report = {
        "what": meta["what"], "params": params, "build_id": build_id, "created": meta["created"],
        "seconds": round(seconds, 1), "inputs": inputs,
        "fit_games": model.n_fit, "fit_games_by_level": meta["fit_games_by_level"],
        "fit_games_by_number_of_moves_kept": {str(k): v for k, v in sorted(fit_moves.items())},
        "keys_counted": {"contexts": len(model.ctx_index), "entries": len(model.cnt_index),
                         "contexts_seen_in_fit": int((model.ctx_tot > 0).sum()),
                         "entries_seen_in_fit": int((model.cnt_tot > 0).sum())},
        "inner": {"games": len(inner), "with_score": int(has_i.sum()),
                  "mean_terms": float(nt_i.mean()),
                  "terms_histogram": {str(k): int(v) for k, v in zip(*np.unique(nt_i, return_counts=True))},
                  "per_game_argmax_accuracy": acc},
        "val": {"slots": len(val), "questions": len({v[0] for v in val}), "with_score": int(has_v.sum()),
                "mean_terms": float(nt_v.mean()),
                "terms_histogram": {str(k): int(v) for k, v in zip(*np.unique(nt_v, return_counts=True))}},
        "outputs_sha256": {n: sha256_file(os.path.join(out_dir, n))
                           for n in ("opening_inner.npz", "opening_val.npz")},
    }
    write_json(os.path.join(out_dir, "opening_report.json"), report)
    return report


def load_opening(path, part):
    """Read opening_inner.npz / opening_val.npz (used by p02_rank_fuse.py)."""
    keys = ["rank", "game_id", "player_id"] if part == "inner" else ["question_id", "slot"]
    z = load_npz(path, keys + ["color", "o", "has", "n_terms", "meta"])
    meta = json.loads(str(z["meta"]))
    if meta.get("part") != part:
        raise P02Error(f"{path}: meta part is {meta.get('part')!r}, expected {part!r}")
    if meta["params"]["M"] != M or meta["params"]["alpha"] != ALPHA:
        raise P02Error(f"{path}: built with M={meta['params']['M']}, alpha={meta['params']['alpha']}")
    z["meta"] = meta
    z["sha256"] = sha256_file(path)
    return z


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--train-dir", default=DEFAULTS["train_dir"])
    ap.add_argument("--split", default=DEFAULTS["split"])
    ap.add_argument("--local-dir", default=DEFAULTS["local_dir"])
    ap.add_argument("--questions", default=QUESTIONS_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    try:
        check_out_dir(a.out_dir, a.smoke)
        os.makedirs(a.out_dir, exist_ok=True)
        inner, val, model, si, sv, fit_moves, secs = build(a.train_dir, a.split, a.local_dir, a.questions,
                                                            log=lambda s: print(s, flush=True))
        if len(inner) != EXPECTED["inner"]["games"] or model.n_fit != EXPECTED["fit"]["games"]:
            raise P02Error(f"unexpected counts: inner {len(inner)}, fit {model.n_fit}")
        inputs = {"players_split_sha256": sha256_file(a.split),
                  "questions": os.path.abspath(a.questions), "questions_sha256": sha256_file(a.questions)}
        rep = write_outputs(a.out_dir, inner, val, model, si, sv, fit_moves, inputs, secs)
    except P02Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"inner games {rep['inner']['games']} (with score {rep['inner']['with_score']}), "
          f"val slots {rep['val']['slots']} (with score {rep['val']['with_score']})")
    print(f"per-game argmax accuracy of the opening alone on inner: {rep['inner']['per_game_argmax_accuracy']}")
    print(f"build_id {rep['build_id']}; wrote opening_inner.npz, opening_val.npz, opening_report.json "
          f"in {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
