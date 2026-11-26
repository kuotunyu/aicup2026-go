"""prereg56 addition 4 (H9, written before running): is a test game literally a training game recorded under the
OTHER player (same moves, colour-agnostic)?  Exact move-sequence match, also on the first 60 moves.  CPU."""
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(R, "results", "prereg49", "explore"))
import p49_run  # noqa: E402,F401  (sys.path of the older modules)
import p49_lib as P49  # noqa: E402
import p36_lib as L  # noqa: E402

csv.field_size_limit(1 << 30)


def h(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).digest()


def main():
    cache = P49.Cache49()
    codes, off = cache.codes, cache.off
    full, pre = {}, {}
    n = cache.n
    for i in range(n):
        c = np.asarray(codes[off[i]:off[i + 1]])
        full.setdefault(h(c), []).append(i)
        if len(c) >= 60:
            pre.setdefault(h(c[:60]), []).append(i)
    res = {"training_games": int(n), "distinct_full": len(full), "distinct_prefix60": len(pre),
           "training_internal_dup_groups": int(sum(1 for v in full.values() if len(v) > 1))}
    pid, col, rk = np.asarray(cache.pid), np.asarray(cache.color), np.asarray(cache.rank)

    def scan(name, games):
        hit_f = hit_p = same_col = 0
        ex = []
        for sgf, colour in games:
            c = L.moves_of(sgf)
            f = full.get(h(c))
            p = pre.get(h(c[:60])) if len(c) >= 60 else None
            hit_f += f is not None
            hit_p += p is not None
            if f is not None:
                same_col += any(str(col[j]).upper() == colour.upper() for j in f)
                if len(ex) < 5:
                    ex.append({"test_colour": colour, "train": [(str(pid[j]), str(col[j]), str(rk[j])) for j in f]})
        res[name] = {"games": len(games), "full_match": int(hit_f), "prefix60_match": int(hit_p),
                     "full_match_same_colour": int(same_col), "examples": ex}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "examples"}), ex[:2], flush=True)

    for sp in ("public", "private"):
        rows = list(csv.DictReader(open(os.path.join(R, "data", "test_v2_download", "rank",
                                                     "rank_prediction_test_%s.csv" % sp), encoding="utf-8", newline="")))
        scan("rank_%s" % sp, [(r["sgf_%d" % i], r["color_%d" % i]) for r in rows
                              for i in range(1, int(r["num_games"]) + 1)])
        T = os.path.join(R, "data", "test_v2_download", "x", "tests")
        for tag in ("test2", "test"):
            rows = list(csv.DictReader(open(os.path.join(T, "player_identification_%s_%s.csv" % (tag, sp)),
                                            encoding="utf-8", newline="")))
            scan("x_%s_%s" % (tag, sp), [(r["sgf_%d" % i], r["color_%d" % i]) for r in rows
                                         for i in range(1, int(r["num_games"]) + 1)])
        rows = list(csv.DictReader(open(os.path.join(T, "player_identification_candidates2_%s.csv" % sp),
                                        encoding="utf-8", newline="")))
        scan("x_candidates2_%s" % sp, [(r["sgf_content"], r["color"]) for r in rows])
    with open(os.path.join(HERE, "dup.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: res[k] for k in ("training_games", "distinct_full", "distinct_prefix60",
                                          "training_internal_dup_groups")}))


if __name__ == "__main__":
    main()
