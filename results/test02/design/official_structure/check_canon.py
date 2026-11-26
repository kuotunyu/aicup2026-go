"""Sanity: how often do canon6 (explore) and p02_moves.canon(mv, 10) give different move lists?"""
import sys
sys.dont_write_bytecode = True
import os
import re
import random

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows  # noqa: E402
import p02_moves  # noqa: E402

MV = re.compile(r";([BW])\[([a-s]{2})\]")


def tf(i, x, y):
    if i & 4:
        x, y = y, x
    if i & 1:
        x = 18 - x
    if i & 2:
        y = 18 - y
    return x, y


def enc(x, y):
    return chr(97 + x) + chr(97 + y)


def canon6(mv, K=6):
    best = None
    for i in range(8):
        s = "".join(enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97)) for _, m in mv[:K])
        if best is None or s < best[0]:
            best = (s, i)
    i = best[1]
    return [(c, enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97))) for c, m in mv]


rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
random.Random(0).shuffle(rows)
n = diff_all = diff20 = 0
for r in rows[:100000]:
    mv = MV.findall(r["sgf_content"][:2000])[:60]
    a, b = canon6(mv), p02_moves.canon(mv, 10)
    n += 1
    diff_all += a != b
    diff20 += a[:20] != b[:20]
print("games %d  canon6 != canon10: whole %d, first 20 moves %d" % (n, diff_all, diff20))
