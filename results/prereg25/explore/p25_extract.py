"""prereg25 (GPU): results/prereg08/p1/p1_extract.py, unchanged, with one extra game source (PLAN.md 2.3).

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; GPU python .venv-gpu/Scripts/python.exe
    results/prereg25/explore/p25_extract.py run --set probe --sets <sets.json> --out-dir <dir> --workers 12 [...]

Every argument goes to p1_extract.main. The only change: p1_extract.set_games is wrapped so that a sets.json whose
probe has source 'val' takes its games from p25_lib.load_val_rows (the split v1 validation players' games in the official
training CSVs; player and colour checked against sets.json, ref = index in the probe list). Any other source ('fit',
'inner') goes to the original function unchanged (seam G re-runs the P3 probe through this same entry point).
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p25_lib as L  # noqa: E402  (puts tools/ and results/prereg08/p1 on sys.path)
import p1_extract as E  # noqa: E402

_ORIGINAL = E.set_games


def set_games(sets, name):
    s = sets[name]
    if s.get("source") != L.SOURCE:
        return _ORIGINAL(sets, name)
    rows = {(r["rank"], r["game_id"]): r for r in L.load_val_rows()}
    out = []
    for j, g in enumerate(s["games"]):
        r = rows[(g["rank"], g["game_id"])]
        if r["player_id"] != g["player_id"] or r["color"] != g["color"]:
            raise E.P.P1Error(f"V game {j}: player or colour differs from sets.json")
        out.append({**r, "ref": j})
    return out


E.set_games = set_games

if __name__ == "__main__":
    sys.exit(E.main())
