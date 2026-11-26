# No answers are read: uses only numbers already in results/score_ledger.csv.
import csv, json, math, sys
from fractions import Fraction
E1 = math.exp(-1)
rows = list(csv.DictReader(open("results/score_ledger.csv", encoding="utf-8")))
def pick(note, sha):
    return [r for r in rows if r["note"] == note and r["pred_sha256"] == sha][0]
cases = [("rank prereg01 B", "prereg01 H1", "prereg02 H2R (A = prereg01 B, reference re-read)",
          "d42a0b1182f1acf34c35edbd06a1ede3f40e23f90a269a82bbcad25181ae4940", 1589),
         ("player prereg01", "prereg01 player baseline", "prereg02 H2P (A = prereg01 player, reference re-read)",
          "0f5761a11b9f68ab07b2bc91b9650236f85b8cbeff45d459753dbd24459d3ec7", 400)]
print("python", sys.version.split()[0])
for name, n1, n2, sha, n in cases:
    a, b = pick(n1, sha), pick(n2, sha)
    da, db = json.loads(a["details"]), json.loads(b["details"])
    print(f"\n{name}: details identical: {da == db}  {da}")
    print(f"  mean  then {a['mean']}  now {b['mean']}  diff {float(b['mean']) - float(a['mean']):.3e}")
    if a["rank_balanced_mean"]:
        print(f"  rank-balanced then {a['rank_balanced_mean']}  now {b['rank_balanced_mean']}  diff {float(b['rank_balanced_mean']) - float(a['rank_balanced_mean']):.3e}")
    if "exact_rate" in da:
        ne, na = round(da["exact_rate"] * n), round(da["adjacent_rate"] * n)
        vals = [1.0] * ne + [E1] * na + [0.0] * (n - ne - na)
        exact = float((Fraction(ne) + Fraction(E1) * na) / n)
    else:
        # player: rank r scores e^-(r-1); only top1/top5 rates are logged, so the exact sum is not recoverable
        exact = None
    if exact is not None:
        print(f"  correctly rounded mean from the counts ({ne} exact, {na} adjacent): {exact!r}")
        print(f"  now == correctly rounded: {float(b['mean']) == exact}; then == correctly rounded: {float(a['mean']) == exact}")
# 3.12 changed builtin sum() for floats to compensated (Neumaier) summation:
xs = [0.1] * 10
print("\nsum([0.1]*10) =", repr(sum(xs)), " math.fsum =", repr(math.fsum(xs)))
