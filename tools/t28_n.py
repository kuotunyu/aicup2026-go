"""PREREG_28 (competition 1): 06-ON for N x N pools, 400 <= N <= 800. A wrapper around the frozen tools/t06_player.py;
it never edits it. Rules: docs/PREREG_28_06ON_N_BY_N.md.

    PY tools/t28_n.py submit <the same arguments as tools/t06_player.py submit>

Runs t06_player.py submit unchanged except for the PREREG_06 5.3 size condition ('pools_400_by_400'):
* the literal holds (both pools 400 x 400 by 06's and test02's counts) -> t06's own entry is kept unchanged, so at
  N = 400 everything t06 writes is the same as `t06_player.py submit`;
* otherwise the entry is replaced by the PREREG_28 rule: in each pool 06's question and candidate counts and test02's
  question and candidate counts are one and the same N with N_MIN <= N <= N_MAX (the two pools may differ). The entry
  keeps its name (tools/official_log.py checks the names) and its detail records the literal, the rule and this tool.
* V3_USABLE_ALL_N (prereg28 eval): False -> a pool with N != 400 whose selected combination is V3 never opens (the
  first condition is set false). Every other condition, the detector, the frozen thresholds and the public veto are
  t06's own, unchanged.
Only 'submit' is wrapped; feats / seam / apply / veto / status are run with tools/t06_player.py itself.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t06_player as t6  # noqa: E402

N_MIN, N_MAX = 400, 800
V3_USABLE_ALL_N = True        # prereg28 eval_report.json: judgement.V3_usable_at_every_N
REGISTRATION = "docs/PREREG_28_06ON_N_BY_N.md"
EVAL_REPORT = "results/prereg28/explore/eval_report.json"
COND = "pools_400_by_400"
FIRST = "private_switch_usable_and_detector_open"
SELF_REL = "tools/t28_n.py"


def _sha_lf(rel):
    with open(os.path.join(REPO, rel), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def size_rule(pub, priv):
    """PREREG_28 2: per pool, 06's and test02's question / candidate counts are one N in [N_MIN, N_MAX]."""
    pools = {}
    for name, r in (("public", pub), ("private", priv)):
        r = r or {}
        t02 = r.get("test02") or {}
        nums = [r.get("n_questions"), r.get("n_candidates"), t02.get("n_questions"), t02.get("n_candidates")]
        ints = all(isinstance(x, int) and not isinstance(x, bool) for x in nums)
        same = ints and len(set(nums)) == 1
        n = nums[0] if same else None
        pools[name] = {"counts": nums, "n": n, "ok": bool(same and N_MIN <= n <= N_MAX)}
    return {"ok": all(p["ok"] for p in pools.values()), "pools": pools, "range": [N_MIN, N_MAX],
            "rule": "each pool: 06's and test02's question and candidate counts are one N, %d <= N <= %d (PREREG_28)"
                    % (N_MIN, N_MAX)}


def wrap(orig):
    def on_conditions(pub, priv, h6p, t2s, aidea_checked, aidea_not_o2o, now):
        out = orig(pub, priv, h6p, t2s, aidea_checked, aidea_not_o2o, now)
        lit = out[COND]
        if lit["ok"]:
            return out                          # 400 x 400: t06's own result, byte for byte
        sz = size_rule(pub, priv)
        out[COND] = {"ok": sz["ok"], "detail": {"literal_400_by_400": lit, "prereg28": sz,
                                                "registration": REGISTRATION, "tool": SELF_REL,
                                                "sha256_lf": {rel: _sha_lf(rel) for rel in
                                                              (SELF_REL, REGISTRATION, EVAL_REPORT)}}}
        if sz["ok"] and not V3_USABLE_ALL_N and (priv or {}).get("selected") == "V3" \
                and sz["pools"]["private"]["n"] != 400:
            first = dict(out[FIRST])
            first["ok"] = False
            first["prereg28"] = "V3 is not usable at every simulated N (prereg28 eval): no switch at N != 400"
            out[FIRST] = first
        return out
    on_conditions.prereg28 = True
    return on_conditions


def committed_clean(rels):
    bad = []
    for rel in rels:
        st_ = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", rel], capture_output=True,
                             text=True).stdout.strip()
        tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--", rel], capture_output=True,
                                 text=True).stdout.strip()
        if st_ or not tracked:
            bad.append(rel)
    return bad


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "submit":
        print("t28_n: only 'submit' is wrapped; run the other commands with tools/t06_player.py (exit 2)",
              file=sys.stderr)
        return 2
    if "--dry-run" not in argv:
        bad = committed_clean([SELF_REL, REGISTRATION, EVAL_REPORT])
        if bad:
            print("t28_n: not committed and clean: %s (exit 2)" % bad, file=sys.stderr)
            return 2
    if not getattr(t6.on_conditions, "prereg28", False):
        t6.on_conditions = wrap(t6.on_conditions)
    return t6.main(argv)


if __name__ == "__main__":
    sys.exit(main())
