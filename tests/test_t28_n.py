"""PREREG_28: tools/t28_n.py (06-ON size condition for N x N pools) wraps the frozen tools/t06_player.py.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q tests/test_t28_n.py
"""
import copy
import datetime as dt
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t06_player as t6  # noqa: E402
import t28_n as t28  # noqa: E402

TW = dt.timezone(dt.timedelta(hours=8))
NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=TW)
ORIG = t6.on_conditions


def pools(n=400, sel="V2"):
    rep = {"used06": True, "selected": sel, "switch_allowed_at_this_level": True, "n_questions": n,
           "n_candidates": n, "test02": {"n_questions": n, "n_candidates": n, "guard_level": 1},
           "detector": {"usable": True, "opens": True, "r_forced_cost": 1.2, "r_star": 2.25}}
    return copy.deepcopy(rep), copy.deepcopy(rep)


def t2():
    return {"c_copies": {"player_identification_test_public.csv": 0, "player_identification_test_private.csv": 0},
            "abd_failed": [], "units_present": ["a", "b", "c", "d"], "verdict": "PASS", "fake": False}


def run(pub, priv, aidea_not_o2o=False):
    return t28.wrap(ORIG)(pub, priv, {"secondary_not_negative": True}, t2(), "page read", aidea_not_o2o, NOW)


class SizeRule(unittest.TestCase):
    def test_400_identical_to_t06(self):
        pub, priv = pools(400)
        self.assertEqual(run(pub, priv), ORIG(pub, priv, {"secondary_not_negative": True}, t2(), "page read",
                                               False, NOW))
        pub, priv = pools(400)
        pub["n_candidates"] = 399                  # literal false and not N x N -> false
        c = run(pub, priv)
        self.assertFalse(c[t28.COND]["ok"])
        self.assertFalse(c[t28.COND]["detail"]["literal_400_by_400"]["ok"])

    def test_n_by_n_in_range_on(self):
        for n in (440, 500, 600, 800):
            pub, priv = pools(n)
            c = run(pub, priv)
            self.assertTrue(all(v["ok"] for v in c.values()), (n, c))
            self.assertEqual(sorted(c), sorted(t6.ON_CONDITIONS))
            self.assertEqual(c[t28.COND]["detail"]["prereg28"]["pools"]["private"]["n"], n)
        pub, _ = pools(500)
        _, priv = pools(600)                        # the two pools may differ
        self.assertTrue(run(pub, priv)[t28.COND]["ok"])

    def test_mixed_pools_and_none(self):
        pub, _ = pools(400)
        _, priv = pools(500, sel="V3")
        c = run(pub, priv)
        self.assertTrue(all(v["ok"] for v in c.values()), c)
        d = c[t28.COND]["detail"]
        self.assertEqual(d["prereg28"]["pools"]["public"]["n"], 400)
        self.assertEqual(d["prereg28"]["pools"]["private"]["n"], 500)
        self.assertFalse(d["literal_400_by_400"]["ok"])
        self.assertEqual(sorted(d["sha256_lf"]), sorted([t28.SELF_REL, t28.REGISTRATION, t28.EVAL_REPORT]))
        self.assertFalse(t28.size_rule(None, None)["ok"])
        _, priv = pools(500)
        self.assertFalse(t28.size_rule(None, priv)["ok"])

    def test_400_all_literal_cases_identical(self):
        h = {"secondary_not_negative": True}
        for mut in (lambda r: None, lambda r: r["detector"].update(opens=False), lambda r: r.update(used06=False),
                    lambda r: r["test02"].update(guard_level=2)):
            pub, priv = pools(400)
            mut(priv)
            a = ORIG(copy.deepcopy(pub), copy.deepcopy(priv), h, t2(), "page read", False, NOW)
            self.assertEqual(run(pub, priv), a)

    def test_official_log_accepts_wrapped_on_conditions(self):
        import official_log as ol
        pub, priv = pools(500)
        rep = {"on_conditions": run(pub, priv), "all_on_conditions": True, "on_produced": True}
        probs = ol.t06_report_problems(rep, "x", None, None, "on")
        self.assertFalse([p for p in probs if "06-ON conditions" in p], probs)
        pub, priv = pools(900)
        rep = {"on_conditions": run(pub, priv), "all_on_conditions": True, "on_produced": True}
        probs = ol.t06_report_problems(rep, "x", None, None, "on")
        self.assertTrue([p for p in probs if "06-ON conditions" in p and t28.COND in p], probs)

    def test_out_of_range_or_not_square_off(self):
        for n in (399, 380, 801, 1000):
            pub, priv = pools(n)
            self.assertFalse(run(pub, priv)[t28.COND]["ok"], n)
        cases = [lambda r: r.update(n_candidates=400), lambda r: r.update(n_questions=600),
                 lambda r: r["test02"].update(n_questions=499), lambda r: r["test02"].update(n_candidates=None),
                 lambda r: r.update(n_questions=500.0), lambda r: r.update(n_questions=True)]
        for f in cases:
            pub, priv = pools(500)
            f(priv)
            self.assertFalse(run(pub, priv)[t28.COND]["ok"])
            pub, priv = pools(500)
            f(pub)
            self.assertFalse(run(pub, priv)[t28.COND]["ok"])

    def test_other_conditions_untouched(self):
        pub, priv = pools(500)
        priv["detector"]["opens"] = False
        c = run(pub, priv)
        self.assertTrue(c[t28.COND]["ok"])
        self.assertFalse(c[t28.FIRST]["ok"])
        pub, priv = pools(500)
        self.assertFalse(run(pub, priv, aidea_not_o2o=True)["aidea_page_not_against_one_to_one"]["ok"])
        pub, priv = pools(500)
        pub["test02"]["guard_level"] = 2
        self.assertFalse(run(pub, priv)["both_pools_06_at_guard_level_1"]["ok"])

    def test_v3_rule(self):
        pub, priv = pools(500, sel="V3")
        saved = t28.V3_USABLE_ALL_N
        try:
            t28.V3_USABLE_ALL_N = False
            self.assertFalse(run(pub, priv)[t28.FIRST]["ok"])
            pub, priv = pools(400, sel="V3")
            self.assertTrue(run(pub, priv)[t28.FIRST]["ok"])
            t28.V3_USABLE_ALL_N = True
            pub, priv = pools(500, sel="V3")
            self.assertTrue(run(pub, priv)[t28.FIRST]["ok"])
        finally:
            t28.V3_USABLE_ALL_N = saved

    def test_official_log_names(self):
        import official_log as ol
        self.assertEqual(tuple(ol.T06_ON_CONDITIONS), tuple(t6.ON_CONDITIONS))
        self.assertIn(t28.COND, ol.T06_ON_CONDITIONS)


class Wrapper(unittest.TestCase):
    def test_t06_unchanged_on_disk(self):
        out = subprocess.run(["git", "-C", REPO, "diff", "--name-only", "testset-v1-final", "--",
                              "tools/t06_player.py"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "")

    def test_submit_reaches_t06_with_the_wrapped_rule(self):
        import tempfile
        d = tempfile.mkdtemp(prefix="c10_t28_")
        try:
            code = t28.main(["submit", "--dry-run", "--out-dir", os.path.join(d, "o6"), "--t02-results-dir",
                             os.path.join(d, "t02"), "--tests-dir", os.path.join(d, "tests")])
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            wrapped = t6.on_conditions
            t6.on_conditions = ORIG
        self.assertEqual(code, 2)                       # t06 refuses the empty tree as usual
        t6.on_conditions = wrapped                       # a second main() does not wrap twice
        try:
            t28.main(["submit", "--dry-run", "--out-dir", os.path.join(d, "o6")])
            self.assertIs(t6.on_conditions, wrapped)
        finally:
            t6.on_conditions = ORIG
        self.assertTrue(getattr(wrapped, "prereg28", False))
        self.assertIs(t6.cmd_submit.__globals__, vars(t6))

    def test_real_mode_needs_committed_files(self):
        self.assertEqual(t28.committed_clean([t28.SELF_REL, t28.REGISTRATION]), [])
        self.assertEqual(t28.committed_clean(["tools/no_such_file_t28.py"]), ["tools/no_such_file_t28.py"])

    def test_only_submit(self):
        self.assertEqual(t28.main(["apply", "--pool", "public"]), 2)
        self.assertEqual(t28.main([]), 2)

    def test_constants(self):
        import json
        rep = json.load(open(os.path.join(REPO, t28.EVAL_REPORT), encoding="utf-8"))["judgement"]
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["V3_usable_at_every_N"], t28.V3_USABLE_ALL_N)
        self.assertEqual((t28.N_MIN, t28.N_MAX), (400, 800))
        self.assertEqual(t6.N_OFFICIAL, 400)


if __name__ == "__main__":
    unittest.main()
