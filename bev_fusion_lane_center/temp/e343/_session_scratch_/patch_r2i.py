import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''        assert i["run"] == "e324_s_s0" and ctl["variants"]["snapped"]["curves"] == 7, "the control card is read at the start"
'''
new = '''        assert i["run"] == "e324_s_s0" and ctl["variants"]["snapped"]["curves"] == 7, "the control card is read at the start"
        seen = []
        with patched((J, "check_seed", lambda run, seed: seen.append((run, seed)))):
            J.precheck(ns(control_card=card), setup, 1, True)
        assert seen == [("e324_s_s1", 1)], "the seed of the pair run is checked in the precheck"
'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
