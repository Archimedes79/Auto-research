import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''    with patched((J, "pyramid_same", lambda *x: False)):
        assert raises(SystemExit, J.precheck, ns(control_card=card), setup, 0, True), "arm k fails the pyramid check: stop, to be run again with --no-stack"
        assert J.precheck(ns(control_card=card), setup, 0, False)[0]["run"] == "e324_s_s0", "--no-stack does not look at the pyramid"
'''
new = '''        with patched((J, "pyramid_same", lambda *x: False)):
            assert raises(SystemExit, J.precheck, ns(control_card=card), setup, 0, True), "arm k fails the pyramid check: stop, to be run again with --no-stack"
            assert J.precheck(ns(control_card=card), setup, 0, False)[0]["run"] == "e324_s_s0", "--no-stack does not look at the pyramid"
'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
