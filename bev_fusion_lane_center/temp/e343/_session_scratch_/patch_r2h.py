import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        for rel in J.PROTECTED + ("temp/e343",):
            os.makedirs(os.path.join(d, rel, "test"))'''
new = '''    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "Mixed_Case_Root")                                                        # the real root has capitals too (Auto-research): both sides of the comparison are case-folded
        os.makedirs(d)
        _write_helper_checks(d)


def _write_helper_checks(d):
    with patched((J, "ROOT", d)):
        for rel in J.PROTECTED + ("temp/e343",):
            os.makedirs(os.path.join(d, rel, "test"))'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
