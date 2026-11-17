import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''        real = J.code_sha256()
'''
new = '''        real = J.code_sha256(entry=os.path.join(J.ROOT, "evaluation", "jtest_eval.py"))                      # the real tree, from the real file (a copy of the module elsewhere must give the same keys)
'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
