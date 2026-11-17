import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()
old = '''            todo += [os.path.realpath(h) for h in hits]'''
new = '''            todo += hits                                                                               # the folders are real paths already (code_dirs): no realpath per hit, it is a lstat per path component'''
assert s.count(old) == 1
open(P, "w", encoding="utf-8").write(s.replace(old, new))
