import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(p, encoding="utf-8").read()
old = '''and all(v in names for v in st["arms"].values()) and set(st["stored"]) <= set(names)'''
new = '''and all(v in names for v in st["arms"].values()) and set(st["stored"]) <= set(names) and set(st["stored_counts"]) <= set(st["stored"])'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
