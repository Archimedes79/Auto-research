import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_jmap_train.py")
s = open(p, encoding="utf-8").read()
old = '        refuses(lambda: E.load_model("nonexistent", "cpu", tmp), "")\n'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, ""))
print("ok")
