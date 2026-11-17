import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3/patch_tl.py")
s = open(p, encoding="utf-8").read()
a = s.index("sub('''    (\"a duplicate is near any node\",''', '''")
b = s.index("open(p, \"w\"")
s = s[:a] + "@@MORE@@\n" + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
