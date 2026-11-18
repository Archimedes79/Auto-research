import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "landing", "land_patch.py")
s = open(p, encoding="utf-8").read()
bad = 'fh.write("\n".join(sorted(changed)) + "\n")'
assert s.count(bad) == 1
s = s.replace(bad, 'fh.write(chr(10).join(sorted(changed)) + chr(10))')
open(p, "w", encoding="utf-8").write(s)
print("fixed")
