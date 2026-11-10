import re
p = "bev_fusion/evaluation/README.md"
s = open(p, encoding="utf-8").read()
m = re.search(r"It replaced `temp/t4/compare_front.py` \([^)]*\)\.?", s)
assert m, "pattern"
print("replacing:", m.group(0))
s = s.replace(m.group(0), "It replaced the scratch script `compare_front.py`.", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
