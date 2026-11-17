import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_lane_map.py")
s = open(p, encoding="utf-8").read()
old = '''("the unknown class is 10 % and the absent class 8 %", '("unknown", 0.08), ("absent", 0.10)', '("unknown", 0.10), ("absent", 0.08)'),'''
new = '''("the unknown class is 10 % and the absent class 8 %", '("unknown", 0.08), ("absent", 0.10), ("gross", 0.10))     # the draw', '("unknown", 0.10), ("absent", 0.08), ("gross", 0.10))     # the draw'),'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
