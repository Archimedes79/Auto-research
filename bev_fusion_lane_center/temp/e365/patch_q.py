import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = '''for name, extra in (("p0", []), ("p3", ["--token-drop", "0.3"])):'''
assert s.count(a) == 1
s = s.replace(a, '''for name, extra in (("p0", []), ("p3", ["--token-drop", "0.3"]), ("w5", ["--none-weight", "0.5"])):''')
a = '''assert math.isfinite(rows["p3"]) and rows["p3"] != rows["p0"], rows'''
assert s.count(a) == 1
s = s.replace(a, '''assert all(math.isfinite(v) for v in rows.values()) and rows["p3"] != rows["p0"] and rows["w5"] != rows["p0"], rows''')
open(p, "w", encoding="utf-8").write(s)
