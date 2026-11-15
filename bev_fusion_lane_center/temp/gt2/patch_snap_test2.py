p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = "ln((9.6, 2.2), (19.6, 2.2))]"
assert old in s
s = s.replace(old, "ln((8.7, 2.3), (18.7, 2.3))]", 1)
s = s.replace("# 6 heading +x at (9.6, 2.2) (wrong direction)", "# 6 heading +x at (8.7, 2.3) (0.2 m: the nearest, wrong direction)")
old = "and (0, 7) not in ra[\"edges\"] and (2, 7) not in ra[\"edges\"], ra[\"edges\"]"
assert old in s
s = s.replace(old, "and (0, 7) not in ra[\"edges\"] and (2, 7) not in ra[\"edges\"] and (7, 6) not in ra[\"edges\"], ra[\"edges\"]", 1)
open(p, "w", encoding="utf-8").write(s)
